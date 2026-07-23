"""
Subprocess worker: speaker diarization via pyannote.audio.
Usage: python _diarization_worker.py <audio_path> <output_json>

Running in a separate process means:
  - pyannote + lightning + torch._dynamo are never loaded in the main process
  - All memory is freed automatically when this process exits
  - LazyModule import errors are isolated here and patched before any import
"""

import json
import os
import sys
import types


def _silence_speechbrain_lazy_errors() -> None:
    """
    speechbrain 1.x wraps optional integrations (k2, wordemb, etc.) in LazyModule
    objects. Accessing any attribute on a LazyModule whose package isn't installed
    raises ImportError via ensure_module(). This happens even for innocent hasattr()
    checks made by inspect.getmodule() inside torch._dynamo during `import pyannote`.

    Fix: patch LazyModule.ensure_module to return an empty stub on failure instead
    of raising. Must run before any pyannote / lightning / speechbrain usage.
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:
        return  # speechbrain not installed — nothing to patch

    original_ensure = LazyModule.ensure_module

    def _safe_ensure(self, stacklevel=0):
        # Return already-loaded module if available
        lazy_mod = getattr(self, "lazy_module", None)
        if lazy_mod is not None:
            return lazy_mod
        try:
            return original_ensure(self, stacklevel)
        except Exception:
            target = str(getattr(self, "target", None) or "speechbrain._optional_stub")
            stub = types.ModuleType(target)
            stub.__file__ = None
            stub.__spec__ = None
            # Store result so repeated access doesn't retry the failed import
            object.__setattr__(self, "lazy_module", stub)
            sys.modules[target] = stub
            return stub

    LazyModule.ensure_module = _safe_ensure

    # DeprecatedModuleRedirect is a subclass that also has ensure_module;
    # patching the base class covers it via MRO, but force it just in case.
    try:
        from speechbrain.utils.importutils import DeprecatedModuleRedirect
        if DeprecatedModuleRedirect.ensure_module is not LazyModule.ensure_module:
            DeprecatedModuleRedirect.ensure_module = _safe_ensure
    except Exception:
        pass


def _bypass_plda() -> None:
    """
    Patch get_plda to return None so pipeline init skips community-model download.
    pyannote/speaker-diarization-community-1 is a gated repo; without PLDA the
    pipeline uses cosine similarity instead (slightly lower speaker-confusion accuracy).

    Must patch both the getter module AND the local binding in speaker_diarization,
    since Python's 'from x import y' creates a separate name binding per module.
    """
    noop = lambda *a, **kw: None  # noqa: E731
    # Patch source module
    try:
        import pyannote.audio.pipelines.utils.getter as _g
        _g.get_plda = noop
    except Exception:
        pass
    # Patch the already-imported local name in speaker_diarization
    try:
        import pyannote.audio.pipelines.speaker_diarization as _sd
        _sd.get_plda = noop
    except Exception:
        pass


def _load_pipeline(hf_token: str):
    """Load pyannote pipeline, falling back gracefully if PLDA model is gated."""
    from pyannote.audio import Pipeline

    def _pretrained(**kwargs):
        try:
            return Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=hf_token, **kwargs
            )
        except TypeError:
            return Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token, **kwargs
            )

    try:
        return _pretrained()
    except Exception as exc:
        if "speaker-diarization-community-1" in str(exc) or "GatedRepo" in type(exc).__name__:
            print(
                "NOTE: pyannote/speaker-diarization-community-1 is not accessible "
                "(gated repo — you haven't accepted its terms yet).\n"
                "For best speaker-separation accuracy, visit:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "Proceeding without PLDA (cosine-similarity fallback — works fine for "
                "most dialogue).",
                file=sys.stderr,
            )
            _bypass_plda()
            return _pretrained()
        raise


def _as_annotation(result):
    """
    Return an object with .itertracks(yield_label=True) and .get_overlap().

    pyannote 3.x pipelines return pyannote.core.Annotation directly.
    pyannote 4.x may return a DiarizeOutput wrapper — unwrap it.
    """
    if hasattr(result, "itertracks"):
        return result
    # pyannote 4.x DiarizeOutput wraps the Annotation in .speaker_diarization
    # (also try other common attribute names for forward-compatibility)
    for attr in (
        "speaker_diarization",           # pyannote 4.x DiarizeOutput
        "exclusive_speaker_diarization", # alternative 4.x field
        "_annotation", "annotation",     # legacy / other wrappers
        "diarization", "_diarization",
    ):
        ann = getattr(result, attr, None)
        if ann is not None and hasattr(ann, "itertracks"):
            return ann
    # Last resort: maybe it IS iterable in a different way; give a useful error
    attrs = [a for a in dir(result) if not a.startswith("_")]
    raise RuntimeError(
        f"Cannot extract Annotation from {type(result).__name__}. "
        f"Public attributes: {attrs}"
    )


def _run_diarization(audio_path: str) -> dict:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set in environment")

    pipeline = _load_pipeline(hf_token)

    raw = pipeline(audio_path)
    diarization = _as_annotation(raw)

    segments = [
        {"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk}
        for seg, _, spk in diarization.itertracks(yield_label=True)
    ]

    overlap_regions: list[dict] = []
    try:
        for seg in diarization.get_overlap():
            if seg.duration < 0.3:
                continue
            active = sorted({
                spk
                for seg2, _, spk in diarization.itertracks(yield_label=True)
                if seg2.overlaps(seg)
            })
            overlap_regions.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speakers": active,
            })
    except Exception:
        pass

    # Fallback overlap detection from raw segments
    if not overlap_regions:
        sorted_segs = sorted(segments, key=lambda s: s["start"])
        for i, a in enumerate(sorted_segs):
            for b in sorted_segs[i + 1:]:
                if b["start"] >= a["end"]:
                    break
                if a["speaker"] == b["speaker"]:
                    continue
                ovlp_start = max(a["start"], b["start"])
                ovlp_end = min(a["end"], b["end"])
                if ovlp_end - ovlp_start >= 0.3:
                    overlap_regions.append({
                        "start": round(ovlp_start, 3),
                        "end": round(ovlp_end, 3),
                        "speakers": sorted({a["speaker"], b["speaker"]}),
                    })

    return {
        "segments": segments,
        "overlap_regions": overlap_regions,
        "num_speakers": len({s["speaker"] for s in segments}),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: _diarization_worker.py <audio_path> <output_json>", file=sys.stderr)
        sys.exit(1)

    audio_path, output_path = sys.argv[1], sys.argv[2]

    # Load .env if HF_TOKEN wasn't inherited from parent (e.g. standalone test runs)
    if not os.environ.get("HF_TOKEN"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

    # Patch BEFORE any pyannote / lightning / speechbrain import
    _silence_speechbrain_lazy_errors()

    try:
        result = _run_diarization(audio_path)
    except Exception as exc:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(3)

    with open(output_path, "w") as fh:
        json.dump(result, fh)

    print(
        f"OK: {result['num_speakers']} speaker(s), "
        f"{len(result['segments'])} segments, "
        f"{len(result['overlap_regions'])} overlap region(s)",
        file=sys.stderr,
    )
