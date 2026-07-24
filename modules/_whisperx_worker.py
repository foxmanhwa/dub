"""
WhisperX subprocess worker — transcription + word-alignment + speaker diarization.
Usage: python _whisperx_worker.py <audio_path> <result_json> [language]

Runs in a subprocess so heavy ML models (whisper, pyannote, torch) are freed
automatically when this process exits, before translation starts.
"""

import json
import os
import sys
import types


def _silence_speechbrain_lazy_errors() -> None:
    """
    Same patch as in _diarization_worker.py — silence speechbrain LazyModule
    errors that fire when pyannote imports speechbrain optional integrations.
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:
        return

    original_ensure = LazyModule.ensure_module

    def _safe_ensure(self, stacklevel=0):
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
            object.__setattr__(self, "lazy_module", stub)
            sys.modules[target] = stub
            return stub

    LazyModule.ensure_module = _safe_ensure

    try:
        from speechbrain.utils.importutils import DeprecatedModuleRedirect
        if DeprecatedModuleRedirect.ensure_module is not LazyModule.ensure_module:
            DeprecatedModuleRedirect.ensure_module = _safe_ensure
    except Exception:
        pass


def _run(audio_path: str, result_path: str, language: str | None) -> None:
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_name = os.environ.get("WHISPERX_MODEL", "large-v2")
    hf_token = os.environ.get("HF_TOKEN")

    print(
        f"[whisperx] device={device}  compute_type={compute_type}  model={model_name}",
        file=sys.stderr,
    )

    # ── 1. Transcription ──────────────────────────────────────────────────────
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,  # None → auto-detect
    )
    audio = whisperx.load_audio(audio_path)
    batch_size = 16 if device == "cuda" else 4
    result = model.transcribe(audio, batch_size=batch_size)
    del model  # free GPU/CPU memory before alignment

    detected_language = result.get("language") or language or "en"
    print(f"[whisperx] detected language: {detected_language}", file=sys.stderr)

    # ── 2. Word-level alignment ───────────────────────────────────────────────
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        del model_a
        print("[whisperx] word-level alignment complete", file=sys.stderr)
    except Exception as exc:
        print(
            f"[whisperx] alignment failed ({exc}) — keeping segment-level timestamps",
            file=sys.stderr,
        )

    # ── 3. Speaker diarization (optional — requires HF_TOKEN) ────────────────
    speakers_assigned = False
    if hf_token:
        try:
            print("[whisperx] running speaker diarization…", file=sys.stderr)
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segs = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segs, result)
            speakers_assigned = True
            print("[whisperx] speaker assignment complete", file=sys.stderr)
        except Exception as exc:
            print(
                f"[whisperx] diarization failed ({exc}) — skipping speaker assignment",
                file=sys.stderr,
            )

    # ── 4. Normalise output ───────────────────────────────────────────────────
    # audio is a 1-D float32 array at 16 kHz
    duration = float(len(audio)) / 16000.0

    segments = []
    for seg in result.get("segments", []):
        s: dict = {
            "text": seg.get("text", "").strip(),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
        }
        if speakers_assigned and "speaker" in seg:
            s["speaker"] = seg["speaker"]
        segments.append(s)

    output = {
        "text": " ".join(s["text"] for s in segments),
        "duration": duration,
        "segments": segments,
        "language": detected_language,
        "speakers_assigned": speakers_assigned,
    }

    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False)

    print(
        f"[whisperx] wrote {len(segments)} segments  "
        f"speakers_assigned={speakers_assigned}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: _whisperx_worker.py <audio_path> <result_json> [language]",
            file=sys.stderr,
        )
        sys.exit(1)

    _audio_path = sys.argv[1]
    _result_path = sys.argv[2]
    _language = sys.argv[3] if len(sys.argv) > 3 else None
    if _language in ("auto", ""):
        _language = None

    # Load .env if HF_TOKEN wasn't inherited (e.g. standalone test runs)
    if not os.environ.get("HF_TOKEN"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

    # Must patch before any pyannote / speechbrain import
    _silence_speechbrain_lazy_errors()

    try:
        _run(_audio_path, _result_path, _language)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
