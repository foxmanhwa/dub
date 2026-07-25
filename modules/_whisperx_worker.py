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


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _probe_device() -> str:
    """
    Return "cuda" if available AND this PyTorch build can dispatch kernels on
    the detected GPU.  Falls back to "cpu" on any CUDA error so the pipeline
    never hard-crashes due to a compute-capability mismatch (e.g. Blackwell
    sm_120 with a cu121 build that only supports up to sm_90).
    """
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.zeros(1, device="cuda") + torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as exc:
        _log(
            f"[whisperx] CUDA kernel dispatch failed: {exc}\n"
            "  Falling back to CPU — transcription will be slower.\n"
            "  To fix: reinstall PyTorch with a newer CUDA build:\n"
            "    pip install torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
        return "cpu"


def _run(audio_path: str, result_path: str, language: str | None) -> None:
    import torch
    import whisperx

    device = _probe_device()
    compute_type = "float16" if device == "cuda" else "int8"

    # CPU default is "small" — large-v2 on CPU can take 30+ minutes for a short
    # clip and is rarely worth the wait without a GPU.  GPU users get large-v2.
    # Override either way with WHISPERX_MODEL in .env.
    _default_model = "large-v2" if device == "cuda" else "small"
    model_name = os.environ.get("WHISPERX_MODEL", _default_model)
    hf_token = os.environ.get("HF_TOKEN")

    # batch_size=1 is optimal on CPU — larger batches don't parallelise on CPU
    # and waste RAM without speed benefit.
    batch_size = 16 if device == "cuda" else 1

    _log(
        f"[whisperx] device={device}  model={model_name}  "
        f"compute_type={compute_type}  batch_size={batch_size}"
    )
    if device == "cpu":
        _log(
            "[whisperx] Running on CPU — typical time for a 1-2 min clip: "
            "~3-5 min with 'small', 30+ min with 'large-v2'. "
            "Set WHISPERX_MODEL=small or ASR_BACKEND=fish to override."
        )

    # ── 1. Transcription ──────────────────────────────────────────────────────
    _log(f"[whisperx] Loading model '{model_name}'… (first run downloads weights)")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,  # None → auto-detect
    )
    _log("[whisperx] Model loaded.")

    audio = whisperx.load_audio(audio_path)
    audio_duration = float(len(audio)) / 16000.0
    _log(f"[whisperx] Transcribing {audio_duration:.1f}s of audio…")
    result = model.transcribe(audio, batch_size=batch_size)
    n_raw = len(result.get("segments", []))
    _log(f"[whisperx] Transcription complete: {n_raw} raw segment(s).")
    del model  # free GPU/CPU memory before alignment

    detected_language = result.get("language") or language or "en"
    _log(f"[whisperx] Detected language: {detected_language}")

    # ── 2. Word-level alignment ───────────────────────────────────────────────
    _log("[whisperx] Loading alignment model…")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
        _log("[whisperx] Aligning words…")
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        del model_a
        _log("[whisperx] Word-level alignment complete.")
    except Exception as exc:
        _log(
            f"[whisperx] Alignment failed ({exc}) — keeping segment-level timestamps."
        )

    # ── 3. Speaker diarization (optional — requires HF_TOKEN) ────────────────
    speakers_assigned = False
    if hf_token:
        _log("[whisperx] Running speaker diarization (pyannote)…")
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            # Pass a pre-converted tensor dict to bypass any file-path audio
            # loading inside pyannote (same fix as _diarization_worker.py).
            # audio is numpy float32 [samples] at 16 kHz from whisperx.load_audio().
            _audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
            diarize_segs = diarize_model({"waveform": _audio_tensor, "sample_rate": 16000})
            result = whisperx.assign_word_speakers(diarize_segs, result)
            speakers_assigned = True
            _log("[whisperx] Speaker assignment complete.")
        except Exception as exc:
            _log(
                f"[whisperx] Diarization failed ({exc}) — skipping speaker assignment."
            )
    else:
        _log("[whisperx] No HF_TOKEN — speaker diarization skipped.")

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
