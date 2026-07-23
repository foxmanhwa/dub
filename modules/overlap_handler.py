"""
Overlap region processing: separate → ASR each stream → translate → TTS → mix.
Called by pipeline.py for each detected overlap window.
"""

import subprocess
from pathlib import Path

from .asr import transcribe
from .translation import translate_chunk
from .tts import synthesize_segments


def process_overlap_region(
    region: dict,
    original_audio_path: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
    ref_id: str | None,
    target_language: str,
    source_language: str | None,
    model: str,
    work_dir: str,
    region_idx: int,
    content_context: str | None = None,
) -> dict:
    """
    Process one overlap window end-to-end.

    Returns:
      {
        "start": float,
        "end":   float,
        "mixed_tts_path": str | None,   # ready to drop into assembly
        "speaker_data":   list[dict],   # per-stream ASR+translation info
        "error":          str | None,
      }
    """
    out = Path(work_dir) / f"overlap_{region_idx:03d}"
    out.mkdir(parents=True, exist_ok=True)

    start = region["start"]
    end = region["end"]
    duration = end - start

    result: dict = {
        "start": start,
        "end": end,
        "mixed_tts_path": None,
        "speaker_data": [],
        "error": None,
    }

    try:
        # ── 1. Extract the overlap window from the original mix ───────────────
        chunk_path = str(out / "chunk.wav")
        _extract(original_audio_path, chunk_path, start, duration)

        # ── 2. Separate into two streams ──────────────────────────────────────
        from .separation import separate_speakers
        path_a, path_b = separate_speakers(chunk_path, str(out / "streams"))

        tts_paths: list[str] = []

        for stream_label, stream_path in [("A", path_a), ("B", path_b)]:
            stream_out = out / f"stream_{stream_label}"
            stream_out.mkdir(exist_ok=True)

            # ── 3. ASR on each separated stream ──────────────────────────────
            try:
                lang = source_language if source_language and source_language != "auto" else None
                asr = transcribe(stream_path, language=lang)
                text = asr.get("text", "").strip()
            except Exception as e:
                text = ""
                result["speaker_data"].append({
                    "stream": stream_label,
                    "text": "",
                    "translated_text": "",
                    "tts_path": None,
                    "asr_error": str(e),
                })
                continue

            if not text:
                result["speaker_data"].append({
                    "stream": stream_label,
                    "text": "",
                    "translated_text": "",
                    "tts_path": None,
                })
                continue

            # ── 4. Translate ──────────────────────────────────────────────────
            fake_seg = [{"text": text, "start": 0.0, "end": duration}]
            try:
                translated = translate_chunk(
                    chunk=fake_seg,
                    chunk_offset=0,
                    target_language=target_language,
                    source_language=source_language,
                    model=model,
                    content_context=content_context,
                )
                translated_text = (translated[0].get("translated_text") or "").strip()
            except Exception:
                translated_text = text   # fall back to source text

            # ── 5. TTS ────────────────────────────────────────────────────────
            tts_path = None
            if translated_text:
                tts_seg = [{
                    "text": text,
                    "translated_text": translated_text,
                    "start": 0.0,
                    "end": duration,
                }]
                try:
                    tts_result = synthesize_segments(
                        tts_seg,
                        reference_audio_bytes=ref_audio_bytes,
                        reference_text=ref_text,
                        reference_id=ref_id,
                        output_dir=str(stream_out / "tts"),
                    )
                    tts_path = tts_result[0].get("tts_path") if tts_result else None
                except Exception:
                    tts_path = None

            result["speaker_data"].append({
                "stream": stream_label,
                "text": text,
                "translated_text": translated_text,
                "tts_path": tts_path,
            })
            if tts_path:
                tts_paths.append(tts_path)

        # ── 6. Mix the separated TTS streams ─────────────────────────────────
        if len(tts_paths) >= 2:
            mixed = str(out / "mixed.wav")
            _mix(tts_paths[0], tts_paths[1], mixed, duration)
            result["mixed_tts_path"] = mixed
        elif len(tts_paths) == 1:
            # Only one stream had audible speech — use it directly
            result["mixed_tts_path"] = tts_paths[0]

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _extract(src: str, dst: str, start: float, duration: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-ar", "44100", "-ac", "1", dst],
        check=True, capture_output=True,
    )


def _mix(path_a: str, path_b: str, out: str, target_dur: float) -> None:
    """Mix two clips at -3 dB each into a single WAV of exactly target_dur seconds."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", path_a, "-i", path_b,
         "-filter_complex",
         (
             "[0]volume=0.7[va];"
             "[1]volume=0.7[vb];"
             f"[va][vb]amix=inputs=2:duration=longest:normalize=0,"
             f"apad=whole_dur={target_dur:.3f}"
         ),
         "-ar", "44100", "-ac", "1",
         "-t", f"{target_dur:.3f}",
         out],
        check=True, capture_output=True,
    )
