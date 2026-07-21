"""
End-to-end redubbing pipeline orchestrator.
Yields progress log lines (str) for Gradio streaming, and returns final paths.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

from .asr import transcribe
from .translation import translate_segments
from .tts import build_voice_reference, synthesize_segments
from .audio_assembly import (
    extract_audio,
    extract_reference_clip,
    assemble_audio_track,
    mux_audio_into_video,
    _video_duration,
)
from .output_writers import write_transcript_json, write_srt


# How long a reference voice clip to extract (seconds)
REFERENCE_CLIP_DURATION = 12.0
# Minimum length of transcript to use as reference text
MAX_REFERENCE_TEXT_CHARS = 300


def run_pipeline(
    video_path: str,
    source_language: str | None,
    target_language: str,
    output_dir: str,
) -> Generator[str | dict, None, None]:
    """
    Generator that yields strings (progress messages) and finally a dict of output paths:
      {
        "video": str,
        "transcript": str,
        "srt": str,
      }
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    yield "Extracting audio from video…"
    audio_wav = str(work / "audio.wav")
    extract_audio(video_path, audio_wav)
    yield "Audio extracted."

    yield "Transcribing with Fish Audio ASR…"
    lang = source_language if source_language and source_language != "auto" else None
    asr_result = transcribe(audio_wav, language=lang)
    segments = asr_result.get("segments", [])
    total_duration = asr_result.get("duration") or _video_duration(video_path)

    if not segments:
        raise ValueError("ASR returned no segments. Check that the video has audible speech.")

    yield f"Transcribed {len(segments)} segments ({total_duration:.1f}s total)."

    yield f"Translating to {target_language}…"
    segments = translate_segments(segments, target_language, source_language)
    yield "Translation complete."

    yield "Extracting voice reference clip…"
    ref_wav = str(work / "reference.wav")
    # Try to start reference a bit in (skip potential intro silence)
    ref_start = min(2.0, segments[0]["start"]) if segments else 0.0
    extract_reference_clip(audio_wav, ref_wav, start=ref_start, duration=REFERENCE_CLIP_DURATION)
    ref_audio_bytes, _ = build_voice_reference(ref_wav)

    # Use beginning of transcript as reference text hint
    ref_text = " ".join(s["text"] for s in segments)[:MAX_REFERENCE_TEXT_CHARS]
    yield "Voice reference ready."

    yield f"Synthesizing TTS for {len(segments)} segments…"
    tts_dir = str(work / "tts")
    segments = synthesize_segments(
        segments,
        reference_audio_bytes=ref_audio_bytes,
        reference_text=ref_text,
        output_dir=tts_dir,
    )
    done_count = sum(1 for s in segments if s.get("tts_path"))
    yield f"TTS done ({done_count}/{len(segments)} segments synthesized)."

    yield "Assembling dubbed audio track…"
    dubbed_wav = str(work / "dubbed_audio.wav")
    assemble_audio_track(
        segments=segments,
        original_audio_path=audio_wav,
        total_duration=total_duration,
        output_path=dubbed_wav,
        work_dir=str(work / "assembly"),
    )
    yield "Audio assembly complete."

    yield "Muxing audio into video…"
    out_video = str(out / "redubbed.mp4")
    mux_audio_into_video(video_path, dubbed_wav, out_video)
    yield "Video muxed."

    yield "Writing transcript and subtitles…"
    transcript_path = str(out / "transcript.json")
    srt_path = str(out / "translated.srt")
    write_transcript_json(segments, transcript_path)
    write_srt(segments, srt_path)
    yield "All output files written."

    yield {
        "video": out_video,
        "transcript": transcript_path,
        "srt": srt_path,
    }
