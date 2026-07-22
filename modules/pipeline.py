"""
End-to-end redubbing pipeline orchestrator.
Yields progress log strings for Gradio streaming, then yields a final dict
of output paths: {"video": str, "transcript": str, "srt": str}.
"""

from pathlib import Path
from typing import Generator

from .asr import transcribe
from .segment_merger import merge_segments
from .translation import translate_chunk, _tail_context, CHUNK_SIZE, back_translate_segments
from .tts import build_voice_reference, synthesize_segments
from .audio_assembly import (
    extract_audio,
    extract_reference_clip,
    assemble_audio_track,
    mux_audio_into_video,
    get_duration,
)
from .output_writers import write_transcript_json, write_srt, write_timing_csv, write_backtrans_csv
from .timing_report import build_timing_report, timing_report_to_df, backtrans_to_df


REFERENCE_CLIP_DURATION = 12.0
MAX_REFERENCE_TEXT_CHARS = 300
MIN_CLONE_DURATION = 15.0  # seconds below which clone quality is likely poor


def run_pipeline(
    video_path: str,
    source_language: str | None,
    target_language: str,
    output_dir: str,
    voice_mode: str = "clone",          # "clone" | "library" | "saved"
    voice_id: str | None = None,        # Fish model id (library) or saved voice name (saved)
    run_backtranslation: bool = False,  # QA: back-translate to English for drift detection
) -> Generator[str | dict, None, None]:
    """
    Generator yielding str (progress) then dict (output paths) at the end.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    # ── Step 1: extract audio ─────────────────────────────────────────────────
    yield "Extracting audio from video…"
    audio_wav = str(work / "audio.wav")
    extract_audio(video_path, audio_wav)
    yield "Audio extracted."

    # ── Step 2: ASR transcription ─────────────────────────────────────────────
    yield "Transcribing with Fish Audio ASR…"
    lang = source_language if source_language and source_language != "auto" else None
    asr_result = transcribe(audio_wav, language=lang)
    segments = asr_result.get("segments", [])
    total_duration = asr_result.get("duration") or get_duration(video_path)

    if not segments:
        raise ValueError(
            "ASR returned no segments. Ensure the video contains audible speech."
        )

    yield f"Transcribed {len(segments)} segments ({total_duration:.1f} s total)."

    # ── Step 2b: merge word-level fragments into phrase-level groups ──────────
    raw_count = len(segments)
    segments = merge_segments(segments)
    yield f"Merged {raw_count} fragments → {len(segments)} phrase-level segments."

    # ── Step 3: translation (chunked so each call finishes in bounded time) ──
    import os as _os
    model = _os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    total_segs = len(segments)
    n_chunks = (total_segs + CHUNK_SIZE - 1) // CHUNK_SIZE
    yield f"Translating {total_segs} segments to {target_language} ({n_chunks} chunk(s) of ≤{CHUNK_SIZE})…"

    translated: list[dict] = []
    prev_context: list[str] = []
    for chunk_start in range(0, total_segs, CHUNK_SIZE):
        chunk = segments[chunk_start: chunk_start + CHUNK_SIZE]
        chunk_end = chunk_start + len(chunk)
        yield f"  Translating segments {chunk_start + 1}–{chunk_end} of {total_segs}…"
        done = translate_chunk(
            chunk=chunk,
            chunk_offset=chunk_start,
            target_language=target_language,
            source_language=source_language,
            prev_context=prev_context,
            model=model,
        )
        translated.extend(done)
        prev_context = _tail_context(done)

    segments = translated
    yield "Translation complete."

    # ── Step 3b (optional): back-translation QA ───────────────────────────────
    if run_backtranslation:
        yield "Running back-translation QA (English → target → English)…"
        import os as _os2
        bt_model = _os2.environ.get("OLLAMA_MODEL", "llama3.2:latest")
        segments = back_translate_segments(segments, model=bt_model)
        flagged = sum(
            1 for s in segments
            if s.get("back_trans_similarity") is not None and s["back_trans_similarity"] < 0.5
        )
        yield f"Back-translation done. {flagged} segment(s) flagged for meaning drift."

    # ── Step 4: voice reference ───────────────────────────────────────────────
    ref_audio_bytes: bytes | None = None
    ref_text: str = ""
    ref_id: str | None = None
    ref_wav: str | None = None  # only set in clone mode; returned for save-voice feature

    if voice_mode == "clone":
        if total_duration < MIN_CLONE_DURATION:
            yield (
                f"WARNING: Video is only {total_duration:.1f}s of audio — voice cloning may "
                f"produce poor quality. Consider using a Fish Audio library voice or a saved "
                f"voice instead (select in 'Voice Selection' above)."
            )
        yield "Extracting voice reference clip for cloning…"
        ref_wav = str(work / "reference.wav")
        ref_start = min(2.0, segments[0]["start"]) if segments else 0.0
        extract_reference_clip(audio_wav, ref_wav, start=ref_start, duration=REFERENCE_CLIP_DURATION)
        ref_audio_bytes, _ = build_voice_reference(ref_wav)
        ref_text = " ".join(s["text"] for s in segments)[:MAX_REFERENCE_TEXT_CHARS]
        yield "Voice reference ready."

    elif voice_mode == "library":
        if not voice_id:
            raise ValueError("Library voice mode selected but no voice ID provided.")
        ref_id = voice_id
        yield f"Using Fish Audio library voice (id={voice_id})."

    elif voice_mode == "saved":
        if not voice_id:
            raise ValueError("Saved voice mode selected but no voice name provided.")
        from .voice_library import load_saved_voices
        voices = load_saved_voices()
        if voice_id not in voices:
            raise ValueError(f"Saved voice '{voice_id}' not found. Run a clone job and save it first.")
        entry = voices[voice_id]
        ref_wav = entry["audio_path"]
        ref_audio_bytes, _ = build_voice_reference(ref_wav)
        ref_text = entry.get("reference_text", "")
        yield f"Using saved voice: '{voice_id}'."

    else:
        raise ValueError(f"Unknown voice_mode: {voice_mode!r}")

    # ── Step 5: TTS synthesis ─────────────────────────────────────────────────
    n_seg = len(segments)
    yield f"Synthesizing TTS for {n_seg} segments…"
    tts_dir = str(work / "tts")

    def _tts_progress(done: int, total: int) -> None:
        pass  # could yield here if we refactor to async; kept simple for now

    segments = synthesize_segments(
        segments,
        reference_audio_bytes=ref_audio_bytes,
        reference_text=ref_text,
        reference_id=ref_id,
        output_dir=tts_dir,
        progress_cb=_tts_progress,
    )
    done_count = sum(1 for s in segments if s.get("tts_path"))
    yield f"TTS complete: {done_count}/{n_seg} segments synthesized."

    failures = [(i, s) for i, s in enumerate(segments) if not s.get("tts_path")]
    for i, s in failures:
        err = s.get("tts_error", "unknown reason")
        orig = s.get("text", "").strip()
        xlat = s.get("translated_text", "").strip()
        yield f"  WARNING seg {i}: {err} | orig={orig!r} xlat={xlat!r}"

    # ── Step 6: audio assembly ────────────────────────────────────────────────
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

    # ── Step 7: mux ───────────────────────────────────────────────────────────
    yield "Muxing audio into video…"
    out_video = str(out / "redubbed.mp4")
    mux_audio_into_video(video_path, dubbed_wav, out_video)
    yield "Video muxed."

    # ── Step 8: output files ──────────────────────────────────────────────────
    yield "Writing transcript and subtitles…"
    transcript_path = str(out / "transcript.json")
    srt_path = str(out / "translated.srt")
    write_transcript_json(segments, transcript_path)
    write_srt(segments, srt_path)

    # Timing report (always produced; TTS files still on disk here)
    timing_rows = build_timing_report(segments)
    timing_csv_path = str(out / "timing_report.csv")
    write_timing_csv(timing_rows, timing_csv_path)
    timing_df = timing_report_to_df(timing_rows)
    n_timing_warn = sum(1 for r in timing_rows if r["status"].startswith("⚠"))
    yield f"Timing report written ({n_timing_warn} segment(s) outside safe ratio)."

    # Back-translation CSV (only if back-translation was run)
    backtrans_csv_path: str | None = None
    backtrans_df = None
    if run_backtranslation:
        backtrans_csv_path = str(out / "back_translation.csv")
        write_backtrans_csv(segments, backtrans_csv_path)
        backtrans_df = backtrans_to_df(segments)

    yield "All files written."

    yield {
        "video": out_video,
        "transcript": transcript_path,
        "srt": srt_path,
        "ref_wav": ref_wav,
        "timing_df": timing_df,
        "timing_csv": timing_csv_path,
        "backtrans_df": backtrans_df,
        "backtrans_csv": backtrans_csv_path,
    }
