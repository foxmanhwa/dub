"""
End-to-end redubbing pipeline orchestrator (Phase A + Phase B).

Phase A — single-speaker:  ASR → merge → translate → voice ref → TTS → assemble → mux
Phase B — multi-speaker:   adds diarization, overlap detection, speech separation,
                            per-stream ASR/translate/TTS, and audio mixing for
                            simultaneous-speech windows.

Yields progress log strings for Gradio streaming, then a final dict of output paths.
"""

import gc
import os
import socket
import sys
import time
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
MIN_CLONE_DURATION = 15.0


def _ram_info() -> tuple[float, float] | None:
    """Returns (available_gb, total_gb) or None if psutil is not installed."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / 1e9, vm.total / 1e9
    except ImportError:
        return None


def _ollama_ping() -> tuple[bool, float]:
    """TCP connect to Ollama port; returns (reachable, elapsed_seconds)."""
    t0 = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=5):
            pass
        return True, time.monotonic() - t0
    except OSError:
        return False, time.monotonic() - t0


def run_pipeline(
    video_path: str,
    source_language: str | None,
    target_language: str,
    output_dir: str,
    voice_mode: str = "clone",
    voice_id: str | None = None,
    run_backtranslation: bool = False,
    handle_overlaps: bool = False,
    preserve_music: bool = False,
    content_context: str | None = None,
) -> Generator[str | dict, None, None]:
    """
    Generator yielding str (progress) then dict (output paths) at the end.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

    # ── Step 1: extract audio ─────────────────────────────────────────────────
    yield "Extracting audio from video…"
    audio_wav = str(work / "audio.wav")
    extract_audio(video_path, audio_wav)
    yield "Audio extracted."

    # ── Step 1b (optional): separate vocals from background music ─────────────
    vocals_wav = audio_wav   # default: process the full mix
    music_wav: str | None = None

    if preserve_music:
        yield "Separating vocals from background music (Demucs htdemucs)…"
        yield "  First run: downloading ~200 MB of model weights — subsequent runs are instant."
        try:
            from .music_separation import check_available as _ms_check, separate_music
            _ms_issues = _ms_check()
            if _ms_issues:
                yield "Music separation disabled — " + "; ".join(_ms_issues)
            else:
                vocals_wav, music_wav = separate_music(audio_wav, str(work / "demucs"))
                yield "Music separation complete — pipeline will run on vocals stem only."
        except Exception as exc:
            yield f"Music separation failed ({exc}); falling back to full-audio mode."
            vocals_wav = audio_wav
            music_wav = None

    # ── Step 2: ASR transcription ─────────────────────────────────────────────
    yield "Transcribing with Fish Audio ASR…"
    lang = source_language if source_language and source_language != "auto" else None
    asr_result = transcribe(vocals_wav, language=lang)
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

    # ── Step 2c (Phase B): speaker diarization + overlap detection ────────────
    overlap_regions: list[dict] = []

    if handle_overlaps:
        yield "Phase B: running speaker diarization…"
        try:
            from .diarization import check_available, run_diarization, assign_speakers_to_segments
            issues = check_available()
            if issues:
                yield "Phase B disabled — " + "; ".join(issues)
            else:
                dia = run_diarization(vocals_wav)
                num_spk = dia["num_speakers"]
                overlap_regions = dia["overlap_regions"]
                segments = assign_speakers_to_segments(
                    segments, dia["segments"], overlap_regions
                )
                yield (
                    f"Diarization complete: {num_spk} speaker(s), "
                    f"{len(overlap_regions)} overlap region(s)."
                )
                for r in overlap_regions:
                    spk_str = ", ".join(r.get("speakers", []))
                    yield (
                        f"  ⚡ OVERLAP {r['start']:.2f}s–{r['end']:.2f}s  "
                        f"[{spk_str}] — will separate + dual-dub"
                    )
                if not overlap_regions:
                    yield "  No overlapping speech detected — proceeding as single-speaker."
        except Exception as exc:
            yield f"Diarization failed ({exc}); falling back to single-speaker mode."
            overlap_regions = []

    # Release diarization model before translation — pyannote + torch consume
    # significant RAM that otherwise starves Ollama's inference.
    if handle_overlaps:
        try:
            from .diarization import release_models as _release_dia
            _release_dia()
        except Exception:
            pass

    # ── Pre-translation diagnostics ───────────────────────────────────────────
    # Explicit GC before Ollama call — subprocess workers should have already
    # freed their memory, but collect any Python-side cycles just in case.
    gc.collect()

    # 1) Check which heavy ML modules leaked into the main process.
    #    With the subprocess approach none of these should be True.
    _mod_flags = {
        "torch": "torch" in sys.modules,
        "pyannote": "pyannote" in sys.modules,
        "speechbrain": "speechbrain" in sys.modules,
        "demucs": "demucs" in sys.modules,
    }
    _loaded = [k for k, v in _mod_flags.items() if v]
    if _loaded:
        yield f"[MEM] WARNING: heavy modules in main process: {', '.join(_loaded)} — subprocess isolation may have failed"
    else:
        yield "[MEM] Module isolation OK — no heavy ML modules loaded in main process"

    # 2) System RAM snapshot.
    _ram = _ram_info()
    if _ram:
        _avail_gb, _total_gb = _ram
        _used_pct = 100 * (1 - _avail_gb / _total_gb)
        yield (
            f"[MEM] RAM: {_avail_gb:.1f} GB available / {_total_gb:.1f} GB total "
            f"({_used_pct:.0f}% used)"
        )
        if _avail_gb < 2.0:
            yield (
                f"[MEM] WARNING: only {_avail_gb:.1f} GB free — "
                "Ollama may need to reload the model from disk, which takes 60-120s"
            )
    else:
        yield "[MEM] psutil not installed — install it for RAM diagnostics (pip install psutil)"

    # 3) Ollama reachability ping.
    _oll_ok, _oll_ms = _ollama_ping()
    if _oll_ok:
        yield f"[OLLAMA] Port 11434 reachable (TCP connect: {_oll_ms * 1000:.0f} ms)"
    else:
        yield "[OLLAMA] WARNING: cannot reach port 11434 — Ollama may not be running"

    # ── Step 3: translation ───────────────────────────────────────────────────
    total_segs = len(segments)
    n_chunks = (total_segs + CHUNK_SIZE - 1) // CHUNK_SIZE
    yield f"Translating {total_segs} segments to {target_language} ({n_chunks} chunk(s))…"

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
            model=ollama_model,
            content_context=content_context,
        )
        translated.extend(done)
        prev_context = _tail_context(done)

    segments = translated
    yield "Translation complete."

    # ── Step 3b (optional): back-translation QA ──────────────────────────────
    if run_backtranslation:
        yield "Running back-translation QA…"
        segments = back_translate_segments(segments, model=ollama_model)
        flagged = sum(
            1 for s in segments
            if s.get("back_trans_similarity") is not None and s["back_trans_similarity"] < 0.5
        )
        yield f"Back-translation done. {flagged} segment(s) flagged for meaning drift."

    # ── Step 4: voice reference ───────────────────────────────────────────────
    ref_audio_bytes: bytes | None = None
    ref_text: str = ""
    ref_id: str | None = None
    ref_wav: str | None = None

    if voice_mode == "clone":
        if total_duration < MIN_CLONE_DURATION:
            yield (
                f"WARNING: Video is only {total_duration:.1f}s — voice cloning may produce "
                "poor quality. Consider a library or saved voice."
            )
        yield "Extracting voice reference clip…"
        ref_wav = str(work / "reference.wav")
        ref_start = min(2.0, segments[0]["start"]) if segments else 0.0
        # Use vocals_wav when available — cleaner signal without background music
        extract_reference_clip(vocals_wav, ref_wav, start=ref_start, duration=REFERENCE_CLIP_DURATION)
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
            raise ValueError(f"Saved voice '{voice_id}' not found.")
        entry = voices[voice_id]
        ref_wav = entry["audio_path"]
        ref_audio_bytes, _ = build_voice_reference(ref_wav)
        ref_text = entry.get("reference_text", "")
        yield f"Using saved voice: '{voice_id}'."

    else:
        raise ValueError(f"Unknown voice_mode: {voice_mode!r}")

    # ── Step 4b (Phase B): process overlap regions ────────────────────────────
    processed_overlaps: list[dict] = []

    if handle_overlaps and overlap_regions:
        from .separation import check_available as sep_check
        sep_issues = sep_check()
        if sep_issues:
            yield "Overlap separation disabled — " + "; ".join(sep_issues)
        else:
            from .overlap_handler import process_overlap_region
            n_ovlp = len(overlap_regions)
            yield f"Phase B: processing {n_ovlp} overlap region(s) through speech separation…"

            for i, region in enumerate(overlap_regions):
                yield (
                    f"  Overlap {i + 1}/{n_ovlp}: "
                    f"{region['start']:.2f}s–{region['end']:.2f}s — separating…"
                )
                proc = process_overlap_region(
                    region=region,
                    original_audio_path=vocals_wav,
                    ref_audio_bytes=ref_audio_bytes,
                    ref_text=ref_text,
                    ref_id=ref_id,
                    target_language=target_language,
                    source_language=source_language,
                    model=ollama_model,
                    work_dir=str(work / "overlaps"),
                    region_idx=i,
                    content_context=content_context,
                )
                if proc.get("error"):
                    yield f"  ⚠ Overlap {i + 1} error: {proc['error']} — using normal path."
                elif proc.get("mixed_tts_path"):
                    stream_summaries = [
                        f"[{sd['stream']}] {sd.get('translated_text', '')[:50]!r}"
                        for sd in proc["speaker_data"]
                        if sd.get("translated_text")
                    ]
                    yield f"  ✓ Overlap {i + 1} dubbed: {' ⊕ '.join(stream_summaries)}"
                    processed_overlaps.append(proc)
                else:
                    yield f"  Overlap {i + 1}: no separable speech — using normal path."

            yield (
                f"Phase B complete: {len(processed_overlaps)}/{n_ovlp} "
                "overlap region(s) successfully dual-dubbed."
            )

            # Free SepFormer model immediately — it's no longer needed
            try:
                from .separation import release_model as _release_sep
                _release_sep()
            except Exception:
                pass

    # ── Step 5: TTS for non-overlap segments ──────────────────────────────────
    # Exclude normal segments that fall inside successfully processed overlap windows
    if processed_overlaps:
        n_before = len(segments)
        processed_spans = [(p["start"], p["end"]) for p in processed_overlaps]
        segments = [
            s for s in segments
            if not any(
                s["start"] >= span[0] - 0.1 and s["end"] <= span[1] + 0.1
                for span in processed_spans
            )
        ]
        n_excluded = n_before - len(segments)
        if n_excluded:
            yield f"Excluded {n_excluded} segment(s) inside overlap windows from normal TTS."

    n_seg = len(segments)
    yield f"Synthesizing TTS for {n_seg} non-overlap segment(s)…"
    tts_dir = str(work / "tts")

    segments = synthesize_segments(
        segments,
        reference_audio_bytes=ref_audio_bytes,
        reference_text=ref_text,
        reference_id=ref_id,
        output_dir=tts_dir,
    )
    done_count = sum(1 for s in segments if s.get("tts_path"))
    yield f"TTS complete: {done_count}/{n_seg} segments synthesized."

    for i, s in enumerate(segments):
        if not s.get("tts_path"):
            err = s.get("tts_error", "unknown")
            yield f"  WARNING seg {i}: {err} | {s.get('text', '')!r}"

    # Inject processed overlap windows as pseudo-segments with pre-mixed audio
    if processed_overlaps:
        for proc in processed_overlaps:
            spk_text = " / ".join(
                f"[{sd['stream']}] {sd.get('text', '')}"
                for sd in proc["speaker_data"] if sd.get("text")
            )
            spk_xlat = " / ".join(
                f"[{sd['stream']}] {sd.get('translated_text', '')}"
                for sd in proc["speaker_data"] if sd.get("translated_text")
            )
            segments.append({
                "start": proc["start"],
                "end": proc["end"],
                "text": f"[OVERLAP] {spk_text}",
                "translated_text": f"[OVERLAP] {spk_xlat}",
                "tts_path": proc["mixed_tts_path"],
                "is_overlap": True,
                "speaker_data": proc["speaker_data"],
            })
        segments.sort(key=lambda s: s["start"])
        yield f"Injected {len(processed_overlaps)} overlap mixed clip(s) into assembly timeline."

    # ── Step 6: audio assembly ────────────────────────────────────────────────
    yield "Assembling dubbed audio track…"
    dubbed_wav = str(work / "dubbed_audio.wav")
    assemble_audio_track(
        segments=segments,
        original_audio_path=vocals_wav,   # gaps filled with vocals stem (or full audio)
        total_duration=total_duration,
        output_path=dubbed_wav,
        work_dir=str(work / "assembly"),
    )
    yield "Audio assembly complete."

    # ── Step 6b (optional): mix dubbed vocals with original background music ──
    use_stereo_mux = False
    if preserve_music and music_wav:
        yield "Mixing dubbed vocals with original background music…"
        try:
            from .music_separation import mix_vocals_with_music
            mixed_wav = str(work / "dubbed_mixed.wav")
            mix_vocals_with_music(dubbed_wav, music_wav, mixed_wav, total_duration)
            dubbed_wav = mixed_wav
            use_stereo_mux = True
            yield "Music mix complete — stereo output ready."
        except Exception as exc:
            yield f"Music mix failed ({exc}); outputting vocals-only audio."

    # ── Step 7: mux ───────────────────────────────────────────────────────────
    yield "Muxing audio into video…"
    out_video = str(out / "redubbed.mp4")
    mux_audio_into_video(video_path, dubbed_wav, out_video, stereo=use_stereo_mux)
    yield "Video muxed."

    # ── Step 8: output files ──────────────────────────────────────────────────
    yield "Writing transcript and subtitles…"
    transcript_path = str(out / "transcript.json")
    srt_path = str(out / "translated.srt")
    write_transcript_json(segments, transcript_path)
    write_srt(segments, srt_path)

    timing_rows = build_timing_report(segments)
    timing_csv_path = str(out / "timing_report.csv")
    write_timing_csv(timing_rows, timing_csv_path)
    timing_df = timing_report_to_df(timing_rows)
    n_timing_warn = sum(1 for r in timing_rows if r["status"].startswith("⚠"))
    yield f"Timing report: {n_timing_warn} segment(s) outside safe ratio."

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
