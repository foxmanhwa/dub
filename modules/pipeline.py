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
from .translation import translate_chunk, _tail_context, CHUNK_SIZE, back_translate_segments, retranslate_for_duration
from .tts import build_voice_reference, synthesize_segments
from .audio_assembly import (
    extract_audio,
    extract_reference_clip,
    extract_speaker_reference_clips,
    assemble_audio_track,
    mux_audio_into_video,
    get_duration,
)
from .output_writers import write_transcript_json, write_srt, write_timing_csv, write_backtrans_csv
from .timing_report import build_timing_report, timing_report_to_df, backtrans_to_df


REFERENCE_CLIP_DURATION = 12.0


def _detect_overlaps_from_segments(
    dia_segments: list[dict], min_duration: float = 0.3
) -> list[dict]:
    """Detect time windows where two different speakers are active simultaneously."""
    sorted_segs = sorted(dia_segments, key=lambda s: s["start"])
    raw: list[dict] = []
    n = len(sorted_segs)
    for i in range(n):
        a = sorted_segs[i]
        for j in range(i + 1, n):
            b = sorted_segs[j]
            if b["start"] >= a["end"]:
                break
            if a.get("speaker") == b.get("speaker"):
                continue
            ovlp_start = max(a["start"], b["start"])
            ovlp_end = min(a["end"], b["end"])
            if ovlp_end - ovlp_start >= min_duration:
                raw.append({
                    "start": round(ovlp_start, 3),
                    "end": round(ovlp_end, 3),
                    "speakers": sorted({a.get("speaker", "SPEAKER_00"), b.get("speaker", "SPEAKER_01")}),
                })
    if not raw:
        return []
    # Merge adjacent overlaps
    raw.sort(key=lambda x: x["start"])
    merged = [raw[0]]
    for ov in raw[1:]:
        last = merged[-1]
        if ov["start"] <= last["end"]:
            last["end"] = max(last["end"], ov["end"])
            last["speakers"] = sorted(set(last["speakers"] + ov["speakers"]))
        else:
            merged.append(ov)
    return merged


MAX_REFERENCE_TEXT_CHARS = 300
MIN_CLONE_DURATION = 15.0
MIN_REF_DURATION = float(os.environ.get("MIN_REF_DURATION_SECS", "6.0"))


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
    fallback_voice_id: str | None = None,
    min_ref_duration: float | None = None,
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
                # Demucs outputs stereo WAV; convert to mono so ASR, diarization,
                # reference extraction, and assembly all get a consistent format.
                vocals_mono = str(work / "vocals_mono.wav")
                extract_audio(vocals_wav, vocals_mono)
                vocals_wav = vocals_mono
                yield "Music separation complete — pipeline will run on vocals stem only."
        except Exception as exc:
            yield f"Music separation failed ({exc}); falling back to full-audio mode."
            vocals_wav = audio_wav
            music_wav = None

    # ── Step 2: ASR transcription ─────────────────────────────────────────────
    _asr_backend = os.environ.get("ASR_BACKEND", "whisperx").lower()
    if _asr_backend == "whisperx":
        yield "Transcribing with WhisperX (local model)…"
        if os.environ.get("HF_TOKEN"):
            yield "  HF_TOKEN present — speaker diarization will run inside WhisperX."
        else:
            yield "  No HF_TOKEN — speaker diarization skipped (set HF_TOKEN to enable)."
    else:
        yield "Transcribing with Fish Audio ASR…"
    lang = source_language if source_language and source_language != "auto" else None
    asr_result = transcribe(vocals_wav, language=lang)
    segments = asr_result.get("segments", [])
    total_duration = asr_result.get("duration") or get_duration(video_path)

    if not segments:
        raise ValueError(
            "ASR returned no segments. Ensure the video contains audible speech."
        )

    _whisperx_diarized = asr_result.get("speakers_assigned", False)
    yield f"Transcribed {len(segments)} segments ({total_duration:.1f} s total)."
    if _whisperx_diarized:
        _n_spk_asr = len({s.get("speaker") for s in segments if s.get("speaker")})
        yield f"  WhisperX speaker assignment complete — {_n_spk_asr} speaker(s) detected."

    # ── Step 2b: merge word-level fragments into phrase-level groups ──────────
    raw_count = len(segments)
    segments = merge_segments(segments)
    yield f"Merged {raw_count} fragments → {len(segments)} phrase-level segments."

    # ── Step 2c: speaker diarization + overlap detection ─────────────────────
    overlap_regions: list[dict] = []
    dia_segments: list[dict] = []  # speaker time-ranges used for reference clip extraction

    # When WhisperX already assigned speakers, derive dia_segments immediately so
    # per-speaker reference extraction works even when handle_overlaps is False.
    if _whisperx_diarized:
        dia_segments = [
            {"start": s["start"], "end": s["end"], "speaker": s.get("speaker", "SPEAKER_00")}
            for s in segments
            if s.get("speaker")
        ]

    if handle_overlaps:
        if _whisperx_diarized:
            yield "Phase B: WhisperX already performed diarization — skipping separate pyannote run."
            overlap_regions = _detect_overlaps_from_segments(dia_segments)
            num_spk = len({d["speaker"] for d in dia_segments})
            # Tag segments with is_overlap
            for seg in segments:
                seg["is_overlap"] = any(
                    r["start"] - 0.1 <= seg["start"] and seg["end"] <= r["end"] + 0.1
                    for r in overlap_regions
                )
            yield (
                f"  {num_spk} speaker(s), {len(overlap_regions)} overlap region(s)."
            )
            for r in overlap_regions:
                spk_str = ", ".join(r.get("speakers", []))
                yield (
                    f"  ⚡ OVERLAP {r['start']:.2f}s–{r['end']:.2f}s  "
                    f"[{spk_str}] — will separate + dual-dub"
                )
            if not overlap_regions:
                yield "  No overlapping speech detected — proceeding as single-speaker."
        else:
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
                    dia_segments = dia["segments"]
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
        segments = back_translate_segments(segments, model=ollama_model, source_language=source_language)
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
    speaker_references: dict[str, tuple[bytes | None, str, str | None]] | None = None
    speaker_ref_wavs: dict[str, str] = {}  # speaker_id → wav_path (for UI save panel)

    if voice_mode == "clone":
        if total_duration < MIN_CLONE_DURATION:
            yield (
                f"WARNING: Video is only {total_duration:.1f}s — voice cloning may produce "
                "poor quality. Consider a library or saved voice."
            )

        detected_speakers = {s.get("speaker", "SPEAKER_00") for s in segments}
        use_per_speaker = len(detected_speakers) > 1 and bool(dia_segments)

        if use_per_speaker:
            yield f"Building per-speaker voice references ({len(detected_speakers)} speakers)…"
            speaker_clip_results = extract_speaker_reference_clips(
                vocals_wav,
                dia_segments,
                work_dir=str(work / "speaker_refs"),
                target_duration=REFERENCE_CLIP_DURATION,
                overlap_regions=overlap_regions,
            )

            _fallback_vid = fallback_voice_id or os.environ.get("FALLBACK_VOICE_ID")
            _min_ref_dur = min_ref_duration if min_ref_duration is not None else MIN_REF_DURATION
            speaker_references = {}
            for spk, (wav_path, duration) in speaker_clip_results.items():
                if duration >= _min_ref_dur:
                    ref_bytes, _ = build_voice_reference(wav_path)
                    spk_text = " ".join(
                        s["text"] for s in segments if s.get("speaker") == spk
                    )[:MAX_REFERENCE_TEXT_CHARS]
                    speaker_references[spk] = (ref_bytes, spk_text, None)
                    speaker_ref_wavs[spk] = wav_path
                    yield f"  {spk} reference: {duration:.1f}s [OK — cloning]"
                elif _fallback_vid:
                    speaker_references[spk] = (None, "", _fallback_vid)
                    yield (
                        f"  {spk} reference: {duration:.1f}s — too short "
                        f"(< {_min_ref_dur:.0f}s); using library voice '{_fallback_vid}' instead"
                    )
                else:
                    ref_bytes, _ = build_voice_reference(wav_path)
                    spk_text = " ".join(
                        s["text"] for s in segments if s.get("speaker") == spk
                    )[:MAX_REFERENCE_TEXT_CHARS]
                    speaker_references[spk] = (ref_bytes, spk_text, None)
                    speaker_ref_wavs[spk] = wav_path
                    quality_tag = "⚠ very short" if duration < 3.0 else "⚠ short"
                    yield (
                        f"  {spk} reference: {duration:.1f}s [{quality_tag} — cloning anyway; "
                        "set a Fallback voice ID in the UI (or FALLBACK_VOICE_ID in .env) "
                        "for automatic library fallback]"
                    )

            # Any speaker tagged in segments but missing from diarization clips
            # (e.g. very few lines with no long-enough segment) gets the global fallback.
            for spk in detected_speakers - set(speaker_references):
                yield f"  {spk}: no usable audio found — will use best available reference as fallback"

            # Primary ref_wav = longest clone-mode speaker (used for the save panel and
            # as global fallback for any segment whose speaker isn't in speaker_references).
            if speaker_ref_wavs:
                primary_spk = max(
                    speaker_ref_wavs,
                    key=lambda k: speaker_clip_results.get(k, (None, 0.0))[1],
                )
                ref_wav = speaker_ref_wavs[primary_spk]
                ref_audio_bytes, _ = build_voice_reference(ref_wav)
                ref_text = speaker_references[primary_spk][1]

            yield "Per-speaker voice references ready."

            # ── Step 4c: speaker embedding analysis ───────────────────────────
            if speaker_ref_wavs:
                try:
                    from .speaker_embeddings import (
                        check_available as _emb_check,
                        extract_embeddings,
                        verify_speaker_distinctness,
                        match_against_library,
                    )
                    from .voice_library import load_saved_voice_embeddings
                    _emb_issues = _emb_check()
                    if _emb_issues:
                        yield "Speaker embeddings skipped — " + "; ".join(_emb_issues)
                    else:
                        yield "Extracting speaker embeddings (ECAPA-TDNN)…"
                        det_embs = extract_embeddings(speaker_ref_wavs)
                        if det_embs:
                            # Consistency check
                            if len(det_embs) > 1:
                                warns = verify_speaker_distinctness(det_embs)
                                if warns:
                                    for w in warns:
                                        yield f"  ⚠ {w}"
                                else:
                                    yield f"  ✓ All {len(det_embs)} speakers have distinct embeddings."
                            # Library matching
                            saved_embs = load_saved_voice_embeddings()
                            if saved_embs:
                                suggestions = match_against_library(det_embs, saved_embs)
                                if suggestions:
                                    yield "  Voice library matches:"
                                    for spk, (vname, sim) in suggestions.items():
                                        yield f"    {spk} → '{vname}' (similarity {sim:.2f})"
                                else:
                                    yield "  No close matches found in saved voice library."
                            else:
                                yield "  No saved voices in library to compare against."
                        else:
                            yield "  Embedding extraction returned no results."
                except Exception as _emb_exc:
                    yield f"Speaker embedding analysis failed ({_emb_exc}) — continuing."

        else:
            yield "Extracting voice reference clip…"
            ref_wav = str(work / "reference.wav")
            ref_start = min(2.0, segments[0]["start"]) if segments else 0.0
            # Use vocals_wav when available — cleaner signal without background music
            extract_reference_clip(vocals_wav, ref_wav, start=ref_start, duration=REFERENCE_CLIP_DURATION)
            ref_audio_bytes, _ = build_voice_reference(ref_wav)
            ref_text = " ".join(s["text"] for s in segments)[:MAX_REFERENCE_TEXT_CHARS]
            if ref_wav:
                speaker_ref_wavs["SPEAKER_00"] = ref_wav
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

    def _translate_retry(translated_text: str, original_text: str, target_secs: float, direction: str) -> str:
        return retranslate_for_duration(
            translated_text=translated_text,
            original_text=original_text,
            target_seconds=target_secs,
            target_language=target_language,
            direction=direction,
            source_language=source_language,
            model=ollama_model,
            content_context=content_context,
        )

    segments = synthesize_segments(
        segments,
        reference_audio_bytes=ref_audio_bytes,
        reference_text=ref_text,
        reference_id=ref_id,
        speaker_references=speaker_references,
        output_dir=tts_dir,
        translate_retry_fn=_translate_retry,
        max_fit_retries=2,
    )
    done_count = sum(1 for s in segments if s.get("tts_path"))
    retried = [(i, s) for i, s in enumerate(segments) if s.get("fit_retries", 0) > 0]
    yield f"TTS complete: {done_count}/{n_seg} segments synthesized."
    if retried:
        yield f"  {len(retried)} segment(s) needed translation retry for timing fit:"
        for idx, s in retried:
            n = s["fit_retries"]
            yield (
                f"    seg {idx} (↻{n}): {s.get('translated_text', '')[:70]!r}"
            )

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
        "speaker_ref_wavs": speaker_ref_wavs,
        "timing_df": timing_df,
        "timing_csv": timing_csv_path,
        "backtrans_df": backtrans_df,
        "backtrans_csv": backtrans_csv_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Two-stage pipeline API
# ══════════════════════════════════════════════════════════════════════════════


def analyze_video(
    video_path: str,
    source_language: str | None,
    output_dir: str,
    preserve_music: bool = False,
    handle_overlaps: bool = False,
) -> Generator[str | dict, None, None]:
    """
    Stage 1 — extract, transcribe, diarize, extract speaker clips.

    Yields progress strings, then a final result dict:
      {
        "video_path":         str,
        "work_dir":           str,
        "vocals_wav":         str,
        "music_wav":          str | None,
        "segments":           list[dict],
        "dia_segments":       list[dict],
        "overlap_regions":    list[dict],
        "total_duration":     float,
        "speaker_ref_wavs":   dict[str, str],   # speaker_id → wav_path
        "_whisperx_diarized": bool,
        "embedding_suggestions": dict,          # speaker_id → (voice_name, sim)
      }
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    # ── Extract audio ─────────────────────────────────────────────────────────
    yield "Extracting audio from video…"
    audio_wav = str(work / "audio.wav")
    extract_audio(video_path, audio_wav)
    yield "Audio extracted."

    # ── Music separation (optional) ───────────────────────────────────────────
    vocals_wav = audio_wav
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
                vocals_mono = str(work / "vocals_mono.wav")
                extract_audio(vocals_wav, vocals_mono)
                vocals_wav = vocals_mono
                yield "Music separation complete."
        except Exception as exc:
            yield f"Music separation failed ({exc}); falling back to full-audio mode."
            vocals_wav = audio_wav
            music_wav = None

    # ── ASR transcription ─────────────────────────────────────────────────────
    _asr_backend = os.environ.get("ASR_BACKEND", "whisperx").lower()
    if _asr_backend == "whisperx":
        yield "Transcribing with WhisperX (local model)…"
        yield ("  HF_TOKEN present — diarization enabled." if os.environ.get("HF_TOKEN")
               else "  No HF_TOKEN — diarization skipped inside WhisperX.")
    else:
        yield "Transcribing with Fish Audio ASR…"

    lang = source_language if source_language and source_language != "auto" else None
    asr_result = transcribe(vocals_wav, language=lang)
    segments = asr_result.get("segments", [])
    total_duration = asr_result.get("duration") or get_duration(video_path)

    if not segments:
        raise ValueError("ASR returned no segments. Ensure the video contains audible speech.")

    _whisperx_diarized = asr_result.get("speakers_assigned", False)
    yield f"Transcribed {len(segments)} segments ({total_duration:.1f} s total)."
    if _whisperx_diarized:
        _n = len({s.get("speaker") for s in segments if s.get("speaker")})
        yield f"  WhisperX speaker assignment: {_n} speaker(s)."

    # ── Merge word fragments ──────────────────────────────────────────────────
    raw_count = len(segments)
    segments = merge_segments(segments)
    yield f"Merged {raw_count} fragments → {len(segments)} phrase-level segments."

    # ── Speaker diarization / overlap detection ───────────────────────────────
    overlap_regions: list[dict] = []
    dia_segments: list[dict] = []

    if _whisperx_diarized:
        dia_segments = [
            {"start": s["start"], "end": s["end"], "speaker": s.get("speaker", "SPEAKER_00")}
            for s in segments if s.get("speaker")
        ]

    if handle_overlaps:
        if _whisperx_diarized:
            yield "WhisperX diarization covers Phase B — detecting overlaps from word data."
            overlap_regions = _detect_overlaps_from_segments(dia_segments)
            num_spk = len({d["speaker"] for d in dia_segments})
            for seg in segments:
                seg["is_overlap"] = any(
                    r["start"] - 0.1 <= seg["start"] and seg["end"] <= r["end"] + 0.1
                    for r in overlap_regions
                )
            yield f"  {num_spk} speaker(s), {len(overlap_regions)} overlap region(s)."
            for r in overlap_regions:
                yield f"  ⚡ OVERLAP {r['start']:.2f}s–{r['end']:.2f}s [{', '.join(r.get('speakers', []))}]"
            if not overlap_regions:
                yield "  No overlapping speech detected."
        else:
            yield "Running speaker diarization (pyannote)…"
            try:
                from .diarization import check_available, run_diarization, assign_speakers_to_segments
                _dia_issues = check_available()
                if _dia_issues:
                    yield "Diarization disabled — " + "; ".join(_dia_issues)
                else:
                    _dia = run_diarization(vocals_wav)
                    overlap_regions = _dia["overlap_regions"]
                    dia_segments = _dia["segments"]
                    segments = assign_speakers_to_segments(segments, _dia["segments"], overlap_regions)
                    yield (f"Diarization: {_dia['num_speakers']} speaker(s), "
                           f"{len(overlap_regions)} overlap region(s).")
                    for r in overlap_regions:
                        yield f"  ⚡ OVERLAP {r['start']:.2f}s–{r['end']:.2f}s [{', '.join(r.get('speakers', []))}]"
                    if not overlap_regions:
                        yield "  No overlapping speech."
            except Exception as exc:
                yield f"Diarization failed ({exc}) — proceeding without it."

    if handle_overlaps:
        try:
            from .diarization import release_models as _rel
            _rel()
        except Exception:
            pass

    # ── Extract speaker reference clips ───────────────────────────────────────
    speaker_ref_wavs: dict[str, str] = {}
    detected_speakers = {s.get("speaker", "SPEAKER_00") for s in segments}
    use_per_speaker = len(detected_speakers) > 1 and bool(dia_segments)

    if use_per_speaker:
        yield f"Extracting per-speaker reference clips ({len(detected_speakers)} speakers)…"
        _clip_results = extract_speaker_reference_clips(
            vocals_wav,
            dia_segments,
            work_dir=str(work / "speaker_refs"),
            target_duration=REFERENCE_CLIP_DURATION,
            overlap_regions=overlap_regions,
        )
        for spk, (wav_path, dur) in _clip_results.items():
            speaker_ref_wavs[spk] = wav_path
            yield f"  {spk}: {dur:.1f}s reference clip."
        yield "Speaker reference clips ready."
    else:
        yield "Extracting voice reference clip (single speaker)…"
        _ref_wav = str(work / "reference.wav")
        _ref_start = min(2.0, segments[0]["start"]) if segments else 0.0
        extract_reference_clip(vocals_wav, _ref_wav, start=_ref_start, duration=REFERENCE_CLIP_DURATION)
        speaker_ref_wavs["SPEAKER_00"] = _ref_wav
        yield "Voice reference ready."

    # ── Energy tagging (for emotion-aware translation) ────────────────────────
    try:
        from .energy import tag_segments_with_energy
        segments = tag_segments_with_energy(vocals_wav, segments)
        n_loud = sum(1 for s in segments if s.get("energy") == "loud")
        n_quiet = sum(1 for s in segments if s.get("energy") == "quiet")
        if n_loud or n_quiet:
            yield f"Energy analysis: {n_loud} loud, {n_quiet} soft/whispered segment(s) tagged."
    except Exception as _e_exc:
        yield f"Energy tagging failed ({_e_exc}) — using neutral tone for all segments."

    # ── Speaker embedding analysis ────────────────────────────────────────────
    embedding_suggestions: dict[str, tuple] = {}
    if speaker_ref_wavs:
        try:
            from .speaker_embeddings import (
                check_available as _emb_check, extract_embeddings,
                verify_speaker_distinctness, match_against_library,
            )
            from .voice_library import load_saved_voice_embeddings
            if not _emb_check():
                yield "Extracting speaker embeddings (ECAPA-TDNN)…"
                _det_embs = extract_embeddings(speaker_ref_wavs)
                if _det_embs:
                    if len(_det_embs) > 1:
                        _warns = verify_speaker_distinctness(_det_embs)
                        for w in _warns:
                            yield f"  ⚠ {w}"
                        if not _warns:
                            yield f"  ✓ All {len(_det_embs)} speakers are distinct."
                    _saved_embs = load_saved_voice_embeddings()
                    if _saved_embs:
                        embedding_suggestions = match_against_library(_det_embs, _saved_embs)
                        if embedding_suggestions:
                            yield "  Voice library suggestions:"
                            for spk, (vname, sim) in embedding_suggestions.items():
                                yield f"    {spk} → '{vname}' (similarity {sim:.2f})"
                        else:
                            yield "  No close saved-voice matches found."
        except Exception as _exc:
            yield f"Embedding analysis failed ({_exc}) — continuing."

    _n_spk = len(detected_speakers)
    yield f"Analysis complete: {len(segments)} segments, {_n_spk} speaker(s)."

    yield {
        "video_path": video_path,
        "work_dir": str(work),
        "vocals_wav": vocals_wav,
        "music_wav": music_wav,
        "segments": segments,
        "dia_segments": dia_segments,
        "overlap_regions": overlap_regions,
        "total_duration": total_duration,
        "speaker_ref_wavs": speaker_ref_wavs,
        "_whisperx_diarized": _whisperx_diarized,
        "embedding_suggestions": dict(embedding_suggestions),
    }


def generate_from_analysis(
    analysis: dict,
    speaker_assignments: dict[str, dict] | None,
    target_language: str,
    source_language: str | None,
    voice_mode: str = "clone",
    voice_id: str | None = None,
    run_backtranslation: bool = False,
    content_context: str | None = None,
    fallback_voice_id: str | None = None,
    min_ref_duration: float | None = None,
) -> Generator[str | dict, None, None]:
    """
    Stage 2 — translate, TTS, assemble, mux.

    analysis:            dict returned by analyze_video()
    speaker_assignments: {speaker_id: {"mode": "clone"|"saved"|"library",
                                       "name": str,      # saved mode
                                       "voice_id": str}} # library mode
                         None → use voice_mode / voice_id as global fallback
    """
    video_path = analysis["video_path"]
    work = Path(analysis["work_dir"])
    out = work.parent
    vocals_wav = analysis["vocals_wav"]
    music_wav = analysis.get("music_wav")
    segments = list(analysis["segments"])
    dia_segments = analysis.get("dia_segments", [])
    overlap_regions = analysis.get("overlap_regions", [])
    total_duration = analysis["total_duration"]
    speaker_ref_wavs = analysis.get("speaker_ref_wavs", {})

    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    _min_ref_dur = min_ref_duration if min_ref_duration is not None else MIN_REF_DURATION
    _fallback_vid = fallback_voice_id or os.environ.get("FALLBACK_VOICE_ID")

    # ── Pre-translation diagnostics ───────────────────────────────────────────
    gc.collect()
    _loaded = [k for k in ("torch", "pyannote", "speechbrain", "demucs") if k in sys.modules]
    if _loaded:
        yield f"[MEM] WARNING: heavy modules in main process: {', '.join(_loaded)}"
    else:
        yield "[MEM] Module isolation OK"

    _ram = _ram_info()
    if _ram:
        _avail, _total = _ram
        yield f"[MEM] RAM: {_avail:.1f} GB available / {_total:.1f} GB total"
        if _avail < 2.0:
            yield f"[MEM] WARNING: only {_avail:.1f} GB free"
    _oll_ok, _oll_ms = _ollama_ping()
    if _oll_ok:
        yield f"[OLLAMA] Port 11434 reachable ({_oll_ms * 1000:.0f} ms)"
    else:
        yield "[OLLAMA] WARNING: cannot reach port 11434"

    # ── Translation ───────────────────────────────────────────────────────────
    total_segs = len(segments)
    n_chunks = (total_segs + CHUNK_SIZE - 1) // CHUNK_SIZE
    yield f"Translating {total_segs} segments to {target_language} ({n_chunks} chunk(s))…"

    translated: list[dict] = []
    prev_context: list[str] = []
    for chunk_start in range(0, total_segs, CHUNK_SIZE):
        chunk = segments[chunk_start: chunk_start + CHUNK_SIZE]
        yield f"  Segments {chunk_start + 1}–{chunk_start + len(chunk)} of {total_segs}…"
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

    if run_backtranslation:
        yield "Running back-translation QA…"
        segments = back_translate_segments(
            segments, model=ollama_model, source_language=source_language
        )
        flagged = sum(
            1 for s in segments
            if s.get("back_trans_similarity") is not None and s["back_trans_similarity"] < 0.5
        )
        yield f"Back-translation done. {flagged} segment(s) flagged."

    # ── Build voice references ────────────────────────────────────────────────
    ref_audio_bytes: bytes | None = None
    ref_text: str = ""
    ref_id: str | None = None
    ref_wav: str | None = None
    speaker_references: dict[str, tuple] | None = None
    detected_speakers = {s.get("speaker", "SPEAKER_00") for s in segments}

    if speaker_assignments:
        # Per-speaker mode from UI assignments
        yield "Building per-speaker voice references from assignments…"
        from .voice_library import load_saved_voices as _lsv
        _sv = _lsv()
        speaker_references = {}

        for spk in detected_speakers:
            _asgn = speaker_assignments.get(spk, {"mode": "clone"})
            _mode = _asgn.get("mode", "clone")

            if _mode == "saved":
                _nm = _asgn.get("name", "")
                if _nm and _nm in _sv:
                    _e = _sv[_nm]
                    _rb, _ = build_voice_reference(_e["audio_path"])
                    speaker_references[spk] = (_rb, _e.get("reference_text", ""), None)
                    yield f"  {spk} → saved voice '{_nm}'"
                    continue
                yield f"  {spk}: saved voice '{_nm}' not found — falling back to clone"
                _mode = "clone"

            if _mode == "library":
                _vid = _asgn.get("voice_id", "").strip()
                if _vid:
                    speaker_references[spk] = (None, "", _vid)
                    yield f"  {spk} → library voice '{_vid}'"
                    continue
                yield f"  {spk}: library voice ID empty — falling back to clone"
                _mode = "clone"

            # clone (or fallback from above)
            _wp = speaker_ref_wavs.get(spk)
            if _wp and Path(_wp).exists():
                _rb, _ = build_voice_reference(_wp)
                _rt = " ".join(s["text"] for s in segments if s.get("speaker") == spk)[:MAX_REFERENCE_TEXT_CHARS]
                speaker_references[spk] = (_rb, _rt, None)
                yield f"  {spk} → cloned from video"
            elif _fallback_vid:
                speaker_references[spk] = (None, "", _fallback_vid)
                yield f"  {spk} → no clip, using fallback voice '{_fallback_vid}'"

        # Global ref (for overlap handler + single-speaker fallback)
        if speaker_ref_wavs:
            _p_spk = max(speaker_ref_wavs, key=lambda k: Path(speaker_ref_wavs[k]).stat().st_size
                         if Path(speaker_ref_wavs[k]).exists() else 0)
            ref_wav = speaker_ref_wavs[_p_spk]
            if Path(ref_wav).exists():
                ref_audio_bytes, _ = build_voice_reference(ref_wav)
                ref_text = (speaker_references.get(_p_spk) or ("", "", None))[1]

        yield "Voice references ready."

    elif voice_mode == "clone":
        if total_duration < MIN_CLONE_DURATION:
            yield f"WARNING: only {total_duration:.1f}s — clone quality may be poor."

        use_per_speaker = len(detected_speakers) > 1 and bool(dia_segments)
        if use_per_speaker:
            yield f"Auto-building per-speaker references ({len(detected_speakers)} speakers)…"
            speaker_references = {}
            for spk, _wp in speaker_ref_wavs.items():
                if not Path(_wp).exists():
                    continue
                try:
                    import subprocess as _sp
                    _dur_out = _sp.check_output(
                        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                         "-of", "csv=p=0", _wp], text=True
                    )
                    _dur = float(_dur_out.strip())
                except Exception:
                    _dur = _min_ref_dur  # assume OK if we can't check

                if _dur >= _min_ref_dur:
                    _rb, _ = build_voice_reference(_wp)
                    _rt = " ".join(s["text"] for s in segments if s.get("speaker") == spk)[:MAX_REFERENCE_TEXT_CHARS]
                    speaker_references[spk] = (_rb, _rt, None)
                    yield f"  {spk}: {_dur:.1f}s [cloning]"
                elif _fallback_vid:
                    speaker_references[spk] = (None, "", _fallback_vid)
                    yield f"  {spk}: {_dur:.1f}s too short → fallback '{_fallback_vid}'"
                else:
                    _rb, _ = build_voice_reference(_wp)
                    _rt = " ".join(s["text"] for s in segments if s.get("speaker") == spk)[:MAX_REFERENCE_TEXT_CHARS]
                    speaker_references[spk] = (_rb, _rt, None)
                    yield f"  {spk}: {_dur:.1f}s [short — cloning anyway]"

            if speaker_ref_wavs:
                _p_spk = max(speaker_ref_wavs, key=lambda k: Path(speaker_ref_wavs[k]).stat().st_size
                             if Path(speaker_ref_wavs[k]).exists() else 0)
                ref_wav = speaker_ref_wavs[_p_spk]
                if Path(ref_wav).exists():
                    ref_audio_bytes, _ = build_voice_reference(ref_wav)
                    ref_text = (speaker_references.get(_p_spk) or ("", "", None))[1]
            yield "Per-speaker references ready."
        else:
            _wp = (speaker_ref_wavs.get("SPEAKER_00")
                   or next(iter(speaker_ref_wavs.values()), None))
            if _wp and Path(_wp).exists():
                ref_wav = _wp
                ref_audio_bytes, _ = build_voice_reference(ref_wav)
                ref_text = " ".join(s["text"] for s in segments)[:MAX_REFERENCE_TEXT_CHARS]
                yield "Using single-speaker voice reference."
            else:
                raise ValueError("No reference audio available for cloning.")

    elif voice_mode == "library":
        if not voice_id:
            raise ValueError("Library voice mode requires a voice ID.")
        ref_id = voice_id
        yield f"Using Fish Audio library voice (id={voice_id})."

    elif voice_mode == "saved":
        if not voice_id:
            raise ValueError("Saved voice mode requires a voice name.")
        from .voice_library import load_saved_voices as _lsv2
        _sv2 = _lsv2()
        if voice_id not in _sv2:
            raise ValueError(f"Saved voice '{voice_id}' not found.")
        _e2 = _sv2[voice_id]
        ref_wav = _e2["audio_path"]
        ref_audio_bytes, _ = build_voice_reference(ref_wav)
        ref_text = _e2.get("reference_text", "")
        yield f"Using saved voice: '{voice_id}'."

    else:
        raise ValueError(f"Unknown voice_mode: {voice_mode!r}")

    # ── Phase B: overlap processing ───────────────────────────────────────────
    processed_overlaps: list[dict] = []

    if overlap_regions:
        from .separation import check_available as sep_check
        sep_issues = sep_check()
        if sep_issues:
            yield "Overlap separation disabled — " + "; ".join(sep_issues)
        else:
            from .overlap_handler import process_overlap_region
            n_ovlp = len(overlap_regions)
            yield f"Phase B: processing {n_ovlp} overlap region(s)…"
            for i, region in enumerate(overlap_regions):
                yield f"  {i+1}/{n_ovlp}: {region['start']:.2f}s–{region['end']:.2f}s"
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
                    yield f"  ⚠ error: {proc['error']}"
                elif proc.get("mixed_tts_path"):
                    processed_overlaps.append(proc)
                    yield f"  ✓ dual-dubbed"
                else:
                    yield f"  no separable speech — using normal path."
            yield f"Phase B: {len(processed_overlaps)}/{n_ovlp} done."
            try:
                from .separation import release_model as _rel_sep
                _rel_sep()
            except Exception:
                pass

    # ── TTS ───────────────────────────────────────────────────────────────────
    if processed_overlaps:
        processed_spans = [(p["start"], p["end"]) for p in processed_overlaps]
        n_before = len(segments)
        segments = [
            s for s in segments
            if not any(s["start"] >= sp[0] - 0.1 and s["end"] <= sp[1] + 0.1
                       for sp in processed_spans)
        ]
        if n_before - len(segments):
            yield f"Excluded {n_before - len(segments)} segment(s) inside overlap windows."

    n_seg = len(segments)
    yield f"Synthesizing TTS for {n_seg} segment(s)…"
    tts_dir = str(work / "tts")

    def _translate_retry(translated_text: str, original_text: str, target_secs: float, direction: str) -> str:
        return retranslate_for_duration(
            translated_text=translated_text,
            original_text=original_text,
            target_seconds=target_secs,
            target_language=target_language,
            direction=direction,
            source_language=source_language,
            model=ollama_model,
            content_context=content_context,
        )

    segments = synthesize_segments(
        segments,
        reference_audio_bytes=ref_audio_bytes,
        reference_text=ref_text,
        reference_id=ref_id,
        speaker_references=speaker_references,
        output_dir=tts_dir,
        translate_retry_fn=_translate_retry,
        max_fit_retries=2,
    )
    done_count = sum(1 for s in segments if s.get("tts_path"))
    retried = [(i, s) for i, s in enumerate(segments) if s.get("fit_retries", 0) > 0]
    yield f"TTS complete: {done_count}/{n_seg} segments synthesized."
    if retried:
        yield f"  {len(retried)} segment(s) needed timing retry:"
        for idx, s in retried:
            yield f"    seg {idx} (↻{s['fit_retries']}): {s.get('translated_text', '')[:70]!r}"
    for i, s in enumerate(segments):
        if not s.get("tts_path"):
            yield f"  WARNING seg {i}: {s.get('tts_error', 'unknown')} | {s.get('text', '')!r}"

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
                "start": proc["start"], "end": proc["end"],
                "text": f"[OVERLAP] {spk_text}",
                "translated_text": f"[OVERLAP] {spk_xlat}",
                "tts_path": proc["mixed_tts_path"],
                "is_overlap": True,
                "speaker_data": proc["speaker_data"],
            })
        segments.sort(key=lambda s: s["start"])
        yield f"Injected {len(processed_overlaps)} overlap clip(s) into timeline."

    # ── Audio assembly ────────────────────────────────────────────────────────
    yield "Assembling dubbed audio track…"
    dubbed_wav = str(work / "dubbed_audio.wav")
    assemble_audio_track(
        segments=segments,
        original_audio_path=vocals_wav,
        total_duration=total_duration,
        output_path=dubbed_wav,
        work_dir=str(work / "assembly"),
    )
    yield "Audio assembly complete."

    use_stereo_mux = False
    if music_wav and Path(music_wav).exists():
        yield "Mixing dubbed vocals with background music…"
        try:
            from .music_separation import mix_vocals_with_music
            mixed_wav = str(work / "dubbed_mixed.wav")
            mix_vocals_with_music(dubbed_wav, music_wav, mixed_wav, total_duration)
            dubbed_wav = mixed_wav
            use_stereo_mux = True
            yield "Music mix complete."
        except Exception as exc:
            yield f"Music mix failed ({exc})."

    yield "Muxing audio into video…"
    out_video = str(out / "redubbed.mp4")
    mux_audio_into_video(video_path, dubbed_wav, out_video, stereo=use_stereo_mux)
    yield "Video muxed."

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
    yield f"Timing: {n_timing_warn} segment(s) outside safe ratio."

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
        "speaker_ref_wavs": speaker_ref_wavs,
        "timing_df": timing_df,
        "timing_csv": timing_csv_path,
        "backtrans_df": backtrans_df,
        "backtrans_csv": backtrans_csv_path,
        # Full segment list (used by the segment editor / rebuild flow)
        "segments": segments,
    }


def rebuild_dubbed_video(
    segments: list[dict],
    analysis: dict,
    output_dir: str | None = None,
) -> str:
    """
    Re-assemble dubbed audio from existing TTS clips and mux back into the video.
    Used by the segment editor "Rebuild output" button after user edits.

    Returns the path to the new video file.
    """
    work = Path(analysis["work_dir"])
    out = work.parent if output_dir is None else Path(output_dir)
    vocals_wav = analysis["vocals_wav"]
    music_wav = analysis.get("music_wav")
    total_duration = analysis["total_duration"]
    video_path = analysis["video_path"]

    dubbed_wav = str(work / "dubbed_audio_rebuilt.wav")
    assemble_audio_track(
        segments=segments,
        original_audio_path=vocals_wav,
        total_duration=total_duration,
        output_path=dubbed_wav,
        work_dir=str(work / "assembly_rebuild"),
    )

    use_stereo_mux = False
    if music_wav and Path(music_wav).exists():
        try:
            from .music_separation import mix_vocals_with_music
            mixed_wav = str(work / "dubbed_mixed_rebuilt.wav")
            mix_vocals_with_music(dubbed_wav, music_wav, mixed_wav, total_duration)
            dubbed_wav = mixed_wav
            use_stereo_mux = True
        except Exception:
            pass

    out_video = str(out / "redubbed.mp4")
    mux_audio_into_video(video_path, dubbed_wav, out_video, stereo=use_stereo_mux)
    return out_video
