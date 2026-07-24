"""
Video Redubbing / Localization App
Gradio UI — two-stage flow: Analyze → [speaker assignment] → Generate
"""

import os
import tempfile
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from modules.pipeline import analyze_video, generate_from_analysis, run_pipeline, rebuild_dubbed_video
from modules.voice_library import (
    list_fish_voices,
    format_voice_label,
    load_saved_voices,
    save_cloned_voice,
)


# ── Language options ─────────────────────────────────────────────────────────

SOURCE_LANGUAGES = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("Chinese (Mandarin)", "zh"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Vietnamese", "vi"),
    ("Thai", "th"),
    ("Indonesian", "id"),
]

TARGET_LANGUAGES = [
    ("English", "English"),
    ("Chinese (Mandarin)", "Mandarin Chinese"),
    ("Spanish", "Spanish"),
    ("French", "French"),
    ("German", "German"),
    ("Japanese", "Japanese"),
    ("Korean", "Korean"),
    ("Portuguese (Brazilian)", "Brazilian Portuguese"),
    ("Russian", "Russian"),
    ("Arabic", "Arabic"),
    ("Hindi", "Hindi"),
    ("Italian", "Italian"),
    ("Dutch", "Dutch"),
    ("Polish", "Polish"),
    ("Turkish", "Turkish"),
    ("Vietnamese", "Vietnamese"),
    ("Thai", "Thai"),
    ("Indonesian", "Indonesian"),
]

LIB_LANGUAGE_FILTER = [
    ("All languages", ""),
    ("English", "en"),
    ("Chinese", "zh"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Italian", "it"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
]

MAX_SPEAKER_SLOTS = 6  # pre-created UI slots; covers virtually all real-world clips


def check_env() -> list[str]:
    warnings = []
    if not os.environ.get("FISH_AUDIO_API_KEY"):
        warnings.append("FISH_AUDIO_API_KEY not set")
    return warnings


def check_overlap_env() -> list[str]:
    import importlib.util
    issues = []
    if not os.environ.get("HF_TOKEN"):
        issues.append("HF_TOKEN not set")
    if importlib.util.find_spec("pyannote") is None:
        issues.append("pyannote.audio not installed")
    return issues


def check_music_env() -> list[str]:
    import importlib.util
    if importlib.util.find_spec("demucs") is None:
        return ["demucs not installed  (pip install demucs)"]
    return []


# ── Voice library helpers ─────────────────────────────────────────────────────

def _refresh_library_voices(language: str):
    try:
        items = list_fish_voices(language=language, page_size=50)
        choices = [(format_voice_label(item), item["_id"]) for item in items]
        if not choices:
            choices = [("No voices found — try a different language filter", "")]
            items = []
        return gr.update(choices=choices, value=None, interactive=True), items
    except Exception as e:
        return gr.update(choices=[(f"Error loading voices: {e}", "")], value=None), []


def _samples_for_voice(voice_id: str, voices: list[dict]) -> list[dict]:
    item = next((v for v in voices if v.get("_id") == voice_id), None)
    if not item:
        return []
    return [s for s in item.get("samples", []) if s.get("audio")]


def on_library_voice_select(voice_id: str | None, voices: list[dict]):
    if not voice_id or not voices:
        return (
            gr.update(visible=False, choices=[], value=None),
            gr.update(visible=False, value=None),
        )
    samples = _samples_for_voice(voice_id, voices)
    if not samples:
        return (
            gr.update(visible=False, choices=[], value=None),
            gr.update(visible=True, value=None, label="Preview (no samples)"),
        )
    first_url = samples[0]["audio"]
    if len(samples) == 1:
        return (
            gr.update(visible=False, choices=[], value=None),
            gr.update(visible=True, value=first_url, label="Preview"),
        )
    choices = [(s.get("title") or f"Sample {i + 1}", s["audio"]) for i, s in enumerate(samples)]
    return (
        gr.update(visible=True, choices=choices, value=first_url),
        gr.update(visible=True, value=first_url, label="Preview"),
    )


def on_sample_select(sample_url: str | None):
    if not sample_url:
        return gr.update(visible=False, value=None)
    return gr.update(visible=True, value=sample_url)


def _refresh_saved_voices() -> gr.update:
    voices = load_saved_voices()
    names = list(voices.keys())
    if names:
        return gr.update(choices=names, value=names[0], interactive=True)
    return gr.update(choices=[], value=None, interactive=True)


# ── Stage 1: Analyze ─────────────────────────────────────────────────────────

def run_analyze(
    video_file,
    source_lang_code: str,
    handle_overlaps: bool,
    preserve_music: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Yields N-tuples:
      (log, analysis_state, *speaker_slot_updates)
    where speaker_slot_updates is 6 × (group_visible, label_md, audio_val, mode_val,
                                       saved_choices, saved_val, lib_val, embed_hint)
    = 48 component updates + 2 base = 50 outputs total.
    """
    log_lines: list[str] = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines)

    def _blank_state():
        # analysis_state=None, speaker_panel hidden, all slots hidden
        slot_updates = []
        for _ in range(MAX_SPEAKER_SLOTS):
            slot_updates.extend([
                gr.update(visible=False),  # group
                gr.update(value=""),       # label
                gr.update(value=None),     # audio sample
                gr.update(value="Auto-clone (from video)"),  # mode radio
                gr.update(choices=[], value=None, visible=False),  # saved dd
                gr.update(value="", visible=False),           # lib textbox
                gr.update(value=None, visible=False),         # upload audio
                gr.update(value=""),                          # embed hint
            ])
        return (log(""), None, gr.update(visible=False), *slot_updates)

    missing = check_env()
    if missing:
        yield (log("Missing env: " + ", ".join(missing)), None, gr.update(visible=False),
               *([gr.update()] * (MAX_SPEAKER_SLOTS * 8)))
        return

    if video_file is None:
        yield _blank_state()
        return

    video_path = video_file if isinstance(video_file, str) else video_file.name
    output_dir = tempfile.mkdtemp(prefix="dub_analyze_")
    analysis_result = None

    try:
        gen = analyze_video(
            video_path=video_path,
            source_language=source_lang_code if source_lang_code != "auto" else None,
            output_dir=output_dir,
            preserve_music=preserve_music,
            handle_overlaps=handle_overlaps,
        )
        for item in gen:
            if isinstance(item, str):
                yield (log(item), None, gr.update(visible=False),
                       *([gr.update()] * (MAX_SPEAKER_SLOTS * 8)))
            elif isinstance(item, dict):
                analysis_result = item

        if not analysis_result:
            yield (log("Analyze returned no result."), None, gr.update(visible=False),
                   *([gr.update()] * (MAX_SPEAKER_SLOTS * 8)))
            return

        # Build speaker slot updates
        speaker_ref_wavs = analysis_result.get("speaker_ref_wavs", {})
        suggestions = analysis_result.get("embedding_suggestions", {})
        saved_voices = list(load_saved_voices().keys())
        all_speakers = sorted(speaker_ref_wavs.keys())

        slot_updates = []
        for i in range(MAX_SPEAKER_SLOTS):
            if i < len(all_speakers):
                spk = all_speakers[i]
                wav_path = speaker_ref_wavs.get(spk, "")
                sugg = suggestions.get(spk)
                hint = (f"_Closest saved voice: **{sugg[0]}** (similarity {sugg[1]:.2f})_"
                        if sugg else "")
                # Pre-fill mode based on suggestion
                _mode_val = "Auto-clone (from video)"
                _saved_val = None
                if sugg and sugg[1] >= 0.75:
                    _mode_val = "Use saved voice"
                    _saved_val = sugg[0]
                slot_updates.extend([
                    gr.update(visible=True),   # group
                    gr.update(value=f"### {spk}"),  # label
                    gr.update(value=wav_path if Path(wav_path).exists() else None),  # audio sample
                    gr.update(value=_mode_val),  # mode radio
                    gr.update(choices=saved_voices, value=_saved_val,
                              visible=(_mode_val == "Use saved voice")),  # saved dd
                    gr.update(value="", visible=(_mode_val == "Use library voice ID")),  # lib textbox
                    gr.update(value=None, visible=False),  # upload audio (always start empty)
                    gr.update(value=hint),     # embed hint
                ])
            else:
                slot_updates.extend([
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=None),
                    gr.update(value="Auto-clone (from video)"),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=""),
                ])

        final_log = log(f"Analysis complete. {len(all_speakers)} speaker(s) detected. "
                        "Assign voices below then click Generate Redub.")
        yield (final_log, analysis_result, gr.update(visible=True), *slot_updates)

    except Exception as e:
        tb = traceback.format_exc()
        yield (log(f"Analyze error: {e}\n\n{tb}"), None, gr.update(visible=False),
               *([gr.update()] * (MAX_SPEAKER_SLOTS * 8)))


# ── Stage 2: Generate ─────────────────────────────────────────────────────────

def run_generate(
    analysis_state,
    target_lang_name: str,
    source_lang_code: str,
    voice_mode: str,
    library_voice_id: str | None,
    saved_voice_name: str | None,
    run_backtranslation: bool,
    content_context: str,
    fallback_voice_id: str,
    min_ref_duration: float,
    # Per-speaker assignment fields (MAX_SPEAKER_SLOTS × 3 = 18)
    *slot_fields,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Yields N-tuples:
      (log, video, transcript, srt, orig_video,
       speaker_refs_state, save_group, save_speaker_dd, save_status,
       timing_table, backtrans_table, timing_csv, backtrans_csv,
       segments_state, gen_ctx_state, seg_editor_update, seg_df_update)
    = 17 outputs
    """
    log_lines: list[str] = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines)

    _EMPTY_TAIL = (
        {}, gr.update(visible=False), gr.update(visible=False, choices=[]), "",
        None, None, None, None,
        [], None, gr.update(visible=False), None,
    )

    def _yield_progress(msg):
        return (log(msg), None, None, None, None,
                *_EMPTY_TAIL)

    if analysis_state is None:
        yield log("Please run Analyze first."), None, None, None, None, *_EMPTY_TAIL
        return

    missing = check_env()
    if missing:
        yield log("Missing env: " + ", ".join(missing)), None, None, None, None, *_EMPTY_TAIL
        return

    slot_modes   = slot_fields[0::4]
    slot_saved   = slot_fields[1::4]
    slot_libs    = slot_fields[2::4]
    slot_uploads = slot_fields[3::4]

    speaker_ref_wavs = analysis_state.get("speaker_ref_wavs", {})
    all_speakers = sorted(speaker_ref_wavs.keys())
    speaker_assignments: dict[str, dict] = {}
    for i, spk in enumerate(all_speakers):
        if i >= MAX_SPEAKER_SLOTS:
            break
        _m = slot_modes[i] if i < len(slot_modes) else "Auto-clone (from video)"
        if _m == "Upload custom sample":
            _up = slot_uploads[i] if i < len(slot_uploads) else None
            if _up:
                speaker_assignments[spk] = {"mode": "upload", "wav_path": str(_up)}
            else:
                speaker_assignments[spk] = {"mode": "clone"}
        elif _m == "Use saved voice":
            _sn = slot_saved[i] if i < len(slot_saved) else None
            speaker_assignments[spk] = {"mode": "saved", "name": _sn or ""}
        elif _m == "Use library voice ID":
            _lid = slot_libs[i] if i < len(slot_libs) else ""
            speaker_assignments[spk] = {"mode": "library", "voice_id": (_lid or "").strip()}
        else:
            speaker_assignments[spk] = {"mode": "clone"}

    global_voice_id: str | None = None
    if not all_speakers:
        if voice_mode == "library":
            global_voice_id = library_voice_id or None
            if not global_voice_id:
                yield log("Please select a library voice first."), None, None, None, None, *_EMPTY_TAIL
                return
        elif voice_mode == "saved":
            global_voice_id = saved_voice_name or None
            if not global_voice_id:
                yield log("Please select a saved voice."), None, None, None, None, *_EMPTY_TAIL
                return
        speaker_assignments = None

    video_path = analysis_state["video_path"]
    out_video = transcript = srt = None
    timing_df = backtrans_df = timing_csv = backtrans_csv = None
    final_segments: list[dict] = []

    # Build generation context for the segment editor (voice refs by WAV path)
    gen_ctx = {
        "speaker_ref_wavs": speaker_ref_wavs,
        "speaker_assignments": speaker_assignments or {},
        "target_language": target_lang_name,
        "source_language": source_lang_code if source_lang_code != "auto" else None,
        "voice_mode": voice_mode,
        "voice_id": global_voice_id,
        "content_context": (content_context or "").strip() or None,
        "fallback_voice_id": (fallback_voice_id or "").strip() or None,
    }

    try:
        gen = generate_from_analysis(
            analysis=analysis_state,
            speaker_assignments=speaker_assignments if speaker_assignments else None,
            target_language=target_lang_name,
            source_language=source_lang_code if source_lang_code != "auto" else None,
            voice_mode=voice_mode,
            voice_id=global_voice_id,
            run_backtranslation=run_backtranslation,
            content_context=(content_context or "").strip() or None,
            fallback_voice_id=(fallback_voice_id or "").strip() or None,
            min_ref_duration=min_ref_duration,
        )
        for item in gen:
            if isinstance(item, str):
                yield (log(item), None, None, None, video_path,
                       *_EMPTY_TAIL)
            elif isinstance(item, dict):
                out_video = item.get("video")
                transcript = item.get("transcript")
                srt = item.get("srt")
                timing_df = item.get("timing_df")
                timing_csv = item.get("timing_csv")
                backtrans_df = item.get("backtrans_df")
                backtrans_csv = item.get("backtrans_csv")
                final_segments = item.get("segments", [])

        final_log = log("Done! Redubbed video is ready. Edit translations below if needed.")
        can_save = voice_mode == "clone" and bool(speaker_ref_wavs)
        speakers = sorted(speaker_ref_wavs.keys())
        multi = len(speakers) > 1
        speaker_dd_update = gr.update(
            choices=speakers, value=speakers[0] if speakers else None, visible=multi
        )

        # Build segment editor DataFrame
        seg_df = _segments_to_df(final_segments)

        yield (final_log, out_video, transcript, srt, video_path,
               speaker_ref_wavs, gr.update(visible=can_save), speaker_dd_update, "",
               timing_df, backtrans_df, timing_csv, backtrans_csv,
               final_segments, gen_ctx, gr.update(visible=True), seg_df)

    except Exception as e:
        tb = traceback.format_exc()
        yield (log(f"Error: {e}\n\n{tb}"), None, None, None,
               analysis_state.get("video_path"),
               {}, gr.update(visible=False), gr.update(visible=False, choices=[]), "",
               None, None, None, None,
               [], None, gr.update(visible=False), None)


# ── Save voice handler ────────────────────────────────────────────────────────

def handle_save_voice(name: str, speaker_id: str | None, speaker_refs: dict):
    name = (name or "").strip()
    if not name:
        return gr.update(), "Please enter a name for the voice."
    if not speaker_refs:
        return gr.update(), "No reference audio available to save."

    wav_path: str | None = None
    if speaker_id and speaker_id in speaker_refs:
        wav_path = speaker_refs[speaker_id]
    else:
        wav_path = next(iter(speaker_refs.values()), None)

    if not wav_path or not Path(wav_path).exists():
        return gr.update(), "Reference audio file not found on disk."
    try:
        save_cloned_voice(name, wav_path)
        voices = load_saved_voices()
        return gr.update(choices=list(voices.keys()), value=name), f"Voice **{name}** saved!"
    except Exception as e:
        return gr.update(), f"Error saving voice: {e}"


# ── Segment editor helpers ────────────────────────────────────────────────────

import pandas as pd


def _segments_to_df(segments: list[dict]):
    """Convert segments list to a DataFrame for the editor."""
    rows = []
    for i, s in enumerate(segments):
        if s.get("is_overlap"):
            continue
        rows.append({
            "#": i,
            "Speaker": s.get("speaker", "SPEAKER_00"),
            "Start": round(s.get("start", 0.0), 2),
            "End": round(s.get("end", 0.0), 2),
            "Original": s.get("text", ""),
            "Translation": s.get("translated_text", ""),
            "TTS": "✓" if s.get("tts_path") else ("⚠" if s.get("tts_error") else "—"),
        })
    if not rows:
        return pd.DataFrame(columns=["#", "Speaker", "Start", "End", "Original", "Translation", "TTS"])
    return pd.DataFrame(rows)


def regenerate_segment_tts(
    seg_df,
    row_idx: int,
    segments_state: list[dict],
    gen_ctx: dict | None,
    analysis_state: dict | None,
):
    """Re-synthesize TTS for a single segment (row_idx into segments_state)."""
    if gen_ctx is None or analysis_state is None:
        return seg_df, segments_state, "Run Generate first."
    if not (0 <= row_idx < len(segments_state)):
        return seg_df, segments_state, f"Row {row_idx} out of range (0–{len(segments_state)-1})."

    seg = segments_state[row_idx]
    if seg.get("is_overlap"):
        return seg_df, segments_state, "Cannot regenerate overlap segments."

    # Get updated translation from the DataFrame (user may have edited it)
    try:
        if seg_df is not None and hasattr(seg_df, "iloc"):
            # Find the row in the df where "#" == row_idx
            if "#" in seg_df.columns:
                mask = seg_df["#"] == row_idx
                if mask.any():
                    new_text = str(seg_df.loc[mask, "Translation"].iloc[0]).strip()
                    if new_text:
                        seg = {**seg, "translated_text": new_text}
    except Exception:
        pass

    text = seg.get("translated_text", "").strip()
    if not text:
        return seg_df, segments_state, f"Segment {row_idx} has no translated text to synthesize."

    # Resolve voice reference for this speaker
    from modules.tts import synthesize, build_voice_reference
    spk = seg.get("speaker", "SPEAKER_00")
    asgn = (gen_ctx.get("speaker_assignments") or {}).get(spk, {"mode": "clone"})
    _mode = asgn.get("mode", "clone")
    ref_bytes = ref_text = ref_id = None

    if _mode == "library":
        ref_id = asgn.get("voice_id") or gen_ctx.get("voice_id")
    elif _mode == "upload":
        _up_path = asgn.get("wav_path", "")
        if _up_path and Path(_up_path).exists():
            ref_bytes, _ = build_voice_reference(_up_path)
        else:
            _mode = "clone"  # uploaded file gone — fall through
    elif _mode == "saved":
        _nm = asgn.get("name", "")
        from modules.voice_library import load_saved_voices
        _sv = load_saved_voices()
        if _nm and _nm in _sv:
            ref_bytes, _ = build_voice_reference(_sv[_nm]["audio_path"])
            ref_text = _sv[_nm].get("reference_text", "")
        else:
            _mode = "clone"
    if _mode == "clone":
        _wp = (gen_ctx.get("speaker_ref_wavs") or {}).get(spk)
        if _wp and Path(_wp).exists():
            ref_bytes, _ = build_voice_reference(_wp)
        elif gen_ctx.get("voice_id"):
            ref_id = gen_ctx["voice_id"]

    try:
        audio = synthesize(
            text=text,
            reference_audio_bytes=ref_bytes,
            reference_text=ref_text or "",
            reference_id=ref_id,
        )
        work = Path(analysis_state["work_dir"])
        tts_dir = work / "tts"
        tts_dir.mkdir(exist_ok=True)
        out_path = tts_dir / f"seg_{row_idx:04d}_edit.mp3"
        out_path.write_bytes(audio)

        updated_segments = list(segments_state)
        updated_segments[row_idx] = {
            **seg,
            "tts_path": str(out_path),
            "tts_error": None,
        }
        updated_df = _segments_to_df(updated_segments)
        return updated_df, updated_segments, f"Segment {row_idx} re-synthesized."

    except Exception as exc:
        return seg_df, segments_state, f"TTS error for segment {row_idx}: {exc}"


def rebuild_output_handler(
    segments_state: list[dict],
    seg_df,
    analysis_state: dict | None,
):
    """Reassemble dubbed video from current segments (after editing)."""
    if analysis_state is None:
        return None, "Run Generate first."
    if not segments_state:
        return None, "No segments available."

    # Flush any edited translations from the DataFrame back into segments
    segments = list(segments_state)
    try:
        if seg_df is not None and hasattr(seg_df, "iterrows"):
            for _, row in seg_df.iterrows():
                idx = int(row.get("#", -1))
                if 0 <= idx < len(segments) and not segments[idx].get("is_overlap"):
                    new_text = str(row.get("Translation", "")).strip()
                    if new_text:
                        segments[idx] = {**segments[idx], "translated_text": new_text}
    except Exception:
        pass

    try:
        new_video = rebuild_dubbed_video(segments, analysis_state)
        return new_video, "Output rebuilt successfully."
    except Exception as exc:
        return None, f"Rebuild error: {exc}"


# ── Build UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    src_choices = [(label, code) for label, code in SOURCE_LANGUAGES]
    tgt_choices = [(label, name) for label, name in TARGET_LANGUAGES]

    warnings = check_env()
    warn_text = "⚠ " + " | ".join(warnings) if warnings else ""

    with gr.Blocks(title="Video Redubbing") as demo:
        gr.Markdown("# Video Redubbing & Localization")
        gr.Markdown(
            "Two-stage workflow: **Analyze** detects speakers and extracts samples, "
            "then you assign voices per character before **Generate Redub** dubs the video. "
            "Powered by **WhisperX** (ASR + diarization) + **Fish Audio** (TTS) + "
            "**Demucs** (music separation) + **Groq / Gemini / Ollama** (translation)."
        )
        if warn_text:
            gr.Markdown(f"**{warn_text}**")

        # State shared between stages
        analysis_state = gr.State(None)  # dict from analyze_video()

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Step 1 — Input")
                video_input = gr.Video(label="Upload video", sources=["upload"])

                with gr.Row():
                    source_lang = gr.Dropdown(
                        label="Source language",
                        choices=src_choices,
                        value="auto",
                    )
                    target_lang = gr.Dropdown(
                        label="Target language",
                        choices=tgt_choices,
                        value="English",
                    )

                content_context_box = gr.Textbox(
                    label="Content context (optional)",
                    placeholder=(
                        "Briefly describe the video: characters, setting, tone, "
                        "or terminology. E.g. 'Action anime — protagonist Ryuu "
                        "is brash; keep honorifics like -san.'"
                    ),
                    lines=2,
                    max_lines=5,
                )

                _overlap_issues = check_overlap_env()
                _overlap_label = (
                    "⚡ Detect & handle overlapping speakers"
                    if not _overlap_issues
                    else "⚡ Detect & handle overlapping speakers — ⚠ " + "; ".join(_overlap_issues)
                )
                handle_overlaps_check = gr.Checkbox(
                    label=_overlap_label,
                    value=not bool(_overlap_issues),
                    info="Phase B: splits simultaneous speech with SepFormer, dubs each stream separately.",
                )

                _music_issues = check_music_env()
                _music_label = (
                    "Preserve background music (Demucs)"
                    if not _music_issues
                    else "Preserve background music (Demucs) — ⚠ " + "; ".join(_music_issues)
                )
                preserve_music_check = gr.Checkbox(
                    label=_music_label,
                    value=not bool(_music_issues),
                    info="Separates vocals from background music; dubs vocals only then re-mixes.",
                )

                analyze_btn = gr.Button("Analyze", variant="secondary", size="lg")

            # ── Right column: progress ───────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Progress")
                log_box = gr.Textbox(
                    label="Pipeline log",
                    lines=14,
                    max_lines=22,
                    interactive=False,
                )

        # ── Step 2: Speaker assignment panel ─────────────────────────────────
        with gr.Group(visible=False) as speaker_panel:
            gr.Markdown("### Step 2 — Assign voices per speaker")
            gr.Markdown(
                "Each detected speaker has a sample clip. Choose how to voice them:\n"
                "- **Auto-clone** — clone directly from the video (default)\n"
                "- **Use saved voice** — pick a voice you saved from a previous run\n"
                "- **Use library voice ID** — paste a Fish Audio library voice ID"
            )

            saved_voices_initial = list(load_saved_voices().keys())

            # Pre-create MAX_SPEAKER_SLOTS slots
            spk_groups   = []
            spk_labels   = []
            spk_audios   = []
            spk_modes    = []
            spk_saveds   = []
            spk_libs     = []
            spk_uploads  = []
            spk_hints    = []

            for _i in range(MAX_SPEAKER_SLOTS):
                with gr.Group(visible=False) as _grp:
                    _lbl = gr.Markdown("")
                    _aud = gr.Audio(label="Sample clip", interactive=False, visible=True)
                    _hint = gr.Markdown("")
                    _mode = gr.Radio(
                        choices=[
                            "Auto-clone (from video)",
                            "Use saved voice",
                            "Use library voice ID",
                            "Upload custom sample",
                        ],
                        value="Auto-clone (from video)",
                        label="Voice assignment",
                    )
                    with gr.Row(visible=False) as _saved_row:
                        _saved = gr.Dropdown(
                            label="Saved voice",
                            choices=saved_voices_initial,
                            value=None,
                            interactive=True,
                            visible=False,
                        )
                    with gr.Row(visible=False) as _lib_row:
                        _lib = gr.Textbox(
                            label="Fish Audio library voice ID",
                            placeholder="Paste voice ID from fish.audio/model",
                            visible=False,
                        )
                    _upload = gr.Audio(
                        label="Upload your own voice sample",
                        sources=["upload"],
                        type="filepath",
                        interactive=True,
                        visible=False,
                    )
                spk_groups.append(_grp)
                spk_labels.append(_lbl)
                spk_audios.append(_aud)
                spk_modes.append(_mode)
                spk_saveds.append(_saved)
                spk_libs.append(_lib)
                spk_uploads.append(_upload)
                spk_hints.append(_hint)

            # Wire each slot's mode radio to show/hide sub-inputs
            def _make_mode_handler(idx):
                def _on_mode(mode):
                    return (
                        gr.update(visible=(mode == "Use saved voice")),
                        gr.update(visible=(mode == "Use library voice ID")),
                        gr.update(visible=(mode == "Upload custom sample")),
                    )
                return _on_mode

            for _i, (_mode_comp, _saved_comp, _lib_comp, _upload_comp) in enumerate(
                zip(spk_modes, spk_saveds, spk_libs, spk_uploads)
            ):
                _mode_comp.change(
                    fn=_make_mode_handler(_i),
                    inputs=[_mode_comp],
                    outputs=[_saved_comp, _lib_comp, _upload_comp],
                )

        # ── Step 3: Generation settings + button ─────────────────────────────
        with gr.Accordion("Voice selection (global fallback)", open=False):
            voice_mode_radio = gr.Radio(
                choices=[
                    ("Clone speaker from video", "clone"),
                    ("Use a Fish Audio library voice", "library"),
                    ("Use a saved voice", "saved"),
                ],
                value="clone",
                label="Global voice mode (used when speaker assignments panel is empty)",
            )

            with gr.Group(visible=True) as clone_group:
                gr.Markdown(
                    "_Clones the speaker's voice from your video. Best with 15s+ of clean speech._"
                )
                with gr.Accordion("Advanced clone settings", open=False):
                    min_ref_dur_slider = gr.Slider(
                        label="Min reference duration (s)",
                        minimum=2.0, maximum=15.0, step=0.5, value=6.0,
                        info="Speakers below this threshold use the fallback voice instead of cloning.",
                    )
                    fallback_voice_input = gr.Textbox(
                        label="Fallback voice ID",
                        placeholder="Fish Audio voice ID",
                        info="Fish Audio voice to use when reference clip is too short.",
                    )

            with gr.Group(visible=False) as library_group:
                lib_voices_state = gr.State([])
                with gr.Row():
                    lib_lang_filter = gr.Dropdown(
                        label="Filter by language",
                        choices=LIB_LANGUAGE_FILTER, value="", scale=3,
                    )
                    lib_load_btn = gr.Button("Load voices", size="sm", scale=1)
                library_voice_dd = gr.Dropdown(
                    label="Select library voice", choices=[], value=None, interactive=True,
                )
                lib_sample_dd = gr.Dropdown(
                    label="Sample clip", choices=[], value=None, visible=False, interactive=True,
                )
                lib_preview_audio = gr.Audio(label="Preview", visible=False, interactive=False)

            with gr.Group(visible=False) as saved_group:
                initial_saved = list(load_saved_voices().keys())
                with gr.Row():
                    saved_voice_dd = gr.Dropdown(
                        label="Select saved voice",
                        choices=initial_saved,
                        value=initial_saved[0] if initial_saved else None,
                        interactive=True, scale=3,
                    )
                    saved_refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                if not initial_saved:
                    gr.Markdown("_No saved voices yet._")

        back_trans_check = gr.Checkbox(
            label="Run back-translation QA (extra round-trip)",
            value=False,
        )

        generate_btn = gr.Button("Generate Redub", variant="primary", size="lg")

        # ── Results ──────────────────────────────────────────────────────────
        gr.Markdown("### Results")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Original**")
                original_player = gr.Video(label="Original video", interactive=False)
            with gr.Column():
                gr.Markdown("**Redubbed**")
                redubbed_player = gr.Video(label="Redubbed video", interactive=False)

        with gr.Row():
            transcript_dl = gr.File(label="transcript.json", interactive=False)
            srt_dl = gr.File(label="translated.srt", interactive=False)

        with gr.Accordion("QA Reports", open=False):
            with gr.Tabs():
                with gr.Tab("Timing Report"):
                    gr.Markdown(
                        "Ratio = TTS raw duration ÷ original slot. "
                        "Values outside 0.85–1.15× are flagged ⚠."
                    )
                    timing_table = gr.DataFrame(label="Segment timing", interactive=False, wrap=False)
                    timing_csv_dl = gr.File(label="timing_report.csv", interactive=False)

                with gr.Tab("Back-translation QA"):
                    gr.Markdown(
                        "Back-translates dubbed text → original language. "
                        "Low-similarity segments (<0.50) are flagged ⚠."
                    )
                    backtrans_table = gr.DataFrame(
                        label="Back-translation", interactive=False, wrap=True,
                    )
                    backtrans_csv_dl = gr.File(label="back_translation.csv", interactive=False)

        # ── Segment editor (Step 4) ───────────────────────────────────────────
        segments_state = gr.State([])
        gen_ctx_state = gr.State(None)

        with gr.Group(visible=False) as seg_editor_group:
            gr.Markdown("### Segment Editor")
            gr.Markdown(
                "Edit translations in the **Translation** column, then click "
                "**Regenerate TTS for row N** to re-synthesize just that segment, "
                "or **Rebuild Output** to reassemble the full video."
            )
            seg_df_editor = gr.Dataframe(
                label="Segments",
                interactive=True,
                wrap=True,
                column_widths=["4%", "8%", "5%", "5%", "34%", "34%", "5%"],
            )
            with gr.Row():
                regen_row_input = gr.Number(
                    label="Row # to regenerate",
                    value=0,
                    precision=0,
                    minimum=0,
                    step=1,
                    scale=1,
                )
                regen_seg_btn = gr.Button("Regenerate TTS for row N", scale=2)
                rebuild_btn = gr.Button("Rebuild Output", variant="primary", scale=2)
            regen_status = gr.Markdown("")

        # ── Save cloned voice panel ───────────────────────────────────────────
        speaker_refs_state = gr.State(value={})

        with gr.Group(visible=False) as save_voice_group:
            gr.Markdown("---\n**Save cloned voice(s) for future use:**")
            save_speaker_dd = gr.Dropdown(
                label="Speaker to save", choices=[], value=None,
                visible=False, interactive=True,
                info="Multiple speakers detected — select which one to save.",
            )
            with gr.Row():
                save_name_input = gr.Textbox(
                    label="Voice name", placeholder="e.g. Male lead — English", scale=3,
                )
                save_btn = gr.Button("Save voice", variant="secondary", scale=1)
            save_status = gr.Markdown("")

        # ── Event wiring ──────────────────────────────────────────────────────

        def on_mode_change(mode):
            return (
                gr.update(visible=mode == "clone"),
                gr.update(visible=mode == "library"),
                gr.update(visible=mode == "saved"),
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
            )

        voice_mode_radio.change(
            fn=on_mode_change,
            inputs=[voice_mode_radio],
            outputs=[clone_group, library_group, saved_group,
                     lib_preview_audio, lib_sample_dd],
        )

        lib_load_btn.click(
            fn=_refresh_library_voices,
            inputs=[lib_lang_filter],
            outputs=[library_voice_dd, lib_voices_state],
        )
        lib_lang_filter.change(
            fn=_refresh_library_voices,
            inputs=[lib_lang_filter],
            outputs=[library_voice_dd, lib_voices_state],
        )
        library_voice_dd.change(
            fn=on_library_voice_select,
            inputs=[library_voice_dd, lib_voices_state],
            outputs=[lib_sample_dd, lib_preview_audio],
        )
        lib_sample_dd.change(
            fn=on_sample_select,
            inputs=[lib_sample_dd],
            outputs=[lib_preview_audio],
        )
        saved_refresh_btn.click(
            fn=_refresh_saved_voices,
            inputs=[],
            outputs=[saved_voice_dd],
        )

        # ── Analyze button ────────────────────────────────────────────────────
        _analyze_outputs = (
            [log_box, analysis_state, speaker_panel]
            + [c for quad in zip(spk_groups, spk_labels, spk_audios,
                                 spk_modes, spk_saveds, spk_libs, spk_uploads, spk_hints)
               for c in quad]
        )
        analyze_btn.click(
            fn=run_analyze,
            inputs=[video_input, source_lang, handle_overlaps_check, preserve_music_check],
            outputs=_analyze_outputs,
        )

        # ── Generate button ───────────────────────────────────────────────────
        # Per-speaker inputs: mode, saved, lib, upload — for each slot
        _slot_inputs = [c for quad in zip(spk_modes, spk_saveds, spk_libs, spk_uploads) for c in quad]

        generate_btn.click(
            fn=run_generate,
            inputs=[
                analysis_state,
                target_lang, source_lang,
                voice_mode_radio, library_voice_dd, saved_voice_dd,
                back_trans_check, content_context_box,
                fallback_voice_input, min_ref_dur_slider,
            ] + _slot_inputs,
            outputs=[
                log_box, redubbed_player, transcript_dl, srt_dl, original_player,
                speaker_refs_state, save_voice_group, save_speaker_dd, save_status,
                timing_table, backtrans_table, timing_csv_dl, backtrans_csv_dl,
                # Segment editor outputs
                segments_state, gen_ctx_state, seg_editor_group, seg_df_editor,
            ],
        )

        regen_seg_btn.click(
            fn=regenerate_segment_tts,
            inputs=[seg_df_editor, regen_row_input, segments_state, gen_ctx_state, analysis_state],
            outputs=[seg_df_editor, segments_state, regen_status],
        )

        rebuild_btn.click(
            fn=rebuild_output_handler,
            inputs=[segments_state, seg_df_editor, analysis_state],
            outputs=[redubbed_player, regen_status],
        )

        save_btn.click(
            fn=handle_save_voice,
            inputs=[save_name_input, save_speaker_dd, speaker_refs_state],
            outputs=[saved_voice_dd, save_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        theme=gr.themes.Soft(),
    )
