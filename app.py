"""
Video Redubbing / Localization App — Phase A
Gradio UI wrapping the end-to-end Fish Audio pipeline.
"""

import os
import tempfile
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from modules.pipeline import run_pipeline
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


def check_env() -> list[str]:
    warnings = []
    if not os.environ.get("FISH_AUDIO_API_KEY"):
        warnings.append("FISH_AUDIO_API_KEY not set")
    return warnings


def check_overlap_env() -> list[str]:
    """Non-fatal warnings shown in the overlap checkbox label.
    Uses find_spec only — never imports pyannote/speechbrain at startup, since
    those drag in lightning + torch._dynamo which permanently occupy hundreds of MB.
    """
    import importlib.util
    issues = []
    if not os.environ.get("HF_TOKEN"):
        issues.append("HF_TOKEN not set (needed for pyannote diarization)")
    if importlib.util.find_spec("pyannote") is None:
        issues.append("pyannote.audio not installed")
    if importlib.util.find_spec("speechbrain") is None:
        issues.append("speechbrain not installed")
    return issues


def check_music_env() -> list[str]:
    """Non-fatal warnings shown in the music preservation checkbox label."""
    import importlib.util
    if importlib.util.find_spec("demucs") is None:
        return ["demucs not installed  (pip install demucs)"]
    return []


# ── Voice library helpers ─────────────────────────────────────────────────────

def _refresh_library_voices(language: str):
    """Fetch Fish Audio public voices. Returns (dropdown_update, raw_items_list)."""
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
    """Return the samples list for the selected voice from the cached items."""
    item = next((v for v in voices if v.get("_id") == voice_id), None)
    if not item:
        return []
    return [s for s in item.get("samples", []) if s.get("audio")]


def on_library_voice_select(voice_id: str | None, voices: list[dict]):
    """Update the sample picker and audio preview when a library voice is chosen."""
    if not voice_id or not voices:
        return (
            gr.update(visible=False, choices=[], value=None),
            gr.update(visible=False, value=None),
        )
    samples = _samples_for_voice(voice_id, voices)
    if not samples:
        return (
            gr.update(visible=False, choices=[], value=None),
            gr.update(visible=True, value=None, label="Preview (no samples available)"),
        )
    first_url = samples[0]["audio"]
    if len(samples) == 1:
        return (
            gr.update(visible=False, choices=[], value=None),
            gr.update(visible=True, value=first_url, label="Preview"),
        )
    # Multiple samples: populate picker and show first
    choices = [(s.get("title") or f"Sample {i + 1}", s["audio"]) for i, s in enumerate(samples)]
    return (
        gr.update(visible=True, choices=choices, value=first_url),
        gr.update(visible=True, value=first_url, label="Preview"),
    )


def on_sample_select(sample_url: str | None):
    """Update audio player when user picks a different sample."""
    if not sample_url:
        return gr.update(visible=False, value=None)
    return gr.update(visible=True, value=sample_url)


def _refresh_saved_voices() -> gr.update:
    voices = load_saved_voices()
    names = list(voices.keys())
    if names:
        return gr.update(choices=names, value=names[0], interactive=True)
    return gr.update(choices=[], value=None, interactive=True)


# ── Main pipeline handler ─────────────────────────────────────────────────────

def generate_redub(
    video_file,
    source_lang_code: str,
    target_lang_name: str,
    voice_mode: str,
    library_voice_id: str | None,
    saved_voice_name: str | None,
    run_backtranslation: bool,
    handle_overlaps: bool,
    preserve_music: bool,
    content_context: str,
    fallback_voice_id: str,
    min_ref_duration: float,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Generator yielding 13-tuple:
      log, out_video, transcript, srt, orig_video,
      speaker_refs_state, save_group_update, save_speaker_dd_update, save_status,
      timing_table, backtrans_table, timing_csv_dl, backtrans_csv_dl
    """
    log_lines: list[str] = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Placeholder for all outputs after log when something goes wrong early
    _EMPTY = (None, None, None, None,
              {}, gr.update(visible=False), gr.update(visible=False, choices=[]), "",
              None, None, None, None)

    missing = check_env()
    if missing:
        err = "Missing environment variables: " + ", ".join(missing)
        yield err, *_EMPTY
        return

    if video_file is None:
        yield "Please upload a video file.", *_EMPTY
        return

    video_path = video_file if isinstance(video_file, str) else video_file.name

    # Resolve voice_id from whichever sub-picker is active
    voice_id: str | None = None
    if voice_mode == "library":
        voice_id = library_voice_id or None
        if not voice_id:
            yield log("Please select a library voice first."), *_EMPTY
            return
    elif voice_mode == "saved":
        voice_id = saved_voice_name or None
        if not voice_id:
            yield log("Please select a saved voice, or run a clone job first."), *_EMPTY
            return

    output_dir = tempfile.mkdtemp(prefix="dub_output_")

    out_video = transcript = srt = None
    speaker_ref_wavs: dict[str, str] = {}
    timing_df = backtrans_df = timing_csv = backtrans_csv = None

    try:
        gen = run_pipeline(
            video_path=video_path,
            source_language=source_lang_code,
            target_language=target_lang_name,
            output_dir=output_dir,
            voice_mode=voice_mode,
            voice_id=voice_id,
            run_backtranslation=run_backtranslation,
            handle_overlaps=handle_overlaps,
            preserve_music=preserve_music,
            content_context=(content_context or "").strip() or None,
            fallback_voice_id=(fallback_voice_id or "").strip() or None,
            min_ref_duration=min_ref_duration,
        )
        for item in gen:
            if isinstance(item, str):
                current_log = log(item)
                yield (current_log, None, None, None, video_path,
                       {}, gr.update(visible=False), gr.update(visible=False, choices=[]), "",
                       None, None, None, None)
            elif isinstance(item, dict):
                out_video = item.get("video")
                transcript = item.get("transcript")
                srt = item.get("srt")
                speaker_ref_wavs = item.get("speaker_ref_wavs") or {}
                timing_df = item.get("timing_df")
                timing_csv = item.get("timing_csv")
                backtrans_df = item.get("backtrans_df")
                backtrans_csv = item.get("backtrans_csv")

        final_log = log("Done! Redubbed video is ready.")
        can_save = voice_mode == "clone" and bool(speaker_ref_wavs)
        speakers = sorted(speaker_ref_wavs.keys())
        multi = len(speakers) > 1
        speaker_dd_update = gr.update(
            choices=speakers,
            value=speakers[0] if speakers else None,
            visible=multi,
        )
        yield (final_log, out_video, transcript, srt, video_path,
               speaker_ref_wavs, gr.update(visible=can_save), speaker_dd_update, "",
               timing_df, backtrans_df, timing_csv, backtrans_csv)

    except Exception as e:
        tb = traceback.format_exc()
        error_log = log(f"Error: {e}\n\n{tb}")
        yield (error_log, None, None, None, video_path,
               {}, gr.update(visible=False), gr.update(visible=False, choices=[]), "",
               None, None, None, None)


# ── Save voice handler ────────────────────────────────────────────────────────

def handle_save_voice(name: str, speaker_id: str | None, speaker_refs: dict):
    """Save a cloned reference WAV under a user-chosen name."""
    name = (name or "").strip()
    if not name:
        return gr.update(), "Please enter a name for the voice."
    if not speaker_refs:
        return gr.update(), "No reference audio available to save."

    # Pick the requested speaker's WAV, or the first available one
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


# ── Build UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    src_choices = [(label, code) for label, code in SOURCE_LANGUAGES]
    tgt_choices = [(label, name) for label, name in TARGET_LANGUAGES]

    warnings = check_env()
    warn_text = "⚠ " + " | ".join(warnings) if warnings else ""

    with gr.Blocks(title="Video Redubbing — Fish Audio") as demo:
        gr.Markdown("# Video Redubbing & Localization")
        gr.Markdown(
            "Upload a video and get a full dubbed version in your target language. "
            "Supports **vocal/music separation** (Demucs) to preserve background music while dubbing dialogue, "
            "and handles **overlapping multi-speaker** audio by separating simultaneous speech into "
            "clean per-speaker streams, dubbing each independently, and re-mixing. "
            "Powered by **Fish Audio** (ASR + TTS) + **Demucs** (music separation) + "
            "**pyannote** (diarization) + **Groq / Gemini / Ollama** (translation)."
        )

        if warn_text:
            gr.Markdown(f"**{warn_text}**")

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")
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
                        "or terminology to preserve. E.g. 'Action anime — protagonist Ryuu "
                        "is brash; keep honorifics like -san. Avoid casual slang.'"
                    ),
                    lines=2,
                    max_lines=5,
                )

                # ── Voice selection ──────────────────────────────────────────
                with gr.Accordion("Voice Selection", open=True):
                    voice_mode_radio = gr.Radio(
                        choices=[
                            ("Clone speaker from this video", "clone"),
                            ("Use a Fish Audio library voice", "library"),
                            ("Use a saved voice", "saved"),
                        ],
                        value="clone",
                        label="Voice source",
                    )

                    # Clone info hint + advanced settings
                    with gr.Group(visible=True) as clone_group:
                        gr.Markdown(
                            "_Clones the speaker's voice from your video. "
                            "Best with **15 s+** of clean speech. "
                            "Reference clips are loudness-normalized automatically._"
                        )
                        with gr.Accordion("Advanced clone settings", open=False):
                            min_ref_dur_slider = gr.Slider(
                                label="Min reference duration (s)",
                                minimum=2.0,
                                maximum=15.0,
                                step=0.5,
                                value=6.0,
                                info=(
                                    "Multi-speaker mode: speakers with less clean audio "
                                    "than this threshold use the fallback voice instead of cloning."
                                ),
                            )
                            fallback_voice_input = gr.Textbox(
                                label="Fallback voice ID (for short references)",
                                placeholder="Fish Audio voice ID",
                                info=(
                                    "When a speaker's reference is below the threshold, "
                                    "use this Fish Audio library voice instead of a weak clone. "
                                    "Find IDs at fish.audio/model — copy the ID from the model URL. "
                                    "Leave blank to clone anyway with a quality warning."
                                ),
                            )

                    # Library voice picker
                    with gr.Group(visible=False) as library_group:
                        lib_voices_state = gr.State([])
                        with gr.Row():
                            lib_lang_filter = gr.Dropdown(
                                label="Filter by language",
                                choices=LIB_LANGUAGE_FILTER,
                                value="",
                                scale=3,
                            )
                            lib_load_btn = gr.Button("Load voices", size="sm", scale=1)
                        library_voice_dd = gr.Dropdown(
                            label="Select library voice",
                            choices=[],
                            value=None,
                            interactive=True,
                        )
                        lib_sample_dd = gr.Dropdown(
                            label="Sample clip",
                            choices=[],
                            value=None,
                            visible=False,
                            interactive=True,
                        )
                        lib_preview_audio = gr.Audio(
                            label="Preview",
                            visible=False,
                            interactive=False,
                        )

                    # Saved voice picker
                    with gr.Group(visible=False) as saved_group:
                        initial_saved = list(load_saved_voices().keys())
                        with gr.Row():
                            saved_voice_dd = gr.Dropdown(
                                label="Select saved voice",
                                choices=initial_saved,
                                value=initial_saved[0] if initial_saved else None,
                                interactive=True,
                                scale=3,
                            )
                            saved_refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                        if not initial_saved:
                            gr.Markdown(
                                "_No saved voices yet. Run a clone job and save the voice using "
                                "the panel that appears after the run._"
                            )

                back_trans_check = gr.Checkbox(
                    label="Run back-translation QA (adds an extra translation round-trip)",
                    value=False,
                )

                _overlap_issues = check_overlap_env()
                _overlap_label = (
                    "⚡ Detect & handle overlapping speakers (Phase B)"
                    if not _overlap_issues
                    else (
                        "⚡ Detect & handle overlapping speakers (Phase B) "
                        "— ⚠ " + "; ".join(_overlap_issues)
                    )
                )
                handle_overlaps_check = gr.Checkbox(
                    label=_overlap_label,
                    value=not bool(_overlap_issues),
                    info=(
                        "Uses pyannote diarization + SepFormer separation to split "
                        "overlapping-speech windows into clean per-speaker streams, "
                        "then dubs each independently and re-mixes. "
                        "Requires HF_TOKEN in .env, pyannote.audio, and speechbrain."
                    ),
                )

                _music_issues = check_music_env()
                _music_label = (
                    "Preserve background music (Demucs vocal separation)"
                    if not _music_issues
                    else (
                        "Preserve background music (Demucs) — ⚠ "
                        + "; ".join(_music_issues)
                    )
                )
                preserve_music_check = gr.Checkbox(
                    label=_music_label,
                    value=not bool(_music_issues),
                    info=(
                        "Separates the audio into a vocals stem (processed: ASR → translate → TTS) "
                        "and a background music / instrumental stem (kept untouched), "
                        "then re-mixes them into a stereo output. "
                        "First run downloads ~200 MB of Demucs weights; subsequent runs are instant. "
                        "Disable for clips with no background music to skip the separation step."
                    ),
                )

                generate_btn = gr.Button("Generate Redub", variant="primary", size="lg")

            # ── Right column: progress ───────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Progress")
                log_box = gr.Textbox(
                    label="Pipeline log",
                    lines=14,
                    max_lines=22,
                    interactive=False,
                )

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

        # ── QA Reports ────────────────────────────────────────────────────────
        with gr.Accordion("QA Reports", open=False) as qa_accordion:
            with gr.Tabs():
                with gr.Tab("Timing Report"):
                    gr.Markdown(
                        "Ratio = TTS raw duration ÷ original slot. "
                        "Values outside **0.85 – 1.15×** are flagged ⚠ "
                        "(audio was stretched or compressed beyond the inaudible range; "
                        "the pipeline retranslated these segments up to 2× before clamping). "
                        "**↻N** in the Retries column shows how many retranslation attempts were made."
                    )
                    timing_table = gr.DataFrame(
                        label="Segment timing",
                        interactive=False,
                        wrap=False,
                    )
                    timing_csv_dl = gr.File(label="timing_report.csv", interactive=False)

                with gr.Tab("Back-translation QA"):
                    gr.Markdown(
                        "Back-translates each segment's dubbed text back into English. "
                        "Segments where the back-translation is very different from the "
                        "original (similarity < 0.50) are flagged ⚠. "
                        "Enable the checkbox above before running to populate this tab."
                    )
                    backtrans_table = gr.DataFrame(
                        label="Back-translation comparison",
                        interactive=False,
                        wrap=True,
                    )
                    backtrans_csv_dl = gr.File(label="back_translation.csv", interactive=False)

        # ── Save cloned voice panel (shown after successful clone run) ────────
        speaker_refs_state = gr.State(value={})  # {speaker_id: wav_path}

        with gr.Group(visible=False) as save_voice_group:
            gr.Markdown("---\n**Save cloned voice(s) for future use:**")
            save_speaker_dd = gr.Dropdown(
                label="Speaker to save",
                choices=[],
                value=None,
                visible=False,
                interactive=True,
                info="Multiple speakers detected — select which one to save.",
            )
            with gr.Row():
                save_name_input = gr.Textbox(
                    label="Voice name",
                    placeholder="e.g. Male lead — English",
                    scale=3,
                )
                save_btn = gr.Button("Save voice", variant="secondary", scale=1)
            save_status = gr.Markdown("")

        # ── Event wiring ──────────────────────────────────────────────────────

        # Voice mode toggle — also clears preview when leaving library mode
        def on_mode_change(mode):
            return (
                gr.update(visible=mode == "clone"),
                gr.update(visible=mode == "library"),
                gr.update(visible=mode == "saved"),
                gr.update(visible=False, value=None),  # lib_preview_audio
                gr.update(visible=False, value=None),  # lib_sample_dd
            )

        voice_mode_radio.change(
            fn=on_mode_change,
            inputs=[voice_mode_radio],
            outputs=[clone_group, library_group, saved_group, lib_preview_audio, lib_sample_dd],
        )

        # Library voice load button and language filter change
        _lib_load_outputs = [library_voice_dd, lib_voices_state]

        lib_load_btn.click(
            fn=_refresh_library_voices,
            inputs=[lib_lang_filter],
            outputs=_lib_load_outputs,
        )
        lib_lang_filter.change(
            fn=_refresh_library_voices,
            inputs=[lib_lang_filter],
            outputs=_lib_load_outputs,
        )

        # Voice selection → update sample picker + audio preview
        library_voice_dd.change(
            fn=on_library_voice_select,
            inputs=[library_voice_dd, lib_voices_state],
            outputs=[lib_sample_dd, lib_preview_audio],
        )

        # Sample picker → update audio preview
        lib_sample_dd.change(
            fn=on_sample_select,
            inputs=[lib_sample_dd],
            outputs=[lib_preview_audio],
        )

        # Saved voice refresh
        saved_refresh_btn.click(
            fn=_refresh_saved_voices,
            inputs=[],
            outputs=[saved_voice_dd],
        )

        # Main generate button
        generate_btn.click(
            fn=generate_redub,
            inputs=[
                video_input, source_lang, target_lang,
                voice_mode_radio, library_voice_dd, saved_voice_dd,
                back_trans_check, handle_overlaps_check, preserve_music_check,
                content_context_box, fallback_voice_input, min_ref_dur_slider,
            ],
            outputs=[
                log_box, redubbed_player, transcript_dl, srt_dl, original_player,
                speaker_refs_state, save_voice_group, save_speaker_dd, save_status,
                timing_table, backtrans_table, timing_csv_dl, backtrans_csv_dl,
            ],
        )

        # Save voice button
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
