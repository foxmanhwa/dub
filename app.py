"""
Video Redubbing / Localization App — Phase A
Gradio UI wrapping the end-to-end Fish Audio pipeline.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from modules.pipeline import run_pipeline


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


def check_env() -> list[str]:
    warnings = []
    if not os.environ.get("FISH_AUDIO_API_KEY"):
        warnings.append("FISH_AUDIO_API_KEY not set")
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set")
    return warnings


def generate_redub(
    video_file,
    source_lang_code: str,
    target_lang_name: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple:
    """
    Main handler — runs the pipeline and streams log updates.
    Returns: (log_text, output_video_path, transcript_path, srt_path, original_video_path)
    """
    log_lines = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Validate inputs
    missing = check_env()
    if missing:
        err = "Missing environment variables: " + ", ".join(missing)
        return err, None, None, None, None

    if video_file is None:
        return "Please upload a video file.", None, None, None, None

    # Gradio may give us a file path string or a temp file object
    video_path = video_file if isinstance(video_file, str) else video_file.name

    output_dir = tempfile.mkdtemp(prefix="dub_output_")

    out_video = None
    transcript = None
    srt = None

    try:
        gen = run_pipeline(
            video_path=video_path,
            source_language=source_lang_code,
            target_language=target_lang_name,
            output_dir=output_dir,
        )
        for item in gen:
            if isinstance(item, str):
                current_log = log(item)
                # Yield intermediate progress (Gradio generator pattern)
                yield current_log, None, None, None, video_path
            elif isinstance(item, dict):
                out_video = item.get("video")
                transcript = item.get("transcript")
                srt = item.get("srt")

        final_log = log("Done! Redubbed video is ready.")
        yield final_log, out_video, transcript, srt, video_path

    except Exception as e:
        tb = traceback.format_exc()
        error_log = log(f"Error: {e}\n\n{tb}")
        yield error_log, None, None, None, video_path


# ── Build UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    src_choices = [(label, code) for label, code in SOURCE_LANGUAGES]
    tgt_choices = [(label, name) for label, name in TARGET_LANGUAGES]

    warnings = check_env()
    warn_text = (
        "⚠ " + " | ".join(warnings)
        if warnings
        else ""
    )

    with gr.Blocks(title="Video Redubbing — Fish Audio") as demo:
        gr.Markdown("# Video Redubbing & Localization")
        gr.Markdown(
            "Upload a single-speaker video. The app transcribes, translates, "
            "and regenerates speech in the target language using voice cloning — "
            "powered by **Fish Audio** + an LLM."
        )

        if warn_text:
            gr.Markdown(f"**{warn_text}**")

        with gr.Row():
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
                generate_btn = gr.Button("Generate Redub", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Progress")
                log_box = gr.Textbox(
                    label="Pipeline log",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                )

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

        generate_btn.click(
            fn=generate_redub,
            inputs=[video_input, source_lang, target_lang],
            outputs=[log_box, redubbed_player, transcript_dl, srt_dl, original_player],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        theme=gr.themes.Soft(),
    )
