# Video Redubbing & Localization (Phase A)

A Gradio web app that redubs single-speaker videos into a target language using:
- **Fish Audio** — ASR transcription (with timestamps) + TTS with instant voice cloning
- **Ollama** — local, offline, context-aware translation (no API key needed)

## Pipeline

```
Video → FFmpeg extract audio → Fish Audio ASR (with timestamps)
      → Ollama LLM translation (duration-conscious) → Fish Audio TTS (voice-cloned)
      → FFmpeg time-fit each clip → reassemble full audio → mux back to MP4
```

Output files per run:
- `redubbed.mp4` — final dubbed video
- `transcript.json` — original + translated text with timestamps
- `translated.srt` — subtitle file in the target language

## Setup

### 1. Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
- [Ollama](https://ollama.com) installed and running

**Install FFmpeg:**
```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows: download from https://ffmpeg.org/download.html and add to PATH
```

**Install Ollama and pull the translation model:**
```bash
# Install from https://ollama.com, then:
ollama pull llama3.2
```

The app defaults to `llama3.2:latest`. Any model already in `ollama list` works —
set `OLLAMA_MODEL` in `.env` to override.

### 2. Clone & install

```bash
git clone https://github.com/foxmanhwa/dub.git
cd dub
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API key

```bash
cp .env.example .env
```

Edit `.env` — only one key needed:

```env
FISH_AUDIO_API_KEY=your_fish_audio_api_key_here
```

**Fish Audio API key**: sign up at [fish.audio](https://fish.audio) → Dashboard → API Keys

Translation runs entirely locally via Ollama — no OpenAI account or billing required.

### 4. Start Ollama, then run

```bash
ollama serve   # if not already running as a background service
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Usage

1. Upload a video file (MP4, MOV, MKV, etc.)
2. Select the source language (or leave on Auto-detect)
3. Choose the target language
4. Click **Generate Redub**
5. Watch the pipeline log — when complete, the redubbed video appears alongside the original
6. Download `transcript.json` and `translated.srt` if needed

## Project structure

```
dub/
├── app.py                    # Gradio UI entry point
├── modules/
│   ├── asr.py                # Fish Audio ASR transcription
│   ├── translation.py        # Ollama-based translation
│   ├── tts.py                # Fish Audio TTS + voice cloning
│   ├── audio_assembly.py     # Time-fitting + audio reassembly (FFmpeg)
│   ├── output_writers.py     # Write transcript.json + .srt
│   └── pipeline.py           # End-to-end orchestrator
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `FISH_AUDIO_API_KEY` | — | **Required.** Fish Audio API key |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model for translation. `llama3.2:1b` is faster but may merge segments. |
| `PORT` | `7860` | Port to run the Gradio server on |

Ollama must be reachable at `http://localhost:11434` (the default after `ollama serve`).

Translation is split into chunks of 10 segments per Ollama call so each request
completes in bounded time on CPU, with per-chunk progress shown in the pipeline log.
The first chunk of a session may take longer while Ollama loads the model into RAM.

## Notes & limitations (Phase A)

- **Single speaker only** — no speaker diarization or overlap detection yet (Phase B)
- Best results with clean audio: minimal background noise, single speaker
- Very short segments (< 1 second) may produce lower-quality TTS
- Time-fitting uses `atempo` (0.5–2.0× speed), with truncation/padding outside that range
- The voice reference clip is extracted from the first ~12 seconds of the source audio
- Translation speed depends on your hardware; a GPU makes Ollama significantly faster

## Phase B roadmap (not yet implemented)

- Speaker diarization (pyannote.audio) for multi-speaker content
- Per-speaker voice cloning with separate reference clips
- Overlap detection and graceful handling
- Background music/SFX preservation (source separation)
