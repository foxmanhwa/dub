# Video Redubbing & Localization

## Quick setup

Prerequisites: **Python 3.11+** and **FFmpeg** on your PATH.

```bat
git clone https://github.com/foxmanhwa/dub.git
cd dub

setup.bat        # Windows
bash setup.sh    # Mac / Linux
```

The script installs everything, detects your GPU, prompts for API keys, and launches the app. That's it.

---

A Gradio web app that redubs single-speaker videos into a target language using:
- **Fish Audio** — ASR transcription (with timestamps) + TTS with instant voice cloning
- **Gemini** — fast, free-tier translation via Google AI Studio (default)
- **Ollama** — fully-local, offline translation fallback (optional)
- **Demucs** — vocal/music stem separation to preserve background music
- **pyannote** — multi-speaker diarization (Phase B)

## Pipeline

```
Video → FFmpeg extract audio
      → [optional] Demucs — split vocals + background music
      → Fish Audio ASR (vocals stem, with timestamps)
      → Gemini / Ollama translation (duration-conscious, chunked)
      → Fish Audio TTS (voice-cloned, per segment)
      → FFmpeg time-fit each clip → reassemble dubbed audio
      → [optional] FFmpeg mix dubbed vocals + original music
      → mux back to MP4
```

Output files per run:
- `redubbed.mp4` — final dubbed video
- `transcript.json` — original + translated text with timestamps
- `translated.srt` — subtitle file in the target language

## Manual setup (step-by-step)

### 1. Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`

**Install FFmpeg:**
```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows: download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Clone & install

```bash
git clone https://github.com/foxmanhwa/dub.git
cd dub
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
FISH_AUDIO_API_KEY=your_fish_audio_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

- **Fish Audio API key**: sign up at [fish.audio](https://fish.audio) → Dashboard → API Keys
- **Gemini API key**: free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no billing required

### 4. Run

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Usage

1. Upload a video file (MP4, MOV, MKV, etc.)
2. Select the source language (or leave on Auto-detect)
3. Choose the target language
4. Optionally enable **Preserve background music** (runs Demucs stem separation)
5. Click **Generate Redub**
6. Watch the pipeline log — when complete, the redubbed video appears alongside the original

## Configuration

| Env var | Default | Description |
|---|---|---|
| `FISH_AUDIO_API_KEY` | — | **Required.** Fish Audio API key |
| `GEMINI_API_KEY` | — | Gemini API key (free). When set, Gemini is the translation backend. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model. `gemini-2.5-flash` for higher quality. |
| `TRANSLATION_BACKEND` | auto | `gemini` or `ollama`. Auto-selects Gemini if `GEMINI_API_KEY` is set. |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model (only used when `TRANSLATION_BACKEND=ollama`). |
| `HF_TOKEN` | — | Hugging Face token for pyannote diarization (Phase B). |
| `PORT` | `7860` | Port to run the Gradio server on. |

## Translation backends

**Gemini (default)** — recommended:
- Fast (~2–5 seconds per chunk vs 60–120 s cold-start with Ollama on CPU)
- Free tier: 15 requests/minute, 1 M tokens/minute — ample for typical videos
- No local GPU or model download needed
- Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

**Ollama (fallback)** — fully local, no API key:
```bash
# Install from https://ollama.com, then:
ollama pull llama3.2
```
Set `TRANSLATION_BACKEND=ollama` in `.env` to use it. Cold-start on CPU takes 60–120 s
to load the model; subsequent chunks are faster once it's in RAM.

## Project structure

```
dub/
├── app.py                        # Gradio UI entry point
├── modules/
│   ├── asr.py                    # Fish Audio ASR transcription
│   ├── translation.py            # Gemini / Ollama translation
│   ├── tts.py                    # Fish Audio TTS + voice cloning
│   ├── audio_assembly.py         # Time-fitting + audio reassembly (FFmpeg)
│   ├── music_separation.py       # Demucs vocal/music stem separation
│   ├── diarization.py            # pyannote speaker diarization (subprocess)
│   ├── _diarization_worker.py    # Isolated subprocess worker for pyannote
│   ├── output_writers.py         # Write transcript.json + .srt
│   └── pipeline.py               # End-to-end orchestrator
├── requirements.txt
├── .env.example
└── README.md
```

## Notes & limitations

- Best results with clean audio: minimal background noise
- Very short segments (< 1 second) may produce lower-quality TTS
- Time-fitting uses `atempo` (0.5–2.0× speed), with truncation/padding outside that range
- The voice reference clip is extracted from the first ~12 seconds of the source audio
- Demucs first run downloads ~200 MB of model weights; subsequent runs use the cache
- Phase B (multi-speaker overlap handling) requires `HF_TOKEN`, `pyannote.audio`, and `speechbrain`
