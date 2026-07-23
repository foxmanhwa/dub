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

**GPU users (NVIDIA 10 GB+ VRAM or Apple Silicon):** the script will offer to install [Ollama](https://ollama.com) and pull `qwen2.5:14b` for fully local, private, offline-capable translation — no cloud API key needed for translation. Just answer `y` at the prompt.

---

A Gradio web app that redubs single-speaker videos into a target language using:
- **Fish Audio** — ASR transcription (with timestamps) + TTS with instant voice cloning
- **Groq** — fast, free-tier translation (default; Gemini and Ollama are fallbacks)
- **Ollama** — fully-local, offline translation (optional)
- **Demucs** — vocal/music stem separation to preserve background music
- **pyannote** — multi-speaker diarization (Phase B)

## Pipeline

```
Video → FFmpeg extract audio
      → [optional] Demucs — split vocals + background music
      → Fish Audio ASR (vocals stem, with timestamps)
      → Groq / Gemini / Ollama translation (duration-conscious, chunked)
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
| `GROQ_API_KEY` | — | Groq API key (free tier). **Auto-selected as default translation backend when set.** |
| `GEMINI_API_KEY` | — | Gemini API key (free). Used when `GROQ_API_KEY` is absent. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model override. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model override. `gemini-2.5-flash` for higher quality. |
| `TRANSLATION_BACKEND` | auto | Force a backend: `groq`, `gemini`, or `ollama`. Auto-selects Groq → Gemini → Ollama by which key is present. |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model (only used when backend is `ollama`). |
| `HF_TOKEN` | — | Hugging Face token for pyannote diarization (Phase B). |
| `FALLBACK_VOICE_ID` | — | Fish Audio library voice ID to use when a speaker's reference clip is shorter than `MIN_REF_DURATION_SECS`. Avoids forcing a poor clone through. Find IDs at fish.audio/model. |
| `MIN_REF_DURATION_SECS` | `6.0` | Minimum seconds of clean reference audio required before attempting voice cloning. Speakers below this threshold fall back to `FALLBACK_VOICE_ID` (if set) or clone with a quality warning. |
| `HOST` | `127.0.0.1` | Server bind address. Set to `0.0.0.0` to expose on the local network. |
| `PORT` | `7860` | Port to run the Gradio server on. |

## Translation backends

**Groq (default)** — recommended:
- Fast (~1–3 s per chunk), generous free tier, no billing required
- Get a free key at [console.groq.com](https://console.groq.com)

**Gemini (fallback when no Groq key)** — also free:
- Free tier: 15 requests/minute, 1 M tokens/day
- Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

**Ollama — local, no API key:**

*GPU path (recommended for NVIDIA 10 GB+ VRAM or Apple Silicon M-series):*
```bash
ollama pull qwen2.5:14b   # ~9 GB download; strong multilingual model
```
```env
TRANSLATION_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:14b
```
The setup script handles this automatically when a capable GPU is detected.

*CPU fallback (slower, lower quality):*
```bash
ollama pull llama3.2      # ~2 GB; workable on CPU
```
```env
TRANSLATION_BACKEND=ollama
OLLAMA_MODEL=llama3.2:latest
```
CPU cold-start takes 60–120 s to load the model; subsequent chunks are faster once it's in RAM.

Backend auto-selection order: `TRANSLATION_BACKEND` env var (explicit) → `GROQ_API_KEY` present → `GEMINI_API_KEY` present → Ollama.

## Project structure

```
dub/
├── app.py                        # Gradio UI entry point
├── modules/
│   ├── asr.py                    # Fish Audio ASR transcription
│   ├── translation.py            # Groq / Gemini / Ollama translation
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
