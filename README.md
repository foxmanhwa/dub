# Video Redubbing & Localization (Phase A)

A Gradio web app that redubs single-speaker videos into a target language using:
- **Fish Audio** — ASR transcription (with timestamps) + TTS with instant voice cloning
- **OpenAI** — context-aware, duration-conscious translation

## Pipeline

```
Video → FFmpeg extract audio → Fish Audio ASR (with timestamps)
      → LLM translation (duration-conscious) → Fish Audio TTS (voice-cloned)
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
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Debian
  sudo apt install ffmpeg
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
OPENAI_API_KEY=your_openai_api_key_here
```

- **Fish Audio API key**: sign up at [fish.audio](https://fish.audio) → Dashboard → API Keys
- **OpenAI API key**: [platform.openai.com](https://platform.openai.com) → API Keys

### 4. Run

```bash
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
│   ├── translation.py        # LLM-based translation
│   ├── tts.py                # Fish Audio TTS + voice cloning
│   ├── audio_assembly.py     # Time-fitting + audio reassembly (FFmpeg)
│   ├── output_writers.py     # Write transcript.json + .srt
│   └── pipeline.py           # End-to-end orchestrator
├── output/                   # Default output directory
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `FISH_AUDIO_API_KEY` | — | Required. Fish Audio API key |
| `OPENAI_API_KEY` | — | Required. OpenAI API key for translation |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for translation |
| `PORT` | `7860` | Port to run the Gradio server on |

## Notes & limitations (Phase A)

- **Single speaker only** — no speaker diarization or overlap detection yet (Phase B)
- Best results with clean audio: minimal background noise, single speaker
- Very short segments (< 1 second) may produce lower-quality TTS
- Time-fitting uses `atempo` (0.5–2.0× speed), with truncation/padding outside that range
- The voice reference clip is extracted from the first ~12 seconds of the source audio

## Phase B roadmap (not yet implemented)

- Speaker diarization (pyannote.audio) for multi-speaker content
- Per-speaker voice cloning with separate reference clips
- Overlap detection and graceful handling
- Background music/SFX preservation (source separation)
