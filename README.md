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

A Gradio web app that redubs single- and multi-speaker videos into a target language using:
- **WhisperX** — local ASR with word-level timestamps + speaker diarization in one pass (default; Fish Audio cloud API is the fallback)
- **Fish Audio** — TTS with instant voice cloning or library voices (free `s2.1-pro-free` model by default — zero cost when ASR_BACKEND=whisperx)
- **Groq** — fast, free-tier translation (default; Gemini and Ollama are fallbacks)
- **Ollama** — fully-local, offline translation (optional)
- **Demucs** — vocal/music stem separation to preserve background music
- **SpeechBrain ECAPA-TDNN** — speaker embedding verification and voice library matching

## Pipeline

The app runs in two stages with a pause in between so you can review and adjust speaker assignments before generating the dub.

### Stage 1 — Analyze

```
Video → FFmpeg extract audio
      → [optional] Demucs — split vocals + background music
      → WhisperX ASR (local, GPU-accelerated) — transcription + word timestamps + speaker diarization
        [or Fish Audio ASR if ASR_BACKEND=fish]
      → Energy analysis — tag each segment as loud / normal / quiet
      → Speaker reference clips extracted and normalized
      → ECAPA-TDNN embeddings — verify speaker distinctness, suggest voice library matches
      → Speaker panel shown (up to 6 speakers with sample audio)
```

### Stage 2 — Generate

```
      → LLM translation — duration-conscious, energy-annotated, chunked
      → Fish Audio TTS — per-speaker voice (clone / saved / library / uploaded sample)
      → Iterative retranslation — re-asks LLM to shorten/lengthen if TTS overshoots slot
      → FFmpeg time-fit each clip → reassemble dubbed audio
      → [optional] FFmpeg mix dubbed vocals + original music
      → Mux back to MP4
      → Segment editor shown — review, edit, and selectively regenerate
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

**WhisperX note:** WhisperX uses `faster-whisper` under the hood. On NVIDIA GPUs, install the CUDA-enabled PyTorch before running `pip install -r requirements.txt`:
```bash
pip install torch "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
The `setup.bat` / `setup.sh` scripts handle this automatically. The CUDA 12.8 build supports GPUs from Maxwell through Blackwell (sm_52–sm_120), including RTX 50-series. torchaudio is pinned to 2.8.x — 2.9+ removes APIs that pyannote.audio and whisperx depend on.

### 3. API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
FISH_AUDIO_API_KEY=your_fish_audio_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional but recommended — enables speaker diarization in WhisperX
# HF_TOKEN=hf_...
```

- **Fish Audio API key**: free signup at [fish.audio](https://fish.audio) → Dashboard → API Keys — **no payment method required, no balance needed**. When using the default `ASR_BACKEND=whisperx`, Fish Audio is only called for TTS and uses the free `s2.1-pro-free` model, so your usage cost is $0.
- **Groq API key**: free at [console.groq.com](https://console.groq.com) — no billing required
- **HF_TOKEN**: needed for WhisperX speaker diarization. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then accept model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). Without this, WhisperX still transcribes but all audio is treated as a single speaker.

### 4. Run

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Usage

### Stage 1 — Analyze

1. Upload a video file (MP4, MOV, MKV, etc.)
2. Select the source language (or leave on Auto-detect)
3. Optionally enable **Preserve background music** (runs Demucs stem separation)
4. Click **Analyze** and wait for the pipeline log to finish

When analysis completes, the **Speaker panel** appears — one row per detected speaker (up to 6). Each row shows:
- A short audio sample of that speaker (the aggregated reference clip)
- An embedding hint if a close match was found in your saved voice library
- A **Voice mode** radio with four options:

| Mode | What it does |
|---|---|
| **Auto-clone (from video)** | Clones the speaker's voice from audio extracted during Analyze. Default. |
| **Use saved voice** | Picks a voice you previously saved from an earlier run. |
| **Use library voice ID** | Pastes any Fish Audio public voice ID directly. |
| **Upload custom sample** | Uploads your own audio file — bypasses all extraction, uses exactly what you provide. Takes priority over everything else. |

If ECAPA-TDNN embeddings found a close match to a saved voice (similarity ≥ 0.75), the mode is pre-filled to "Use saved voice" automatically.

Adjust voice assignments if needed, then proceed to Stage 2.

### Stage 2 — Generate

5. Choose the **target language**
6. Optionally paste **content context** (a short description of the video topic — helps the LLM translator stay accurate)
7. Optionally enable **Handle speaker overlaps** (attenuates the non-primary speaker during overlap windows)
8. Click **Generate Dub**

When generation completes:
- The dubbed video and original appear side-by-side
- The **Segment editor** table shows every segment with its original text, translation, and speaker

### Segment editor

The segment table lets you review and fix the dub without re-running the full pipeline:

- **Edit the Translation column** directly in the table to fix a mistranslation
- Enter a **row number** and click **Regenerate segment TTS** to re-synthesize just that one segment using the edited text
- Click **Rebuild output** to reassemble the final video from the current segment table — no re-transcription, no re-translation, no re-TTS for unchanged rows

## Configuration

| Env var | Default | Description |
|---|---|---|
| `FISH_AUDIO_API_KEY` | — | **Required.** Fish Audio API key. Free account, no payment needed. Used for TTS (always) and ASR (only when `ASR_BACKEND=fish`). |
| `FISH_TTS_MODEL` | `s2.1-pro-free` | Fish Audio TTS model. Default is free (zero cost). Use `s2.1-pro` for paid higher-quality output. |
| `GROQ_API_KEY` | — | Groq API key (free tier). Auto-selected as default translation backend when set. |
| `GEMINI_API_KEY` | — | Gemini API key (free). Used when `GROQ_API_KEY` is absent. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model override. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model override. `gemini-2.5-flash` for higher quality. |
| `TRANSLATION_BACKEND` | auto | Force a backend: `groq`, `gemini`, or `ollama`. Auto-selects Groq → Gemini → Ollama by which key is present. |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model (only used when backend is `ollama`). |
| `ASR_BACKEND` | `whisperx` | ASR backend: `whisperx` (local, GPU-accelerated) or `fish` (Fish Audio cloud API). |
| `WHISPERX_MODEL` | `large-v2` | WhisperX model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. Smaller = faster, less accurate. |
| `HF_TOKEN` | — | Hugging Face token. Needed for WhisperX speaker diarization (and pyannote Phase B). Without it, all audio is treated as one speaker. |
| `FALLBACK_VOICE_ID` | — | Fish Audio library voice ID to use when a speaker's reference clip is shorter than `MIN_REF_DURATION_SECS`. Find IDs at fish.audio/model. |
| `MIN_REF_DURATION_SECS` | `6.0` | Minimum seconds of clean reference audio required before attempting voice cloning. Below this threshold the speaker falls back to `FALLBACK_VOICE_ID` (if set) or clones with a quality warning. |
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
├── app.py                          # Gradio UI — two-stage Analyze → Generate flow
├── modules/
│   ├── pipeline.py                 # End-to-end orchestrator (analyze_video / generate_from_analysis)
│   ├── asr.py                      # ASR dispatch — WhisperX (default) or Fish Audio
│   ├── asr_whisperx.py             # WhisperX subprocess interface
│   ├── _whisperx_worker.py         # Isolated subprocess worker (loads WhisperX + pyannote)
│   ├── translation.py              # Groq / Gemini / Ollama translation (energy-annotated)
│   ├── tts.py                      # Fish Audio TTS + voice cloning
│   ├── audio_assembly.py           # Time-fitting + audio reassembly (FFmpeg)
│   ├── energy.py                   # Per-segment RMS energy analysis (pure Python)
│   ├── speaker_embeddings.py       # ECAPA-TDNN embedding extraction + matching
│   ├── _embeddings_worker.py       # Isolated subprocess worker (loads SpeechBrain)
│   ├── voice_library.py            # Save / load cloned voices with embeddings
│   ├── music_separation.py         # Demucs vocal/music stem separation
│   ├── diarization.py              # pyannote speaker diarization (subprocess)
│   ├── _diarization_worker.py      # Isolated subprocess worker for pyannote
│   └── output_writers.py           # Write transcript.json + .srt
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

**GPU not being used / "no kernel image" CUDA error**

If you have an NVIDIA GPU but the pipeline runs on CPU, or you see `CUDA error: no kernel image is available for execution on the device`, your PyTorch build doesn't support your GPU's compute capability. This most commonly affects newer GPUs (RTX 50-series / Blackwell, sm_120).

Fix — reinstall with the CUDA 12.8 build, pinned to torchaudio 2.8.0:
```bash
pip install --upgrade torch "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
```

torchaudio must stay on 2.8.x — version 2.9+ removes the `AudioMetaData` API that pyannote.audio depends on, and violates whisperx's own `torchaudio~=2.8.0` constraint. The cu128 build of torchaudio 2.8.0 satisfies both Blackwell GPU support and the API compatibility requirement simultaneously.

The app handles CUDA arch mismatches gracefully: if the kernel dispatch test fails at startup, WhisperX falls back to CPU automatically with a log message explaining the issue. Nothing crashes — it just runs slower until you do the reinstall.

The setup scripts already install the correct pinned version. This only affects machines that ran `setup.bat` / `setup.sh` before this fix was shipped.

---

## Notes & limitations

- Best results with clean audio: minimal background noise
- Very short TTS output segments (< 1 second) may sound slightly robotic — this is a Fish Audio constraint, not a pipeline one
- Time-fitting uses `atempo` (0.85–1.15× range) with trim/pad outside that range; iterative retranslation tries to bring segments into range before clamping
- WhisperX first run downloads model weights (~1–3 GB depending on model size); subsequent runs use the cache
- Speaker diarization in WhisperX requires `HF_TOKEN` — without it all speech is treated as one speaker
- The ECAPA-TDNN embedding model (`speechbrain/spkrec-ecapa-voxceleb`) downloads ~100 MB on first use
- Demucs first run downloads ~200 MB of model weights; subsequent runs use the cache
- Up to 6 speakers are shown in the speaker panel; videos with more speakers use the first 6 detected
- Energy analysis (loud/quiet tagging) runs per-speaker so volume differences between speakers don't affect classification

**Voice reference extraction:**
- Fragments as short as 0.1 s are included in the aggregated reference — short utterances like "yes", "ok", "hmm" carry usable voice timbre and add up across back-and-forth dialogue. Only genuine near-zero artefacts (< 0.1 s) are discarded.
- The pipeline log reports per-speaker aggregated totals so you can verify accumulation on realistic content, e.g.:
  ```
  SPEAKER_00: 12.0s reference clip (42 fragments, 18.7s total — target reached)
  SPEAKER_01: 4.2s reference clip (11 fragments, 4.2s total available)
  ```
- If a speaker's total falls below `MIN_REF_DURATION_SECS` (default 6 s) even after aggregating the whole clip, the fallback chain is: `FALLBACK_VOICE_ID` (if set) → clone anyway with a quality warning. The "Upload custom sample" option in the speaker panel bypasses all of this — use it when you have a clean external recording of the speaker.
