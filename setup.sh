#!/usr/bin/env bash
set -euo pipefail

echo
echo "======================================================"
echo "  Dub -- Video Redubbing Setup"
echo "======================================================"
echo

OS=$(uname -s)

# ── 1. Python version check ──────────────────────────────────────────────────

PY_CMD=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PY_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    echo "  ERROR: Python 3.11+ not found."
    echo
    if [ "$OS" = "Darwin" ]; then
        echo "  Install via Homebrew:  brew install python@3.12"
        echo "  Or download from:      https://www.python.org/downloads/"
    else
        echo "  Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv"
        echo "  Or download from: https://www.python.org/downloads/"
    fi
    echo
    exit 1
fi

PY_VER=$("$PY_CMD" --version 2>&1 | awk '{print $2}')
echo "  [OK] Python $PY_VER ($PY_CMD)"
echo

# ── 2. FFmpeg check ──────────────────────────────────────────────────────────

if command -v ffmpeg &>/dev/null; then
    echo "  [OK] FFmpeg found."
else
    echo "  WARNING: FFmpeg not found on your PATH."
    echo
    if [ "$OS" = "Darwin" ]; then
        echo "  Install with:  brew install ffmpeg"
    else
        echo "  Install with:  sudo apt install ffmpeg"
    fi
    echo
    echo "  FFmpeg is required -- the app will not work without it."
    echo "  You can finish setup now and install FFmpeg separately."
    echo
    read -rp "  Press Enter to continue anyway, or Ctrl+C to cancel: "
fi
echo

# ── 3. Virtual environment ────────────────────────────────────────────────────

if [ ! -d .venv ]; then
    echo "  Creating virtual environment..."
    "$PY_CMD" -m venv .venv
    echo "  [OK] Virtual environment created."
else
    echo "  [OK] Virtual environment already exists, reusing it."
fi
echo

# shellcheck source=/dev/null
source .venv/bin/activate

# ── 4. PyTorch -- GPU or CPU ─────────────────────────────────────────────────

echo "  Detecting GPU..."

if [ "$OS" = "Darwin" ]; then
    # macOS: no CUDA; MPS (Apple Silicon) or CPU handled transparently by torch
    echo "  [INFO] macOS detected. Installing standard PyTorch (MPS-enabled for Apple Silicon)."
    echo
    pip install torch torchaudio
elif command -v nvidia-smi &>/dev/null; then
    echo "  [OK] NVIDIA GPU detected! Installing CUDA-enabled PyTorch (CUDA 12.1)..."
    echo
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  [INFO] No NVIDIA GPU detected."
    echo "         Installing CPU-only PyTorch. Demucs stem separation will be slower."
    echo
    pip install torch torchaudio
fi
echo

# ── 5. Remaining dependencies ─────────────────────────────────────────────────

echo "  Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo
echo "  [OK] All dependencies installed."
echo

# ── 6. API keys ───────────────────────────────────────────────────────────────

echo "======================================================"
echo "  API Key Setup"
echo "======================================================"
echo

if [ -f .env ]; then
    echo "  [INFO] .env already exists -- skipping key setup."
    echo "         Edit .env manually to update your keys."
    echo
else
    echo "  Two keys are required; one is optional."
    echo
    echo "  [1] FISH_AUDIO_API_KEY  (required)"
    echo "      Handles transcription and voice-cloned TTS."
    echo "      Sign up at: https://fish.audio  →  Dashboard → API Keys"
    echo
    read -rp "  Enter Fish Audio API key: " FISH_KEY
    echo

    echo "  [2] GROQ_API_KEY  (required for translation)"
    echo "      Free tier, no billing required."
    echo "      Get one at: https://console.groq.com"
    echo
    read -rp "  Enter Groq API key: " GROQ_KEY
    echo

    echo "  [3] HF_TOKEN  (optional)"
    echo "      Only needed for Phase B multi-speaker / overlapping-speaker dubbing."
    echo "      For single-speaker videos, press Enter to skip."
    echo
    echo "      If you need it:"
    echo "        Token:        https://huggingface.co/settings/tokens"
    echo "        Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo
    read -rp "  Hugging Face token (Enter to skip): " HF_KEY
    echo

    echo "  Writing .env..."

    cat > .env <<EOF
FISH_AUDIO_API_KEY=${FISH_KEY}

# Translation -- Groq (default), Gemini, or Ollama
GROQ_API_KEY=${GROQ_KEY}
# GEMINI_API_KEY=your_gemini_api_key_here

# Phase B -- multi-speaker diarization (optional)
EOF

    if [ -n "${HF_KEY:-}" ]; then
        echo "HF_TOKEN=${HF_KEY}" >> .env
    else
        echo "# HF_TOKEN=hf_..." >> .env
    fi

    echo "  [OK] .env written."
    echo
fi

# ── 7. Launch ────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  Setup complete!"
echo "======================================================"
echo
echo "  Starting the app now..."
echo "  Open your browser at: http://localhost:7860"
echo
echo "  To start the app again later:"
echo "    source .venv/bin/activate"
echo "    python app.py"
echo
echo "  Press Ctrl+C to stop the app."
echo
python app.py
