#!/usr/bin/env bash
set -euo pipefail

echo
echo "======================================================"
echo "  Dub -- Video Redubbing Setup"
echo "======================================================"
echo

OS=$(uname -s)
ARCH=$(uname -m)

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

# ── 4. PyTorch + optional Ollama setup ───────────────────────────────────────

GPU_FOUND=0
USE_OLLAMA=n
OLLAMA_READY=0

echo "  Detecting GPU..."

_install_ollama_and_model() {
    # Install Ollama binary if needed
    if command -v ollama &>/dev/null; then
        echo "  [OK] Ollama already installed."
        OLLAMA_READY=1
    else
        echo "  Installing Ollama..."
        if [ "$OS" = "Darwin" ]; then
            if command -v brew &>/dev/null; then
                brew install ollama
                OLLAMA_READY=1
            else
                echo "  Homebrew not found. Install Ollama manually:"
                echo "    https://ollama.com/download"
                echo "  Then run:  ollama pull qwen2.5:14b"
                OLLAMA_READY=0
                return
            fi
        else
            # Linux: official install script
            if command -v curl &>/dev/null; then
                curl -fsSL https://ollama.com/install.sh | sh
                OLLAMA_READY=1
            else
                echo "  curl not found. Install Ollama manually:"
                echo "    https://ollama.com/download"
                OLLAMA_READY=0
                return
            fi
        fi
        echo "  [OK] Ollama installed."
    fi
}

if [ "$OS" = "Darwin" ]; then
    echo "  [INFO] macOS detected. Installing standard PyTorch (MPS-enabled for Apple Silicon)."
    echo
    pip install torch torchaudio

    # Apple Silicon: Metal GPU can run Ollama models efficiently
    if [ "$ARCH" = "arm64" ]; then
        echo
        echo "  --------------------------------------------------------"
        echo "   LOCAL TRANSLATION (optional)"
        echo "  --------------------------------------------------------"
        echo "   Apple Silicon detected. Ollama runs large models via"
        echo "   Metal, keeping translations private and offline."
        echo
        echo "   Model: qwen2.5:14b  (~9 GB one-time download)"
        echo "   Alternatively, skip this and use Groq (cloud, free)."
        echo "  --------------------------------------------------------"
        echo
        read -rp "  Use local Ollama translation? (y/n): " USE_OLLAMA
        USE_OLLAMA="${USE_OLLAMA:-n}"
        echo
        if [[ "$USE_OLLAMA" =~ ^[Yy]$ ]]; then
            USE_OLLAMA=y
            GPU_FOUND=1
            _install_ollama_and_model
        else
            USE_OLLAMA=n
        fi
    fi

elif command -v nvidia-smi &>/dev/null; then
    GPU_FOUND=1
    echo "  [OK] NVIDIA GPU detected! Installing CUDA-enabled PyTorch (CUDA 12.8 — supports Blackwell RTX 5000 series)..."
    echo
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

    echo "  Verifying GPU kernel dispatch..."
    if python3 -c "import torch; torch.zeros(1,device='cuda')+torch.zeros(1,device='cuda')" 2>/dev/null; then
        echo "  [OK] GPU kernel dispatch works — CUDA acceleration active."
    else
        echo
        echo "  WARNING: CUDA build installed but kernel dispatch failed for your GPU."
        echo "           Your GPU may be newer than this PyTorch CUDA build supports."
        echo "           The app will fall back to CPU automatically — everything still"
        echo "           works, just slower. To force GPU support:"
        echo "             pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128"
        echo
    fi
    echo
    echo "  --------------------------------------------------------"
    echo "   LOCAL TRANSLATION (optional)"
    echo "  --------------------------------------------------------"
    echo "   Your GPU can run a local translation model via Ollama,"
    echo "   keeping translations private and working fully offline."
    echo
    echo "   Model: qwen2.5:14b  (~9 GB one-time download)"
    echo "   Requires ~10 GB VRAM. Works great on RTX 3060 12 GB+."
    echo
    echo "   Alternatively, skip this and use Groq (cloud, free)."
    echo "  --------------------------------------------------------"
    echo
    read -rp "  Use local Ollama translation? (y/n): " USE_OLLAMA
    USE_OLLAMA="${USE_OLLAMA:-n}"
    echo
    if [[ "$USE_OLLAMA" =~ ^[Yy]$ ]]; then
        USE_OLLAMA=y
        _install_ollama_and_model
    else
        USE_OLLAMA=n
    fi

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

# ── 5b. Pre-download WhisperX model ──────────────────────────────────────────
echo "  Pre-downloading WhisperX model weights (avoids a silent wait on first use)..."
"$PY_CMD" - << 'PYEOF'
try:
    import os, torch
    d = "cuda" if torch.cuda.is_available() else "cpu"
    m = os.environ.get("WHISPERX_MODEL", "large-v2" if d == "cuda" else "small")
    ct = "float16" if d == "cuda" else "int8"
    print(f"  Downloading Whisper '{m}' for {d}...")
    import whisperx
    whisperx.load_model(m, d, compute_type=ct)
    print(f"  [OK] Whisper '{m}' cached.")
except Exception as e:
    print(f"  [WARN] WhisperX model pre-download failed: {e}")
    print("  It will download on first use instead.")
    print("  Or set ASR_BACKEND=fish in .env to use Fish Audio cloud ASR.")
PYEOF
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
    echo "  [1] FISH_AUDIO_API_KEY  (required)"
    echo "      Used for voice-cloned TTS (free s2.1-pro-free model — zero cost by default)."
    echo "      Free signup, no payment method required."
    echo "      Sign up at: https://fish.audio  ->  Dashboard -> API Keys"
    echo
    read -rp "  Enter Fish Audio API key: " FISH_KEY
    echo

    if [[ "$USE_OLLAMA" =~ ^[Yy]$ ]]; then
        echo "  [2] GROQ_API_KEY  (optional -- cloud fallback if Ollama is unavailable)"
        echo "      Leave blank to use Ollama only."
        echo "      Get one at: https://console.groq.com"
        echo
        read -rp "  Groq API key (Enter to skip): " GROQ_KEY
    else
        echo "  [2] GROQ_API_KEY  (required for translation)"
        echo "      Free tier, no billing required."
        echo "      Get one at: https://console.groq.com"
        echo
        read -rp "  Enter Groq API key: " GROQ_KEY
    fi
    echo

    echo "  [3] HF_TOKEN  (recommended)"
    echo "      Enables speaker diarization in WhisperX (the default ASR backend)."
    echo "      Without it, all speech is attributed to a single speaker."
    echo
    echo "      Setup:"
    echo "        Token:        https://huggingface.co/settings/tokens"
    echo "        Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "                      https://huggingface.co/pyannote/segmentation-3.0"
    echo
    read -rp "  Hugging Face token (Enter to skip): " HF_KEY
    echo

    echo "  Writing .env..."

    if [[ "$USE_OLLAMA" =~ ^[Yy]$ ]]; then
        cat > .env <<EOF
FISH_AUDIO_API_KEY=${FISH_KEY}

# Translation -- local Ollama (GPU-accelerated, offline)
TRANSLATION_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:14b
EOF
        if [ -n "${GROQ_KEY:-}" ]; then
            cat >> .env <<EOF

# Cloud fallback -- set TRANSLATION_BACKEND=groq to use instead of Ollama
GROQ_API_KEY=${GROQ_KEY}
EOF
        else
            echo "# GROQ_API_KEY=  (optional cloud fallback -- set TRANSLATION_BACKEND=groq to use)" >> .env
        fi
    else
        cat > .env <<EOF
FISH_AUDIO_API_KEY=${FISH_KEY}

# Translation -- Groq (default), Gemini, or Ollama
GROQ_API_KEY=${GROQ_KEY}
# GEMINI_API_KEY=your_gemini_api_key_here
EOF
    fi

    cat >> .env <<EOF

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

# ── 7. Pull Ollama model (after keys so user can walk away) ──────────────────

if [[ "$USE_OLLAMA" =~ ^[Yy]$ ]] && [ "$OLLAMA_READY" = "1" ]; then
    echo "======================================================"
    echo "  Pulling translation model"
    echo "======================================================"
    echo
    echo "  Downloading qwen2.5:14b (~9 GB) -- this may take 10-30 minutes."
    echo "  You can step away; the app will launch automatically when done."
    echo

    # On Linux, start the Ollama service if it isn't running yet
    if [ "$OS" = "Linux" ]; then
        if command -v systemctl &>/dev/null && systemctl is-enabled ollama &>/dev/null 2>&1; then
            sudo systemctl start ollama 2>/dev/null || true
        elif ! pgrep -x ollama &>/dev/null; then
            ollama serve &>/dev/null &
            sleep 3
        fi
    fi

    if ollama pull qwen2.5:14b; then
        echo
        echo "  [OK] qwen2.5:14b ready."
        echo
    else
        echo
        echo "  WARNING: Model pull failed. To retry:"
        echo "    ollama pull qwen2.5:14b"
        echo "  Then start the app:"
        echo "    python app.py"
        echo
    fi
fi

# ── 8. Launch ────────────────────────────────────────────────────────────────

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
