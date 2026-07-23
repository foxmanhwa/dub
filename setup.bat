@echo off
setlocal enabledelayedexpansion

echo.
echo ======================================================
echo   Dub -- Video Redubbing Setup
echo ======================================================
echo.

REM ── 1. Python version check ──────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found on your PATH.
    echo.
    echo  Install Python 3.11 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  During installation, check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)

if !PYMAJ! LSS 3 goto :pyver_bad
if !PYMAJ! EQU 3 if !PYMIN! LSS 11 goto :pyver_bad
goto :pyver_ok

:pyver_bad
echo  ERROR: Python 3.11+ required. Found !PYVER!
echo.
echo  Download a newer version from:
echo    https://www.python.org/downloads/
echo.
pause
exit /b 1

:pyver_ok
echo  [OK] Python !PYVER!
echo.

REM ── 2. FFmpeg check ──────────────────────────────────────────────────────────
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo  WARNING: FFmpeg not found on your PATH.
    echo.
    echo  FFmpeg is required -- the app will not work without it.
    echo  Download from: https://ffmpeg.org/download.html
    echo    - Extract the zip and add the bin\ folder to your system PATH.
    echo.
    echo  You can finish setup now and add FFmpeg later.
    echo  Press any key to continue anyway, or close this window to stop.
    echo.
    pause
) else (
    echo  [OK] FFmpeg found.
    echo.
)

REM ── 3. Virtual environment ────────────────────────────────────────────────────
if not exist .venv (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
) else (
    echo  [OK] Virtual environment already exists, reusing it.
)
echo.

call .venv\Scripts\activate.bat

REM ── 4. PyTorch -- GPU (CUDA) or CPU ──────────────────────────────────────────
echo  Detecting GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  [INFO] No NVIDIA GPU detected.
    echo         Installing CPU-only PyTorch.
    echo         Demucs stem separation will be slower on CPU.
    echo.
    pip install torch torchaudio
) else (
    echo  [OK] NVIDIA GPU detected! Installing CUDA-enabled PyTorch (CUDA 12.1^)...
    echo.
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
)
echo.

REM ── 5. Remaining dependencies ─────────────────────────────────────────────────
echo  Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo  ERROR: pip install failed. See output above.
    pause
    exit /b 1
)
echo.
echo  [OK] All dependencies installed.
echo.

REM ── 6. API keys ───────────────────────────────────────────────────────────────
echo ======================================================
echo   API Key Setup
echo ======================================================
echo.

if exist .env (
    echo  [INFO] .env already exists -- skipping key setup.
    echo         Edit .env manually to update your keys.
    echo.
    goto :launch
)

echo  Two keys are required; one is optional.
echo.
echo  [1] FISH_AUDIO_API_KEY  (required^)
echo      Handles transcription and voice-cloned TTS.
echo      Sign up at: https://fish.audio
echo      Dashboard - API Keys
echo.
set /p FISH_KEY=  Enter Fish Audio API key:
echo.

echo  [2] GROQ_API_KEY  (required for translation^)
echo      Free tier, no billing required.
echo      Get one at: https://console.groq.com
echo.
set /p GROQ_KEY=  Enter Groq API key:
echo.

echo  [3] HF_TOKEN  (optional^)
echo      Only needed for Phase B multi-speaker / overlapping-speaker dubbing.
echo      For single-speaker videos, press Enter to skip.
echo.
echo      If you need it:
echo        Token:      https://huggingface.co/settings/tokens
echo        Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
echo.
set /p HF_KEY=  Hugging Face token (Enter to skip^):
echo.

echo  Writing .env...

(
    echo FISH_AUDIO_API_KEY=!FISH_KEY!
    echo.
    echo # Translation -- Groq ^(default^), Gemini, or Ollama
    echo GROQ_API_KEY=!GROQ_KEY!
    echo # GEMINI_API_KEY=your_gemini_api_key_here
    echo.
    echo # Phase B -- multi-speaker diarization ^(optional^)
) > .env

if "!HF_KEY!"=="" (
    echo # HF_TOKEN=hf_...>> .env
) else (
    echo HF_TOKEN=!HF_KEY!>> .env
)

echo  [OK] .env written.
echo.

:launch
echo ======================================================
echo   Setup complete!
echo ======================================================
echo.
echo  Starting the app now...
echo  Open your browser at: http://localhost:7860
echo.
echo  To start the app again later:
echo    .venv\Scripts\activate
echo    python app.py
echo.
echo  Press Ctrl+C to stop the app.
echo.
python app.py
