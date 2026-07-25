@echo off
setlocal enabledelayedexpansion

echo.
echo ======================================================
echo   Dub -- Video Redubbing Setup
echo ======================================================
echo.

REM -- 1. Python version check ---------------------------------------------------
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

REM -- 2. FFmpeg check -----------------------------------------------------------
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

REM -- 3. Virtual environment ---------------------------------------------------
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

REM -- 4. PyTorch + optional Ollama setup ---------------------------------------
set GPU_FOUND=0
set USE_OLLAMA=n
set OLLAMA_READY=0

echo  Detecting GPU...

REM -- Try nvidia-smi first (present when CUDA driver is on PATH)
set GPU_NAME=
nvidia-smi --query-gpu=name --format=csv,noheader >"%TEMP%\_dub_gpu.txt" 2>nul
if not errorlevel 1 (
    set /p GPU_NAME=<"%TEMP%\_dub_gpu.txt"
)
if not "!GPU_NAME!"=="" (
    echo  [OK] NVIDIA GPU detected via nvidia-smi: !GPU_NAME!
    goto :gpu_detected
)

REM -- wmic fallback: works even when nvidia-smi is not on PATH
echo  nvidia-smi not found on PATH -- checking via wmic...
wmic path Win32_VideoController get Name /format:list 2>nul >"%TEMP%\_dub_wmic.txt"
for /f "tokens=2 delims==" %%g in ('findstr /i "NVIDIA" "%TEMP%\_dub_wmic.txt" 2^>nul') do set GPU_NAME=%%g
if not "!GPU_NAME!"=="" (
    echo  [OK] NVIDIA GPU detected via wmic: !GPU_NAME!
    goto :gpu_detected
)

goto :no_gpu

REM === NVIDIA GPU FOUND ========================================================
:gpu_detected
set GPU_FOUND=1
echo  Installing CUDA-enabled PyTorch (CUDA 12.8 -- supports Blackwell RTX 5000 series^)...
echo.
pip install --upgrade torch "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
echo.

REM -- Verify the CUDA build actually installed (pip can silently downgrade)
echo  Verifying CUDA build...
python -c "import torch; ok='+cu' in torch.__version__; print('  torch', torch.__version__, '-- CUDA available:', torch.cuda.is_available()); exit(0 if ok else 1)"
if errorlevel 1 (
    echo.
    echo  WARNING: pip installed a CPU-only PyTorch. Forcing CUDA reinstall...
    pip install --upgrade --force-reinstall torch "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; print('  torch', torch.__version__, '-- CUDA available:', torch.cuda.is_available())"
)
echo.

REM -- Verify this build can actually dispatch kernels on the detected GPU
REM    (catches compute-capability mismatches, e.g. sm_120 Blackwell on an old cu build)
echo  Verifying GPU kernel dispatch...
python -c "import torch; torch.zeros(1,device='cuda')+torch.zeros(1,device='cuda'); print('  [OK] GPU kernel dispatch works -- CUDA acceleration active')"
if errorlevel 1 (
    echo.
    echo  WARNING: CUDA build installed but kernel dispatch failed for your GPU.
    echo           !GPU_NAME!
    echo           This usually means your GPU is newer than this PyTorch CUDA build supports.
    echo           The app will fall back to CPU automatically -- everything still works,
    echo           just slower. If you want GPU acceleration, try:
    echo             pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo.
)
echo.

echo  --------------------------------------------------------
echo   LOCAL TRANSLATION (optional^)
echo  --------------------------------------------------------
echo   Your GPU can run a local translation model via Ollama,
echo   keeping translations private and working fully offline.
echo.
echo   Model: qwen2.5:14b  (~9 GB one-time download^)
echo   Requires ~10 GB VRAM. Works great on RTX 3060 12 GB+.
echo.
echo   Alternatively, skip this and use Groq (cloud, free^).
echo  --------------------------------------------------------
echo.
set /p USE_OLLAMA=  Use local Ollama translation? (y/n):
echo.

if /i not "!USE_OLLAMA!"=="y" goto :torch_done

REM --- Install Ollama binary (fast; model pull happens after keys) -------------
echo  Checking for Ollama...
ollama --version >nul 2>&1
if not errorlevel 1 (
    echo  [OK] Ollama already installed.
    set OLLAMA_READY=1
    goto :torch_done
)

echo  Ollama not found -- downloading installer...
powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe'" ^
  " -OutFile ([IO.Path]::Combine($env:TEMP, 'OllamaSetup.exe'))" ^
  " -UseBasicParsing"
if errorlevel 1 (
    echo.
    echo  WARNING: Could not download the Ollama installer.
    echo           Check your internet connection and try again, or install manually:
    echo             https://ollama.com/download
    echo           Then re-run setup.bat.
    echo.
    set USE_OLLAMA=n
    goto :torch_done
)

echo  Running Ollama installer (silent^)...
start /wait "" "%TEMP%\OllamaSetup.exe" /S

REM Refresh PATH from user registry so ollama.exe is findable this session
for /f "skip=2 tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do (
    if not "%%b"=="" set "PATH=!PATH!;%%b"
)

ollama --version >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Ollama installed but not yet on PATH in this terminal session.
    echo         The model will be pulled on first app launch, or run manually:
    echo           ollama pull qwen2.5:14b
) else (
    echo  [OK] Ollama installed.
    set OLLAMA_READY=1
)
goto :torch_done

REM === NO GPU ==================================================================
:no_gpu
set GPU_FOUND=0
echo  [INFO] No NVIDIA GPU detected.
echo         Installing CPU-only PyTorch.
echo         Demucs stem separation will be slower on CPU.
echo.
pip install torch torchaudio

:torch_done
echo.

REM -- 5. Remaining dependencies -------------------------------------------------
echo  Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo  ERROR: pip install failed. See output above.
    pause
    exit /b 1
)

REM -- requirements.txt can pull in CPU torch as a transitive dep; fix if so
if "!GPU_FOUND!"=="1" (
    python -c "import torch; exit(0 if '+cu' in torch.__version__ else 1)" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo  [WARN] requirements.txt overwrote PyTorch with a CPU build. Reinstalling CUDA build...
        pip install --upgrade --force-reinstall torch "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    )
)
echo.
echo  [OK] All dependencies installed.
echo.

REM -- 5b. Pre-download WhisperX model ──────────────────────────────────────
echo  Pre-downloading WhisperX model weights (avoids a silent wait on first use)...
echo  (This may take a few minutes on first run.)
echo.
echo import torch > "%TEMP%\_dub_wx.py"
echo d = "cuda" if torch.cuda.is_available() else "cpu" >> "%TEMP%\_dub_wx.py"
echo import os >> "%TEMP%\_dub_wx.py"
echo default = "large-v2" if d == "cuda" else "small" >> "%TEMP%\_dub_wx.py"
echo m = os.environ.get("WHISPERX_MODEL", default) >> "%TEMP%\_dub_wx.py"
echo ct = "float16" if d == "cuda" else "int8" >> "%TEMP%\_dub_wx.py"
echo print(f"  Downloading Whisper '{m}' for {d}...") >> "%TEMP%\_dub_wx.py"
echo import whisperx >> "%TEMP%\_dub_wx.py"
echo whisperx.load_model(m, d, compute_type=ct) >> "%TEMP%\_dub_wx.py"
echo print(f"  [OK] Whisper '{m}' cached.") >> "%TEMP%\_dub_wx.py"
python "%TEMP%\_dub_wx.py"
if errorlevel 1 (
    echo.
    echo  [WARN] WhisperX model pre-download failed -- it will download on first use instead.
    echo         Or set ASR_BACKEND=fish in .env to use Fish Audio cloud ASR.
    echo.
) else (
    echo.
)
del "%TEMP%\_dub_wx.py" 2>nul

REM -- 6. API keys ---------------------------------------------------------------
echo ======================================================
echo   API Key Setup
echo ======================================================
echo.

if exist .env (
    echo  [INFO] .env already exists -- skipping key setup.
    echo         Edit .env manually to update your keys.
    echo.
    goto :pull_model
)

echo  [1] FISH_AUDIO_API_KEY  (required^)
echo      Used for voice-cloned TTS ^(free s2.1-pro-free model -- zero cost by default^).
echo      Free signup, no payment method required.
echo      Sign up at: https://fish.audio  -- Dashboard -- API Keys
echo.
set /p FISH_KEY=  Enter Fish Audio API key:
echo.

if /i "!USE_OLLAMA!"=="y" (
    echo  [2] GROQ_API_KEY  (optional -- cloud fallback if Ollama is unavailable^)
    echo      Leave blank to use Ollama only.
    echo      Get one at: https://console.groq.com
    echo.
    set /p GROQ_KEY=  Groq API key (Enter to skip^):
) else (
    echo  [2] GROQ_API_KEY  (required for translation^)
    echo      Free tier, no billing required.
    echo      Get one at: https://console.groq.com
    echo.
    set /p GROQ_KEY=  Enter Groq API key:
)
echo.

echo  [3] HF_TOKEN  (recommended^)
echo      Enables speaker diarization in WhisperX ^(the default ASR backend^).
echo      Without it, all speech is attributed to a single speaker.
echo.
echo      Setup:
echo        Token:      https://huggingface.co/settings/tokens
echo        Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
echo                      https://huggingface.co/pyannote/segmentation-3.0
echo.
set /p HF_KEY=  Hugging Face token (Enter to skip^):
echo.

echo  Writing .env...

if /i "!USE_OLLAMA!"=="y" (
    (
        echo FISH_AUDIO_API_KEY=!FISH_KEY!
        echo.
        echo # Translation -- local Ollama ^(GPU-accelerated, offline^)
        echo TRANSLATION_BACKEND=ollama
        echo OLLAMA_MODEL=qwen2.5:14b
    ) > .env
    if "!GROQ_KEY!"=="" (
        echo # GROQ_API_KEY=  ^(optional cloud fallback -- set TRANSLATION_BACKEND=groq to use^)>> .env
    ) else (
        echo.>> .env
        echo # Cloud fallback -- set TRANSLATION_BACKEND=groq to use instead of Ollama>> .env
        echo GROQ_API_KEY=!GROQ_KEY!>> .env
    )
) else (
    (
        echo FISH_AUDIO_API_KEY=!FISH_KEY!
        echo.
        echo # Translation -- Groq ^(default^), Gemini, or Ollama
        echo GROQ_API_KEY=!GROQ_KEY!
        echo # GEMINI_API_KEY=your_gemini_api_key_here
    ) > .env
)

(
    echo.
    echo # Phase B -- multi-speaker diarization ^(optional^)
) >> .env

if "!HF_KEY!"=="" (
    echo # HF_TOKEN=hf_...>> .env
) else (
    echo HF_TOKEN=!HF_KEY!>> .env
)

echo  [OK] .env written.
echo.

REM -- 7. Pull Ollama model (after keys so user can walk away) ------------------
:pull_model
if /i not "!USE_OLLAMA!"=="y" goto :launch
if "!OLLAMA_READY!"=="0" goto :launch

echo ======================================================
echo   Pulling translation model
echo ======================================================
echo.
echo  Downloading qwen2.5:14b (~9 GB^) -- this may take 10-30 minutes.
echo  You can step away; the app will launch automatically when done.
echo.
ollama pull qwen2.5:14b
if errorlevel 1 (
    echo.
    echo  WARNING: Model pull failed. To retry manually:
    echo    ollama pull qwen2.5:14b
    echo  Then start the app:
    echo    python app.py
    echo.
) else (
    echo.
    echo  [OK] qwen2.5:14b ready.
    echo.
)

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
