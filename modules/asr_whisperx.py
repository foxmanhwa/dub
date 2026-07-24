"""
WhisperX ASR — runs in an isolated subprocess (same pattern as diarization.py).

Why subprocess?
  - whisperx loads faster-whisper + pyannote + torch, hundreds of MB that would
    otherwise occupy the main process for the rest of the run.
  - Subprocess exit frees all model memory before translation starts.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_WORKER = str(Path(__file__).parent / "_whisperx_worker.py")
_TIMEOUT = 1800  # 30 min — large model on CPU can be slow for long clips


def check_available() -> list[str]:
    """Return a list of missing prerequisites (empty = all good)."""
    import importlib.util
    issues = []
    if importlib.util.find_spec("whisperx") is None:
        issues.append("whisperx not installed  (pip install whisperx)")
    return issues


def transcribe(audio_path: str | Path, language: str | None = None) -> dict:
    """
    Transcribe audio using WhisperX in a subprocess.

    Returns:
      {
        "text": "full transcript",
        "duration": 12.3,
        "segments": [
          {"text": "...", "start": 0.0, "end": 1.5},          # no HF_TOKEN
          {"text": "...", "start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},  # with HF_TOKEN
          ...
        ],
        "language": "en",
        "speakers_assigned": bool,
      }

    language: ISO 639-1 code ("en", "zh", "es" …) or None for auto-detect.
    """
    audio_path = Path(audio_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name

    cmd = [sys.executable, _WORKER, str(audio_path), result_path]
    if language and language != "auto":
        cmd.append(language)

    env = os.environ.copy()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=env,
        )
        # Forward stderr from the worker (contains progress lines)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(line, file=sys.stderr)
        if proc.stdout:
            for line in proc.stdout.splitlines():
                print(line, file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"WhisperX worker exited with code {proc.returncode}.\n"
                f"STDERR:\n{proc.stderr or '(empty)'}"
            )
        with open(result_path, encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        Path(result_path).unlink(missing_ok=True)
