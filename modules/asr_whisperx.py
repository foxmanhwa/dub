"""
WhisperX ASR — runs in an isolated subprocess (same pattern as diarization.py).

Why subprocess?
  - whisperx loads faster-whisper + pyannote + torch, hundreds of MB that would
    otherwise occupy the main process for the rest of the run.
  - Subprocess exit frees all model memory before translation starts.

transcribe() is a generator: it yields progress strings from the worker in real
time (so the pipeline log stays live), then yields the result dict as the final
item.  Callers iterate and dispatch on type:

    for item in transcribe(path, language):
        if isinstance(item, str):
            yield item          # pipeline log
        else:
            asr_result = item   # final dict
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


_WORKER = str(Path(__file__).parent / "_whisperx_worker.py")
# Ceiling timeout — with the CPU default of "small", a 1-2 min clip finishes in
# 3-8 min.  30 min covers worst-case first-run model download + inference on CPU.
_TIMEOUT = 1800


def check_available() -> list[str]:
    """Return a list of missing prerequisites (empty = all good)."""
    import importlib.util
    issues = []
    if importlib.util.find_spec("whisperx") is None:
        issues.append("whisperx not installed  (pip install whisperx)")
    return issues


def transcribe(audio_path: str | Path, language: str | None = None):
    """
    Generator: yields progress strings from the worker subprocess in real time,
    then yields the result dict as the last item.

    language: ISO 639-1 code or None for auto-detect.
    Raises TimeoutError / RuntimeError on failure.
    """
    audio_path = Path(audio_path)
    result_fd, result_path = tempfile.mkstemp(suffix=".json")
    os.close(result_fd)

    cmd = [sys.executable, _WORKER, str(audio_path), result_path]
    if language and language != "auto":
        cmd.append(language)

    env = os.environ.copy()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,   # worker writes nothing to stdout
        stderr=subprocess.PIPE,      # progress lines come through stderr
        text=True,
        env=env,
    )

    # Drain stderr in a background thread — avoids pipe-buffer deadlock and lets
    # us do non-blocking reads with a timeout.
    line_q: queue.Queue[str | None] = queue.Queue()

    def _drain() -> None:
        for line in proc.stderr:
            line_q.put(line.rstrip())
        line_q.put(None)  # sentinel: stream exhausted

    threading.Thread(target=_drain, daemon=True).start()

    deadline = time.monotonic() + _TIMEOUT
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                raise TimeoutError(
                    f"WhisperX timed out after {_TIMEOUT}s. "
                    "On CPU, try a smaller model (WHISPERX_MODEL=small in .env) "
                    "or switch to the Fish Audio cloud ASR (ASR_BACKEND=fish)."
                )
            try:
                line = line_q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue
            if line is None:
                break
            yield line  # forward worker progress to the pipeline log

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"WhisperX worker exited with code {proc.returncode}. "
                "Check the pipeline log above for the error."
            )

        with open(result_path, encoding="utf-8") as fh:
            yield json.load(fh)  # final item: the result dict

    finally:
        Path(result_path).unlink(missing_ok=True)
