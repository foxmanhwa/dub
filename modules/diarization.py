"""
Speaker diarization via pyannote.audio — runs in an isolated subprocess.

Why subprocess?
  - `import pyannote.audio` drags in lightning + torch._dynamo (hundreds of MB).
  - Those modules are never freed from a Python process once imported.
  - Running in a subprocess means all that memory is released the moment diarization
    finishes, before translation starts.
  - It also isolates speechbrain's LazyModule import errors (k2, wordemb, etc.) that
    fire during `import pyannote` due to hasattr() probes from inspect.getmodule().
    The worker patches LazyModule.ensure_module before any import, so those errors
    are silenced inside the worker without affecting the main process at all.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKER = str(Path(__file__).parent / "_diarization_worker.py")
_DIARIZATION_TIMEOUT = 900  # 15 min — plenty for CPU inference on long clips


def check_available() -> list[str]:
    """Return a list of missing prerequisites (empty = all good)."""
    issues = []
    # Check for pyannote via metadata only — importing it loads lightning+torch._dynamo
    # which permanently occupies hundreds of MB. Use importlib.util.find_spec instead.
    import importlib.util
    if importlib.util.find_spec("pyannote") is None:
        issues.append("pyannote.audio not installed  (pip install pyannote.audio)")
    if not os.environ.get("HF_TOKEN"):
        issues.append("HF_TOKEN not set (required for pyannote models)")
    return issues


def run_diarization(audio_path: str) -> dict:
    """
    Run pyannote/speaker-diarization-3.1 in a subprocess and return:
      {
        "segments":        [{"start", "end", "speaker"}],
        "overlap_regions": [{"start", "end", "speakers": [str, ...]}],
        "num_speakers":    int,
      }
    The subprocess exits when done, so all pyannote/torch memory is freed before
    the caller continues.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        result_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, _WORKER, audio_path, result_path],
            capture_output=True,
            text=True,
            timeout=_DIARIZATION_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Diarization worker exited with code {proc.returncode}.\n"
                f"STDERR:\n{proc.stderr or '(empty)'}\n"
                f"STDOUT:\n{proc.stdout or '(empty)'}"
            )
        with open(result_path) as fh:
            return json.load(fh)
    finally:
        Path(result_path).unlink(missing_ok=True)


def release_models() -> None:
    """No-op: memory is freed automatically when the worker subprocess exits."""
    pass


# ── Pure-Python helpers (no ML, run in main process) ─────────────────────────

def assign_speakers_to_segments(
    asr_segments: list[dict],
    dia_segments: list[dict],
    overlap_regions: list[dict],
) -> list[dict]:
    """
    Tag each ASR segment with the dominant diarization speaker and whether it
    lies inside an overlap region.
    """
    result = []
    for seg in asr_segments:
        speaker = "SPEAKER_00"
        best = 0.0
        for d in dia_segments:
            ovlp = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
            if ovlp > best:
                best = ovlp
                speaker = d["speaker"]

        in_overlap = any(
            r["start"] - 0.1 <= seg["start"] and seg["end"] <= r["end"] + 0.1
            for r in overlap_regions
        )
        result.append({**seg, "speaker": speaker, "is_overlap": in_overlap})
    return result
