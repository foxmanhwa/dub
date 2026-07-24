"""
Speaker embedding extraction and matching via SpeechBrain ECAPA-TDNN.

Runs in a subprocess to keep SpeechBrain + torch out of the main process.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


_WORKER = str(Path(__file__).parent / "_embeddings_worker.py")
_TIMEOUT = 300  # 5 min — model download on first run may take a moment
_MATCH_THRESHOLD = 0.75  # cosine similarity above which we suggest a saved-voice match
_DISTINCT_THRESHOLD = 0.85  # above this, two "different" speakers look suspiciously similar


def check_available() -> list[str]:
    """Return missing prerequisites (empty = all good)."""
    import importlib.util
    issues = []
    if importlib.util.find_spec("speechbrain") is None:
        issues.append("speechbrain not installed  (pip install speechbrain)")
    if importlib.util.find_spec("torchaudio") is None:
        issues.append("torchaudio not installed")
    return issues


def extract_embeddings(label_wav: dict[str, str]) -> dict[str, list[float]]:
    """
    Extract ECAPA-TDNN speaker embeddings for a set of WAV files.

    label_wav: {label: wav_path}
    Returns:   {label: [float, ...]}   (empty dict on failure)
    """
    if not label_wav:
        return {}

    pairs = [{"label": k, "wav_path": str(v)} for k, v in label_wav.items()]

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as inp,
        tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out,
    ):
        json.dump(pairs, inp)
        inp_path, out_path = inp.name, out.name

    try:
        proc = subprocess.run(
            [sys.executable, _WORKER, inp_path, out_path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=os.environ.copy(),
        )
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(line, file=sys.stderr)
        if proc.returncode != 0:
            print(
                f"[embeddings] worker exited {proc.returncode} — skipping",
                file=sys.stderr,
            )
            return {}
        with open(out_path, encoding="utf-8") as fh:
            results: list[dict] = json.load(fh)
        return {r["label"]: r["embedding"] for r in results}
    except Exception as exc:
        print(f"[embeddings] extraction failed ({exc})", file=sys.stderr)
        return {}
    finally:
        Path(inp_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def verify_speaker_distinctness(
    detected_embeddings: dict[str, list[float]],
) -> list[str]:
    """
    Return warning strings for any pair of 'different' speakers whose
    embeddings are suspiciously similar (possible diarization error).
    """
    warnings = []
    spk_list = list(detected_embeddings)
    for i, a in enumerate(spk_list):
        for b in spk_list[i + 1:]:
            sim = cosine_similarity(detected_embeddings[a], detected_embeddings[b])
            if sim > _DISTINCT_THRESHOLD:
                warnings.append(
                    f"{a} and {b} have high embedding similarity ({sim:.2f}) "
                    "— possible diarization error or the same speaker labeled twice"
                )
    return warnings


def match_against_library(
    detected_embeddings: dict[str, list[float]],
    saved_embeddings: dict[str, list[float]],
) -> dict[str, tuple[str, float]]:
    """
    For each detected speaker, find the closest saved voice.

    Returns {speaker_label: (voice_name, similarity)} only for matches
    above _MATCH_THRESHOLD.
    """
    suggestions: dict[str, tuple[str, float]] = {}
    for spk, emb in detected_embeddings.items():
        best_name, best_sim = None, -1.0
        for name, saved_emb in saved_embeddings.items():
            sim = cosine_similarity(emb, saved_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name is not None and best_sim >= _MATCH_THRESHOLD:
            suggestions[spk] = (best_name, best_sim)
    return suggestions
