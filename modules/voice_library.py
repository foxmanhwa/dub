"""
Voice library — Fish Audio public models + locally saved cloned voices.
"""

import json
import os
import re
import shutil
from pathlib import Path

import httpx


FISH_MODEL_URL = "https://api.fish.audio/model"
SAVED_VOICES_FILE = Path.home() / ".dub" / "saved_voices.json"
SAVED_VOICES_AUDIO_DIR = Path.home() / ".dub" / "voices"


def list_fish_voices(
    language: str = "",
    title: str = "",
    page_size: int = 50,
) -> list[dict]:
    """
    Fetch public Fish Audio voice models.
    Returns list of dicts with at least {id, title, languages, visibility}.
    """
    api_key = os.environ.get("FISH_AUDIO_API_KEY", "")
    params: dict = {"page_size": page_size, "page_number": 1, "sort_by": "task_count"}
    if language:
        params["language"] = language
    if title:
        params["title"] = title

    resp = httpx.get(
        FISH_MODEL_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def format_voice_label(item: dict) -> str:
    """Human-readable label for a Fish Audio model item."""
    title = item.get("title", item.get("_id", "Unknown"))
    langs = item.get("languages", [])
    if langs:
        return f"{title}  [{', '.join(langs)}]"
    return title


def load_saved_voices() -> dict:
    """Return {name: {audio_path, reference_text}} from local JSON. Thread-safe read."""
    if SAVED_VOICES_FILE.exists():
        try:
            with open(SAVED_VOICES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cloned_voice(name: str, ref_audio_path: str, reference_text: str = "") -> None:
    """Permanently copy a reference WAV and register it in saved_voices.json."""
    SAVED_VOICES_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = SAVED_VOICES_AUDIO_DIR / f"{_slugify(name)}.wav"
    shutil.copy2(ref_audio_path, dest)

    # Compute speaker embedding for future similarity matching (best-effort).
    embedding: list[float] | None = None
    try:
        from .speaker_embeddings import extract_embeddings, check_available as _emb_check
        if not _emb_check():
            embs = extract_embeddings({name: str(dest)})
            embedding = embs.get(name)
    except Exception:
        pass

    voices = load_saved_voices()
    SAVED_VOICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry: dict = {"audio_path": str(dest), "reference_text": reference_text}
    if embedding is not None:
        entry["embedding"] = embedding
    voices[name] = entry
    with open(SAVED_VOICES_FILE, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)


def load_saved_voice_embeddings() -> dict[str, list[float]]:
    """
    Return {name: embedding} for saved voices that have a stored embedding.
    Voices without a stored embedding are excluded (they'll be computed on demand).
    """
    voices = load_saved_voices()
    return {
        name: entry["embedding"]
        for name, entry in voices.items()
        if entry.get("embedding")
    }


def delete_saved_voice(name: str) -> None:
    """Remove a saved voice entry and its audio file."""
    voices = load_saved_voices()
    entry = voices.pop(name, None)
    if entry:
        p = Path(entry.get("audio_path", ""))
        if p.exists():
            p.unlink(missing_ok=True)
    SAVED_VOICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVED_VOICES_FILE, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)


def _slugify(name: str) -> str:
    return re.sub(r"[^\w-]", "_", name.lower())[:64]
