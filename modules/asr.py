"""Fish Audio ASR module — transcribes audio with word/segment timestamps."""

import os
import httpx
from pathlib import Path


FISH_ASR_URL = "https://api.fish.audio/v1/asr"


def transcribe(audio_path: str | Path, language: str | None = None) -> dict:
    """
    Transcribe audio using Fish Audio ASR.

    Returns a dict matching Fish Audio's response:
      {
        "text": "...",
        "duration": 12.3,
        "segments": [
          {"text": "...", "start": 0.0, "end": 1.5},
          ...
        ]
      }

    language: BCP-47 code like "en", "zh", "es", or None for auto-detect.
    """
    api_key = os.environ["FISH_AUDIO_API_KEY"]
    audio_path = Path(audio_path)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {"audio": (audio_path.name, audio_bytes, _mime(audio_path))}
    data = {"ignore_timestamps": "false"}
    if language and language != "auto":
        data["language"] = language

    resp = httpx.post(
        FISH_ASR_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(ext, "audio/mpeg")
