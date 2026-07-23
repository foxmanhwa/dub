"""Fish Audio ASR — transcribes audio with segment timestamps."""

import os
import sys
import httpx
from pathlib import Path


FISH_ASR_URL = "https://api.fish.audio/v1/asr"


def transcribe(audio_path: str | Path, language: str | None = None) -> dict:
    """
    Transcribe audio using Fish Audio ASR.

    Returns the Fish Audio response dict:
      {
        "text": "...",
        "duration": 12.3,
        "segments": [
          {"text": "...", "start": 0.0, "end": 1.5},
          ...
        ]
      }

    language: BCP-47 code ("en", "zh", "es" …) or None for auto-detect.
    """
    api_key = os.environ["FISH_AUDIO_API_KEY"]
    audio_path = Path(audio_path)

    file_size = audio_path.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Audio file is empty (0 bytes): {audio_path}")

    mime = _mime(audio_path)
    size_mb = file_size / 1e6
    print(
        f"[ASR] sending {audio_path.name}  {size_mb:.1f} MB  mime={mime}",
        file=sys.stderr,
    )

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {"audio": (audio_path.name, audio_bytes, mime)}
    data = {"ignore_timestamps": "false"}
    if language and language != "auto":
        data["language"] = language

    resp = httpx.post(
        FISH_ASR_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=180,
    )

    if not resp.is_success:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:1000]
        raise RuntimeError(
            f"Fish Audio ASR {resp.status_code}: {detail}"
        )

    return resp.json()


def _mime(path: Path) -> str:
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }.get(path.suffix.lower(), "audio/mpeg")
