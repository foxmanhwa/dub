"""Fish Audio TTS — instant voice cloning via msgpack API."""

import os
import time
from pathlib import Path

import httpx
import msgpack


FISH_TTS_URL = "https://api.fish.audio/v1/tts"
_RETRY_DELAYS = [2, 5, 10]  # seconds between retries


def build_voice_reference(audio_path: str) -> tuple[bytes, str]:
    """Return (audio_bytes, "") ready for the Fish Audio TTS references field."""
    with open(audio_path, "rb") as f:
        return f.read(), ""


def synthesize(
    text: str,
    reference_audio_bytes: bytes,
    reference_text: str = "",
    output_format: str = "mp3",
    latency: str = "balanced",
) -> bytes:
    """
    Generate speech for text, cloned from reference_audio_bytes.
    Returns raw audio bytes in the requested format.
    Retries up to 3 times on transient network errors.
    """
    api_key = os.environ["FISH_AUDIO_API_KEY"]

    payload = {
        "text": text,
        "references": [{"audio": reference_audio_bytes, "text": reference_text}],
        "format": output_format,
        "latency": latency,
        "streaming": False,
    }
    body = msgpack.packb(payload, use_bin_type=True)

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = httpx.post(
                FISH_TTS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/msgpack",
                },
                content=body,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise  # 4xx errors are not retryable
    raise RuntimeError(f"TTS failed after {len(_RETRY_DELAYS) + 1} attempts") from last_exc


def synthesize_segments(
    segments: list[dict],
    reference_audio_bytes: bytes,
    reference_text: str = "",
    output_dir: str = "output",
    progress_cb=None,
) -> list[dict]:
    """
    Synthesize TTS for each translated segment.
    Adds "tts_path" key to each segment.  Segments without translated_text
    get tts_path=None.

    progress_cb: optional callable(done, total) for progress reporting.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = sum(1 for s in segments if s.get("translated_text", "").strip())
    done = 0
    result = []

    for i, seg in enumerate(segments):
        text = seg.get("translated_text", "").strip()
        if not text:
            result.append({**seg, "tts_path": None})
            continue

        audio = synthesize(
            text=text,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=reference_text,
        )

        path = out / f"seg_{i:04d}.mp3"
        path.write_bytes(audio)
        done += 1
        if progress_cb:
            progress_cb(done, total)
        result.append({**seg, "tts_path": str(path)})

    return result
