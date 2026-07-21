"""Fish Audio TTS module — instant voice cloning + speech synthesis."""

import io
import os
import struct
import httpx
import msgpack


FISH_TTS_URL = "https://api.fish.audio/v1/tts"


def build_voice_reference(audio_path: str, reference_text: str = "") -> tuple[bytes, str]:
    """
    Read a reference audio clip and return (audio_bytes, text) for voice cloning.
    The text is optional — helps the model understand the speaker's pronunciation.
    """
    with open(audio_path, "rb") as f:
        return f.read(), reference_text


def synthesize(
    text: str,
    reference_audio_bytes: bytes,
    reference_text: str = "",
    output_format: str = "mp3",
    latency: str = "balanced",
) -> bytes:
    """
    Generate speech for `text` cloned from reference_audio_bytes.
    Returns raw audio bytes in the requested format.
    """
    api_key = os.environ["FISH_AUDIO_API_KEY"]

    payload = {
        "text": text,
        "references": [
            {
                "audio": reference_audio_bytes,
                "text": reference_text,
            }
        ],
        "format": output_format,
        "latency": latency,
        "streaming": False,
    }

    body = msgpack.packb(payload, use_bin_type=True)

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


def synthesize_segments(
    segments: list[dict],
    reference_audio_bytes: bytes,
    reference_text: str = "",
    output_dir: str = "output",
) -> list[dict]:
    """
    Synthesize TTS for each translated segment.
    Adds "tts_path" key to each segment dict.
    Skips segments with empty translated_text.
    """
    import os
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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
        result.append({**seg, "tts_path": str(path)})

    return result
