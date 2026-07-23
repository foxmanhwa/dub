"""Fish Audio TTS — instant voice cloning via msgpack API."""

import os
import time
from pathlib import Path

import httpx
import msgpack

from .audio_assembly import get_duration as _get_audio_duration, SAFE_RATIO_MIN, SAFE_RATIO_MAX


FISH_TTS_URL = "https://api.fish.audio/v1/tts"
_RETRY_DELAYS = [2, 5, 10]  # seconds between retries


def build_voice_reference(audio_path: str) -> tuple[bytes, str]:
    """Return (audio_bytes, "") ready for the Fish Audio TTS references field."""
    with open(audio_path, "rb") as f:
        return f.read(), ""


def synthesize(
    text: str,
    reference_audio_bytes: bytes | None = None,
    reference_text: str = "",
    reference_id: str | None = None,
    output_format: str = "mp3",
    latency: str = "balanced",
) -> bytes:
    """
    Generate speech for text.  Supply EITHER:
      - reference_audio_bytes (+ optional reference_text) for instant cloning, OR
      - reference_id for a Fish Audio library / saved model voice.
    Returns raw audio bytes in the requested format.
    Retries up to 3 times on transient network errors.
    """
    api_key = os.environ["FISH_AUDIO_API_KEY"]

    if reference_id:
        # Library / saved model — JSON body (no binary, so no need for msgpack)
        payload_json = {
            "text": text,
            "reference_id": reference_id,
            "format": output_format,
            "latency": latency,
            "streaming": False,
        }
        import json as _json
        body = _json.dumps(payload_json).encode()
        content_type = "application/json"
    else:
        payload = {
            "text": text,
            "references": [{"audio": reference_audio_bytes, "text": reference_text}],
            "format": output_format,
            "latency": latency,
            "streaming": False,
        }
        body = msgpack.packb(payload, use_bin_type=True)
        content_type = "application/msgpack"

    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = httpx.post(
                FISH_TTS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": content_type,
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
    reference_audio_bytes: bytes | None = None,
    reference_text: str = "",
    reference_id: str | None = None,
    speaker_references: "dict[str, tuple[bytes | None, str, str | None]] | None" = None,
    output_dir: str = "output",
    progress_cb=None,
    translate_retry_fn=None,
    max_fit_retries: int = 2,
) -> list[dict]:
    """
    Synthesize TTS for each translated segment.

    Supply either reference_audio_bytes (clone) or reference_id (library/saved model)
    as the global fallback reference.

    speaker_references: optional dict {speaker_id: (audio_bytes_or_None, ref_text, reference_id_or_None)}.
      - (bytes, ref_text, None)  → instant voice clone for this speaker
      - (None, "", voice_id)     → Fish Audio library voice fallback for this speaker
    Falls back to the global reference when a segment's speaker is absent from the dict.

    translate_retry_fn: optional callable(translated_text, original_text, target_secs, direction) -> str.
      When provided, segments whose TTS duration falls outside [SAFE_RATIO_MIN, SAFE_RATIO_MAX]
      relative to their original slot are retranslated and re-synthesized up to max_fit_retries times.
      direction is "shorter" (TTS ran over) or "longer" (TTS was too short).

    Adds "tts_path" and "fit_retries" keys to each segment.
    Segments without translated_text get tts_path=None.
    progress_cb: optional callable(done, total).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = sum(1 for s in segments if s.get("translated_text", "").strip())
    done = 0
    result = []

    for i, seg in enumerate(segments):
        text = seg.get("translated_text", "").strip()
        if not text:
            orig = seg.get("text", "").strip()
            reason = (
                f"translation is empty (original: {orig!r})"
                if orig
                else "both original text and translation are empty"
            )
            print(f"[TTS skip] seg {i}: {reason}")
            result.append({**seg, "tts_path": None, "tts_error": f"empty translation — {reason}"})
            continue

        # Per-speaker reference lookup
        seg_ref_bytes = reference_audio_bytes
        seg_ref_text = reference_text
        seg_ref_id = reference_id
        if speaker_references:
            speaker = seg.get("speaker")
            if speaker and speaker in speaker_references:
                spk_bytes, spk_text, spk_id = speaker_references[speaker]
                if spk_id is not None:
                    # Library voice fallback for this speaker
                    seg_ref_id = spk_id
                    seg_ref_bytes = None
                    seg_ref_text = ""
                else:
                    # Instant clone for this speaker
                    seg_ref_bytes = spk_bytes
                    seg_ref_text = spk_text
                    seg_ref_id = None
            elif speaker:
                print(f"[TTS] seg {i}: speaker {speaker!r} not in speaker_references — using global ref")

        try:
            current_text = text
            audio = synthesize(
                text=current_text,
                reference_audio_bytes=seg_ref_bytes,
                reference_text=seg_ref_text,
                reference_id=seg_ref_id,
            )
            path = out / f"seg_{i:04d}.mp3"
            path.write_bytes(audio)

            # Iterative fit: retranslate if TTS duration is outside safe stretch bounds
            fit_retries = 0
            if translate_retry_fn is not None and max_fit_retries > 0:
                orig_dur = seg.get("end", 0.0) - seg.get("start", 0.0)
                if orig_dur > 0:
                    for _ in range(max_fit_retries):
                        try:
                            tts_dur = _get_audio_duration(str(path))
                        except Exception:
                            break
                        ratio = tts_dur / orig_dur
                        if SAFE_RATIO_MIN <= ratio <= SAFE_RATIO_MAX:
                            break
                        direction = "shorter" if ratio > SAFE_RATIO_MAX else "longer"
                        print(
                            f"[TTS fit] seg {i}: ratio {ratio:.2f} — "
                            f"requesting {direction} retranslation"
                        )
                        new_text = translate_retry_fn(
                            current_text, seg.get("text", ""), orig_dur, direction
                        )
                        if not new_text or new_text == current_text:
                            print(f"[TTS fit] seg {i}: retranslation unchanged — stopping")
                            break
                        new_audio = synthesize(
                            text=new_text,
                            reference_audio_bytes=seg_ref_bytes,
                            reference_text=seg_ref_text,
                            reference_id=seg_ref_id,
                        )
                        path.write_bytes(new_audio)
                        current_text = new_text
                        fit_retries += 1
                        try:
                            new_dur = _get_audio_duration(str(path))
                            print(
                                f"[TTS fit] seg {i}: retry {fit_retries} → "
                                f"ratio {new_dur / orig_dur:.2f}"
                            )
                        except Exception:
                            pass

            done += 1
            if progress_cb:
                progress_cb(done, total)
            result.append({
                **seg,
                "tts_path": str(path),
                "translated_text": current_text,
                "fit_retries": fit_retries,
            })
        except Exception as exc:
            print(f"[TTS error] seg {i}: {exc!r}  text={text!r}")
            result.append({**seg, "tts_path": None, "tts_error": str(exc), "fit_retries": 0})

    return result
