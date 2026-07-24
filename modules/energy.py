"""
Per-segment audio energy analysis — pure Python, no extra dependencies.

Used to:
  1. Filter whispered/quiet segments from speaker reference clip selection.
  2. Tag segments as loud/normal/quiet for emotion-aware translation prompts.
"""

import math
import struct
import wave
from pathlib import Path


def compute_segment_rms(wav_path: str, start: float, end: float) -> float | None:
    """
    Return normalised RMS (0.0–1.0) for the audio in [start, end) seconds.

    Reads only the relevant portion from the WAV file — fast even for long files.
    Returns None on error (unsupported format, out-of-range window, etc.).
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()  # bytes per sample

            start_frame = max(0, int(start * sr))
            end_frame = min(wf.getnframes(), int(end * sr))
            n_frames = end_frame - start_frame
            if n_frames <= 0:
                return None

            wf.setpos(start_frame)
            raw = wf.readframes(n_frames)

        # Only handle 16-bit signed PCM (by far the most common output from ffmpeg)
        if sw != 2:
            return None

        n_samples = len(raw) // 2
        if n_samples == 0:
            return None

        samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])

        # For multi-channel audio, take only the first channel
        if n_ch > 1:
            samples = samples[::n_ch]

        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        return rms / 32768.0  # normalise to 0.0–1.0

    except Exception:
        return None


def compute_energy_profile(
    wav_path: str, segments: list[dict]
) -> list[float | None]:
    """
    Compute RMS energy for each segment in the list.
    Returns a parallel list of floats (or None for failures).
    """
    wav_path = str(wav_path)
    if not Path(wav_path).exists():
        return [None] * len(segments)
    return [
        compute_segment_rms(wav_path, s.get("start", 0.0), s.get("end", 0.0))
        for s in segments
    ]


def classify_energy(
    rms: float | None,
    mean_rms: float,
    std_rms: float,
    loud_threshold: float = 1.5,
    quiet_threshold: float = 0.5,
) -> str:
    """
    Tag a segment as "loud", "normal", or "quiet" relative to the speaker's mean.

    loud_threshold / quiet_threshold are Z-score multiples of std_rms.
    """
    if rms is None or mean_rms <= 0:
        return "normal"
    if std_rms <= 0:
        std_rms = mean_rms * 0.1  # fallback to 10% of mean
    z = (rms - mean_rms) / std_rms
    if z >= loud_threshold:
        return "loud"
    if z <= -quiet_threshold:
        return "quiet"
    return "normal"


def tag_segments_with_energy(
    wav_path: str,
    segments: list[dict],
) -> list[dict]:
    """
    Add an "energy" field ("loud" | "normal" | "quiet") to each segment.
    Operates per-speaker so the classification is relative to each speaker's
    typical volume rather than a global threshold.
    """
    rms_values = compute_energy_profile(wav_path, segments)

    # Group by speaker for per-speaker statistics
    by_speaker: dict[str, list[float]] = {}
    for seg, rms in zip(segments, rms_values):
        spk = seg.get("speaker", "SPEAKER_00")
        if rms is not None:
            by_speaker.setdefault(spk, []).append(rms)

    # Compute per-speaker mean and std
    stats: dict[str, tuple[float, float]] = {}
    for spk, vals in by_speaker.items():
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[spk] = (mean, math.sqrt(variance))

    result = []
    for seg, rms in zip(segments, rms_values):
        spk = seg.get("speaker", "SPEAKER_00")
        mean_rms, std_rms = stats.get(spk, (0.0, 0.0))
        tag = classify_energy(rms, mean_rms, std_rms)
        result.append({**seg, "energy": tag})

    return result


def select_representative_segments(
    segments: list[dict],
    wav_path: str,
    target_duration: float = 12.0,
    min_quiet_db_margin: float = 15.0,
) -> list[dict]:
    """
    From a list of segments for one speaker, select up to target_duration of
    audio while preferring segments that have representative (non-whispered) energy.

    Segments whose RMS is more than min_quiet_db_margin dB below the speaker's
    loudest segment are deprioritised (moved to the end of the candidate list).

    This prevents a reference clip built entirely from whispers even if the
    speaker speaks normally in most of the video.
    """
    if not segments:
        return []

    rms_values = [
        compute_segment_rms(wav_path, s["start"], s["end"])
        for s in segments
    ]

    # Compute max RMS among valid segments
    valid_rms = [r for r in rms_values if r is not None and r > 0]
    if not valid_rms:
        # No energy data — fall back to duration-sorted order
        return sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True)

    max_rms = max(valid_rms)
    # Convert margin to linear ratio: margin dB below max
    min_rms = max_rms * (10 ** (-min_quiet_db_margin / 20.0))

    # Split into "good energy" and "too quiet" buckets
    good, quiet = [], []
    for seg, rms in zip(segments, rms_values):
        dur = seg["end"] - seg["start"]
        if rms is None or rms >= min_rms:
            good.append((seg, dur))
        else:
            quiet.append((seg, dur))

    # Sort each bucket by duration (longest first)
    good.sort(key=lambda x: x[1], reverse=True)
    quiet.sort(key=lambda x: x[1], reverse=True)

    # Pick from good first, then quiet as fallback
    selected = []
    accumulated = 0.0
    for seg, dur in good + quiet:
        if accumulated >= target_duration:
            break
        selected.append(seg)
        accumulated += dur

    return selected
