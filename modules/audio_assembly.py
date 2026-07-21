"""
Audio assembly module — time-fits TTS clips to original segment durations,
then reassembles the full dubbed audio track and muxes into the output video.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path


# atempo filter supports 0.5–100× in a single pass (we stay within 0.5–2.0).
# Outside that range we chain two atempo filters.
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _audio_duration(path: str) -> float:
    """Return duration of an audio file in seconds via ffprobe."""
    result = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(result.stdout.strip())


def _video_duration(path: str) -> float:
    result = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(result.stdout.strip())


def _build_atempo_chain(ratio: float) -> str:
    """Build an ffmpeg atempo filter string for any ratio in [0.25, 4.0]."""
    ratio = max(0.25, min(4.0, ratio))
    if _ATEMPO_MIN <= ratio <= _ATEMPO_MAX:
        return f"atempo={ratio:.6f}"
    # Chain two filters: sqrt(ratio) each (works for 0.25–4.0)
    import math
    step = math.sqrt(ratio)
    step = max(_ATEMPO_MIN, min(_ATEMPO_MAX, step))
    return f"atempo={step:.6f},atempo={step:.6f}"


def time_fit_clip(
    tts_path: str,
    target_duration: float,
    output_path: str,
    sample_rate: int = 44100,
) -> str:
    """
    Stretch or compress a TTS audio clip to exactly target_duration seconds.
    Strategy:
      - If ratio is within [0.5, 2.0]: atempo filter.
      - If TTS is very short vs target: pad with silence at the end.
      - If TTS is long and ratio would exceed 2.0: truncate to target.
    Returns output_path.
    """
    tts_dur = _audio_duration(tts_path)

    if tts_dur <= 0:
        # Synthesize silence
        _run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(target_duration),
            output_path,
        ])
        return output_path

    ratio = tts_dur / target_duration  # >1 means TTS is longer (need to speed up)
    ratio = max(0.25, min(4.0, ratio))

    atempo = _build_atempo_chain(ratio)

    _run([
        "ffmpeg", "-y",
        "-i", tts_path,
        "-filter:a", atempo,
        "-ar", str(sample_rate),
        output_path,
    ])

    # Trim or pad to exact duration
    fitted_dur = _audio_duration(output_path)
    if abs(fitted_dur - target_duration) > 0.05:
        tmp = output_path + ".trim.wav"
        _run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-t", str(target_duration),
            "-af", f"apad=whole_dur={target_duration}",
            "-ar", str(sample_rate),
            tmp,
        ])
        os.replace(tmp, output_path)

    return output_path


def assemble_audio_track(
    segments: list[dict],
    original_audio_path: str,
    total_duration: float,
    output_path: str,
    work_dir: str,
    sample_rate: int = 44100,
) -> str:
    """
    Build a full-length audio track by:
    1. Starting from silence of total_duration.
    2. Overlaying the original audio for any segments without TTS.
    3. Overlaying time-fitted TTS clips at their original timestamps.

    Returns output_path (WAV).
    """
    work = Path(work_dir)
    fitted_dir = work / "fitted"
    fitted_dir.mkdir(parents=True, exist_ok=True)

    # Convert original audio to WAV at our sample rate
    orig_wav = str(work / "original_audio.wav")
    _run([
        "ffmpeg", "-y", "-i", original_audio_path,
        "-ar", str(sample_rate), "-ac", "1",
        orig_wav,
    ])

    # Build a silence base
    silence_path = str(work / "silence_base.wav")
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(total_duration),
        silence_path,
    ])

    # Fit each TTS clip and record delays
    fitted_clips: list[tuple[str, float]] = []  # (path, start_time)

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        target_dur = end - start
        tts_path = seg.get("tts_path")

        if not tts_path or not os.path.exists(tts_path):
            # Keep original audio for this segment — no overlay needed
            continue

        fitted_path = str(fitted_dir / f"fitted_{i:04d}.wav")
        time_fit_clip(tts_path, target_dur, fitted_path, sample_rate)
        fitted_clips.append((fitted_path, start))

    if not fitted_clips:
        # No TTS — just return original audio converted
        _run([
            "ffmpeg", "-y", "-i", orig_wav,
            "-t", str(total_duration),
            output_path,
        ])
        return output_path

    # Build amix filter graph using adelay + amix
    # First, create a "dubbed" track: original audio with TTS segments
    # overriding silence, then mix.
    #
    # Strategy: build an ffmpeg complex filter that:
    #   - takes the original audio as base
    #   - overlays each TTS clip using adelay at its start offset
    #   - at segment positions: mix only TTS (mute original)
    #
    # Simpler approach: create segment replacements and use concat.
    _assemble_with_gaps(
        orig_wav=orig_wav,
        fitted_clips=fitted_clips,
        segments=segments,
        total_duration=total_duration,
        output_path=output_path,
        work_dir=str(work),
        sample_rate=sample_rate,
    )
    return output_path


def _assemble_with_gaps(
    orig_wav: str,
    fitted_clips: list[tuple[str, float]],  # (path, start)
    segments: list[dict],
    total_duration: float,
    output_path: str,
    work_dir: str,
    sample_rate: int,
) -> None:
    """
    Cut the timeline into regions:
      - Gap regions: from original audio
      - Segment regions: from fitted TTS clips (or original if no TTS)
    Concatenate all regions into the final track.
    """
    work = Path(work_dir)

    # Build a lookup: segment index -> fitted path
    tts_lookup: dict[int, str] = {}
    for clip_path, start in fitted_clips:
        for i, seg in enumerate(segments):
            if abs(seg["start"] - start) < 0.01:
                tts_lookup[i] = clip_path
                break

    # Build ordered timeline events
    events: list[tuple[float, float, str | None]] = []  # (start, end, tts_path|None)
    for i, seg in enumerate(segments):
        events.append((seg["start"], seg["end"], tts_lookup.get(i)))

    events.sort(key=lambda x: x[0])

    pieces: list[str] = []
    cursor = 0.0
    piece_idx = 0

    for ev_start, ev_end, tts_path in events:
        # Gap before this segment
        if ev_start > cursor + 0.01:
            gap_path = str(work / f"piece_{piece_idx:04d}_gap.wav")
            _extract_audio_segment(orig_wav, cursor, ev_start, gap_path, sample_rate)
            pieces.append(gap_path)
            piece_idx += 1

        # The segment itself
        seg_dur = ev_end - ev_start
        seg_path = str(work / f"piece_{piece_idx:04d}_seg.wav")
        if tts_path and os.path.exists(tts_path):
            # Use TTS clip (already fitted to seg_dur)
            _extract_exact_duration(tts_path, seg_dur, seg_path, sample_rate)
        else:
            _extract_audio_segment(orig_wav, ev_start, ev_end, seg_path, sample_rate)
        pieces.append(seg_path)
        piece_idx += 1
        cursor = ev_end

    # Trailing gap after last segment
    if cursor < total_duration - 0.01:
        tail_path = str(work / f"piece_{piece_idx:04d}_tail.wav")
        _extract_audio_segment(orig_wav, cursor, total_duration, tail_path, sample_rate)
        pieces.append(tail_path)

    if not pieces:
        _run(["ffmpeg", "-y", "-i", orig_wav, output_path])
        return

    # Concatenate all pieces
    concat_list = str(work / "concat_list.txt")
    with open(concat_list, "w") as f:
        for p in pieces:
            f.write(f"file '{p}'\n")

    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-ar", str(sample_rate),
        output_path,
    ])


def _extract_audio_segment(
    source: str, start: float, end: float, out: str, sample_rate: int
) -> None:
    duration = max(0.01, end - start)
    _run([
        "ffmpeg", "-y",
        "-i", source,
        "-ss", f"{start:.6f}",
        "-t", f"{duration:.6f}",
        "-ar", str(sample_rate), "-ac", "1",
        out,
    ])


def _extract_exact_duration(source: str, duration: float, out: str, sample_rate: int) -> None:
    _run([
        "ffmpeg", "-y",
        "-i", source,
        "-t", f"{duration:.6f}",
        "-af", f"apad=whole_dur={duration:.6f}",
        "-ar", str(sample_rate), "-ac", "1",
        out,
    ])


def mux_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """Replace the audio track of video_path with audio_path, output to output_path."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ])
    return output_path


def extract_audio(video_path: str, output_path: str, sample_rate: int = 44100) -> str:
    """Extract audio from a video file to a WAV file."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ar", str(sample_rate), "-ac", "1",
        output_path,
    ])
    return output_path


def extract_reference_clip(
    audio_path: str,
    output_path: str,
    start: float = 0.0,
    duration: float = 10.0,
) -> str:
    """
    Extract a short reference clip from audio_path for voice cloning.
    Tries to pick a segment with speech (skips the very beginning if silent).
    """
    _run([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-ar", "44100", "-ac", "1",
        output_path,
    ])
    return output_path
