"""
Audio assembly — time-fits TTS clips to original durations, reassembles the
full dubbed audio track, and muxes it back into the output video.
"""

import math
import os
import subprocess
from pathlib import Path


# atempo supports 0.5–2.0 in a single pass; outside that we chain two filters.
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def get_duration(path: str) -> float:
    """Return duration of any media file in seconds via ffprobe."""
    result = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(result.stdout.strip())


def extract_audio(video_path: str, output_path: str, sample_rate: int = 44100) -> str:
    """Extract mono audio from a video file to a WAV."""
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", str(sample_rate), "-ac", "1",
        output_path,
    ])
    return output_path


def extract_reference_clip(
    audio_path: str,
    output_path: str,
    start: float = 0.0,
    duration: float = 12.0,
) -> str:
    """Extract a short clip from audio_path for voice cloning."""
    _run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-ar", "44100", "-ac", "1",
        output_path,
    ])
    return output_path


def mux_audio_into_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Replace the audio track of video_path with audio_path."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-ac", "1",          # explicit codec + mono
        "-map", "0:v:0", "-map", "1:a:0",
        "-movflags", "+faststart",            # moov atom at front for browser play
        "-shortest",
        output_path,
    ])
    return output_path


# ── Time-fitting ──────────────────────────────────────────────────────────────

def _atempo_chain(ratio: float) -> str:
    """
    Build an ffmpeg atempo filter string for ratio in [0.25, 4.0].
    Single filter when in [0.5, 2.0]; two chained filters otherwise.
    """
    ratio = max(0.25, min(4.0, ratio))
    if _ATEMPO_MIN <= ratio <= _ATEMPO_MAX:
        return f"atempo={ratio:.6f}"
    step = max(_ATEMPO_MIN, min(_ATEMPO_MAX, math.sqrt(ratio)))
    return f"atempo={step:.6f},atempo={step:.6f}"


def time_fit_clip(
    tts_path: str,
    target_duration: float,
    output_path: str,
    sample_rate: int = 44100,
) -> str:
    """
    Stretch or compress a TTS clip to exactly target_duration seconds.

    Strategy:
    - Zero/negative target_duration: transcode TTS as-is (can't compress to nothing).
    - Ratio in [0.25, 4.0]: atempo filter (chained if needed).
    - Zero-length source: synthesize silence.
    - After atempo: trim/pad to exact target_duration.
    """
    if target_duration <= 0:
        # Zero-duration ASR segment — keep the TTS clip at its natural length.
        _run(["ffmpeg", "-y", "-i", tts_path, "-ar", str(sample_rate), "-ac", "1", output_path])
        return output_path

    tts_dur = get_duration(tts_path)

    if tts_dur <= 0:
        _run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(target_duration), output_path,
        ])
        return output_path

    ratio = max(0.25, min(4.0, tts_dur / target_duration))
    atempo = _atempo_chain(ratio)

    _run([
        "ffmpeg", "-y", "-i", tts_path,
        "-filter:a", atempo, "-ar", str(sample_rate), "-ac", "1",
        output_path,
    ])

    # Trim or pad to be exact
    fitted_dur = get_duration(output_path)
    if abs(fitted_dur - target_duration) > 0.05:
        tmp = output_path + ".trim.wav"
        _run([
            "ffmpeg", "-y", "-i", output_path,
            "-t", str(target_duration),
            "-af", f"apad=whole_dur={target_duration}",
            "-ar", str(sample_rate),
            tmp,
        ])
        os.replace(tmp, output_path)

    return output_path


# ── Audio track assembly ──────────────────────────────────────────────────────

def assemble_audio_track(
    segments: list[dict],
    original_audio_path: str,
    total_duration: float,
    output_path: str,
    work_dir: str,
    sample_rate: int = 44100,
) -> str:
    """
    Build a full-length audio track:
    - Gaps between segments: lifted from the original audio.
    - Segment windows: replaced with time-fitted TTS clips.
    - Segments without TTS: kept from the original audio.

    Returns output_path (WAV).
    """
    work = Path(work_dir)
    fitted_dir = work / "fitted"
    fitted_dir.mkdir(parents=True, exist_ok=True)

    # Normalise original audio
    orig_wav = str(work / "original_audio.wav")
    _run([
        "ffmpeg", "-y", "-i", original_audio_path,
        "-ar", str(sample_rate), "-ac", "1",
        orig_wav,
    ])

    # Time-fit every TTS clip
    tts_lookup: dict[int, str] = {}
    for i, seg in enumerate(segments):
        tts_path = seg.get("tts_path")
        if not tts_path or not os.path.exists(tts_path):
            continue
        target_dur = seg["end"] - seg["start"]
        if target_dur <= 0:
            # Zero-duration ASR segment: no time slot to place audio into — skip.
            continue
        fitted_path = str(fitted_dir / f"fitted_{i:04d}.wav")
        time_fit_clip(tts_path, target_dur, fitted_path, sample_rate)
        tts_lookup[i] = fitted_path

    if not tts_lookup:
        # Nothing to replace — return original audio trimmed to total_duration
        _run([
            "ffmpeg", "-y", "-i", orig_wav,
            "-t", str(total_duration), output_path,
        ])
        return output_path

    # Slice timeline into pieces and concatenate
    pieces: list[str] = []
    cursor = 0.0
    piece_idx = 0

    for i, seg in enumerate(segments):
        ev_start, ev_end = seg["start"], seg["end"]

        if ev_end - ev_start <= 0:
            # Zero-duration segment: no time slot, nothing to place.
            continue

        if ev_start > cursor + 0.01:
            gap_path = str(work / f"piece_{piece_idx:04d}_gap.wav")
            _extract_segment(orig_wav, cursor, ev_start, gap_path, sample_rate)
            pieces.append(gap_path)
            piece_idx += 1

        seg_dur = ev_end - ev_start
        seg_path = str(work / f"piece_{piece_idx:04d}_seg.wav")
        if i in tts_lookup:
            _extract_exact(tts_lookup[i], seg_dur, seg_path, sample_rate)
        else:
            _extract_segment(orig_wav, ev_start, ev_end, seg_path, sample_rate)
        pieces.append(seg_path)
        piece_idx += 1
        cursor = ev_end

    if cursor < total_duration - 0.01:
        tail_path = str(work / f"piece_{piece_idx:04d}_tail.wav")
        _extract_segment(orig_wav, cursor, total_duration, tail_path, sample_rate)
        pieces.append(tail_path)

    concat_list = str(work / "concat_list.txt")
    with open(concat_list, "w") as f:
        for p in pieces:
            f.write(f"file '{p}'\n")

    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-ar", str(sample_rate), "-ac", "1",  # lock to explicit mono
        output_path,
    ])
    return output_path


def _extract_segment(
    source: str, start: float, end: float, out: str, sample_rate: int
) -> None:
    duration = max(0.01, end - start)
    _run([
        "ffmpeg", "-y", "-i", source,
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}",
        "-ar", str(sample_rate), "-ac", "1",
        out,
    ])


def _extract_exact(source: str, duration: float, out: str, sample_rate: int) -> None:
    _run([
        "ffmpeg", "-y", "-i", source,
        "-t", f"{duration:.6f}",
        "-af", f"apad=whole_dur={duration:.6f}",
        "-ar", str(sample_rate), "-ac", "1",
        out,
    ])
