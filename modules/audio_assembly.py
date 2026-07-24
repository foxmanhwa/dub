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

# Maximum stretch we're willing to apply before preferring trim/pad over distortion.
# Iterative retranslation tries to bring segments inside this range first; whatever
# remains after retries is clamped here rather than stretched beyond recognition.
SAFE_RATIO_MIN = 0.85
SAFE_RATIO_MAX = 1.15


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


def normalize_reference_clip(input_path: str, output_path: str) -> str:
    """
    Normalize loudness to −16 LUFS (EBU R128 single-pass) and true peak to −1.5 dBFS.
    Vocals stems from Demucs can have inconsistent gain across segments; levelling before
    cloning gives Fish Audio a consistent signal and improves clone quality.
    """
    _run([
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "44100", "-ac", "1",
        output_path,
    ])
    return output_path


def extract_reference_clip(
    audio_path: str,
    output_path: str,
    start: float = 0.0,
    duration: float = 12.0,
) -> str:
    """Extract a short clip from audio_path for voice cloning, normalized to -16 LUFS."""
    raw = output_path + ".raw.wav"
    _run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-ar", "44100", "-ac", "1",
        raw,
    ])
    try:
        normalize_reference_clip(raw, output_path)
        os.remove(raw)
    except Exception:
        os.replace(raw, output_path)
    return output_path


def extract_speaker_reference_clips(
    audio_path: str,
    diarization_segments: list[dict],
    work_dir: str,
    target_duration: float = 12.0,
    overlap_regions: list[dict] | None = None,
) -> dict[str, tuple[str, float, int, float]]:
    """
    Build a per-speaker voice reference WAV from diarization segments.

    Picks clean, non-overlapping segments per speaker and concatenates them
    until reaching target_duration (or as close as possible).  Fragments as
    short as 0.1 s are accepted — short lines like "yes" / "ok" still carry
    usable timbre and add up across a back-and-forth dialogue.

    Returns {speaker_id: (wav_path, clipped_duration, n_fragments, raw_total)}.
      clipped_duration — seconds of audio in the reference WAV (≤ target_duration)
      n_fragments      — number of source segments included
      raw_total        — sum of all included segment durations before the cap
    Speakers with very little audio are still included; callers check duration.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Build a set of overlap windows to exclude (mixed audio = bad reference)
    overlap_spans: list[tuple[float, float]] = []
    for r in (overlap_regions or []):
        overlap_spans.append((r["start"], r["end"]))

    def _in_overlap(start: float, end: float) -> bool:
        for os, oe in overlap_spans:
            if start < oe and end > os:
                return True
        return False

    # Group diarization segments by speaker
    # 0.1 s floor rejects genuine zero-duration artefacts while keeping short
    # utterances ("yes", "ok", "hmm") that still carry usable voice timbre.
    by_speaker: dict[str, list[dict]] = {}
    for seg in diarization_segments:
        spk = seg.get("speaker", "SPEAKER_00")
        dur = seg["end"] - seg["start"]
        if dur < 0.1 or _in_overlap(seg["start"], seg["end"]):
            continue
        by_speaker.setdefault(spk, []).append(seg)

    result: dict[str, tuple[str, float, int, float]] = {}

    for speaker, segs in by_speaker.items():
        # Prefer segments with representative energy (not whispered if speaker also speaks normally)
        try:
            from .energy import select_representative_segments
            selected = select_representative_segments(segs, audio_path, target_duration)
            accumulated = sum(s["end"] - s["start"] for s in selected)
        except Exception:
            # Fallback: longest segments first
            segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
            selected, accumulated = [], 0.0
            for seg in segs_sorted:
                selected.append(seg)
                accumulated += seg["end"] - seg["start"]
                if accumulated >= target_duration:
                    break

        if not selected:
            continue

        # Extract each chunk to a temp WAV
        chunk_paths: list[str] = []
        for ci, seg in enumerate(selected):
            chunk_path = str(work / f"{speaker}_chunk_{ci:02d}.wav")
            seg_dur = seg["end"] - seg["start"]
            _run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", f"{seg['start']:.3f}", "-t", f"{seg_dur:.3f}",
                "-ar", "44100", "-ac", "1",
                chunk_path,
            ])
            chunk_paths.append(chunk_path)

        ref_path = str(work / f"{speaker}_reference.wav")
        if len(chunk_paths) == 1:
            os.replace(chunk_paths[0], ref_path)
        else:
            concat_list = str(work / f"{speaker}_concat_list.txt")
            with open(concat_list, "w") as f:
                for p in chunk_paths:
                    f.write(f"file '{p}'\n")
            _run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-ar", "44100", "-ac", "1",
                ref_path,
            ])
            for p in chunk_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Normalize loudness in-place — consistent level across concatenated chunks
        norm_path = str(work / f"{speaker}_reference_norm.wav")
        try:
            normalize_reference_clip(ref_path, norm_path)
            os.replace(norm_path, ref_path)
        except Exception:
            pass  # keep un-normalized if loudnorm fails (e.g. near-silent clip)

        actual_dur = min(accumulated, target_duration)
        result[speaker] = (ref_path, actual_dur, len(selected), accumulated)

    return result


def mux_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    stereo: bool = False,
) -> str:
    """Replace the audio track of video_path with audio_path."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
    ]
    if not stereo:
        cmd += ["-ac", "1"]   # lock to mono for voice-only output
    cmd += [
        "-map", "0:v:0", "-map", "1:a:0",
        "-movflags", "+faststart",
        "-shortest",
        output_path,
    ]
    _run(cmd)
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
    Fit a TTS clip into target_duration using modest atempo stretch, then trim/pad.

    Atempo is clamped to [SAFE_RATIO_MIN, SAFE_RATIO_MAX] (0.85–1.15×) so the
    audio never sounds distorted.  When the raw ratio exceeds that range the clip
    is stretched by at most the allowed amount, and the remaining gap is filled
    with silence (too-short TTS) or trimmed (too-long TTS).  Iterative
    retranslation in the TTS layer is the primary mechanism for bringing
    segments into range before this function is called.
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

    ratio = max(SAFE_RATIO_MIN, min(SAFE_RATIO_MAX, tts_dur / target_duration))
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
