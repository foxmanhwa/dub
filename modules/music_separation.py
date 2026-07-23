"""
Vocal / background-music separation via Demucs (htdemucs, --two-stems=vocals).
No model weights are loaded until separate_music() is called for the first time;
first call downloads ~200 MB of weights to the Torch hub cache.
"""

import subprocess
import sys
from pathlib import Path

_MODEL = "htdemucs"


def check_available() -> list[str]:
    issues = []
    try:
        import demucs  # noqa: F401
    except ImportError:
        issues.append("demucs not installed  (pip install demucs)")
    return issues


def separate_music(
    audio_path: str,
    output_dir: str,
    model: str = _MODEL,
) -> tuple[str, str]:
    """
    Split audio into vocals and background music using Demucs two-stems mode.
    Returns (vocals_path, music_path) — both 44100 Hz stereo WAVs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem_name = Path(audio_path).stem  # e.g. "audio" for "audio.wav"

    try:
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "-o", str(out),
                "-n", model,
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"Demucs failed:\n{stderr}") from exc

    model_dir = out / model / stem_name
    vocals_path = str(model_dir / "vocals.wav")
    music_path  = str(model_dir / "no_vocals.wav")

    if not Path(vocals_path).exists():
        raise FileNotFoundError(
            f"Demucs output not found at {vocals_path}. "
            f"Contents of expected dir: {list(model_dir.iterdir()) if model_dir.exists() else 'dir missing'}"
        )

    return vocals_path, music_path


def mix_vocals_with_music(
    dubbed_vocals_path: str,
    music_path: str,
    output_path: str,
    total_duration: float,
) -> None:
    """
    Mix a mono dubbed-vocals WAV with a stereo background-music WAV.
    Output is stereo 44100 Hz with no level normalization so the relative
    loudness matches the original recording.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", dubbed_vocals_path,
            "-i", music_path,
            "-filter_complex",
            (
                # Upmix the mono dubbed track to stereo (same signal in L + R),
                # then mix additively with the stereo music stem.
                "[0]aresample=44100,pan=stereo|FL=c0|FR=c0[v];"
                "[1]aresample=44100[m];"
                "[v][m]amix=inputs=2:normalize=0:dropout_transition=0"
            ),
            "-t", str(total_duration),
            "-ar", "44100",
            "-ac", "2",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
