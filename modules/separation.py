"""
Two-speaker separation via SpeechBrain SepFormer (sepformer-whamr16k, 16 kHz).
Model is downloaded once to ~/.dub/sepformer-whamr16k/ and cached in memory.
"""

import gc
import os
import subprocess
import sys
from pathlib import Path

_model_cache = None
_MODEL_SOURCE = "speechbrain/sepformer-whamr16k"
_MODEL_SAVEDIR = str(Path.home() / ".dub" / "sepformer-whamr16k")
_SAMPLE_RATE = 16_000


def release_model() -> None:
    """Free the SepFormer model from memory after overlap processing."""
    global _model_cache
    _model_cache = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def check_available() -> list[str]:
    issues = []
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        issues.append("speechbrain not installed  (pip install speechbrain)")
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        issues.append("torchaudio not installed  (pip install torchaudio)")
    return issues


def _get_model():
    global _model_cache
    if _model_cache is None:
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation  # <1.0 shim
        _model_cache = SepformerSeparation.from_hparams(
            source=_MODEL_SOURCE,
            savedir=_MODEL_SAVEDIR,
            # CPU by default; set run_opts={"device":"cuda"} for GPU
        )
    return _model_cache


def separate_speakers(
    audio_path: str,
    output_dir: str,
) -> tuple[str, str]:
    """
    Separate a 2-speaker clip into two 16 kHz mono WAV streams.
    Returns (path_speaker_0, path_speaker_1).
    Input is resampled to 16 kHz before separation.
    """
    import torchaudio

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resample input to 16 kHz mono (SepFormer requirement)
    input_16k = str(out / "input_16k.wav")
    _resample(audio_path, input_16k, _SAMPLE_RATE)

    model = _get_model()
    est = model.separate_file(path=input_16k)
    # est shape: [1, num_samples, 2]

    paths = []
    for i in range(est.shape[-1]):
        stream = est[:, :, i].detach().cpu()   # [1, samples]
        p = str(out / f"speaker_{i}.wav")
        torchaudio.save(p, stream, _SAMPLE_RATE)
        paths.append(p)

    # Guard: always return exactly 2 paths
    while len(paths) < 2:
        paths.append(paths[0] if paths else audio_path)

    return paths[0], paths[1]


def _resample(src: str, dst: str, rate: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", str(rate), "-ac", "1", dst],
        check=True, capture_output=True,
    )
