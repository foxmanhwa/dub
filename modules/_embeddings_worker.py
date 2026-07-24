"""
Subprocess worker: extract speaker embeddings using SpeechBrain ECAPA-TDNN.
Usage: python _embeddings_worker.py <input_json> <output_json>

input_json  — list of {"label": str, "wav_path": str}
output_json — list of {"label": str, "embedding": [float, ...]}

Runs in a subprocess so SpeechBrain + torch stay out of the main process.
"""

import json
import sys
import os
from pathlib import Path


def _run(pairs: list[dict]) -> list[dict]:
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=str(Path.home() / ".cache" / "speechbrain" / "ecapa"),
    )
    model.eval()

    results = []
    for pair in pairs:
        label = pair["label"]
        wav_path = pair["wav_path"]
        try:
            signal, fs = torchaudio.load(wav_path)
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            if fs != 16000:
                signal = torchaudio.transforms.Resample(fs, 16000)(signal)
            signal = signal.to(device)
            with torch.no_grad():
                emb = model.encode_batch(signal)
            emb_list = emb.squeeze().cpu().tolist()
            if isinstance(emb_list, float):
                emb_list = [emb_list]
            results.append({"label": label, "embedding": emb_list})
            print(f"[embeddings] {label}: dim={len(emb_list)}", file=sys.stderr)
        except Exception as exc:
            print(f"[embeddings] {label}: failed ({exc})", file=sys.stderr)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: _embeddings_worker.py <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as fh:
        pairs = json.load(fh)

    try:
        results = _run(pairs)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(results, fh)

    print(f"[embeddings] done: {len(results)} embeddings written", file=sys.stderr)
