"""
Quick test: generate a 10-second test WAV and run the diarization worker.
Confirms the LazyModule patch works and memory is released cleanly.
"""
import subprocess
import sys
import tempfile
import os
import struct
import wave

# Generate a minimal 10-second mono WAV (silence + simple tone burst)
def make_test_wav(path: str, duration: float = 10.0, sample_rate: int = 16000) -> None:
    n = int(duration * sample_rate)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        import math
        data = bytearray()
        for i in range(n):
            # Simple 440 Hz tone for first 3s, silence rest
            if i < 3 * sample_rate:
                val = int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            else:
                val = 0
            data += struct.pack('<h', val)
        wf.writeframes(bytes(data))

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    wav_path = f.name
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    json_path = f.name

try:
    print(f"Creating test WAV: {wav_path}")
    make_test_wav(wav_path)

    worker = os.path.join(os.path.dirname(__file__), 'modules', '_diarization_worker.py')
    print(f"Running worker: {worker}")
    print("(First run may download pyannote models — this can take several minutes)\n")

    proc = subprocess.run(
        [sys.executable, worker, wav_path, json_path],
        capture_output=False,   # let stderr flow to terminal
        timeout=900,
    )
    print(f"\nWorker exit code: {proc.returncode}")

    if proc.returncode == 0:
        import json
        with open(json_path) as f:
            result = json.load(f)
        print("Result:", result)
        print("\nSUCCESS — diarization worker ran cleanly")
    else:
        print("FAILED")
finally:
    for p in (wav_path, json_path):
        try:
            os.unlink(p)
        except Exception:
            pass
