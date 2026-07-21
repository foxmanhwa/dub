"""Write transcript.json and translated.srt output files."""

import json
from pathlib import Path


def write_transcript_json(segments: list[dict], path: str) -> str:
    out = [
        {
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "original_text": seg.get("text", ""),
            "translated_text": seg.get("translated_text", ""),
        }
        for i, seg in enumerate(segments)
    ]
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_srt(segments: list[dict], path: str) -> str:
    lines = []
    idx = 1
    for seg in segments:
        text = seg.get("translated_text", "").strip()
        if not text:
            continue
        lines.append(
            f"{idx}\n"
            f"{_ts(seg['start'])} --> {_ts(seg['end'])}\n"
            f"{text}\n"
        )
        idx += 1
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
