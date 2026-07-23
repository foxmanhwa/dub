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


def write_timing_csv(rows: list[dict], path: str) -> str:
    import csv
    fieldnames = ["seg", "start", "end", "orig_dur", "tts_dur", "ratio", "status",
                  "original_text", "translated_text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row = dict(r)
            for k in ("orig_dur", "tts_dur", "ratio"):
                if row[k] is not None:
                    row[k] = f"{row[k]:.3f}"
                else:
                    row[k] = ""
            writer.writerow(row)
    return path


def write_backtrans_csv(segments: list[dict], path: str) -> str:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seg", "start", "end", "original_text", "translated_text",
                         "back_translated_text", "similarity", "flag"])
        for i, seg in enumerate(segments):
            sim = seg.get("back_trans_similarity")
            flag = "warning" if (sim is not None and sim < 0.5) else ""
            writer.writerow([
                i,
                f"{seg['start']:.3f}",
                f"{seg['end']:.3f}",
                seg.get("text", ""),
                seg.get("translated_text", ""),
                seg.get("back_translated_text", ""),
                f"{sim:.3f}" if sim is not None else "",
                flag,
            ])
    return path


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
