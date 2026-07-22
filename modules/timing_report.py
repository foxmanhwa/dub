"""
Timing QA report — compares raw TTS durations to original segment slots
and flags segments where the stretch/compress ratio is outside the safe range.
"""

import difflib

from .audio_assembly import get_duration


RATIO_WARN_LOW = 0.7   # below this → TTS is much shorter, will be slowed down
RATIO_WARN_HIGH = 1.4  # above this → TTS is much longer, will be sped up


def build_timing_report(segments: list[dict]) -> list[dict]:
    """
    Return one row dict per segment with:
      seg, start, end, orig_dur, tts_dur, ratio, status, original_text, translated_text
    Must be called while TTS files are still on disk (before any cleanup).
    """
    rows = []
    for i, seg in enumerate(segments):
        orig_dur = seg["end"] - seg["start"]
        tts_path = seg.get("tts_path")

        tts_dur: float | None = None
        if tts_path:
            try:
                tts_dur = get_duration(tts_path)
            except Exception:
                pass

        ratio: float | None = None
        if tts_dur is not None and orig_dur > 0:
            ratio = tts_dur / orig_dur

        if tts_path is None:
            status = "no audio"
        elif tts_dur is None:
            status = "probe failed"
        elif orig_dur <= 0:
            status = "zero slot"
        elif ratio < RATIO_WARN_LOW:
            status = f"⚠ stretched {ratio:.2f}×"
        elif ratio > RATIO_WARN_HIGH:
            status = f"⚠ compressed {ratio:.2f}×"
        else:
            status = "ok"

        rows.append({
            "seg": i,
            "start": seg["start"],
            "end": seg["end"],
            "orig_dur": orig_dur,
            "tts_dur": tts_dur,
            "ratio": ratio,
            "status": status,
            "original_text": seg.get("text", ""),
            "translated_text": seg.get("translated_text", ""),
        })
    return rows


def timing_report_to_df(rows: list[dict]):
    """Return a pandas DataFrame suitable for gr.DataFrame display."""
    import pandas as pd

    display = []
    for r in rows:
        display.append({
            "Seg": r["seg"],
            "Start": f"{r['start']:.2f}",
            "End": f"{r['end']:.2f}",
            "Orig (s)": f"{r['orig_dur']:.2f}",
            "TTS raw (s)": f"{r['tts_dur']:.2f}" if r["tts_dur"] is not None else "—",
            "Ratio": f"{r['ratio']:.2f}" if r["ratio"] is not None else "—",
            "Status": r["status"],
        })
    return pd.DataFrame(display)


def backtrans_to_df(segments: list[dict]):
    """Return a pandas DataFrame of back-translation QA results."""
    import pandas as pd

    rows = []
    for i, seg in enumerate(segments):
        sim = seg.get("back_trans_similarity")
        flag = "⚠" if (sim is not None and sim < 0.5) else ""
        rows.append({
            "Seg": i,
            "Original": seg.get("text", "")[:120],
            "Translation": seg.get("translated_text", "")[:120],
            "Back-translation": seg.get("back_translated_text", "")[:120],
            "Sim": f"{sim:.2f}" if sim is not None else "—",
            "⚠": flag,
        })
    return pd.DataFrame(rows)
