"""
Merge fine-grained ASR segments into phrase/sentence-level groups before
translation and TTS.  Single-word segments and rapid-fire fragments are
folded into their neighbours so that:
  - translation has enough context per call to produce natural output
  - TTS clips are long enough to synthesize cleanly (> ~0.5 s)
  - the final dub sounds like flowing speech, not word-by-word bursts
"""

import re

# A pause wider than this between two segments is a natural phrase boundary.
MIN_PAUSE_GAP: float = 0.4       # seconds

# Hard caps so we never produce a single enormously long merged segment.
MAX_GROUP_DURATION: float = 8.0  # seconds
MAX_GROUP_WORDS: int = 30

_SENTENCE_END = re.compile(r"[.!?…。！？]$")


def merge_segments(segments: list[dict]) -> list[dict]:
    """
    Merge adjacent ASR segments into phrase-level groups.

    A group boundary is inserted when ANY of these is true:
      1. The pause gap to the next segment exceeds MIN_PAUSE_GAP.
      2. The current segment's text ends with sentence-ending punctuation.
      3. Adding the next segment would push the group past MAX_GROUP_DURATION
         or MAX_GROUP_WORDS.

    Returns a new list with the same dict schema (text / start / end), plus a
    'source_segments' key listing the originals that were folded in.
    """
    if not segments:
        return segments

    groups: list[list[dict]] = []
    current: list[dict] = [segments[0]]

    for prev, nxt in zip(segments, segments[1:]):
        gap = nxt["start"] - prev["end"]
        would_be_dur = nxt["end"] - current[0]["start"]
        would_be_words = len(" ".join(s["text"] for s in current).split()) + len(nxt["text"].split())

        split_here = (
            gap > MIN_PAUSE_GAP
            or _SENTENCE_END.search(prev["text"].strip())
            or would_be_dur > MAX_GROUP_DURATION
            or would_be_words > MAX_GROUP_WORDS
        )

        if split_here:
            groups.append(current)
            current = [nxt]
        else:
            current.append(nxt)

    groups.append(current)
    return [_collapse(g) for g in groups]


def _collapse(group: list[dict]) -> dict:
    return {
        "text": " ".join(s["text"] for s in group),
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "source_segments": group,
    }
