"""LLM-based translation via local Ollama — chunked, context-aware, duration-conscious."""

import os
import re
import httpx


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:latest"  # override with OLLAMA_MODEL env var
CHUNK_SIZE = 10                    # segments per Ollama call
# 300s covers model cold-load (~60-90s) + generation for one chunk.
# Subsequent chunks are fast (~15-30s) once the model is in RAM.
PER_CHUNK_TIMEOUT = 300

# Some models echo the duration hint "(2.50s)" at the start of translated lines.
_DURATION_PREFIX = re.compile(r"^\(\d+(\.\d+)?s\)\s*")


def translate_chunk(
    chunk: list[dict],
    chunk_offset: int,
    target_language: str,
    source_language: str | None = None,
    prev_context: list[str] | None = None,
    model: str | None = None,
) -> list[dict]:
    """
    Translate one chunk of segments.  Returns the same dicts with
    'translated_text' added.

    Segments are numbered locally from [1] within the chunk so small models
    aren't confused by high or non-integer indices.

    prev_context: list of plain-text translated sentences from the tail of the
                  previous chunk, shown for coherence without re-indexing.
    """
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    if prev_context is None:
        prev_context = []

    src_hint = (
        f"from {source_language} " if source_language and source_language != "auto" else ""
    )

    # Previous translations shown as plain prose so the model doesn't confuse
    # them with the current chunk's index numbers.
    context_block = ""
    if prev_context:
        context_block = (
            "Previous speech (already translated, for context only — do NOT output these):\n"
            + "\n".join(f"- {line}" for line in prev_context)
            + "\n\n"
        )

    # Always number locally from 1 so the model can't mix up indices and durations
    numbered = "\n".join(
        f"[{i + 1}] ({seg['end'] - seg['start']:.2f}s) {seg['text']}"
        for i, seg in enumerate(chunk)
    )

    system_prompt = (
        f"You are a professional dubbing translator. "
        f"Translate the speech segments {src_hint}to {target_language}.\n"
        "Rules:\n"
        "1. Keep translations natural and conversational — not literal.\n"
        "2. Each segment shows its duration in seconds. Keep the translation "
        "   concise enough to be spoken within that duration at a natural pace "
        "   (~2–3 words/second depending on language).\n"
        "3. Preserve the speaker's tone, register, and personality.\n"
        "4. Return ONLY the translated lines, one per line, each prefixed with "
        "   the same numbered index in square brackets. No extra commentary.\n"
        "Example output:\n"
        "[1] Translated text here\n"
        "[2] Another translated line\n"
    )

    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_block + numbered},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=PER_CHUNK_TIMEOUT,
    )
    resp.raise_for_status()

    raw = resp.json()["message"]["content"] or ""
    # Parse local [1]-based indices
    translations = _parse_numbered(raw, len(chunk))

    return [{**seg, "translated_text": t} for seg, t in zip(chunk, translations)]


def translate_segments(
    segments: list[dict],
    target_language: str,
    source_language: str | None = None,
) -> list[dict]:
    """
    Convenience wrapper: translates all segments in CHUNK_SIZE batches.
    Use this when you don't need per-chunk progress; otherwise drive the
    loop yourself with translate_chunk() as pipeline.py does.
    """
    if not segments:
        return segments

    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    result: list[dict] = []
    prev_context: list[str] = []

    for chunk_start in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[chunk_start: chunk_start + CHUNK_SIZE]
        translated = translate_chunk(
            chunk=chunk,
            chunk_offset=chunk_start,
            target_language=target_language,
            source_language=source_language,
            prev_context=prev_context,
            model=model,
        )
        result.extend(translated)
        prev_context = _tail_context(translated)

    return result


def back_translate_segments(
    segments: list[dict],
    model: str | None = None,
) -> list[dict]:
    """
    Back-translate each segment's translated_text to English for QA.
    Adds 'back_translated_text' and 'back_trans_similarity' keys.
    Segments without a translation are left with empty back-translation.
    Uses the same chunk size and parser as forward translation.
    """
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    result = [dict(seg) for seg in segments]

    # Index of segments that actually have translated text to process
    to_process = [
        (i, seg) for i, seg in enumerate(result)
        if seg.get("translated_text", "").strip()
    ]

    system_prompt = (
        "Translate the following text to English. "
        "Return each line with the same [N] prefix as in the input. "
        "No explanations, notes, or commentary — only the English translations.\n"
        "Example:\n[1] Hello\n[2] How are you?"
    )

    for chunk_start in range(0, len(to_process), CHUNK_SIZE):
        chunk_pairs = to_process[chunk_start: chunk_start + CHUNK_SIZE]
        numbered = "\n".join(
            f"[{local_i + 1}] {seg['translated_text']}"
            for local_i, (_, seg) in enumerate(chunk_pairs)
        )
        try:
            resp = httpx.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": numbered},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=PER_CHUNK_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"] or ""
            back_texts = _parse_numbered(raw, len(chunk_pairs))
        except Exception:
            back_texts = [""] * len(chunk_pairs)

        for (orig_idx, _), back_text in zip(chunk_pairs, back_texts):
            orig_text = result[orig_idx].get("text", "")
            sim = _word_similarity(orig_text, back_text) if back_text else 0.0
            result[orig_idx]["back_translated_text"] = back_text
            result[orig_idx]["back_trans_similarity"] = round(sim, 3)

    # Fill in empty back-translation for segments not processed
    for seg in result:
        if "back_translated_text" not in seg:
            seg["back_translated_text"] = ""
            seg["back_trans_similarity"] = None

    return result


def _word_similarity(a: str, b: str) -> float:
    """Word-level similarity ratio via SequenceMatcher (stdlib)."""
    import difflib
    words_a = a.lower().split()
    words_b = b.lower().split()
    if not words_a and not words_b:
        return 1.0
    return difflib.SequenceMatcher(None, words_a, words_b).ratio()


def _tail_context(translated: list[dict]) -> list[str]:
    """Return the last 2 translated sentences as plain strings for context carry-over."""
    return [
        seg["translated_text"]
        for seg in translated[-2:]
        if seg.get("translated_text")
    ]


def _parse_numbered(text: str, count: int) -> list[str]:
    """
    Extract translations from [N] prefixed lines.

    Parses positionally (in order of appearance) rather than by index value,
    so small models that mis-number their output are handled correctly.
    Falls back to any non-empty line if too few indexed lines are found.
    """
    indexed: list[str] = []
    fallback: list[str] = []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            try:
                int(line[1:bracket_end])          # must be a number
                content = line[bracket_end + 1:].strip()
                content = _DURATION_PREFIX.sub("", content)
                if content:
                    indexed.append(content)
            except ValueError:
                pass
        else:
            fallback.append(line)

    candidates = indexed if len(indexed) >= count else (indexed or fallback)
    # Pad with empty strings if the model produced fewer lines than expected
    return (candidates + [""] * count)[:count]
