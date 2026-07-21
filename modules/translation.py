"""LLM-based translation — context-aware and duration-conscious."""

import os
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def translate_segments(
    segments: list[dict],
    target_language: str,
    source_language: str | None = None,
) -> list[dict]:
    """
    Translate ASR segments to target_language.

    Input segments: {"text": str, "start": float, "end": float}
    Returns same list with "translated_text" added to each item.

    All segments are sent in a single LLM call for cohesive context.
    """
    if not segments:
        return segments

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    client = _get_client()

    src_hint = f"from {source_language} " if source_language and source_language != "auto" else ""

    numbered = "\n".join(
        f"[{i + 1}] ({seg['end'] - seg['start']:.2f}s) {seg['text']}"
        for i, seg in enumerate(segments)
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
        "4. Return ONLY the translated lines, one per line, prefixed with the "
        "   same [N] index. No extra commentary.\n"
        "Example:\n"
        "[1] Translated text here\n"
        "[2] Another translated line\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": numbered},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content or ""
    translations = _parse_numbered(raw, len(segments))

    return [{**seg, "translated_text": t} for seg, t in zip(segments, translations)]


def _parse_numbered(text: str, count: int) -> list[str]:
    lines: dict[int, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            try:
                idx = int(line[1:bracket_end])
                lines[idx] = line[bracket_end + 1:].strip()
            except ValueError:
                pass

    return [lines.get(i, "") for i in range(1, count + 1)]
