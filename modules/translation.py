"""LLM-based translation module — context-aware, duration-conscious."""

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
    Translate a list of ASR segments to target_language.

    Each input segment: {"text": str, "start": float, "end": float}
    Returns the same list with "translated_text" added to each segment.

    Sends all segments in a single LLM call for context awareness.
    """
    if not segments:
        return segments

    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    client = _get_client()

    src_hint = f"from {source_language} " if source_language and source_language != "auto" else ""

    numbered = "\n".join(
        f"[{i+1}] ({seg['end'] - seg['start']:.2f}s) {seg['text']}"
        for i, seg in enumerate(segments)
    )

    system_prompt = (
        f"You are a professional subtitler/dubbing translator. "
        f"Translate the following speech segments {src_hint}to {target_language}. "
        "Rules:\n"
        "1. Keep translations natural and conversational, not literal.\n"
        "2. Each segment shows its duration in seconds. Keep translated text "
        "   concise enough to be spoken within that duration at a natural pace.\n"
        "   A rough guide: English ~2.5 words/second, adjust for target language.\n"
        "3. Preserve tone, register, and speaker personality.\n"
        "4. Return ONLY the translated lines, one per line, prefixed with the same "
        "   [N] index. No extra commentary.\n"
        "Example output format:\n"
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

    result = []
    for seg, translated in zip(segments, translations):
        result.append({**seg, "translated_text": translated})
    return result


def _parse_numbered(text: str, count: int) -> list[str]:
    """Extract [N] prefixed lines into an ordered list."""
    lines: dict[int, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]")
            try:
                idx = int(line[1:bracket_end])
                content = line[bracket_end + 1:].strip()
                lines[idx] = content
            except ValueError:
                pass

    # Fall back to positional order if parsing is incomplete
    result = []
    for i in range(1, count + 1):
        result.append(lines.get(i, ""))
    return result
