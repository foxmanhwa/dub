"""
LLM-based translation — Groq (default), Gemini, or Ollama fallback.

Backend selection (in priority order):
  TRANSLATION_BACKEND=groq     force Groq (requires GROQ_API_KEY)
  TRANSLATION_BACKEND=gemini   force Gemini (requires GEMINI_API_KEY)
  TRANSLATION_BACKEND=ollama   force Ollama (requires ollama running locally)
  [unset]                      auto: Groq if GROQ_API_KEY set, else Gemini if
                                     GEMINI_API_KEY set, else Ollama
"""

import os
import re
import httpx


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 10
PER_CHUNK_TIMEOUT = 300   # seconds; used for Ollama only (Gemini/Groq have their own timeouts)
TAIL_CONTEXT_SIZE = 5     # translated sentences carried between chunks for coherence

_DURATION_PREFIX = re.compile(r"^\(\d+(\.\d+)?s\)\s*")


def active_backend() -> str:
    """
    Return 'groq', 'gemini', or 'ollama' based on env vars.
    Priority: TRANSLATION_BACKEND (explicit) > GROQ_API_KEY > GEMINI_API_KEY > ollama
    """
    explicit = os.environ.get("TRANSLATION_BACKEND", "").lower().strip()
    if explicit in ("groq", "gemini", "ollama"):
        return explicit
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    return "ollama"


# ── Low-level backend calls ───────────────────────────────────────────────────

def _groq_call(system_prompt: str, user_content: str) -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env or set TRANSLATION_BACKEND explicitly."
        )
    model_name = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def _gemini_call(system_prompt: str, user_content: str) -> str:
    from google import genai
    from google.genai import types as genai_types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or set TRANSLATION_BACKEND=ollama."
        )
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model_name,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
        contents=user_content,
    )
    return resp.text or ""


def _ollama_call(system_prompt: str, user_content: str, model: str) -> str:
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=PER_CHUNK_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"] or ""


def _llm_call(system_prompt: str, user_content: str, ollama_model: str) -> str:
    """Dispatch to the active backend."""
    backend = active_backend()
    if backend == "groq":
        return _groq_call(system_prompt, user_content)
    if backend == "gemini":
        return _gemini_call(system_prompt, user_content)
    return _ollama_call(system_prompt, user_content, ollama_model)


# ── Translation prompt builder ────────────────────────────────────────────────

def _build_translation_prompts(
    chunk: list[dict],
    target_language: str,
    source_language: str | None,
    prev_context: list[str],
    content_context: str | None = None,
) -> tuple[str, str]:
    src_hint = (
        f"from {source_language} " if source_language and source_language != "auto" else ""
    )

    # Detect whether any segment carries a speaker label from diarization
    has_speakers = any(seg.get("speaker") for seg in chunk)

    # Build each segment line: [N] (Xs) [SPEAKER_XX] text
    def _fmt(i: int, seg: dict) -> str:
        dur = f"({seg['end'] - seg['start']:.2f}s)"
        spk = f" [{seg['speaker']}]" if seg.get("speaker") else ""
        return f"[{i + 1}] {dur}{spk} {seg['text']}"

    numbered = "\n".join(_fmt(i, seg) for i, seg in enumerate(chunk))

    context_block = ""
    if prev_context:
        context_block = (
            "Previous speech (already translated, for context only — "
            "do NOT output these again):\n"
            + "\n".join(f"- {line}" for line in prev_context)
            + "\n\n"
        )

    system_prompt = (
        f"You are a professional dubbing translator. "
        f"Translate the speech segments {src_hint}to {target_language}.\n"
    )
    if content_context:
        system_prompt += f"Content context: {content_context}\n"
    system_prompt += (
        "Rules:\n"
        "1. Keep translations natural and conversational — not literal.\n"
        "2. Each segment shows its duration in seconds. Keep the translation "
        "concise enough to be spoken within that duration at a natural pace "
        "(~2–3 words/second depending on language).\n"
        "3. Preserve the speaker's tone, register, and personality.\n"
    )
    if has_speakers:
        system_prompt += (
            "4. Each segment is labeled with a speaker ID in [SPEAKER_XX] format. "
            "Keep each speaker's voice, register, and style consistent across all "
            "their segments throughout the clip.\n"
            "5. Return ONLY the translated lines, one per line, each prefixed with "
            "the same numbered index in square brackets. "
            "Do NOT include speaker labels in your output. No extra commentary.\n"
        )
    else:
        system_prompt += (
            "4. Return ONLY the translated lines, one per line, each prefixed with "
            "the same numbered index in square brackets. No extra commentary.\n"
        )
    system_prompt += "Example output:\n[1] Translated text here\n[2] Another translated line\n"

    return system_prompt, context_block + numbered


# ── Public API ────────────────────────────────────────────────────────────────

def translate_chunk(
    chunk: list[dict],
    chunk_offset: int,
    target_language: str,
    source_language: str | None = None,
    prev_context: list[str] | None = None,
    model: str | None = None,
    content_context: str | None = None,
) -> list[dict]:
    """
    Translate one chunk of segments. Returns the same dicts with 'translated_text' added.

    `model` is the Ollama model name; ignored when backend is Gemini.
    `content_context` is an optional free-text description of the video content.
    Segments are numbered locally from [1] within the chunk.
    """
    if prev_context is None:
        prev_context = []
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    system_prompt, user_content = _build_translation_prompts(
        chunk, target_language, source_language, prev_context, content_context
    )
    raw = _llm_call(system_prompt, user_content, model)
    translations = _parse_numbered(raw, len(chunk))
    return [{**seg, "translated_text": t} for seg, t in zip(chunk, translations)]


def translate_segments(
    segments: list[dict],
    target_language: str,
    source_language: str | None = None,
    content_context: str | None = None,
) -> list[dict]:
    """Convenience wrapper: translates all segments in CHUNK_SIZE batches."""
    if not segments:
        return segments
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
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
            content_context=content_context,
        )
        result.extend(translated)
        prev_context = _tail_context(translated)
    return result


# ISO 639-1 code → display name used in back-translation prompts
_LANG_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "zh": "Chinese (Mandarin)",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
}


def back_translate_segments(
    segments: list[dict],
    model: str | None = None,
    source_language: str | None = None,
) -> list[dict]:
    """
    Back-translate each segment's translated_text back to the source language for QA.

    source_language: ISO 639-1 code (e.g. "en", "es") or None/"auto".
    When None/"auto", defaults to English so the result is still useful for
    English-source videos; a note is added to each segment.

    The back-translated text is compared against seg["text"] (the original
    source transcript), so the comparison is always apples-to-apples only when
    the back-translation target matches the source language.
    """
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    known_source = source_language and source_language not in ("auto", "")
    target_lang_name = _LANG_NAMES.get(source_language or "", "English") if known_source else "English"
    lang_is_known = bool(known_source)

    result = [dict(seg) for seg in segments]
    to_process = [
        (i, seg) for i, seg in enumerate(result)
        if seg.get("translated_text", "").strip()
    ]

    system_prompt = (
        f"Translate the following text to {target_lang_name}. "
        "Return each line with the same [N] prefix as in the input. "
        f"No explanations, notes, or commentary — only the {target_lang_name} translations.\n"
        "Example:\n[1] Hello\n[2] How are you?"
    )

    for chunk_start in range(0, len(to_process), CHUNK_SIZE):
        chunk_pairs = to_process[chunk_start: chunk_start + CHUNK_SIZE]
        numbered = "\n".join(
            f"[{local_i + 1}] {seg['translated_text']}"
            for local_i, (_, seg) in enumerate(chunk_pairs)
        )
        try:
            raw = _llm_call(system_prompt, numbered, model)
            back_texts = _parse_numbered(raw, len(chunk_pairs))
        except Exception:
            back_texts = [""] * len(chunk_pairs)

        for (orig_idx, _), back_text in zip(chunk_pairs, back_texts):
            orig_text = result[orig_idx].get("text", "")
            sim = _word_similarity(orig_text, back_text) if back_text else 0.0
            result[orig_idx]["back_translated_text"] = back_text
            result[orig_idx]["back_trans_similarity"] = round(sim, 3) if lang_is_known else None

    for seg in result:
        if "back_translated_text" not in seg:
            seg["back_translated_text"] = ""
            seg["back_trans_similarity"] = None

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tail_context(translated: list[dict]) -> list[str]:
    return [
        seg["translated_text"]
        for seg in translated[-TAIL_CONTEXT_SIZE:]
        if seg.get("translated_text")
    ]


def _word_similarity(a: str, b: str) -> float:
    import difflib
    words_a = a.lower().split()
    words_b = b.lower().split()
    if not words_a and not words_b:
        return 1.0
    return difflib.SequenceMatcher(None, words_a, words_b).ratio()


def _parse_numbered(text: str, count: int) -> list[str]:
    """
    Extract translations from [N] prefixed lines, positionally (not by index value).
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
                int(line[1:bracket_end])
                content = line[bracket_end + 1:].strip()
                content = _DURATION_PREFIX.sub("", content)
                if content:
                    indexed.append(content)
            except ValueError:
                pass
        else:
            fallback.append(line)
    candidates = indexed if len(indexed) >= count else (indexed or fallback)
    return (candidates + [""] * count)[:count]
