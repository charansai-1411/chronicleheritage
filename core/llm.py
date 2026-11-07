"""Thin Gemini wrapper (google-genai SDK). Fails soft: raises LLMUnavailable when no key / API error."""
from __future__ import annotations

from functools import lru_cache

from config import settings


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    from google import genai
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def generate(prompt: str, temperature: float = 0.3) -> tuple[str, str]:
    """Return (text, model_used). Tries each configured model in order."""
    if not settings.gemini_available():
        raise LLMUnavailable("GEMINI_API_KEY is not set in .env")
    from google.genai import types
    last = None
    for model in settings.GEMINI_MODELS:
        try:
            r = _client().models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature))
            if r.text:
                return r.text.strip(), model
        except Exception as e:  # try the next model name
            last = e
    raise LLMUnavailable(f"Gemini request failed: {last}")
