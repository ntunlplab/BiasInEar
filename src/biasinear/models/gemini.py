"""Google Gemini model provider."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from biasinear.models.base import BaseModel
from biasinear.models._parser import DEFAULT_SYSTEM_PROMPT, parse_answer


class GeminiModel(BaseModel):
    """Gemini model via the ``google-genai`` SDK.

    Args:
        model_name: Gemini model identifier (default ``gemini-2.5-flash``).
        api_key: Gemini API key.  Falls back to the ``GEMINI_API_KEY``
            environment variable when *None*.
        system_prompt: Override the default system prompt.
        **kwargs: Forwarded to ``generate_content`` (e.g. ``temperature``).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        self._system_prompt = kwargs.pop("system_prompt", DEFAULT_SYSTEM_PROMPT)
        super().__init__(model_name, **kwargs)
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    def generate(self, audio: bytes) -> dict:
        audio_part = types.Part.from_bytes(data=audio, mime_type="audio/wav")
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[audio_part, self._system_prompt],
            config=types.GenerateContentConfig(**self.kwargs)
            if self.kwargs
            else None,
        )
        raw = response.text
        return {"answer": parse_answer(raw), "raw_response": raw}
