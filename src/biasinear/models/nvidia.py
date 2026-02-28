"""NVIDIA Build model provider (OpenAI-compatible API)."""

from __future__ import annotations

import base64
import os

from openai import OpenAI

from biasinear.models.base import BaseModel
from biasinear.models._parser import DEFAULT_SYSTEM_PROMPT, parse_answer

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

_GEMMA_MODELS = {"google/gemma-3n-e4b-it", "google/gemma-3n-e2b-it"}
_PHI_MODELS = {"microsoft/phi-4-multimodal-instruct"}


class NvidiaModel(BaseModel):
    """NVIDIA Build model via OpenAI-compatible API.

    Supports Gemma-3n and Phi-4 multimodal models, each with its own
    audio encoding format.

    Args:
        model_name: Model identifier (default ``google/gemma-3n-e4b-it``).
        api_key: NVIDIA API key.  Falls back to the ``NVIDIA_API_KEY``
            environment variable when *None*.
        system_prompt: Override the default system prompt.
        **kwargs: Forwarded to ``chat.completions.create``
            (e.g. ``temperature``).
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3n-e4b-it",
        *,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        self._system_prompt = kwargs.pop("system_prompt", DEFAULT_SYSTEM_PROMPT)
        super().__init__(model_name, **kwargs)
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=_NVIDIA_BASE_URL)

    # ------------------------------------------------------------------
    # Message formatting per model family
    # ------------------------------------------------------------------

    def _gemma_messages(self, b64: str) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": f"data:audio/wav;base64,{b64}",
                    },
                ],
            },
        ]

    def _phi_messages(self, b64: str) -> list[dict]:
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f'<|audio_1|> <audio src="data:audio/wav;base64,{b64}" />'
                ),
            },
        ]

    # ------------------------------------------------------------------

    def generate(self, audio: bytes) -> dict:
        b64 = base64.b64encode(audio).decode()

        if self.model_name in _GEMMA_MODELS:
            messages = self._gemma_messages(b64)
        elif self.model_name in _PHI_MODELS:
            messages = self._phi_messages(b64)
        else:
            # Default to Gemma-style for unknown NVIDIA models
            messages = self._gemma_messages(b64)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.kwargs,
        )
        raw = response.choices[0].message.content
        return {"answer": parse_answer(raw), "raw_response": raw}
