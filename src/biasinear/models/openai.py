"""OpenAI model provider."""

from __future__ import annotations

import base64
import os

from openai import OpenAI

from biasinear.models.base import BaseModel
from biasinear.models._parser import DEFAULT_SYSTEM_PROMPT, parse_answer


class OpenAIModel(BaseModel):
    """OpenAI audio model via the ``openai`` SDK.

    Args:
        model_name: Model identifier (default ``gpt-4o-audio-preview``).
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.
        system_prompt: Override the default system prompt.
        **kwargs: Forwarded to ``chat.completions.create``
            (e.g. ``temperature``).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-audio-preview",
        *,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        self._system_prompt = kwargs.pop("system_prompt", DEFAULT_SYSTEM_PROMPT)
        super().__init__(model_name, **kwargs)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

    def generate(self, audio: bytes) -> dict:
        b64 = base64.b64encode(audio).decode()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        },
                    ],
                },
            ],
            **self.kwargs,
        )
        raw = response.choices[0].message.content
        return {"answer": parse_answer(raw), "raw_response": raw}
