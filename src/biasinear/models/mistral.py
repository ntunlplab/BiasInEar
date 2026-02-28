"""Mistral model provider."""

from __future__ import annotations

import base64
import os

from mistralai import Mistral

from biasinear.models.base import BaseModel
from biasinear.models._parser import DEFAULT_SYSTEM_PROMPT, parse_answer


class MistralModel(BaseModel):
    """Mistral audio model via the ``mistralai`` SDK.

    Args:
        model_name: Model identifier (default ``voxtral-small-2507``).
        api_key: Mistral API key.  Falls back to the ``MISTRAL_API_KEY``
            environment variable when *None*.
        system_prompt: Override the default system prompt.
        **kwargs: Forwarded to ``chat.complete``
            (e.g. ``temperature``).
    """

    def __init__(
        self,
        model_name: str = "voxtral-small-2507",
        *,
        api_key: str | None = None,
        **kwargs,
    ) -> None:
        self._system_prompt = kwargs.pop("system_prompt", DEFAULT_SYSTEM_PROMPT)
        super().__init__(model_name, **kwargs)
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._client = Mistral(api_key=api_key)

    def generate(self, audio: bytes) -> dict:
        b64 = base64.b64encode(audio).decode()
        response = self._client.chat.complete(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "audio": f"data:audio/wav;base64,{b64}",
                        },
                    ],
                },
            ],
            **self.kwargs,
        )
        raw = response.choices[0].message.content
        return {"answer": parse_answer(raw), "raw_response": raw}
