"""Abstract base class for audio language model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract interface for audio language models.

    Subclass this to implement a specific model provider (e.g. OpenAI,
    Google, Anthropic).

    Args:
        model_name: Model identifier string.
        **kwargs: Provider-specific parameters such as ``temperature``,
            ``system_prompt``, ``max_tokens``, etc.
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, audio: bytes) -> dict:
        """Run inference on a single concatenated audio input.

        Args:
            audio: Concatenated audio bytes (question + options) in WAV
                format.

        Returns:
            Dict with at least::

                {
                    "answer": "A",           # Parsed answer (A/B/C/D)
                    "raw_response": "..."     # Full model response
                }
        """
        ...
