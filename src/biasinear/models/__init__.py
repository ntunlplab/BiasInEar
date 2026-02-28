"""Model interfaces for BiasInEar."""

from biasinear.models.base import BaseModel

__all__ = [
    "BaseModel",
    "GeminiModel",
    "OpenAIModel",
    "NvidiaModel",
    "MistralModel",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "GeminiModel": ("biasinear.models.gemini", "GeminiModel"),
    "OpenAIModel": ("biasinear.models.openai", "OpenAIModel"),
    "NvidiaModel": ("biasinear.models.nvidia", "NvidiaModel"),
    "MistralModel": ("biasinear.models.mistral", "MistralModel"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
