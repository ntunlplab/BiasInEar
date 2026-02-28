"""Shared answer parser and default prompt for model providers."""

from __future__ import annotations

import re

DEFAULT_SYSTEM_PROMPT = (
    "You will hear a multiple-choice question followed by four options "
    "labeled A, B, C, and D. Listen carefully and respond with ONLY the "
    "letter of the correct answer."
)

_STANDALONE_RE = re.compile(r"\b([A-D])\b")


def parse_answer(text: str) -> str | None:
    """Extract a single A/B/C/D answer from model output.

    Strategy:
        1. Look for a standalone letter A-D (word boundary on both sides).
        2. Fallback: return the first A-D character found anywhere.
        3. Return ``None`` if nothing matches.
    """
    m = _STANDALONE_RE.search(text)
    if m:
        return m.group(1)
    # Fallback: first occurrence of A-D anywhere
    for ch in text:
        if ch in "ABCD":
            return ch
    return None
