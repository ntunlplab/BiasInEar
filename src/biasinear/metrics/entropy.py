"""Question entropy metric for measuring prediction uncertainty."""

from __future__ import annotations

import math
from collections import Counter


def question_entropy(predictions: list[str], num_categories: int = 4) -> float:
    """Compute Shannon entropy of predictions for a single question.

    Entropy is normalized by log(num_categories) so the result is in [0, 1].

    Args:
        predictions: Predicted answers across configurations for a single
            question (e.g. ["A", "A", "B", "C"]).
        num_categories: Number of possible answer categories (default 4
            for A/B/C/D).

    Returns:
        Normalized entropy in [0, 1]. Returns ``float('nan')`` if
        *predictions* is empty.
    """
    total = len(predictions)
    if total == 0:
        return float("nan")

    counts = Counter(p.strip().upper() for p in predictions)
    base = float(num_categories) if num_categories > 1 else 2.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * (math.log(p) / math.log(base))

    return entropy
