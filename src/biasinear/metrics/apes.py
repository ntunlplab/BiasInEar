"""Average Pairwise Entropy Shift (APES) metric."""

from __future__ import annotations

import math
from itertools import combinations


def apes(entropies: list[float]) -> float:
    """Compute Average Pairwise Entropy Shift.

    APES quantifies the average absolute difference in entropy across
    variable levels (e.g. accents, genders).

    Args:
        entropies: Entropy values for each level of a variable.

    Returns:
        APES value (>= 0). Returns ``float('nan')`` if fewer than 2
        finite entropy values are provided.
    """
    valid = [e for e in entropies if not (math.isnan(e) or math.isinf(e))]
    if len(valid) < 2:
        return float("nan")

    pair_diffs = [abs(a - b) for a, b in combinations(valid, 2)]
    return sum(pair_diffs) / len(pair_diffs)
