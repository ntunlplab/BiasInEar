"""Accuracy metric for multiple-choice question answering."""

from __future__ import annotations


def accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute accuracy between predictions and references.

    Args:
        predictions: Model predicted answers (e.g. ["A", "B", "C", ...]).
        references: Ground-truth answers (e.g. ["B", "B", "B", ...]).

    Returns:
        Accuracy as a float in [0, 1].

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != references ({len(references)})"
        )
    if len(predictions) == 0:
        raise ValueError("Inputs must be non-empty")

    correct = sum(
        p.strip().upper() == r.strip().upper()
        for p, r in zip(predictions, references)
    )
    return correct / len(predictions)
