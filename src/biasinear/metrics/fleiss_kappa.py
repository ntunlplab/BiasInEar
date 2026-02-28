"""Fleiss' kappa for inter-rater agreement."""

from __future__ import annotations

import numpy as np


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa from a ratings count matrix.

    Args:
        ratings_matrix: A 2-D array of shape ``(n_subjects, n_categories)``
            where each entry ``M[i, j]`` is the number of raters who
            assigned subject *i* to category *j*.

    Returns:
        Fleiss' kappa coefficient. Returns ``float('nan')`` when the
        statistic is undefined (e.g. empty input or zero variance).
    """
    M = np.asarray(ratings_matrix, dtype=float)
    if M.size == 0:
        return float("nan")

    n_i = M.sum(axis=1)
    valid = n_i > 1
    M = M[valid]
    n_i = n_i[valid]

    if M.shape[0] == 0:
        return float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        P_i = (M * (M - 1)).sum(axis=1) / (n_i * (n_i - 1))

    total_ratings = n_i.sum()
    if total_ratings <= 0:
        return float("nan")

    p_j = M.sum(axis=0) / total_ratings
    P_bar = float(np.average(P_i, weights=n_i))
    P_e = float((p_j**2).sum())

    denom = 1.0 - P_e
    if denom <= 0:
        return float("nan")

    return (P_bar - P_e) / denom
