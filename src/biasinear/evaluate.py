"""Evaluator for BiasInEar benchmark results."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np

from biasinear.metrics.accuracy import accuracy as compute_accuracy
from biasinear.metrics.apes import apes as compute_apes
from biasinear.metrics.entropy import question_entropy
from biasinear.metrics.fleiss_kappa import fleiss_kappa


class Evaluator:
    """Unified evaluator that computes all BiasInEar metrics.

    Args:
        predictions: Model predicted answers (e.g. ``["A", "B", "C", ...]``).
        references: Ground-truth answers (e.g. ``["B", "B", "B", ...]``).
        question_ids: Base question ID for each sample, used to group
            predictions per question.
        groups: Dict mapping variable names to per-sample level lists.
            Example::

                {
                    "language": ["en", "en", "zh", ...],
                    "accent": ["American", "British", "Beijing", ...],
                    "gender": ["Female", "Female", "Male", ...],
                    "order": ["original", "reversed", ...],
                }
    """

    ANSWER_CATEGORIES = ["A", "B", "C", "D"]

    def __init__(
        self,
        predictions: list[str],
        references: list[str],
        question_ids: list[str],
        groups: dict[str, list[str]],
    ) -> None:
        n = len(predictions)
        if len(references) != n or len(question_ids) != n:
            raise ValueError("predictions, references, and question_ids must have the same length")
        for name, vals in groups.items():
            if len(vals) != n:
                raise ValueError(f"Group '{name}' length ({len(vals)}) != predictions length ({n})")

        self.predictions = [p.strip().upper() for p in predictions]
        self.references = [r.strip().upper() for r in references]
        self.question_ids = question_ids
        self.groups = groups

    def run(self) -> dict:
        """Compute all metrics.

        Returns:
            Dict with keys ``accuracy``, ``entropy``, ``apes``, ``fleiss_kappa``.
        """
        acc = compute_accuracy(self.predictions, self.references)

        # --- per-question grouping ---
        q_preds: dict[str, list[str]] = defaultdict(list)
        q_groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        for i, qid in enumerate(self.question_ids):
            q_preds[qid].append(self.predictions[i])
            for var, vals in self.groups.items():
                q_groups[qid][var].append(vals[i])

        # --- entropy ---
        per_q_entropy: dict[str, float] = {}
        for qid, preds in q_preds.items():
            per_q_entropy[qid] = question_entropy(preds, num_categories=len(self.ANSWER_CATEGORIES))

        entropy_values = [v for v in per_q_entropy.values() if not math.isnan(v)]
        entropy_mean = float(np.mean(entropy_values)) if entropy_values else float("nan")

        # --- APES and Fleiss' kappa per variable ---
        apes_results: dict[str, float] = {}
        kappa_results: dict[str, float] = {}

        for var in self.groups:
            apes_results[var] = self._compute_apes_for_variable(var, q_preds, q_groups)
            kappa_results[var] = self._compute_kappa_for_variable(var, q_preds, q_groups)

        return {
            "accuracy": acc,
            "entropy": {
                "mean": entropy_mean,
                "per_question": per_q_entropy,
            },
            "apes": apes_results,
            "fleiss_kappa": kappa_results,
        }

    def _compute_apes_for_variable(
        self,
        var: str,
        q_preds: dict[str, list[str]],
        q_groups: dict[str, dict[str, list[str]]],
    ) -> float:
        """Compute mean APES across all questions for a variable."""
        apes_per_q: list[float] = []

        for qid, preds in q_preds.items():
            levels = q_groups[qid][var]
            # group predictions by level
            level_preds: dict[str, list[str]] = defaultdict(list)
            for pred, lvl in zip(preds, levels):
                level_preds[lvl].append(pred)

            # entropy per level
            entropies = [
                question_entropy(lp, num_categories=len(self.ANSWER_CATEGORIES))
                for lp in level_preds.values()
            ]

            val = compute_apes(entropies)
            if not math.isnan(val):
                apes_per_q.append(val)

        return float(np.mean(apes_per_q)) if apes_per_q else float("nan")

    def _compute_kappa_for_variable(
        self,
        var: str,
        q_preds: dict[str, list[str]],
        q_groups: dict[str, dict[str, list[str]]],
    ) -> float:
        """Compute mean Fleiss' kappa across all questions for a variable."""
        kappa_per_q: list[float] = []
        cats = self.ANSWER_CATEGORIES

        for qid, preds in q_preds.items():
            levels = q_groups[qid][var]
            other_vars = [v for v in self.groups if v != var]

            # Build items: each combination of other variables is one "subject"
            # and each level of the target variable is one "rater".
            # The rating is the mode answer for that (other-vars-combo, level).
            combo_key = defaultdict(lambda: defaultdict(list))
            for i, (pred, lvl) in enumerate(zip(preds, levels)):
                other_key = tuple(q_groups[qid][ov][i] for ov in other_vars)
                combo_key[other_key][lvl].append(pred)

            items: list[np.ndarray] = []
            for other_key, level_dict in combo_key.items():
                counts = np.zeros(len(cats), dtype=int)
                for lvl, lvl_preds in level_dict.items():
                    # mode answer for this (other-combo, level)
                    mode = _mode(lvl_preds)
                    if mode in cats:
                        counts[cats.index(mode)] += 1
                if counts.sum() > 1:
                    items.append(counts)

            if items:
                k = fleiss_kappa(np.vstack(items))
                if not math.isnan(k):
                    kappa_per_q.append(k)

        return float(np.mean(kappa_per_q)) if kappa_per_q else float("nan")


def _mode(values: list[str]) -> str | None:
    """Return the most frequent value, tie-breaking alphabetically."""
    if not values:
        return None
    from collections import Counter

    counts = Counter(v.strip().upper() for v in values)
    max_count = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == max_count)
    return winners[0]
