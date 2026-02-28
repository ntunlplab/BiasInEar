"""Tests for individual metric functions."""

import math

import numpy as np
import pytest

from biasinear.metrics import accuracy, question_entropy, apes, fleiss_kappa


# --- accuracy ---

class TestAccuracy:
    def test_perfect(self):
        assert accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_all_wrong(self):
        assert accuracy(["A", "A", "A"], ["B", "B", "B"]) == 0.0

    def test_partial(self):
        assert accuracy(["A", "B", "C", "D"], ["A", "B", "A", "A"]) == 0.5

    def test_case_insensitive(self):
        assert accuracy(["a", "b"], ["A", "B"]) == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy(["A"], ["A", "B"])


# --- question_entropy ---

class TestQuestionEntropy:
    def test_uniform(self):
        # All 4 categories equally represented → max entropy = 1.0
        preds = ["A", "B", "C", "D"]
        assert question_entropy(preds, num_categories=4) == pytest.approx(1.0)

    def test_deterministic(self):
        # All same → entropy = 0.0
        preds = ["A", "A", "A", "A"]
        assert question_entropy(preds, num_categories=4) == pytest.approx(0.0)

    def test_binary(self):
        # Half A, half B out of 4 categories
        preds = ["A", "A", "B", "B"]
        expected = -(2 * (0.5 * math.log(0.5) / math.log(4)))
        assert question_entropy(preds, num_categories=4) == pytest.approx(expected)

    def test_empty(self):
        assert math.isnan(question_entropy([], num_categories=4))


# --- apes ---

class TestApes:
    def test_identical_entropies(self):
        assert apes([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_two_values(self):
        assert apes([0.0, 1.0]) == pytest.approx(1.0)

    def test_three_values(self):
        # pairs: |0-0.5|=0.5, |0-1|=1.0, |0.5-1|=0.5 → mean=2/3
        assert apes([0.0, 0.5, 1.0]) == pytest.approx(2.0 / 3.0)

    def test_single_value(self):
        assert math.isnan(apes([0.5]))

    def test_with_nan(self):
        # NaN values are filtered out
        assert apes([0.0, float("nan"), 1.0]) == pytest.approx(1.0)


# --- fleiss_kappa ---

class TestFleissKappa:
    def test_perfect_agreement(self):
        # All raters agree on the same category
        M = np.array([
            [3, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 3, 0],
        ])
        assert fleiss_kappa(M) == pytest.approx(1.0, abs=1e-10)

    def test_no_agreement(self):
        # Each rater picks a different category for each subject
        M = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
        ])
        kappa = fleiss_kappa(M)
        assert kappa < 0.1  # near-chance agreement

    def test_empty(self):
        assert math.isnan(fleiss_kappa(np.array([])))

    def test_single_rater_skipped(self):
        # Rows with n_i <= 1 are filtered
        M = np.array([
            [1, 0, 0, 0],  # only 1 rater → skipped
            [3, 0, 0, 0],  # 3 agree on same category
            [0, 3, 0, 0],  # 3 agree on different category
        ])
        # Two valid subjects, all with perfect agreement
        assert fleiss_kappa(M) == pytest.approx(1.0, abs=1e-10)
