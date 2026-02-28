"""Tests for the Evaluator integration."""

import math

import pytest

from biasinear.evaluate import Evaluator


class TestEvaluator:
    """Test Evaluator with small fixed data."""

    def _make_evaluator(self):
        """Create an evaluator with known data for two questions."""
        # q1: 4 configs, all predict "A" (correct answer "A")
        # q2: 4 configs, 2 predict "A" and 2 predict "B" (correct answer "A")
        predictions = ["A", "A", "A", "A", "A", "B", "A", "B"]
        references = ["A", "A", "A", "A", "A", "A", "A", "A"]
        question_ids = ["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2"]
        groups = {
            "gender": [
                "Female", "Male", "Female", "Male",
                "Female", "Male", "Female", "Male",
            ],
            "order": [
                "original", "original", "reversed", "reversed",
                "original", "original", "reversed", "reversed",
            ],
        }
        return Evaluator(predictions, references, question_ids, groups)

    def test_accuracy(self):
        ev = self._make_evaluator()
        results = ev.run()
        # 6 correct out of 8
        assert results["accuracy"] == pytest.approx(6 / 8)

    def test_entropy_keys(self):
        ev = self._make_evaluator()
        results = ev.run()
        assert "mean" in results["entropy"]
        assert "per_question" in results["entropy"]
        assert "q1" in results["entropy"]["per_question"]
        assert "q2" in results["entropy"]["per_question"]

    def test_q1_zero_entropy(self):
        ev = self._make_evaluator()
        results = ev.run()
        # q1: all predictions are "A" → entropy = 0
        assert results["entropy"]["per_question"]["q1"] == pytest.approx(0.0)

    def test_q2_nonzero_entropy(self):
        ev = self._make_evaluator()
        results = ev.run()
        # q2: 2 "A" and 2 "B" → entropy > 0
        assert results["entropy"]["per_question"]["q2"] > 0

    def test_apes_keys(self):
        ev = self._make_evaluator()
        results = ev.run()
        assert "gender" in results["apes"]
        assert "order" in results["apes"]

    def test_fleiss_kappa_keys(self):
        ev = self._make_evaluator()
        results = ev.run()
        assert "gender" in results["fleiss_kappa"]
        assert "order" in results["fleiss_kappa"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            Evaluator(["A"], ["A", "B"], ["q1"], {"g": ["x"]})

    def test_group_length_mismatch(self):
        with pytest.raises(ValueError):
            Evaluator(["A", "B"], ["A", "B"], ["q1", "q1"], {"g": ["x"]})
