"""Example: Evaluate model predictions with BiasInEar metrics."""

import json
from pathlib import Path

from biasinear import Evaluator, accuracy

# --- Option A: Quick accuracy check ---

predictions = ["A", "B", "C", "D", "A", "B"]
references = ["A", "B", "C", "D", "B", "B"]
print(f"Accuracy: {accuracy(predictions, references):.2%}")

# --- Option B: Full evaluation with Evaluator ---

# Simulated data (in practice, load from your inference results)
predictions = ["A", "A", "B", "A", "A", "B", "A", "A"]
references = ["A", "A", "A", "A", "A", "A", "A", "A"]
question_ids = ["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2"]
groups = {
    "language": ["en", "en", "zh", "zh", "en", "en", "zh", "zh"],
    "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    "order": ["original", "reversed", "original", "reversed", "original", "reversed", "original", "reversed"],
}

evaluator = Evaluator(
    predictions=predictions,
    references=references,
    question_ids=question_ids,
    groups=groups,
)
results = evaluator.run()

print(f"\n=== Evaluation Results ===")
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Mean Entropy: {results['entropy']['mean']:.4f}")
print(f"\nAPES:")
for var, val in results["apes"].items():
    print(f"  {var}: {val:.4f}")
print(f"\nFleiss' Kappa:")
for var, val in results["fleiss_kappa"].items():
    print(f"  {var}: {val:.4f}")
