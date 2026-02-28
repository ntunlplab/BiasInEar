"""Example: Run inference with a custom model.

This example demonstrates how to:
1. Load data from HuggingFace
2. Concatenate question + option audio
3. Call a model and save results

NOTE: You need to implement your own model class that extends BaseModel.
"""

import json
import io
from pathlib import Path

import soundfile as sf

from biasinear import load_dataset
from biasinear.utils import concat_audio
from biasinear.models import BaseModel


# --- Step 1: Implement your model ---

class MyModel(BaseModel):
    """Example model placeholder. Replace with your actual implementation."""

    def generate(self, audio: bytes) -> dict:
        # TODO: Replace with actual API call
        # e.g. send audio bytes to OpenAI, Google, etc.
        return {"answer": "A", "raw_response": "placeholder"}


# --- Step 2: Load data and run inference ---

def audio_dict_to_bytes(audio_dict: dict) -> bytes:
    """Convert HuggingFace audio dict to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_dict["array"], audio_dict["sampling_rate"], format="WAV")
    return buf.getvalue()


def main():
    model = MyModel("my-model")
    dataset = load_dataset(config="en_Female")

    results = []
    for i, sample in enumerate(dataset):
        if i >= 5:  # Process only first 5 for demo
            break

        # Convert audio dicts to bytes
        q_bytes = audio_dict_to_bytes(sample["question"])
        opt_bytes = [
            audio_dict_to_bytes(sample[f"option_{c}"])
            for c in "abcd"
        ]

        # Concatenate into single audio
        combined = concat_audio(question=q_bytes, options=opt_bytes)

        # Run inference
        output = model.generate(combined)

        results.append({
            "sample_id": sample["sample_id"],
            "prediction": output["answer"],
            "reference": sample["answer"],
            "raw_response": output["raw_response"],
        })

        print(f"[{i+1}] {sample['sample_id']}: pred={output['answer']}, ref={sample['answer']}")

    # Save results
    out_path = Path("results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
