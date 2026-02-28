"""Example: Run inference with NVIDIA Build API.

Supports multiple models:
  - google/gemma-3n-e4b-it (default)
  - google/gemma-3n-e2b-it
  - microsoft/phi-4-multimodal-instruct

Prerequisites:
    pip install biasinear[nvidia,data,audio]

    export NVIDIA_API_KEY="your-api-key"
"""

import io

import soundfile as sf

from biasinear import load_dataset
from biasinear.utils import concat_audio
from biasinear.models import NvidiaModel


def audio_dict_to_bytes(audio_dict: dict) -> bytes:
    """Convert HuggingFace audio dict to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_dict["array"], audio_dict["sampling_rate"], format="WAV")
    return buf.getvalue()


def main():
    # --- Try different NVIDIA models ---
    models = [
        NvidiaModel(),  # default: google/gemma-3n-e4b-it
        NvidiaModel("microsoft/phi-4-multimodal-instruct"),
    ]

    dataset = load_dataset(config="en_Female")
    samples = [s for i, s in enumerate(dataset) if i < 3]

    for model in models:
        print(f"=== {model.model_name} ===")
        for i, sample in enumerate(samples):
            q_bytes = audio_dict_to_bytes(sample["question"])
            opt_bytes = [audio_dict_to_bytes(sample[f"option_{c}"]) for c in "abcd"]
            combined = concat_audio(question=q_bytes, options=opt_bytes)

            output = model.generate(combined)

            print(f"[{i+1}] sample_id={sample['sample_id']}")
            print(f"     prediction={output['answer']}  reference={sample['answer']}")
            print(f"     raw_response={output['raw_response']}")
            print()


if __name__ == "__main__":
    main()
