"""Example: Run inference with OpenAI.

Prerequisites:
    pip install biasinear[openai,data,audio]

    export OPENAI_API_KEY="your-api-key"
"""

import io

import soundfile as sf

from biasinear import load_dataset
from biasinear.utils import concat_audio
from biasinear.models import OpenAIModel


def audio_dict_to_bytes(audio_dict: dict) -> bytes:
    """Convert HuggingFace audio dict to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_dict["array"], audio_dict["sampling_rate"], format="WAV")
    return buf.getvalue()


def main():
    model = OpenAIModel()  # defaults to gpt-4o-audio-preview
    dataset = load_dataset(config="en_Female")

    for i, sample in enumerate(dataset):
        if i >= 3:
            break

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
