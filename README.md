# BiasInEar

**Assessing Sensitivity in Audio Language Models Across Linguistic, Demographic, and Positional Variations**

[![arXiv](https://img.shields.io/badge/arXiv-2602.01030-red)](https://arxiv.org/abs/2602.01030)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BiasInEar-yellow)](https://huggingface.co/datasets/ntunlplab/BiasInEar)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

BiasInEar is a benchmark for evaluating speech bias in multilingual multimodal large language models (MLLMs). It provides 11,200 spoken multiple-choice questions across 3 languages, 7 accents, 2 genders, and 2 option orders.

## Installation

```bash
# Core (metrics only)
pip install biasinear

# With data loading
pip install biasinear[data]

# With audio utilities
pip install biasinear[audio]

# With a specific model provider
pip install biasinear[gemini]     # Google Gemini
pip install biasinear[openai]     # OpenAI
pip install biasinear[nvidia]     # NVIDIA Build
pip install biasinear[mistral]    # Mistral

# Everything
pip install biasinear[all]
```

> **Note:** Audio features (`[audio]`) require **FFmpeg** installed on your system.
> Install it via `brew install ffmpeg` (macOS), `apt install ffmpeg` (Ubuntu), or see [ffmpeg.org](https://ffmpeg.org/download.html).

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install biasinear[all]
```

## Model Providers

BiasInEar includes built-in support for several audio language model APIs:

| Provider | Class | Install | Default Model |
|----------|-------|---------|---------------|
| Google Gemini | `GeminiModel` | `pip install biasinear[gemini]` | `gemini-2.5-flash` |
| OpenAI | `OpenAIModel` | `pip install biasinear[openai]` | `gpt-4o-audio-preview` |
| NVIDIA Build | `NvidiaModel` | `pip install biasinear[nvidia]` | `google/gemma-3n-e4b-it` |
| Mistral | `MistralModel` | `pip install biasinear[mistral]` | `voxtral-small-2507` |

### API Keys

Set your API key as an environment variable:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export NVIDIA_API_KEY="..."
export MISTRAL_API_KEY="..."
```

Or pass directly when creating the model:

```python
from biasinear.models import GeminiModel
model = GeminiModel(api_key="your-api-key")
```

### Quick Example (Gemini)

```python
import io
import soundfile as sf
from biasinear import load_dataset
from biasinear.models import GeminiModel
from biasinear.utils import concat_audio

def audio_dict_to_bytes(audio_dict: dict) -> bytes:
    """Convert HuggingFace audio dict to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_dict["array"], audio_dict["sampling_rate"], format="WAV")
    return buf.getvalue()

model = GeminiModel()  # uses GEMINI_API_KEY env var
dataset = load_dataset(config="en_Female")
sample = dataset[0]

q_bytes = audio_dict_to_bytes(sample["question"])
opt_bytes = [audio_dict_to_bytes(sample[f"option_{c}"]) for c in "abcd"]
combined = concat_audio(question=q_bytes, options=opt_bytes)

output = model.generate(combined)
print(output["answer"], output["raw_response"])
```

See [`examples/`](examples/) for complete provider scripts.

## Quick Start

### 1. Load Data

```python
from biasinear import load_dataset

# Load a specific config
dataset = load_dataset(config="en_Female")

# Load all configs merged
dataset = load_dataset()
```

### 2. Run Inference

```python
from biasinear.utils import concat_audio
from biasinear.models import BaseModel

# Implement your model by extending BaseModel
class MyModel(BaseModel):
    def generate(self, audio: bytes) -> dict:
        # Your API call here
        return {"answer": "A", "raw_response": "..."}

model = MyModel("my-model")
output = model.generate(audio_bytes)
```

### 3. Evaluate

```python
from biasinear import Evaluator

evaluator = Evaluator(
    predictions=["A", "B", "A", ...],
    references=["A", "A", "A", ...],
    question_ids=["q1", "q1", "q2", ...],
    groups={
        "language": ["en", "en", "zh", ...],
        "gender": ["Female", "Male", "Female", ...],
        "order": ["original", "reversed", "original", ...],
    },
)
results = evaluator.run()
# {
#     "accuracy": 0.75,
#     "entropy": {"mean": 0.32, "per_question": {...}},
#     "apes": {"language": 0.12, "gender": 0.03, "order": 0.15},
#     "fleiss_kappa": {"language": 0.65, "gender": 0.88, "order": 0.52},
# }
```

### Use Metrics Individually

```python
from biasinear import accuracy, question_entropy, apes, fleiss_kappa

acc = accuracy(predictions, references)
ent = question_entropy(["A", "A", "B", "C"], num_categories=4)
apes_val = apes([0.3, 0.5, 0.4])
kappa = fleiss_kappa(ratings_matrix)
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Standard MCQ correctness |
| **Question Entropy** | Prediction uncertainty across configurations |
| **APES** | Average Pairwise Entropy Shift across variable levels |
| **Fleiss' Kappa** | Inter-rater agreement across perturbations |

See the [paper](https://arxiv.org/abs/2602.01030) for details.

## Citation

```bibtex
@inproceedings{wei-etal-2026-biasinear,
  title={Bias in the Ear of the Listener: Assessing Sensitivity in Audio Language Models Across Linguistic, Demographic, and Positional Variations},
  author={Wei, Sheng-Lun and Liao, Yu-Ling and Chang, Yen-Hua and Huang, Hen-Hsen and Chen, Hsin-Hsi},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2026},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
