"""Example: Load and explore the BiasInEar dataset."""

from biasinear import load_dataset

# Load English female samples
dataset = load_dataset(config="en_Female")
print(f"Loaded {len(dataset)} samples from en_Female")

# Access a sample
sample = dataset[0]
print(f"Sample ID: {sample['sample_id']}")
print(f"Subject: {sample['subject']} ({sample['subject_category']})")
print(f"Accent: {sample['accent']}, Order: {sample['order']}")
print(f"Answer: {sample['answer']}")
print(f"Question text: {sample['question_text'][:100]}...")

# Audio data is a dict with 'array' and 'sampling_rate'
q_audio = sample["question"]
print(f"Question audio: sr={q_audio['sampling_rate']}, length={len(q_audio['array'])} samples")

# Filter by accent
american = dataset.filter(lambda x: x["accent"] == "American")
print(f"\nAmerican accent samples: {len(american)}")

# Filter culturally sensitive items
cs = dataset.filter(lambda x: x["cultural_sensitivity_label"] == "CS")
print(f"Culturally sensitive samples: {len(cs)}")
