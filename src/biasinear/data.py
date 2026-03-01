"""Data loading utilities for BiasInEar dataset."""

from __future__ import annotations

DATASET_ID = "ntunlplab/BiasInEar"
ALL_CONFIGS = ["en_Female", "en_Male", "zh_Female", "zh_Male", "ko_Female", "ko_Male"]


def load_dataset(config: str | None = None, split: str = "train"):
    """Load the BiasInEar dataset from HuggingFace Hub.

    Thin wrapper around ``datasets.load_dataset()``.

    Args:
        config: Dataset configuration name. One of ``"en_Female"``,
            ``"en_Male"``, ``"zh_Female"``, ``"zh_Male"``, ``"ko_Female"``,
            ``"ko_Male"``. If ``None``, all configs are loaded and
            concatenated.
        split: Dataset split (default ``"train"``).

    Returns:
        A ``datasets.Dataset`` object.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
            Install it with ``pip install biasinear[data]``.
    """
    try:
        import datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for loading data. "
            "Install it with: pip install biasinear[data]"
        )

    if config is not None:
        return datasets.load_dataset(DATASET_ID, config, split=split)

    # Load all configs and concatenate
    all_ds = []
    for cfg in ALL_CONFIGS:
        ds = datasets.load_dataset(DATASET_ID, cfg, split=split)
        all_ds.append(ds)

    return datasets.concatenate_datasets(all_ds)
