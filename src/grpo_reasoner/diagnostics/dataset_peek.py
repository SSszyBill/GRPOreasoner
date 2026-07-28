"""Small helpers for peeking Hugging Face datasets."""
from __future__ import annotations

import random
from typing import Any


def dataset_summary(dataset) -> dict[str, Any]:
    return {
        "features": list(dataset.features.keys()),
        "num_rows": len(dataset),
    }


def sample_rows(dataset, n: int = 3, max_chars: int = 600, seed: int | None = None) -> list[dict[str, str]]:
    rng = random.Random(seed)
    indexes = rng.sample(range(len(dataset)), min(n, len(dataset)))
    rows = []
    for index in indexes:
        row = {}
        for key, value in dataset[index].items():
            text = str(value)
            if len(text) > max_chars:
                text = text[:max_chars] + " ..."
            row[key] = text
        rows.append({"_row_index": str(index), **row})
    return rows


def load_dataset_for_peek(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)

