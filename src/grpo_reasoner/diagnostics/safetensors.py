"""safetensors inspection helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path


def list_shards(model_dir: str | Path) -> list[Path]:
    return sorted(Path(model_dir).glob("*.safetensors"))


def dtype_counts(model_dir: str | Path) -> Counter:
    from safetensors import safe_open

    shards = list_shards(model_dir)
    if not shards:
        raise FileNotFoundError(f"no safetensors under {model_dir}")

    dtypes = Counter()
    with safe_open(shards[0], framework="pt", device="cpu") as handle:
        for key in handle.keys():
            dtypes[handle.get_slice(key).get_dtype()] += 1
    return dtypes


def inspect_safetensors(model_dir: str | Path) -> dict:
    shards = list_shards(model_dir)
    return {
        "model_dir": str(model_dir),
        "num_shards": len(shards),
        "shards": [str(path) for path in shards],
        "dtype_counts_first_shard": dict(dtype_counts(model_dir)) if shards else {},
    }

