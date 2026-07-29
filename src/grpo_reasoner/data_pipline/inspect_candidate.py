"""Helpers for inspecting candidate pools."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def count_rows(path: str | Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def subsource_dist(path: str | Path, batch_size: int = 2000) -> Counter:
    counter = Counter()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["extra_info"]):
        for extra_info in batch.column("extra_info").to_pylist():
            subsource = extra_info.get("subsource", "unknown") if isinstance(extra_info, dict) else "unknown"
            counter[subsource] += 1
    return counter

