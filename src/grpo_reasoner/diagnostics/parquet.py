"""Parquet inspection helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parquet_summary(path: str | Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    meta = parquet_file.metadata
    return {
        "path": str(path),
        "num_rows": meta.num_rows,
        "num_columns": meta.num_columns,
        "num_row_groups": meta.num_row_groups,
        "schema": str(parquet_file.schema_arrow),
    }


def first_row_preview(path: str | Path, max_chars: int = 120) -> dict[str, str]:
    parquet_file = pq.ParquetFile(path)
    first_batch = next(parquet_file.iter_batches(batch_size=1))
    sample = first_batch.to_pylist()[0]
    preview = {}
    for key, value in sample.items():
        text = str(value)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        preview[key] = text
    return preview

