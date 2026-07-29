"""Candidate-pool sampling helpers."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write_selected(src_path: str | Path, out_path: str | Path, selected: set[int], batch_size: int, logger=None) -> int:
    parquet_file = pq.ParquetFile(src_path)
    writer = pq.ParquetWriter(out_path, parquet_file.schema_arrow)
    global_index = 0
    kept = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        n_rows = batch.num_rows
        mask = [(global_index + i) in selected for i in range(n_rows)]
        global_index += n_rows
        if any(mask):
            table = pa.Table.from_batches([batch])
            filtered = table.filter(pa.array(mask, type=pa.bool_()))
            if filtered.num_rows > 0:
                writer.write_table(filtered)
                kept += filtered.num_rows
    writer.close()
    if logger is not None:
        logger.info(f"wrote {kept} rows -> {out_path}")
    return kept


def subsample_uniform(
    src_path: str | Path,
    out_path: str | Path,
    n: int,
    seed: int,
    batch_size: int,
    logger=None,
) -> int:
    parquet_file = pq.ParquetFile(src_path)
    total = parquet_file.metadata.num_rows
    n = min(n, total)
    rng = random.Random(seed)
    selected = set(rng.sample(range(total), n))
    if logger is not None:
        logger.info(f"[uniform] {src_path}: total {total} -> sample {n}")
    return _write_selected(src_path, out_path, selected, batch_size, logger)


def subsample_stratified(
    src_path: str | Path,
    out_path: str | Path,
    n: int,
    seed: int,
    batch_size: int,
    logger=None,
) -> int:
    parquet_file = pq.ParquetFile(src_path)
    total = parquet_file.metadata.num_rows
    keys = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["extra_info"]):
        for extra_info in batch.column("extra_info").to_pylist():
            subsource = extra_info.get("subsource", "unknown") if isinstance(extra_info, dict) else "unknown"
            keys.append(subsource)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    rng = random.Random(seed)
    selected: set[int] = set()
    dist = {}
    for key, indexes in sorted(groups.items()):
        take = min(len(indexes), round(n * len(indexes) / total))
        picked = rng.sample(indexes, take)
        selected.update(picked)
        dist[key] = take
    if logger is not None:
        logger.info(f"[stratified] {src_path}: total {total} -> sample {len(selected)}")
        logger.info(f"stratified result: {dist}")
    return _write_selected(src_path, out_path, selected, batch_size, logger)

