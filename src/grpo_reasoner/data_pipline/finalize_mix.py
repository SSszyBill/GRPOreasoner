"""Finalize the training mix from bucketed candidate pools."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import yaml


@dataclass(frozen=True)
class BucketedSource:
    """Where each source's pass_rate parquet lives."""
    name: str
    data_source: str
    input_filename: str


BUCKETED_SOURCES: tuple[BucketedSource, ...] = (
    BucketedSource("gsm8k", "openai/gsm8k", "gsm8k_train_pass_rate.parquet"),
    BucketedSource("numinamath", "numina_math", "numinamath_train_pass_rate.parquet"),
    BucketedSource("countdown", "countdown", "countdown_train_pass_rate.parquet"),
)

SOURCE_NAMES: frozenset[str] = frozenset(source.name for source in BUCKETED_SOURCES)


@dataclass(frozen=True)
class PassRateRange:
    """Half-open filter (min_exclusive, max_exclusive)."""
    min_exclusive: float
    max_exclusive: float

    def matches(self, pass_rate: float) -> bool:
        return self.min_exclusive < pass_rate < self.max_exclusive


@dataclass(frozen=True)
class SourceTakeSpec:
    """How many rows to take from one source after filtering."""
    name: str
    take: int | None


@dataclass(frozen=True)
class MixConfig:
    """Full training-mix config, loaded from YAML."""
    version: str
    seed: int
    pass_rate_range: PassRateRange
    source_takes: tuple[SourceTakeSpec, ...]
    yaml_path: Path
    yaml_sha256: str


def load_mix_config(config_path: Path) -> MixConfig:
    raw_bytes = config_path.read_bytes()
    yaml_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes)

    version = str(raw.get("version", "v1"))
    seed = int(raw.get("seed", 42))

    range_raw = raw.get("pass_rate_range", {})
    pass_rate_range = PassRateRange(
        min_exclusive=float(range_raw.get("min_exclusive", 0.0)),
        max_exclusive=float(range_raw.get("max_exclusive", 1.0)),
    )
    if not (0.0 <= pass_rate_range.min_exclusive < pass_rate_range.max_exclusive <= 1.0):
        raise ValueError(
            f"pass_rate_range invalid: {pass_rate_range} "
            "(need 0 <= min_exclusive < max_exclusive <= 1)"
        )

    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict) or not sources_raw:
        raise ValueError("config.sources must be a non-empty dict")

    unknown = set(sources_raw.keys()) - SOURCE_NAMES
    if unknown:
        raise ValueError(
            f"config.sources has unknown source names {sorted(unknown)}; "
            f"known: {sorted(SOURCE_NAMES)}"
        )

    source_takes: list[SourceTakeSpec] = []
    for source_name, spec in sources_raw.items():
        take_raw = spec.get("take") if isinstance(spec, dict) else spec
        if take_raw == "all" or take_raw is None:
            take: int | None = None
        else:
            take = int(take_raw)
            if take <= 0:
                raise ValueError(
                    f"config.sources.{source_name}.take must be 'all' or a positive int, got {take_raw!r}"
                )
        source_takes.append(SourceTakeSpec(name=source_name, take=take))

    return MixConfig(
        version=version,
        seed=seed,
        pass_rate_range=pass_rate_range,
        source_takes=tuple(source_takes),
        yaml_path=config_path,
        yaml_sha256=yaml_sha256,
    )


def iter_rows(parquet_path: Path, batch_size: int = 512) -> Iterator[dict]:
    parquet_file = pq.ParquetFile(str(parquet_path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


@dataclass(frozen=True)
class SourceResult:
    name: str
    data_source: str
    input_path: Path
    n_pool: int
    n_after_filter: int
    n_taken: int
    take_requested: int | None
    was_shortage: bool
    pass_rate_hist_taken: dict[str, int]
    rows: list[dict]


def finalize_one_source(
    source: BucketedSource,
    input_path: Path,
    take: int | None,
    pass_rate_range: PassRateRange,
    seed: int,
    allow_shortage: bool,
) -> SourceResult:
    pool_rows: list[dict] = list(iter_rows(input_path))
    n_pool = len(pool_rows)

    filtered_rows = [row for row in pool_rows if pass_rate_range.matches(row["pass_rate"])]
    n_after_filter = len(filtered_rows)

    was_shortage = False
    if take is None:
        taken_rows = filtered_rows
    else:
        if take > n_after_filter:
            was_shortage = True
            if not allow_shortage:
                raise ValueError(
                    f"{source.name}: requested take={take} but only {n_after_filter} rows available after filter"
                )
            taken_rows = filtered_rows
        else:
            per_source_rng = random.Random(f"{seed}:{source.name}")
            taken_rows = per_source_rng.sample(filtered_rows, take)

    n_taken = len(taken_rows)
    k = taken_rows[0]["k_samples"] if taken_rows else 0
    pass_rate_hist_taken = {
        str(c): sum(1 for row in taken_rows if row["correct_count"] == c)
        for c in range(k + 1)
    }

    return SourceResult(
        name=source.name,
        data_source=source.data_source,
        input_path=input_path,
        n_pool=n_pool,
        n_after_filter=n_after_filter,
        n_taken=n_taken,
        take_requested=take,
        was_shortage=was_shortage,
        pass_rate_hist_taken=pass_rate_hist_taken,
        rows=taken_rows,
    )

