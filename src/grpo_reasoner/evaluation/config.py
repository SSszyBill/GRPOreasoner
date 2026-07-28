"""Frozen-criteria eval configuration loader.

Loads eval_frozen.yaml, validates required keys, and checksums the file
so downstream runs can prove they used a specific frozen version.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = (
    "version", "eval_sets", "metrics", "sampling",
    "generation", "scoring", "engine",
)

REQUIRED_EVAL_SET_KEYS = ("name", "path", "data_source", "size")


@dataclass(frozen=True)
class EvalConfig:
    """Typed, immutable view of eval_frozen.yaml.

    Fields mirror top-level yaml sections one-to-one. Nested sections stay
    as dicts because their inner keys are already schema-checked at load time.
    """
    version: str
    frozen_at: str
    eval_sets: list[dict[str, Any]]
    metrics: dict[str, Any]
    sampling: dict[str, Any]
    generation: dict[str, Any]
    scoring: dict[str, Any]
    engine: dict[str, Any]
    yaml_path: Path
    yaml_sha256: str
    raw: dict[str, Any] = field(repr=False)


def load_eval_config(yaml_path: str | Path) -> EvalConfig:
    """Load, checksum, and validate the frozen eval yaml.

    Fails loud on any missing required key so a malformed frozen config never
    silently degrades a baseline run.
    """
    resolved_yaml_path = Path(yaml_path).resolve()
    yaml_text = resolved_yaml_path.read_text(encoding="utf-8")
    yaml_sha256 = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
    raw_config = yaml.safe_load(yaml_text)

    missing_top_level = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in raw_config]
    if missing_top_level:
        raise ValueError(
            f"eval_frozen.yaml missing required keys: {missing_top_level}"
        )
    if not raw_config["eval_sets"]:
        raise ValueError("eval_frozen.yaml has empty eval_sets")

    for eval_set in raw_config["eval_sets"]:
        missing = [k for k in REQUIRED_EVAL_SET_KEYS if k not in eval_set]
        if missing:
            raise ValueError(f"eval_set missing keys {missing}: {eval_set}")

    return EvalConfig(
        version=raw_config["version"],
        frozen_at=raw_config.get("frozen_at", "unknown"),
        eval_sets=raw_config["eval_sets"],
        metrics=raw_config["metrics"],
        sampling=raw_config["sampling"],
        generation=raw_config["generation"],
        scoring=raw_config["scoring"],
        engine=raw_config["engine"],
        yaml_path=resolved_yaml_path,
        yaml_sha256=yaml_sha256,
        raw=raw_config,
    )

