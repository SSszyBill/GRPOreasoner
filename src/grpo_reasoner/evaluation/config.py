"""Frozen-criteria eval configuration loader (Hydra-aware).

Loads eval config from a Hydra DictConfig, still validates required keys
and computes SHA256 of the SOURCE yaml for drift detection.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


REQUIRED_TOP_LEVEL_KEYS = (
    "version", "eval_sets", "metrics", "sampling",
    "generation", "scoring", "engine",
)
REQUIRED_EVAL_SET_KEYS = ("name", "path", "data_source", "size")
@dataclass(frozen=True)
class EvalConfig:
    """Typed, immutable view of eval_frozen.yaml. (Unchanged from v1.)"""
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
def load_eval_config(cfg: DictConfig, source_yaml_path: str | Path) -> EvalConfig:
    """Convert a Hydra-loaded DictConfig into a validated EvalConfig.
    source_yaml_path is the ORIGINAL frozen yaml on disk (before hydra
    composition), used for SHA256 drift detection. Hydra can't infer it
    reliably, so we pass it explicitly.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in raw]
    if missing:
        raise ValueError(f"eval config missing required keys: {missing}")
    if not raw["eval_sets"]:
        raise ValueError("eval config has empty eval_sets")
    for eval_set in raw["eval_sets"]:
        eval_missing = [k for k in REQUIRED_EVAL_SET_KEYS if k not in eval_set]
        if eval_missing:
            raise ValueError(f"eval_set missing keys {eval_missing}: {eval_set}")
    resolved_path = Path(source_yaml_path).resolve()
    yaml_bytes = resolved_path.read_bytes()
    yaml_sha256 = hashlib.sha256(yaml_bytes).hexdigest()
    return EvalConfig(
        version=raw["version"],
        frozen_at=raw.get("frozen_at", "unknown"),
        eval_sets=raw["eval_sets"],
        metrics=raw["metrics"],
        sampling=raw["sampling"],
        generation=raw["generation"],
        scoring=raw["scoring"],
        engine=raw["engine"],
        yaml_path=resolved_path,
        yaml_sha256=yaml_sha256,
        raw=raw,
    )