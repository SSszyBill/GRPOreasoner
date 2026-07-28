"""Environment probing helpers."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterable


DEFAULT_MODULES: tuple[str, ...] = (
    "math_verify",
    "sympy",
    "latex2sympy2",
    "antlr4",
    "verl",
    "verl.utils.reward_score",
    "verl.utils.reward_score.math",
    "verl.utils.reward_score.gsm8k",
)


def probe_modules(modules: Iterable[str] = DEFAULT_MODULES) -> list[dict[str, str]]:
    results = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            results.append({"module": module_name, "status": "ok", "error": ""})
        except Exception as exc:
            results.append({
                "module": module_name,
                "status": "missing",
                "error": f"{type(exc).__name__}: {str(exc)[:120]}",
            })
    return results


def reward_score_info() -> dict:
    try:
        import verl.utils.reward_score as reward_score
    except Exception as exc:
        return {
            "status": "missing",
            "error": f"{type(exc).__name__}: {str(exc)[:120]}",
            "path": "",
            "files": [],
            "exports": [],
        }

    directory = Path(os.path.dirname(reward_score.__file__))
    return {
        "status": "ok",
        "error": "",
        "path": str(directory),
        "files": sorted(path.name for path in directory.iterdir() if not path.name.startswith("__")),
        "exports": [name for name in dir(reward_score) if not name.startswith("_")],
    }

