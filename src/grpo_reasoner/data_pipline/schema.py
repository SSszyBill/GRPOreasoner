"""verl-format training/eval sample schema.
verl expects each parquet row to have exactly these five top-level fields:
data_source     : str        — source identifier (e.g. "openai/gsm8k")
prompt          : list[dict] — chat turns, each with {"role", "content"}
ability         : str        — task category (e.g. "math")
reward_model    : dict       — {"style": "rule", "ground_truth": str}
extra_info      : dict       — free-form metadata; must be a dict, not None
Any code that writes verl-format parquet should build rows via
VerlSample.build(...).to_dict() rather than assembling dicts by hand,
so that field names, nesting, and required keys stay consistent across
gsm8k / numina / math500 / countdown / (future sources).
"""
from future import annotations
from dataclasses import dataclass, field
from typing import Any
_REQUIRED_TOP_LEVEL = ("data_source", "prompt", "ability", "reward_model", "extra_info")
_REQUIRED_REWARD_MODEL = ("style", "ground_truth")

@dataclass(frozen=True)
class VerlSample:
    """One verl-format sample. Immutable so it can be safely shared/hashed."""
    data_source: str
    prompt: tuple[dict, ...]
    ability: str
    ground_truth: str
    extra_info: dict = field(default_factory=dict)
    reward_style: str = "rule"

    @classmethod
    def build(
        cls,
        data_source: str,
        user_content: str,
        ground_truth: str,
        ability: str = "math",
        extra_info: dict | None = None,
        reward_style: str = "rule",
        ) -> "VerlSample":
        """Convenience constructor for the common single-user-turn case."""
        return cls(
            data_source=data_source,
            prompt=({"role": "user", "content": user_content},),
            ability=ability,
            ground_truth=str(ground_truth),
            extra_info=dict(extra_info) if extra_info else {},
            reward_style=reward_style,
        )
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a verl-compatible dict (ready for pyarrow.Table.from_pylist)."""
        return {
            "data_source": self.data_source,
            "prompt": [dict(turn) for turn in self.prompt],
            "ability": self.ability,
            "reward_model": {"style": self.reward_style, "ground_truth": self.ground_truth},
            "extra_info": dict(self.extra_info),
        }
def validate(row: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems with row, or [] if OK.
    Use this as a defensive check when consuming parquet files from
    outside sources or when debugging "verl rejects my dataset" issues.
    """
    problems: list[str] = []
    for key in _REQUIRED_TOP_LEVEL:
        if key not in row:
            problems.append(f"missing top-level field: {key}")
    prompt = row.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, list) or not prompt:
            problems.append("prompt must be a non-empty list of {role, content} dicts")
    else:
        for i, turn in enumerate(prompt):
            if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                problems.append(f"prompt[{i}] must have keys 'role' and 'content'")
    reward_model = row.get("reward_model")
    if reward_model is not None:
        if not isinstance(reward_model, dict):
            problems.append("reward_model must be a dict")
        else:
            for key in _REQUIRED_REWARD_MODEL:
                if key not in reward_model:
                    problems.append(f"reward_model missing key: {key}")
    extra_info = row.get("extra_info")
    if extra_info is not None and not isinstance(extra_info, dict):
        problems.append("extra_info must be a dict (use {} for empty, not None)")
    return problems