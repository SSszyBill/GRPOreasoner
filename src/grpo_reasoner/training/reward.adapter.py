"""Reward function entrypoint for verl.

verl loads reward functions by *file path* (see custom_reward_function.path
in your training shell script), not by python import. This module re-exports
grpo_reasoner.scoring.compute_score at module level so verl can grab it.

The one requirement: `grpo_reasoner` must be pip-installed (editable is fine)
in the training environment. This is guaranteed by running
`pip install -e .` from the repo root once — same install Phase 1 sets up.

Usage in a verl shell script:
    custom_reward_function.path=$RL_ROOT/GRPOreasoner/src/grpo_reasoner/training/reward_adapter.py
    custom_reward_function.name=compute_score
"""
from grpo_reasoner.scoring import compute_score  # noqa: F401

__all__ = ["compute_score"]