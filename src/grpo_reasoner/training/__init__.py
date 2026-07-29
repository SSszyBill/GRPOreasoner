"""Training-side entrypoints.

Public API:
    reward_adapter.compute_score  (loaded by verl via file path)
    lora_merge.merge              (also usable as CLI: `python -m grpo_reasoner.training.lora_merge`)

Kept intentionally thin: training-specific glue lives here, not scoring
logic (that's in grpo_reasoner.scoring).
"""