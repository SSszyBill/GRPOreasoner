# 4090_grpo_multisource_v1

**实验目标**：Qwen2.5-1.5B-Base 上基于 GSM8K + NuminaMath + Countdown 三源数据做 GRPO 后训练，对比 GSM8K test / MATH500 在训练前后的 pass@1 与 maj@8。

**关键口径**：见 `configs/eval_frozen.yaml`，冻结于 2026-07-03，不再修改。

**上游依赖**：
- 数据：`data/gsm8k/`、`data/numinamath/train_decontam.parquet`、`data/countdown/`、`data/math500/test.parquet`
- reward：`utils/reward_multi.py`（13/13 单测通过）
- 候选池：`data/candidate/`合计 47,471 条

**时间线**：
- 2026-07-03：冻结评测口径 v1
- 待补：分桶完成 / baseline 完成 / 定稿配比 / 训练完成 / 前后对比

**结果汇总**：见 `results/comparison.md`