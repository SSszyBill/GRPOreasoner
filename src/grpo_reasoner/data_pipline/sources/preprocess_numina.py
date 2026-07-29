"""NuminaMath-CoT conversion helpers for verl-format RL parquet."""
from __future__ import annotations

from collections import Counter
from grpo_reasoner.data_pipline.schema import VerlSample

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DATA_SOURCE = "numina_math"
ABILITY = "math"


def extract_all_boxed(text: str) -> list[str]:
    key = r"\boxed"
    out: list[str] = []
    i = 0
    while True:
        idx = text.find(key, i)
        if idx == -1:
            break
        j = idx + len(key)
        while j < len(text) and text[j] == " ":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = idx + len(key)
            continue
        depth, start = 0, j
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[start + 1:j])
        i = j + 1
    return out


def normalize_answer(s: str) -> str:
    s = s.strip()
    for tok in [r"\left", r"\right", r"\!", r"\,", r"\;", r"\ ", "$", "\n"]:
        s = s.replace(tok, "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return s.strip()


def extract_ground_truth(solution: str | None) -> tuple[str | None, str]:
    boxes = extract_all_boxed(solution or "")
    norm = [normalize_answer(b) for b in boxes]
    uniq = set(norm)
    if len(boxes) == 0:
        return None, "no_boxed"
    if len(uniq) > 1:
        return None, "multi_boxed"
    gt = norm[0]
    if gt == "":
        return None, "empty"
    if len(gt) > 100:
        return None, "too_long"
    return gt, "ok"


def process_example(example: dict, idx: int, split: str) -> dict:
    problem = example.get("problem", "") or ""
    solution = example.get("solution", "") or ""
    subsource = example.get("source", "") or ""
    gt, reason = extract_ground_truth(solution)
    keep = gt is not None
    return VerlSample.build(
        data_source=DATA_SOURCE,
        user_content=problem + "\n" + INSTRUCTION,
        ground_truth=str(gt) if gt is not None else "",
        ability=ABILITY,
        extra_info={
            "split": split,
            "index": idx,
            "subsource": subsource,
            "solution": solution,
            "reason": reason,
            "keep": keep,
        },
    ).to_dict()


def compute_stats(split: str, n_total: int, keeps: list[bool], subs: list[str], reasons: list[str]) -> dict:
    n_keep = sum(1 for keep in keeps if keep)
    total_by_src = Counter(subs)
    kept_by_src = Counter(sub for sub, keep in zip(subs, keeps) if keep)
    drop_reasons = Counter(reason for reason in reasons if reason != "ok")
    per_sub = {}
    for src in sorted(total_by_src, key=lambda item: -total_by_src[item]):
        total, kept = total_by_src[src], kept_by_src[src]
        per_sub[src] = {"total": total, "kept": kept, "rate": round(kept / total, 4) if total else 0.0}
    return {
        "split": split,
        "original": n_total,
        "kept": n_keep,
        "dropped": n_total - n_keep,
        "retention": round(n_keep / n_total, 4) if n_total else 0.0,
        "drop_reasons": dict(drop_reasons),
        "per_subsource": per_sub,
    }


def log_stats(logger, stats: dict) -> None:
    logger.info(
        f"[{stats['split']}] original {stats['original']} | kept {stats['kept']} | "
        f"dropped {stats['dropped']} | retention {stats['retention']:.1%}"
    )
    logger.info(f"[{stats['split']}] drop details: {stats['drop_reasons']}")
    logger.info(f"[{stats['split']}] retention by subsource:")
    for src, value in stats["per_subsource"].items():
        logger.info(f"    {src:16s} {value['kept']:>7d}/{value['total']:<7d} ({value['rate']:.1%})")

