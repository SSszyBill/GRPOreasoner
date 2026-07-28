"""Problem decontamination helpers for NuminaMath."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def norm_exact(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def tokens(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def ngrams(toks: list[str], n: int) -> list[str]:
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def get_field(extra_info, key: str) -> str:
    if isinstance(extra_info, dict):
        return extra_info.get(key, "") or ""
    return ""


def build_reference(ref_specs: list[tuple[str | Path, str, str]], n: int, logger=None) -> tuple[dict[str, str], dict[str, str]]:
    """Build exact and n-gram references from small eval parquets.

    `ref_specs` entries are `(path, problem_key, tag)`.
    """
    exact_ref: dict[str, str] = {}
    gram_to_eval: dict[str, str] = {}
    for path, key, tag in ref_specs:
        tbl = pq.read_table(path, columns=["extra_info"])
        count = 0
        for i, row in enumerate(tbl.to_pylist()):
            prob = get_field(row.get("extra_info"), key)
            if not prob:
                continue
            eid = f"{tag}:{i}"
            exact_ref.setdefault(norm_exact(prob), eid)
            for gram in ngrams(tokens(prob), n):
                gram_to_eval.setdefault(gram, eid)
            count += 1
        if logger is not None:
            logger.info(f"reference {tag}: {count} problems from {path}")
    if logger is not None:
        logger.info(f"references total: exact={len(exact_ref)}, {n}-gram={len(gram_to_eval)}")
    return exact_ref, gram_to_eval


def classify(
    prob: str,
    exact_ref: dict[str, str],
    gram_to_eval: dict[str, str],
    gram_set: set[str],
    n: int,
    threshold: float,
) -> tuple[bool, str, float, str | None]:
    ne = norm_exact(prob)
    if ne and ne in exact_ref:
        return True, "exact", 1.0, exact_ref[ne]
    grams = ngrams(tokens(prob), n)
    if not grams:
        return False, "", 0.0, None
    matched = [gram for gram in grams if gram in gram_set]
    hit = len(matched) / len(grams)
    if hit > threshold:
        eid = Counter(gram_to_eval[gram] for gram in matched).most_common(1)[0][0]
        return True, "ngram", hit, eid
    return False, "", hit, None

