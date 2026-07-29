"""Shared \\boxed{...} extractor used by math + countdown scorers.

Leading underscore in the filename marks this as package-internal:
callers should use `grpo_reasoner.scoring.extract_answer(data_source, ...)`
instead of importing from here directly.
"""
from __future__ import annotations

_BS = chr(92)
_BOXED = _BS + "boxed"
def last_boxed(text: str) -> str | None:
    """Return the content inside the LAST \boxed{...} in text, or None.
    Handles nested braces via depth counting; tolerates a single space
    between \boxed and {.
    """
    idx = text.rfind(_BOXED)
    if idx < 0:
        return None
    i = idx + len(_BOXED)
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    i += 1
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None
def has_boxed(text: str) -> bool:
    return last_boxed(text) is not None