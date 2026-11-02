"""Post-generation grounding check for numeric claims.

Every number stated in an answer should be traceable to the evidence (or the question).
Numbers are compared by value, so "180,895.90", "$180,895.9" and "180895.9" match, and
"31%" matches "31". Citation markers like [2] are ignored. A number that appears nowhere
in the evidence is reported as unsupported. It may be a hallucination, or a value the
model derived, such as a ratio, which a reader should double-check.
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[(?:\d{1,3}|SQL)\]")
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    values = []
    for token in _NUMBER.findall(_CITATION.sub(" ", text)):
        cleaned = token.replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def _matches(value: float, pool: list[float]) -> bool:
    return any(abs(value - p) <= max(0.011, abs(p) * 1e-4) for p in pool)


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-*])|\n+")
_CITED = re.compile(r"\[(\d{1,3}|SQL)\]")


def _check(text: str, pool: list[float], unsupported: list[str]) -> None:
    for token in _NUMBER.findall(_CITATION.sub(" ", text)):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        # Small ordinals ("1." list markers, "top 3") are not factual claims worth flagging.
        if abs(value) < 10 and value.is_integer():
            continue
        if not _matches(value, pool) and token not in unsupported:
            unsupported.append(token)


def unsupported_numbers(
    answer: str,
    evidence: str,
    question: str = "",
    sources: dict[str, str] | None = None,
) -> list[str]:
    """Return numbers in ``answer`` (as written) that the evidence does not support.

    With ``sources`` ({"1": chunk text, ..., "SQL": result text}) the check is
    citation-aware: a sentence citing [n] must find its numbers in source n (or the
    question). Sentences without citations are checked against all evidence.
    """
    question_pool = extract_numbers(question)
    all_pool = extract_numbers(evidence) + question_pool
    unsupported: list[str] = []
    for sentence in filter(None, (s.strip() for s in _SENTENCE.split(answer))):
        cited = [c for c in _CITED.findall(sentence) if sources and c in sources]
        pool = question_pool + [v for c in cited for v in extract_numbers(sources[c])] if cited else all_pool
        _check(sentence, pool, unsupported)
    return unsupported
