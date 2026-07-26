"""Conservative text-only safety gate for LLM subtitle edits."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable


def validate_llm_text(
    original: str,
    candidate: str,
    *,
    peer_texts: Iterable[str] = (),
    min_similarity: float = 0.75,
    max_length_ratio: float = 1.35,
) -> tuple[bool, str]:
    """Allow local corrections while rejecting translation/cross-item moves."""
    source = str(original or "").strip()
    value = str(candidate or "").strip()
    if not source or not value:
        return False, "empty_text"
    if _script(source) != _script(value):
        return False, "script_changed"
    similarity = SequenceMatcher(None, source.casefold(), value.casefold()).ratio()
    if similarity < min_similarity:
        return False, "similarity_below_threshold"
    if len(value) > max(1, int(len(source) * max_length_ratio)):
        return False, "length_ratio_exceeded"
    if _numbers(source) != _numbers(value):
        return False, "protected_numbers_changed"
    normalized = _normalize(value)
    for peer in peer_texts:
        peer_normalized = _normalize(peer)
        if len(peer_normalized) >= 3 and peer_normalized in normalized:
            return False, "cross_event_text"
    return True, "accepted"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def _numbers(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+(?:[.,:]\d+)*", str(value)))


def _script(value: str) -> str:
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in value)
    latin = sum(("a" <= char.lower() <= "z") for char in value)
    if cjk and cjk >= latin:
        return "cjk"
    if latin:
        return "latin"
    return "other"
