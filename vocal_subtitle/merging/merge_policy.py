"""Deterministic semantic boundary policy for merge fallback paths."""

from __future__ import annotations

import re

SECTION_START_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d+[\.\)]\s"),
    re.compile(r"^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)[,.\s]"),
    re.compile(r"^[一二三四五六七八九十][、，.]"),
    re.compile(r"^(Summary\s+(and|&)\s+review)", re.IGNORECASE),
    re.compile(r"^(Example[:]?)", re.IGNORECASE),
    re.compile(r"^(Effective\s+Communication)", re.IGNORECASE),
    re.compile(r"^(Phone\s+Etiquette|Rapid\s+Response)", re.IGNORECASE),
    re.compile(r"^(Answer|Listen|Hang\s+up|Identify)", re.IGNORECASE),
    re.compile(r"^(End\s+with\s+courtesy)", re.IGNORECASE),
]
SECTION_END_MARKERS: list[re.Pattern] = [
    re.compile(r"(^|\s)(and|with)\s+courtesy[.]?\s*$", re.IGNORECASE),
]


def detect_semantic_boundary(current_text: str, next_text: str) -> bool:
    next_stripped = next_text.strip()
    if any(pattern.match(next_stripped) for pattern in SECTION_START_PATTERNS):
        return True
    current_stripped = current_text.strip()
    return any(pattern.search(current_stripped) for pattern in SECTION_END_MARKERS)


_detect_semantic_boundary = detect_semantic_boundary
