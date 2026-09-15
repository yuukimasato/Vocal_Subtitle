"""Deterministic expected/predicted match classification for quality gates.

Shared by the golden gate and provenance attribution so both apply one match
policy. Kept free of imports from other quality modules to preserve a single
dependency direction: matching <- provenance <- golden_gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

LEGACY_MATCH_POLICY_VERSION = "legacy-overlap-v1"
STRICT_MATCH_POLICY_VERSION = "strict-overlap-v1"
DEFAULT_STRICT_MIN_OVERLAP_SECONDS = 0.01


def _text(value: Any) -> str:
    return "".join(str(value or "").split()).casefold()


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    try:
        return max(
            0.0,
            min(float(left.get("end", 0)), float(right.get("end", 0)))
            - max(float(left.get("start", 0)), float(right.get("start", 0))),
        )
    except (TypeError, ValueError):
        return 0.0


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _text(left), _text(right), autojunk=False).ratio()


def _overlap_ratio(expected: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    try:
        duration = max(
            0.0, float(expected.get("end", 0)) - float(expected.get("start", 0))
        )
    except (TypeError, ValueError):
        return 0.0
    return _overlap(expected, candidate) / duration if duration else 0.0


def _temporal_gap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    try:
        left_start = float(left.get("start", 0))
        left_end = float(left.get("end", 0))
        right_start = float(right.get("start", 0))
        right_end = float(right.get("end", 0))
    except (TypeError, ValueError):
        return float("inf")
    if _overlap(left, right) > 0:
        return 0.0
    return min(abs(right_start - left_end), abs(left_start - right_end))


def _candidate_summary(
    expected: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "start": candidate.get("start"),
        "end": candidate.get("end"),
        "text": candidate.get("text", ""),
        "overlap_seconds": round(_overlap(expected, candidate), 6),
        "overlap_ratio": round(_overlap_ratio(expected, candidate), 6),
        "text_similarity": round(
            _similarity(expected.get("text", ""), candidate.get("text", "")),
            6,
        ),
        "temporal_gap_seconds": round(_temporal_gap(expected, candidate), 6),
    }


def classify_expected_match(
    expected: Mapping[str, Any],
    predicted: Sequence[Mapping[str, Any]],
    *,
    min_similarity: float = 0.35,
    nearby_gap_seconds: float = 2.0,
    min_overlap_seconds: float = 0.0,
    min_overlap_ratio: float = 0.0,
    match_policy: str = LEGACY_MATCH_POLICY_VERSION,
) -> dict[str, Any]:
    """Classify one expected event without changing the gate threshold.

    ``legacy-overlap-v1`` keeps the historical positive-overlap behavior.
    Strict callers can require a minimum overlap duration and ratio without
    changing the legacy result used by the release gate.
    """
    if min_overlap_seconds < 0:
        raise ValueError("min_overlap_seconds must be non-negative")
    if not 0.0 <= min_overlap_ratio <= 1.0:
        raise ValueError("min_overlap_ratio must be between 0 and 1")
    candidates = [item for item in predicted if isinstance(item, Mapping)]
    overlap_ranked = sorted(
        candidates,
        key=lambda item: (
            _overlap(expected, item),
            _similarity(expected.get("text", ""), item.get("text", "")),
        ),
        reverse=True,
    )
    text_ranked = sorted(
        candidates,
        key=lambda item: (
            _similarity(expected.get("text", ""), item.get("text", "")),
            -_temporal_gap(expected, item),
            _overlap(expected, item),
        ),
        reverse=True,
    )
    best_overlap = overlap_ranked[0] if overlap_ranked else None
    best_text = text_ranked[0] if text_ranked else None
    gate_matches = [
        item
        for item in candidates
        if _overlap(expected, item) >= min_overlap_seconds
        and _overlap(expected, item) > 0
        and _overlap_ratio(expected, item) >= min_overlap_ratio
        and _similarity(expected.get("text", ""), item.get("text", ""))
        >= min_similarity
    ]
    if gate_matches:
        best_overlap = max(
            gate_matches,
            key=lambda item: (
                _overlap(expected, item),
                _similarity(expected.get("text", ""), item.get("text", "")),
            ),
        )
    best_overlap_seconds = _overlap(expected, best_overlap) if best_overlap else 0.0
    best_overlap_similarity = (
        _similarity(expected.get("text", ""), best_overlap.get("text", ""))
        if best_overlap
        else 0.0
    )
    matched = bool(gate_matches)
    if matched:
        stage = "matched"
    elif best_overlap_seconds > 0:
        stage = (
            "strict_overlap_below_minimum"
            if match_policy == STRICT_MATCH_POLICY_VERSION
            and (
                best_overlap_seconds < min_overlap_seconds
                or _overlap_ratio(expected, best_overlap or {}) < min_overlap_ratio
            )
            and best_overlap_similarity >= min_similarity
            else "asr_text_mismatch"
        )
    elif (
        best_text
        and _similarity(expected.get("text", ""), best_text.get("text", ""))
        >= min_similarity
    ):
        stage = (
            "timeline_shift_or_boundary_mismatch"
            if _temporal_gap(expected, best_text) <= nearby_gap_seconds
            else "speech_candidate_missing"
        )
    elif candidates:
        stage = "speech_candidate_missing"
    else:
        stage = "no_predicted_events"
    return {
        "matched": matched,
        "stage": stage,
        "match_policy": match_policy,
        "match_requirements": {
            "min_similarity": min_similarity,
            "min_overlap_seconds": min_overlap_seconds,
            "min_overlap_ratio": min_overlap_ratio,
        },
        "expected": {
            "start": expected.get("start"),
            "end": expected.get("end"),
            "text": expected.get("text", ""),
            "kind": expected.get("kind", "speech"),
        },
        "best_overlap": _candidate_summary(expected, best_overlap),
        "best_text": _candidate_summary(expected, best_text),
    }


__all__ = [
    "DEFAULT_STRICT_MIN_OVERLAP_SECONDS",
    "LEGACY_MATCH_POLICY_VERSION",
    "STRICT_MATCH_POLICY_VERSION",
    "classify_expected_match",
]
