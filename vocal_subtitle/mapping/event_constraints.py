"""Shared safety rules for subtitle event composition."""

from __future__ import annotations

from typing import Any


HARD_BOUNDARY_WARNINGS = frozenset(
    {
        "physical_envelope_conflict",
        "discontinuous_physical_boundary",
        "speaker_conflict",
        "outside_physical_clip",
        "hard_split",
    }
)


def speaker_compatible(left: Any, right: Any) -> bool:
    """Require equal known IDs; unknown cannot absorb a known speaker."""
    left_id = _value(left, "speaker_id", None)
    right_id = _value(right, "speaker_id", None)
    return left_id == right_id


def physical_owner_compatible(
    left: Any,
    right: Any,
    *,
    max_gap: float = 0.02,
) -> bool:
    """Check that physical spans overlap or meet without a discontinuity."""
    left_spans = _spans(left)
    right_spans = _spans(right)
    left_region = _value(left, "physical_region_id", None)
    right_region = _value(right, "physical_region_id", None)
    if left_region is not None or right_region is not None:
        if left_region != right_region:
            return False
    if not left_spans and not right_spans:
        return True
    if not left_spans or not right_spans:
        return False

    left_clip_ids = {_clip_id(span) for span in left_spans if _clip_id(span)}
    right_clip_ids = {_clip_id(span) for span in right_spans if _clip_id(span)}
    if left_clip_ids and right_clip_ids and not left_clip_ids.intersection(right_clip_ids):
        return False

    ordered = sorted(
        [*left_spans, *right_spans],
        key=lambda span: (_span_start(span), _span_end(span), _clip_id(span)),
    )
    return all(
        _span_start(next_span) - _span_end(span) <= max_gap
        for span, next_span in zip(ordered, ordered[1:])
    )


def warnings_compatible(left: Any, right: Any) -> bool:
    if bool(_value(left, "hard_split_after", False)) or bool(
        _value(right, "hard_split_before", False)
    ):
        return False
    warnings = _warning_set(left) | _warning_set(right)
    return not warnings.intersection(HARD_BOUNDARY_WARNINGS)


def can_merge_records(
    left: Any,
    right: Any,
    *,
    max_gap: float | None = None,
    max_duration: float | None = None,
) -> bool:
    """Dict/object variant used by the LLM merge engine."""
    if not speaker_compatible(left, right):
        return False
    if not warnings_compatible(left, right):
        return False
    physical_gap = max_gap if max_gap is not None else 0.02
    if not physical_owner_compatible(left, right, max_gap=physical_gap):
        return False
    left_end = float(_value(left, "end", 0.0))
    right_start = float(_value(right, "start", 0.0))
    gap = right_start - left_end
    if max_gap is not None and gap > max_gap:
        return False
    if max_duration is not None:
        start = min(float(_value(left, "start", 0.0)), right_start)
        end = max(left_end, float(_value(right, "end", 0.0)))
        if end - start > max_duration:
            return False
    return True


def can_merge_events(
    left: Any,
    right: Any,
    *,
    max_gap: float | None = None,
    max_duration: float | None = None,
) -> bool:
    """Return whether a merge preserves phase-three ownership invariants."""
    return can_merge_records(
        left,
        right,
        max_gap=max_gap,
        max_duration=max_duration,
    )


def merge_event_metadata(left: Any, right: Any) -> None:
    """Merge phase-three provenance fields in place."""
    left_words = list(getattr(left, "words", []) or [])
    right_words = list(getattr(right, "words", []) or [])
    if right_words:
        left.words = [*left_words, *right_words]

    left_ids = list(getattr(left, "source_word_ids", []) or [])
    right_ids = list(getattr(right, "source_word_ids", []) or [])
    left.source_word_ids = list(dict.fromkeys([*left_ids, *right_ids]))

    left_spans = list(getattr(left, "physical_spans", []) or [])
    right_spans = list(getattr(right, "physical_spans", []) or [])
    left.physical_spans = [*left_spans, *right_spans]

    warning_values = [
        getattr(left, "alignment_warning", None),
        getattr(right, "alignment_warning", None),
    ]
    warnings = [
        item
        for value in warning_values
        if value
        for item in str(value).split(";")
        if item
    ]
    left.alignment_warning = ";".join(dict.fromkeys(warnings)) or None

    if getattr(left, "physical_start", None) is None:
        left.physical_start = getattr(right, "physical_start", None)
    elif getattr(right, "physical_start", None) is not None:
        left.physical_start = min(left.physical_start, right.physical_start)
    if getattr(left, "physical_end", None) is None:
        left.physical_end = getattr(right, "physical_end", None)
    elif getattr(right, "physical_end", None) is not None:
        left.physical_end = max(left.physical_end, right.physical_end)
    if getattr(left, "physical_region_id", None) is None:
        left.physical_region_id = getattr(right, "physical_region_id", None)
    if getattr(left, "physical_bin_id", None) and getattr(right, "physical_bin_id", None):
        if left.physical_bin_id != right.physical_bin_id:
            left.physical_bin_id = f"{left.physical_bin_id}+{right.physical_bin_id}"
    elif getattr(left, "physical_bin_id", None) is None:
        left.physical_bin_id = getattr(right, "physical_bin_id", None)
    if getattr(left, "physical_bin_start", None) is None:
        left.physical_bin_start = getattr(right, "physical_bin_start", None)
    elif getattr(right, "physical_bin_start", None) is not None:
        left.physical_bin_start = min(left.physical_bin_start, right.physical_bin_start)
    if getattr(left, "physical_bin_end", None) is None:
        left.physical_bin_end = getattr(right, "physical_bin_end", None)
    elif getattr(right, "physical_bin_end", None) is not None:
        left.physical_bin_end = max(left.physical_bin_end, right.physical_bin_end)
    left_source = getattr(left, "time_source", None)
    right_source = getattr(right, "time_source", None)
    if left_source != right_source and right_source:
        left.time_source = "merged"


def _spans(event: Any) -> list[Any]:
    return list(_value(event, "physical_spans", None) or [])


def _clip_id(span: Any) -> str:
    if isinstance(span, dict):
        return str(span.get("physical_clip_id", span.get("clip_id", "")))
    return str(getattr(span, "clip_id", ""))


def _span_start(span: Any) -> float:
    if isinstance(span, dict):
        return float(span.get("start", 0.0))
    return float(getattr(span, "start", 0.0))


def _span_end(span: Any) -> float:
    if isinstance(span, dict):
        return float(span.get("end", 0.0))
    return float(getattr(span, "end", 0.0))


def _warning_set(event: Any) -> set[str]:
    value = _value(event, "alignment_warning", None)
    return {item for item in str(value or "").split(";") if item}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)
