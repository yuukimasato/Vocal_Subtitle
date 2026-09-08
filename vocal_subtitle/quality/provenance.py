"""Shared provenance and miss attribution helpers for offline reports.

The production chain intentionally keeps provenance as serializable mappings.
This module provides the stable vocabulary and deterministic attribution rules
without coupling report generation to an ASR implementation.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

from .matching import classify_expected_match


TRACE_SCHEMA_VERSION = "offline-trace-v1"
TRACE_KEYS = (
    "source_id",
    "offset_id",
    "window_id",
    "candidate_id",
    "physical_span_ids",
    "decision_id",
    "final_event_ids",
)


def as_mapping(item: Any) -> dict[str, Any]:
    """Return a defensive mapping for a serialized object or dataclass."""
    if isinstance(item, Mapping):
        return dict(item)
    serializer = getattr(item, "to_dict", None)
    if callable(serializer):
        value = serializer()
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def normalize_trace_context(value: Mapping[str, Any] | None = None, **updates: Any) -> dict[str, Any]:
    """Normalize a trace context while retaining forward-compatible fields."""
    context = dict(value or {})
    context.update({key: item for key, item in updates.items() if item is not None})
    context.setdefault("schema_version", TRACE_SCHEMA_VERSION)
    context.setdefault("source_id", None)
    context.setdefault("offset_id", None)
    context.setdefault("window_id", None)
    context.setdefault("candidate_id", None)
    context.setdefault("physical_span_ids", [])
    context.setdefault("decision_id", None)
    context.setdefault("final_event_ids", [])
    # Span payloads from older producers may carry full span dicts instead of
    # id strings; recover the clip/evidence ids so ids stay hashable.
    span_ids: list[str] = []
    for item in context["physical_span_ids"] or []:
        if isinstance(item, str) and item:
            span_ids.append(item)
        elif isinstance(item, Mapping):
            recovered = item.get("physical_clip_id") or item.get("id")
            if recovered:
                span_ids.append(str(recovered))
            else:
                span_ids.extend(
                    str(evidence_id)
                    for evidence_id in item.get("evidence_ids", ()) or ()
                    if evidence_id
                )
    context["physical_span_ids"] = list(dict.fromkeys(span_ids))
    context["final_event_ids"] = list(dict.fromkeys(context["final_event_ids"] or []))
    return context


def trace_context_for(item: Any, *, source: str | None = None) -> dict[str, Any]:
    """Extract the common trace context from a candidate, decision, or event."""
    payload = as_mapping(item)
    context = payload.get("trace_context")
    context = dict(context) if isinstance(context, Mapping) else {}
    if source and not context.get("source_id"):
        context["source_id"] = source
    if not context.get("candidate_id"):
        context["candidate_id"] = payload.get("id")
    if not context.get("decision_id"):
        context["decision_id"] = payload.get("decision_id")
    physical = payload.get("physical_span_ids")
    if physical and not context.get("physical_span_ids"):
        context["physical_span_ids"] = list(physical)
    return normalize_trace_context(context)


def _range(item: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    first = _range(left)
    second = _range(right)
    if first is None or second is None:
        return 0.0
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _gap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    first = _range(left)
    second = _range(right)
    if first is None or second is None:
        return float("inf")
    if _overlap(left, right) > 0:
        return 0.0
    return min(abs(first[0] - second[1]), abs(second[0] - first[1]))


def _similarity(left: Any, right: Any) -> float:
    normalize = lambda value: "".join(str(value or "").split()).casefold()
    return SequenceMatcher(None, normalize(left), normalize(right), autojunk=False).ratio()


def _related(expected: Mapping[str, Any], items: Iterable[Any], *, gap: float = 0.35) -> list[dict[str, Any]]:
    expected_id = expected.get("id")
    result = []
    for raw in items:
        item = as_mapping(raw)
        context = trace_context_for(item)
        refs = set(context.get("physical_span_ids", []))
        if expected_id and expected_id in refs:
            result.append(item)
            continue
        if _overlap(expected, item) > 0 or _gap(expected, item) <= gap:
            result.append(item)
    return result


def _ids(items: Iterable[Mapping[str, Any]], key: str = "id") -> list[str]:
    return [str(item[key]) for item in items if item.get(key) is not None]


def attribute_expected_miss(
    expected: Mapping[str, Any],
    predicted: Sequence[Mapping[str, Any]],
    *,
    physical_spans: Sequence[Any] = (),
    candidates: Sequence[Any] = (),
    decisions: Sequence[Any] = (),
    final_events: Sequence[Any] | None = None,
    match_result: Mapping[str, Any] | None = None,
    strict_match_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign one auditable primary stage to an unmatched expected event."""
    legacy = dict(match_result or classify_expected_match(expected, predicted))
    strict = dict(strict_match_result or {})
    if not strict:
        strict = classify_expected_match(
            expected,
            predicted,
            min_overlap_seconds=0.15,
            min_overlap_ratio=0.3,
            match_policy="strict-overlap-v1",
        )
    physical = _related(expected, physical_spans, gap=0.05)
    candidate_items = _related(expected, candidates)
    decision_items = _related(expected, decisions)
    final_items = _related(expected, final_events if final_events is not None else predicted)

    if legacy.get("matched") and not strict.get("matched"):
        stage = "strict_match_only_failure"
    elif physical_spans and not physical:
        stage = "physical_coverage_missing"
    elif physical and not candidate_items:
        stage = "speech_candidate_missing"
    elif candidate_items and not any(
        _similarity(expected.get("text"), item.get("text", item.get("final_text", ""))) >= 0.35
        for item in candidate_items
    ):
        stage = "asr_text_mismatch"
    elif candidate_items and not decision_items:
        stage = "decision_drop_or_unresolved"
    elif decision_items and not final_items:
        stage = "postprocess_coverage_loss"
    elif not physical and not physical_spans and not candidate_items and not decision_items:
        stage = "unknown_with_evidence_gap"
    elif not legacy.get("matched"):
        stage = str(legacy.get("stage") or "unknown_with_evidence_gap")
    else:
        stage = "unknown_with_evidence_gap"

    return {
        "expected_event_id": expected.get("id"),
        "start": expected.get("start"),
        "end": expected.get("end"),
        "text": expected.get("text", ""),
        "stage": stage,
        "physical_span_ids": _ids(physical),
        "candidate_ids": _ids(candidate_items),
        "decision_ids": [
            str(item.get("decision_id") or item.get("id"))
            for item in decision_items
            if item.get("decision_id") is not None or item.get("id") is not None
        ],
        "final_event_ids": _ids(final_items),
        "best_overlap": legacy.get("best_overlap"),
        "best_text": legacy.get("best_text"),
        "strict_match": strict,
        "legacy_match": legacy,
        "evidence_status": {
            "physical": "present" if physical_spans else "missing",
            "candidate": "present" if candidate_items else "missing",
            "decision": "present" if decision_items else "missing",
            "final_event": "present" if final_items else "missing",
        },
    }


def attribute_case_misses(
    expected_events: Sequence[Mapping[str, Any]],
    predicted_events: Sequence[Mapping[str, Any]],
    *,
    physical_spans: Sequence[Any] = (),
    candidates: Sequence[Any] = (),
    decisions: Sequence[Any] = (),
    final_events: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Return detailed attribution records for legacy and strict misses."""
    result = []
    for expected in expected_events:
        if str(expected.get("kind", "speech")).casefold() in {
            "non_speech", "non-speech", "noise", "hallucination",
        }:
            continue
        legacy = classify_expected_match(expected, predicted_events)
        strict = classify_expected_match(
            expected,
            predicted_events,
            min_overlap_seconds=0.15,
            min_overlap_ratio=0.3,
            match_policy="strict-overlap-v1",
        )
        if not legacy.get("matched") or not strict.get("matched"):
            result.append(attribute_expected_miss(
                expected,
                predicted_events,
                physical_spans=physical_spans,
                candidates=candidates,
                decisions=decisions,
                final_events=final_events,
                match_result=legacy,
                strict_match_result=strict,
            ))
    return result


def attribution_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count primary attribution stages in a stable order."""
    return dict(Counter(str(item.get("stage", "unknown_with_evidence_gap")) for item in records))


__all__ = [
    "TRACE_KEYS",
    "TRACE_SCHEMA_VERSION",
    "as_mapping",
    "attribute_case_misses",
    "attribute_expected_miss",
    "attribution_counts",
    "normalize_trace_context",
    "trace_context_for",
]
