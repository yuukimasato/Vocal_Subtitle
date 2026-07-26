"""Final subtitle event validation for phase four."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .event_constraints import _clip_id, _span_end, _span_start
from .time_mapper import SubtitleEvent


@dataclass
class FinalValidationResult:
    events: list[SubtitleEvent]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_events(
    events: Sequence[SubtitleEvent],
    *,
    audio_duration: float | None = None,
    strict: bool = True,
) -> FinalValidationResult:
    diagnostics: dict[str, Any] = {
        "input_event_count": len(events),
        "output_event_count": 0,
        "physical_clamped_count": 0,
        "removed_count": 0,
        "reasons": {},
        "warning_count": 0,
        "overlap_count": 0,
        "overlap_trimmed_count": 0,
    }
    result: list[SubtitleEvent] = []
    for event in sorted(events, key=lambda item: (item.start, item.end, item.index)):
        reason = _validate_event(event, audio_duration, strict)
        if reason:
            diagnostics["removed_count"] += 1
            diagnostics["reasons"][reason] = diagnostics["reasons"].get(reason, 0) + 1
            continue

        original = (event.start, event.end)
        physical_start = getattr(event, "physical_start", None)
        physical_end = getattr(event, "physical_end", None)
        if physical_start is not None:
            event.start = max(event.start, float(physical_start))
        if physical_end is not None:
            event.end = min(event.end, float(physical_end))
        if audio_duration is not None:
            event.start = max(0.0, min(float(audio_duration), event.start))
            event.end = max(0.0, min(float(audio_duration), event.end))
        else:
            event.start = max(0.0, event.start)
        if (event.start, event.end) != original:
            diagnostics["physical_clamped_count"] += 1
        if event.end <= event.start:
            diagnostics["removed_count"] += 1
            diagnostics["reasons"]["empty_after_clamp"] = diagnostics["reasons"].get("empty_after_clamp", 0) + 1
            continue
        if getattr(event, "alignment_warning", None):
            diagnostics["warning_count"] += len(str(event.alignment_warning).split(";"))
        result.append(event)

    # Quantized subtitle formats can turn adjacent floating-point intervals
    # into small overlaps. Trim the previous event after all physical clamps
    # so the serialized sequence is strictly ordered.
    ordered: list[SubtitleEvent] = []
    for event in result:
        if ordered and event.start < ordered[-1].end:
            diagnostics["overlap_count"] += 1
            previous = ordered[-1]
            previous.end = event.start
            diagnostics["overlap_trimmed_count"] += 1
            if previous.end <= previous.start:
                ordered.pop()
        ordered.append(event)

    result = ordered
    for index, event in enumerate(result, start=1):
        event.index = index
    diagnostics["output_event_count"] = len(result)
    return FinalValidationResult(result, diagnostics)


def _validate_event(
    event: SubtitleEvent,
    audio_duration: float | None,
    strict: bool,
) -> str | None:
    try:
        start = float(event.start)
        end = float(event.end)
    except (TypeError, ValueError):
        return "invalid_time"
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return "invalid_time"
    if start < 0 and end <= 0:
        return "audio_bounds"
    if audio_duration is not None and start >= audio_duration:
        return "audio_bounds"

    physical_start = getattr(event, "physical_start", None)
    physical_end = getattr(event, "physical_end", None)
    if physical_start is not None and physical_end is not None:
        if float(physical_end) <= float(physical_start):
            return "invalid_physical_envelope"
        if strict and (end <= float(physical_start) or start >= float(physical_end)):
            return "outside_physical_envelope"

    spans = list(getattr(event, "physical_spans", None) or [])
    for span in spans:
        try:
            if not _clip_id(span) or _span_end(span) <= _span_start(span):
                return "invalid_physical_span"
        except (TypeError, ValueError):
            return "invalid_physical_span"

    source_ids = list(getattr(event, "source_word_ids", None) or [])
    words = list(getattr(event, "words", None) or [])
    word_ids = {str(getattr(word, "id")) for word in words if getattr(word, "id", None) is not None}
    if strict and word_ids and any(source_id not in word_ids for source_id in source_ids):
        return "dangling_source_word"
    if strict and not str(getattr(event, "text", "")).strip():
        return "empty_text"
    return None
