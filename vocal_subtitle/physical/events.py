"""Event construction from physically allocated global words."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..asr.base import WordTimestamp
from ..mapping.time_mapper import SubtitleEvent
from .allocator import AllocationResult, PhysicalSpan, WordAllocation
from .subtitle_bins import (
    PhysicalSubtitleBin,
    assign_word_to_bin,
)


@dataclass
class GlobalSubtitleEvent:
    index: int
    start: float
    end: float
    text: str
    words: list[Any] = field(default_factory=list)
    speaker_id: int | None = None
    speaker_status: str = "unknown"
    speaker_source: str = "unknown"
    physical_spans: list[PhysicalSpan] = field(default_factory=list)
    source_word_ids: list[str] = field(default_factory=list)
    logical_sentence_id: int | None = None
    alignment_warning: str | None = None
    hard_split_before: bool = False
    physical_bin_id: str | None = None
    physical_bin_start: float | None = None
    physical_bin_end: float | None = None
    time_source: str = "asr_word"
    speaker_split_degraded: bool = False
    revision_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "source_word_ids": list(self.source_word_ids),
            "speaker_id": self.speaker_id,
            "speaker_status": self.speaker_status,
            "speaker_source": self.speaker_source,
            "physical_spans": [span.to_dict() for span in self.physical_spans],
            "logical_sentence_id": self.logical_sentence_id,
            "alignment_warning": self.alignment_warning,
            "hard_split_before": self.hard_split_before,
            "physical_bin_id": self.physical_bin_id,
            "physical_bin_start": self.physical_bin_start,
            "physical_bin_end": self.physical_bin_end,
            "time_source": self.time_source,
            "revision_trace": list(self.revision_trace),
        }

    def to_subtitle_event(self) -> SubtitleEvent:
        physical_start = self.start
        physical_end = self.end
        relative_words = [
            WordTimestamp(
                word=word.text,
                start=word.raw_start - self.start,
                end=word.raw_end - self.start,
                # Preserve missing confidence for the evidence adapter. The
                # global event is a compatibility observation, not a calibrated
                # subtitle confidence source.
                confidence=word.confidence,
                speaker_id=word.speaker_id,
            )
            for word in self.words
        ]
        return SubtitleEvent(
            index=self.index,
            start=self.start,
            end=self.end,
            text=self.text,
            words=relative_words,
            asr_text=self.text,
            speaker_id=self.speaker_id,
            speaker_source=self.speaker_source,
            physical_start=physical_start,
            physical_end=physical_end,
            physical_spans=[span.to_dict() for span in self.physical_spans],
            source_word_ids=list(self.source_word_ids),
            logical_sentence_id=self.logical_sentence_id,
            alignment_warning=self.alignment_warning,
            hard_split_before=self.hard_split_before,
            physical_bin_id=self.physical_bin_id,
            physical_bin_start=self.physical_bin_start,
            physical_bin_end=self.physical_bin_end,
            time_source=self.time_source,
            speaker_split_degraded=self.speaker_split_degraded,
            revision_trace=list(self.revision_trace),
        )


def build_events(
    allocation: AllocationResult,
    *,
    max_word_gap: float = 0.8,
    max_evidence_gap: float = 0.35,
    subtitle_bins: Sequence[PhysicalSubtitleBin] | None = None,
) -> list[GlobalSubtitleEvent]:
    """Build events without splitting individual words."""
    if not isinstance(allocation, AllocationResult):
        raise ValueError("allocation must be an AllocationResult")
    if max_word_gap < 0:
        raise ValueError("max_word_gap must be non-negative")
    if max_evidence_gap < 0:
        raise ValueError("max_evidence_gap must be non-negative")

    if subtitle_bins:
        return _build_bin_events(
            allocation,
            subtitle_bins,
            max_word_gap=max_word_gap,
            max_evidence_gap=max_evidence_gap,
        )

    events: list[GlobalSubtitleEvent] = []
    current: list[WordAllocation] = []
    for item in allocation.accepted:
        if current and not _can_append(
            current[-1], item, max_word_gap, max_evidence_gap
        ):
            events.append(_make_event(len(events) + 1, current))
            current = []
        current.append(item)
    if current:
        events.append(_make_event(len(events) + 1, current))
    return events


def _build_bin_events(
    allocation: AllocationResult,
    subtitle_bins: Sequence[PhysicalSubtitleBin],
    *,
    max_word_gap: float,
    max_evidence_gap: float,
) -> list[GlobalSubtitleEvent]:
    """Aggregate accepted words by physical bin before building events.

    Evidence decisions are an internal review unit, not a subtitle unit. A
    single acoustic bin can therefore contain words from several decisions;
    grouping by decision here would leak those internal boundaries as
    word-level subtitles. Only hard physical or speaker boundaries may split
    a bin.
    """
    ordered_bins = sorted(
        subtitle_bins, key=lambda item: (item.start, item.end, item.id)
    )
    grouped: dict[str, tuple[PhysicalSubtitleBin, list[WordAllocation]]] = {}
    for item in allocation.accepted:
        assigned = assign_word_to_bin(item.word, ordered_bins)
        if assigned is None:
            # With a physical-bin projection, an accepted word without a bin
            # has no safe subtitle container and must not bypass the skeleton.
            continue
        if assigned.id not in grouped:
            grouped[assigned.id] = (assigned, [])
        grouped[assigned.id][1].append(item)

    events: list[GlobalSubtitleEvent] = []
    for bin_item, items in sorted(
        grouped.values(), key=lambda value: (value[0].start, value[0].end, value[0].id)
    ):
        items = _deduplicate_bin_items(items)
        for group in _split_bin_hard_boundaries(items):
            events.append(
                _make_event(
                    len(events) + 1,
                    group,
                    subtitle_bin=bin_item,
                )
            )
    return events


def _deduplicate_bin_items(items: Sequence[WordAllocation]) -> list[WordAllocation]:
    """Remove repeated evidence observations of the same spoken word."""
    ordered = sorted(
        items, key=lambda item: (item.word.raw_start, item.word.raw_end, item.word.id)
    )
    result: list[WordAllocation] = []
    seen_source_ids: set[str] = set()
    for item in ordered:
        source_id = str(item.word.metadata.get("source_word_id", item.word.id))
        if source_id in seen_source_ids:
            continue
        text = _normalize_word_text(item.word.text)
        duplicate = False
        for previous in result:
            if text != _normalize_word_text(previous.word.text):
                continue
            overlap = min(item.word.raw_end, previous.word.raw_end) - max(
                item.word.raw_start, previous.word.raw_start
            )
            shorter = min(
                item.word.raw_end - item.word.raw_start,
                previous.word.raw_end - previous.word.raw_start,
            )
            if shorter > 0 and overlap / shorter >= 0.7:
                duplicate = True
                break
        if duplicate:
            continue
        seen_source_ids.add(source_id)
        result.append(item)
    return result


def _split_bin_hard_boundaries(
    items: Sequence[WordAllocation],
) -> list[list[WordAllocation]]:
    """Split at ownership changes and every material physical gap.

    A subtitle event is allowed to contain adjacent words, but it must not
    claim a physically silent interval.  The bin itself may be broad enough
    to contain several words, so the event boundary must use the allocated
    physical spans rather than the bin envelope.
    """
    groups: list[list[WordAllocation]] = []
    for item in sorted(
        items,
        key=lambda value: (value.word.raw_start, value.word.raw_end, value.word.id),
    ):
        if not groups or _has_bin_hard_boundary(groups[-1][-1], item):
            groups.append([])
        groups[-1].append(item)
    return groups


def _has_bin_hard_boundary(previous: WordAllocation, current: WordAllocation) -> bool:
    if previous.speaker_id != current.speaker_id:
        return True
    if previous.speaker_source == "mixed" or current.speaker_source == "mixed":
        return True
    previous_clips = {span.clip_id for span in previous.physical_spans}
    current_clips = {span.clip_id for span in current.physical_spans}
    if (
        previous_clips
        and current_clips
        and not previous_clips.intersection(current_clips)
    ):
        return True
    previous_end = max(span.end for span in previous.physical_spans)
    current_start = min(span.start for span in current.physical_spans)
    # Preserve natural inter-word pauses inside a subtitle.  A larger gap is
    # the hard physical boundary used by the evidence-gap contract.
    return current_start - previous_end > 0.35


def _normalize_word_text(value: Any) -> str:
    return "".join(str(value or "").split()).casefold()


def _can_append(
    previous: WordAllocation,
    current: WordAllocation,
    max_word_gap: float,
    max_evidence_gap: float,
) -> bool:
    previous_decision = previous.word.metadata.get("decision_id")
    current_decision = current.word.metadata.get("decision_id")
    if previous_decision or current_decision:
        if previous_decision != current_decision:
            return False
    if previous.speaker_id != current.speaker_id:
        return False
    if previous.speaker_source == "mixed" or current.speaker_source == "mixed":
        return False
    gap = current.word.raw_start - previous.word.raw_end
    if gap < -0.01 or gap > max_word_gap:
        return False
    if previous.evidence_spans and current.evidence_spans:
        if not _evidence_sets_touch(
            previous.evidence_spans, current.evidence_spans, max_evidence_gap
        ):
            return False
    previous_spans = previous.physical_spans
    current_spans = current.physical_spans
    if not previous_spans or not current_spans:
        return False
    return _span_sets_touch(previous_spans, current_spans, max_word_gap)


def _evidence_sets_touch(
    left: Sequence[Any], right: Sequence[Any], max_gap: float
) -> bool:
    """Require adjacent words to belong to the same acoustic evidence run."""
    left_end = max(float(span.end) for span in left)
    right_start = min(float(span.start) for span in right)
    return right_start - left_end <= max_gap


def _span_sets_touch(
    left: Sequence[PhysicalSpan], right: Sequence[PhysicalSpan], max_gap: float
) -> bool:
    left_end = max(span.end for span in left)
    right_start = min(span.start for span in right)
    if right_start - left_end > max_gap:
        return False
    left_ids = {span.clip_id for span in left}
    right_ids = {span.clip_id for span in right}
    if left_ids.intersection(right_ids):
        return True
    ordered = sorted([*left, *right], key=lambda span: (span.start, span.end))
    return all(
        next_span.start - span.end <= 0.01
        for span, next_span in zip(ordered, ordered[1:])
    )


def _make_event(
    index: int,
    items: Sequence[WordAllocation],
    *,
    subtitle_bin: PhysicalSubtitleBin | None = None,
) -> GlobalSubtitleEvent:
    words = [item.word for item in items]
    allowed_evidence_ids = (
        set(subtitle_bin.evidence_ids) if subtitle_bin is not None else None
    )
    evidence_spans = [
        PhysicalSpan(
            span.physical_clip_id or item.physical_spans[0].clip_id,
            max(item.word.raw_start, span.start),
            min(item.word.raw_end, span.end),
            (span.id,),
        )
        for item in items
        for span in item.evidence_spans
        if allowed_evidence_ids is None or span.id in allowed_evidence_ids
        if span.physical_clip_id
        and max(item.word.raw_start, span.start) < min(item.word.raw_end, span.end)
        and item.physical_spans
    ]
    spans = _merge_spans(
        evidence_spans or [span for item in items for span in item.physical_spans]
    )
    warnings = [warning for item in items for warning in item.warnings]
    decision_trace: list[dict[str, Any]] = []
    seen_decisions: set[str] = set()
    for word in words:
        metadata = word.metadata or {}
        decision_id = metadata.get("decision_id")
        if not decision_id or decision_id in seen_decisions:
            continue
        seen_decisions.add(decision_id)
        decision_trace.append(
            {
                "stage": "evidence_decision",
                "decision_id": decision_id,
                "decision": metadata.get("decision"),
                "candidate_ids": list(metadata.get("candidate_ids", ())),
                "risk_level": metadata.get("risk_level"),
                "risk_score": metadata.get("risk_score"),
                "evidence_codes": list(metadata.get("evidence_codes", ())),
            }
        )
        if metadata.get("timestamp_clamped"):
            warnings.append("timestamp_clamped")
        if metadata.get("time_source") == "segment_boundary":
            warnings.append("missing_word_timestamps")
        if metadata.get("decision") == "unresolved":
            warnings.append("unresolved evidence conflict")
    speaker_status = (
        "known"
        if items[0].speaker_id is not None
        else (
            "mixed"
            if any(item.speaker_source == "mixed" for item in items)
            else "unknown"
        )
    )
    first = items[0]
    last = items[-1]
    start = (
        first.aligned_start if first.aligned_start is not None else words[0].raw_start
    )
    end = last.aligned_end if last.aligned_end is not None else words[-1].raw_end
    decisions = [
        decision
        for decision in (
            first.start_boundary_decision,
            last.end_boundary_decision,
        )
        if decision is not None
    ]
    degraded = len(decisions) != 2 or any(not item.accepted for item in decisions)
    if degraded:
        warnings.append("timing_degraded")
    boundary_trace = [
        {
            "stage": "boundary_arbitration",
            "boundary": decision.boundary_type,
            "decision": decision.to_dict(),
        }
        for decision in decisions
    ]
    return GlobalSubtitleEvent(
        index=index,
        start=start,
        end=end,
        text=_join_words([word.text for word in words]),
        words=words,
        speaker_id=items[0].speaker_id,
        speaker_status=speaker_status,
        speaker_source=items[0].speaker_source or "unknown",
        physical_spans=spans,
        source_word_ids=[
            str(word.metadata.get("source_word_id", word.id)) for word in words
        ],
        logical_sentence_id=index,
        alignment_warning=";".join(dict.fromkeys(warnings)) or None,
        hard_split_before=index > 1,
        physical_bin_id=subtitle_bin.id if subtitle_bin is not None else None,
        physical_bin_start=subtitle_bin.start if subtitle_bin is not None else None,
        physical_bin_end=subtitle_bin.end if subtitle_bin is not None else None,
        time_source="boundary_decision" if not degraded else "timing_degraded",
        revision_trace=[*decision_trace, *boundary_trace],
    )


def _merge_spans(spans: Sequence[PhysicalSpan]) -> list[PhysicalSpan]:
    grouped: dict[tuple[str, tuple[str, ...]], list[PhysicalSpan]] = {}
    for span in spans:
        grouped.setdefault((span.clip_id, span.evidence_ids), []).append(span)
    merged: list[PhysicalSpan] = []
    for (clip_id, evidence_ids), values in grouped.items():
        for span in sorted(values, key=lambda item: (item.start, item.end)):
            if (
                merged
                and merged[-1].clip_id == clip_id
                and merged[-1].evidence_ids == evidence_ids
                and span.start <= merged[-1].end + 0.005
            ):
                previous = merged.pop()
                merged.append(
                    PhysicalSpan(
                        clip_id,
                        previous.start,
                        max(previous.end, span.end),
                        evidence_ids,
                    )
                )
            else:
                merged.append(span)
    return sorted(merged, key=lambda item: (item.start, item.end, item.clip_id))


def _join_words(words: Sequence[str]) -> str:
    result = ""
    for raw in words:
        token = str(raw).strip()
        if not token:
            continue
        if not result:
            result = token
            continue
        no_space = (
            token[0] in '，。！？；：、,.!?;:)]}）】》”’"'
            or result[-1] in "([{（【《“‘"
            or _is_cjk(token[0])
            or _is_cjk(result[-1])
        )
        result += ("" if no_space else " ") + token
    return result


def _is_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)
