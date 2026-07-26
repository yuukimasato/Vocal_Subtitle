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
        }

    def to_subtitle_event(self) -> SubtitleEvent:
        if self.physical_bin_start is not None:
            physical_start = self.physical_bin_start
        else:
            physical_start = min(
                (span.start for span in self.physical_spans), default=self.start
            )
        if self.physical_bin_end is not None:
            physical_end = self.physical_bin_end
        else:
            physical_end = max(
                (span.end for span in self.physical_spans), default=self.end
            )
        relative_words = [
            WordTimestamp(
                word=word.text,
                start=word.raw_start - self.start,
                end=word.raw_end - self.start,
                confidence=word.confidence if word.confidence is not None else 1.0,
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
    """Fill physical bins while retaining whole-word and evidence boundaries."""
    bin_by_id = {item.id: item for item in subtitle_bins}
    groups: list[tuple[list[WordAllocation], PhysicalSubtitleBin | None]] = []
    current: list[WordAllocation] = []
    current_bin: PhysicalSubtitleBin | None = None
    for item in allocation.accepted:
        assigned = assign_word_to_bin(item.word, subtitle_bins)
        if current and not _can_append(
            current[-1], item, max_word_gap, max_evidence_gap
        ):
            groups.append((current, current_bin))
            current = []
        if not current:
            current_bin = assigned
        elif (
            assigned is not None
            and current_bin is not None
            and assigned.id != current_bin.id
        ):
            groups.append((current, current_bin))
            current = []
            current_bin = assigned
        current.append(item)
    if current:
        groups.append((current, current_bin))

    bin_group_counts: dict[str, int] = {}
    for _, bin_item in groups:
        if bin_item is not None:
            bin_group_counts[bin_item.id] = bin_group_counts.get(bin_item.id, 0) + 1

    events: list[GlobalSubtitleEvent] = []
    for items, bin_item in groups:
        use_bin_bounds = bool(
            bin_item is not None and bin_group_counts.get(bin_item.id) == 1
        )
        events.append(
            _make_event(
                len(events) + 1,
                items,
                subtitle_bin=bin_item if use_bin_bounds else None,
            )
        )
    return _merge_micro_bin_events(events)


def _merge_micro_bin_events(
    events: Sequence[GlobalSubtitleEvent],
    *,
    max_bin_gap: float = 0.15,
    max_word_gap: float = 0.05,
    max_short_text_chars: int = 2,
) -> list[GlobalSubtitleEvent]:
    """Join a short whole-word fragment split by adjacent physical bins.

    Physical evidence can leave a tiny trailing bin around a syllable or an
    initial address.  Keeping that as a one-character subtitle is worse than
    carrying the complete adjacent word stream, but broad bin merging would
    erase real speaker and sentence boundaries.  This narrow rule therefore
    requires a short side, contiguous word timestamps, one physical clip, and
    no hard or punctuation boundary.
    """
    if not events:
        return []
    merged: list[GlobalSubtitleEvent] = [events[0]]
    for current in events[1:]:
        previous = merged[-1]
        previous_word = previous.words[-1] if previous.words else None
        current_word = current.words[0] if current.words else None
        short_side = min(
            _display_char_count(previous.text),
            _display_char_count(current.text),
        ) <= max_short_text_chars
        same_clip = _single_clip(previous) is not None and _single_clip(previous) == _single_clip(current)
        bin_gap = (
            current.physical_bin_start - previous.physical_bin_end
            if current.physical_bin_start is not None
            and previous.physical_bin_end is not None
            else float("inf")
        )
        word_gap = (
            current_word.raw_start - previous_word.raw_end
            if previous_word is not None and current_word is not None
            else float("inf")
        )
        can_merge = bool(
            short_side
            and same_clip
            and previous.speaker_id == current.speaker_id
            and not _has_hard_warning(previous)
            and not _has_hard_warning(current)
            and not _ends_punctuation(previous_word)
            and bin_gap >= 0.0
            and bin_gap <= max_bin_gap
            and word_gap >= -0.01
            and word_gap <= max_word_gap
        )
        if not can_merge:
            merged.append(current)
            continue
        previous.words.extend(current.words)
        previous.text = _join_words([word.text for word in previous.words])
        previous.end = current.end
        previous.source_word_ids.extend(current.source_word_ids)
        previous.physical_spans = _merge_spans(
            [*previous.physical_spans, *current.physical_spans]
        )
        previous.physical_bin_id = "+".join(
            item
            for item in (previous.physical_bin_id, current.physical_bin_id)
            if item
        ) or None
        previous.physical_bin_end = current.physical_bin_end
        previous.alignment_warning = ";".join(
            dict.fromkeys(
                item
                for item in (previous.alignment_warning, current.alignment_warning)
                if item
                for item in str(item).split(";")
            )
        ) or None
    for index, event in enumerate(merged, start=1):
        event.index = index
        event.logical_sentence_id = index
    return merged


def _single_clip(event: GlobalSubtitleEvent) -> str | None:
    clip_ids = {span.clip_id for span in event.physical_spans}
    return next(iter(clip_ids)) if len(clip_ids) == 1 else None


def _display_char_count(text: str) -> int:
    return len(str(text).replace(" ", ""))


def _ends_punctuation(word: Any | None) -> bool:
    if word is None:
        return False
    return str(getattr(word, "text", "")).strip().endswith((".", "!", "?", "。", "！", "？"))


def _has_hard_warning(event: GlobalSubtitleEvent) -> bool:
    warning = str(event.alignment_warning or "")
    return any(
        item in warning.split(";")
        for item in ("speaker_conflict", "discontinuous_physical_boundary")
    )


def _can_append(
    previous: WordAllocation,
    current: WordAllocation,
    max_word_gap: float,
    max_evidence_gap: float,
) -> bool:
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
    evidence_spans = [
        PhysicalSpan(
            span.physical_clip_id or item.physical_spans[0].clip_id,
            max(item.word.raw_start, span.start),
            min(item.word.raw_end, span.end),
            (span.id,),
        )
        for item in items
        for span in item.evidence_spans
        if span.physical_clip_id
        and max(item.word.raw_start, span.start)
        < min(item.word.raw_end, span.end)
        and item.physical_spans
    ]
    spans = _merge_spans(
        evidence_spans
        or [span for item in items for span in item.physical_spans]
    )
    warnings = [warning for item in items for warning in item.warnings]
    speaker_status = (
        "known"
        if items[0].speaker_id is not None
        else (
            "mixed"
            if any(item.speaker_source == "mixed" for item in items)
            else "unknown"
        )
    )
    start = subtitle_bin.start if subtitle_bin is not None else min(word.raw_start for word in words)
    end = subtitle_bin.end if subtitle_bin is not None else max(word.raw_end for word in words)
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
        source_word_ids=[word.id for word in words],
        logical_sentence_id=index,
        alignment_warning=";".join(dict.fromkeys(warnings)) or None,
        hard_split_before=index > 1,
        physical_bin_id=subtitle_bin.id if subtitle_bin is not None else None,
        physical_bin_start=subtitle_bin.start if subtitle_bin is not None else None,
        physical_bin_end=subtitle_bin.end if subtitle_bin is not None else None,
        time_source="physical_bin" if subtitle_bin is not None else "asr_word",
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
