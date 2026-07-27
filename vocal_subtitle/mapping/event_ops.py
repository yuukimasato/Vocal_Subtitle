"""Provenance-preserving event operations.

Pure functions for merging, splitting, cloning, and shifting SubtitleEvents
without losing physical provenance or speaker metadata.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .time_mapper import SubtitleEvent


def clone_event(event: SubtitleEvent, **changes) -> SubtitleEvent:
    """Create a copy of a SubtitleEvent, optionally overriding fields.

    All mutable fields (words, physical_spans, source_word_ids, etc.)
    are deep-copied so the clone is independent of the original.
    """
    fields: Dict[str, Any] = {
        "index": event.index,
        "start": event.start,
        "end": event.end,
        "text": event.text,
        "words": copy.deepcopy(event.words),
        "original_text": event.original_text,
        "speaker_id": event.speaker_id,
        "speaker_label": event.speaker_label,
        "physical_start": event.physical_start,
        "physical_end": event.physical_end,
        "physical_spans": copy.deepcopy(event.physical_spans),
        "source_word_ids": list(event.source_word_ids),
        "logical_sentence_id": event.logical_sentence_id,
        "alignment_warning": event.alignment_warning,
        "physical_region_id": event.physical_region_id,
        "physical_bin_id": event.physical_bin_id,
        "physical_bin_start": event.physical_bin_start,
        "physical_bin_end": event.physical_bin_end,
        "time_source": event.time_source,
        "hard_split_before": event.hard_split_before,
        "speaker_status": event.speaker_status,
        "speaker_source": event.speaker_source,
        "speaker_repair_reason": event.speaker_repair_reason,
        "asr_text": event.asr_text,
        "genuine_overlap": event.genuine_overlap,
        "overlap_group_id": event.overlap_group_id,
        "overlap_tracks": copy.deepcopy(event.overlap_tracks),
        "revision_trace": list(event.revision_trace),
    }
    fields.update(changes)
    return SubtitleEvent(**fields)


def shift_event(event: SubtitleEvent, offset: float) -> SubtitleEvent:
    """Shift all time fields of an event by a given offset.

    Moves display times, physical times, bin times, and physical_spans
    simultaneously. Never only shifts start/end.
    """
    if not offset:
        return clone_event(event)

    new_physical_spans = []
    for span in event.physical_spans:
        s = dict(span) if isinstance(span, dict) else span
        if hasattr(s, "copy"):
            s = copy.deepcopy(s)
        if isinstance(s, dict):
            s["start"] = s.get("start", 0) + offset
            s["end"] = s.get("end", 0) + offset
        new_physical_spans.append(s)

    return clone_event(
        event,
        start=event.start + offset,
        end=event.end + offset,
        physical_start=event.physical_start + offset if event.physical_start is not None else None,
        physical_end=event.physical_end + offset if event.physical_end is not None else None,
        physical_bin_start=event.physical_bin_start + offset if event.physical_bin_start is not None else None,
        physical_bin_end=event.physical_bin_end + offset if event.physical_bin_end is not None else None,
        physical_spans=new_physical_spans,
    )


def can_merge_events(
    events: Sequence[SubtitleEvent],
    *,
    max_gap: float = 0.12,
    max_combined_duration: float = 5.0,
) -> tuple[bool, Optional[str]]:
    """Check whether a group of events can be safely merged.

    Refuses merges that would:
    - Cross a hard boundary (hard_split_before is True on non-first events)
    - Combine events from different physical bins (physical_owner conflict)
    - Combine events with different confirmed speakers
    - Combine events with genuine overlap (would collapse separate tracks)
    - Exceed duration limits

    Args:
        events: Ordered events to consider merging.
        max_gap: Maximum gap between consecutive events for merge eligibility.
        max_combined_duration: Maximum total duration for merged event.

    Returns:
        (allowed, reason) — allowed is True if the merge is safe.
    """
    if len(events) < 2:
        return True, None

    # Sort by start time
    ordered = sorted(events, key=lambda e: e.start)

    # Hard boundary check
    for i, event in enumerate(ordered):
        if i > 0 and event.hard_split_before:
            return False, f"event {event.index} has hard_split_before"

    # Physical bin conflict
    bin_ids = {e.physical_bin_id for e in ordered if e.physical_bin_id is not None}
    if len(bin_ids) > 1:
        return False, f"events span multiple physical bins: {bin_ids}"

    # Confirmed speaker conflict
    confirmed_speakers = {
        e.speaker_id
        for e in ordered
        if e.speaker_id is not None and e.speaker_status == "confirmed"
    }
    if len(confirmed_speakers) > 1:
        return False, f"events have different confirmed speakers: {confirmed_speakers}"

    # Genuine overlap cannot be flattened
    if any(e.genuine_overlap for e in ordered):
        return False, "events contain genuine overlap with separate speaker tracks"

    # Gap check
    for i in range(len(ordered) - 1):
        gap = ordered[i + 1].start - ordered[i].end
        if gap > max_gap:
            return False, f"gap {gap:.3f}s exceeds max {max_gap:.3f}s"

    # Duration check
    combined_duration = ordered[-1].end - ordered[0].start
    if combined_duration > max_combined_duration:
        return False, f"combined duration {combined_duration:.1f}s exceeds max {max_combined_duration:.1f}s"

    return True, None


def merge_event_group(
    events: Sequence[SubtitleEvent],
    text: Optional[str] = None,
    reason: str = "merge",
    *,
    new_index: Optional[int] = None,
) -> SubtitleEvent:
    """Merge a group of events into one, preserving provenance.

    - source_word_ids are deduplicated in order
    - physical_spans are merged sequentially
    - revision_trace records the merge
    - confirmed speaker is preserved (raise on conflict)
    - display times cover the full range
    - physical times use the union of all events' physical ranges

    Args:
        events: Ordered events to merge.
        text: Optional merged text; if None, joined from events.
        reason: Reason code for the revision trace.
        new_index: Optional index for the merged event; if None, uses first event's index.

    Returns:
        A new SubtitleEvent representing the merge.
    """
    if not events:
        raise ValueError("cannot merge empty event list")

    ordered = sorted(events, key=lambda e: e.start)

    # Deduplicate source word IDs in order
    seen_word_ids = set()
    merged_word_ids = []
    for e in ordered:
        for wid in e.source_word_ids:
            if wid not in seen_word_ids:
                seen_word_ids.add(wid)
                merged_word_ids.append(wid)

    # Merge physical spans
    merged_spans = []
    for e in ordered:
        merged_spans.extend(copy.deepcopy(e.physical_spans))

    # Merge revision traces
    merged_trace = []
    for e in ordered:
        merged_trace.extend(copy.deepcopy(e.revision_trace))
    merged_trace.append({
        "op": "merge",
        "reason": reason,
        "merged_indices": [e.index for e in ordered],
        "merged_texts": [e.text for e in ordered],
    })

    # Determine speaker: use confirmed if present; prefer majority otherwise
    speakers = [e.speaker_id for e in ordered if e.speaker_id is not None]
    confirmed = [
        e for e in ordered
        if e.speaker_id is not None and e.speaker_status == "confirmed"
    ]
    if confirmed:
        speaker_id = confirmed[0].speaker_id
        speaker_status = "confirmed"
        speaker_source = confirmed[0].speaker_source
    elif speakers:
        speaker_id = max(set(speakers), key=speakers.count)
        speaker_status = ordered[0].speaker_status if ordered[0].speaker_status else "unknown"
        speaker_source = ordered[0].speaker_source if ordered[0].speaker_source else "unknown"
    else:
        speaker_id = None
        speaker_status = "unknown"
        speaker_source = "unknown"

    # Physical range union
    physical_starts = [e.physical_start for e in ordered if e.physical_start is not None]
    physical_ends = [e.physical_end for e in ordered if e.physical_end is not None]
    physical_start = min(physical_starts) if physical_starts else None
    physical_end = max(physical_ends) if physical_ends else None

    merged_text = text if text is not None else " ".join(e.text for e in ordered)
    idx = new_index if new_index is not None else ordered[0].index

    return SubtitleEvent(
        index=idx,
        start=ordered[0].start,
        end=ordered[-1].end,
        text=merged_text,
        words=copy.deepcopy(ordered[0].words),
        original_text=ordered[0].original_text,
        speaker_id=speaker_id,
        speaker_label=ordered[0].speaker_label,
        physical_start=physical_start,
        physical_end=physical_end,
        physical_spans=merged_spans,
        source_word_ids=merged_word_ids,
        logical_sentence_id=ordered[0].logical_sentence_id,
        alignment_warning=None,
        physical_region_id=ordered[0].physical_region_id,
        physical_bin_id=ordered[0].physical_bin_id,
        physical_bin_start=ordered[0].physical_bin_start,
        physical_bin_end=ordered[-1].physical_bin_end if ordered[-1].physical_bin_end is not None else ordered[-1].physical_bin_end,
        time_source=ordered[0].time_source,
        hard_split_before=ordered[0].hard_split_before,
        speaker_status=speaker_status,
        speaker_source=speaker_source,
        speaker_repair_reason="",
        asr_text=ordered[0].asr_text,
        genuine_overlap=False,
        overlap_group_id=None,
        overlap_tracks=[],
        revision_trace=merged_trace,
    )


def split_event_by_word_ranges(
    event: SubtitleEvent,
    ranges: List[tuple[int, int]],
    reason: str = "split",
) -> List[SubtitleEvent]:
    """Split an event into multiple events based on word index ranges.

    Each range is (start_word_idx, end_word_idx) where end_word_idx is exclusive.
    Words are indexed within the event's words list.
    Source IDs and physical spans are partitioned proportionally.

    Args:
        event: The event to split.
        ranges: List of (start, end) word ranges. Must cover the event text.
        reason: Reason code for the revision trace.

    Returns:
        List of new SubtitleEvents, one per range.
    """
    if not ranges:
        return [clone_event(event)]

    words = event.words
    if not words:
        # No word-level data: split by time proportion
        total_duration = event.end - event.start
        if total_duration <= 0:
            return [clone_event(event, index=event.index + i) for i in range(len(ranges))]

        results = []
        for i, (fraction_start, fraction_end) in enumerate(ranges):
            clone = clone_event(event, index=event.index + i)
            clone.start = event.start + fraction_start * total_duration
            clone.end = event.start + fraction_end * total_duration
            clone.revision_trace.append({
                "op": "split",
                "reason": reason,
                "range": [fraction_start, fraction_end],
            })
            results.append(clone)
        return results

    # Word-level split
    results = []
    for i, (w_start, w_end) in enumerate(ranges):
        w_start = max(0, min(w_start, len(words)))
        w_end = max(w_start, min(w_end, len(words)))
        sub_words = words[w_start:w_end]

        if not sub_words:
            continue

        new_start = min(w.start for w in sub_words)
        new_end = max(w.end for w in sub_words)
        new_text = "".join(w.word for w in sub_words)

        # Partition source word IDs proportionally
        total_words = len(words) if words else 1
        source_start = max(0, int(len(event.source_word_ids) * w_start / total_words))
        source_end = min(len(event.source_word_ids), int(len(event.source_word_ids) * w_end / total_words) + 1)
        sub_source_ids = event.source_word_ids[source_start:source_end]

        trace_entry = {
            "op": "split",
            "reason": reason,
            "word_range": [w_start, w_end],
        }
        new_trace = list(event.revision_trace) + [trace_entry]

        results.append(SubtitleEvent(
            index=event.index + i,
            start=new_start,
            end=new_end,
            text=new_text,
            words=copy.deepcopy(sub_words),
            original_text=event.original_text,
            speaker_id=event.speaker_id,
            speaker_label=event.speaker_label,
            physical_start=new_start,
            physical_end=new_end,
            physical_spans=copy.deepcopy(event.physical_spans),
            source_word_ids=sub_source_ids,
            logical_sentence_id=event.logical_sentence_id,
            alignment_warning=event.alignment_warning,
            physical_region_id=event.physical_region_id,
            physical_bin_id=event.physical_bin_id,
            physical_bin_start=event.physical_bin_start,
            physical_bin_end=event.physical_bin_end,
            time_source=event.time_source,
            hard_split_before=False,
            speaker_status=event.speaker_status,
            speaker_source=event.speaker_source,
            speaker_repair_reason="",
            asr_text=event.asr_text,
            genuine_overlap=event.genuine_overlap,
            overlap_group_id=event.overlap_group_id,
            overlap_tracks=copy.deepcopy(event.overlap_tracks),
            revision_trace=new_trace,
        ))

    return results
