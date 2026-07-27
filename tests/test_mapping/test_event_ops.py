"""Tests for provenance-preserving event operations."""

import pytest

from vocal_subtitle.mapping.event_ops import (
    can_merge_events,
    clone_event,
    merge_event_group,
    shift_event,
    split_event_by_word_ranges,
)
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


def _make_event(index: int, start: float, end: float, text: str, **kwargs) -> SubtitleEvent:
    defaults = {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
    }
    if "physical_start" not in kwargs:
        defaults["physical_start"] = start
    if "physical_end" not in kwargs:
        defaults["physical_end"] = end
    defaults.update(kwargs)
    return SubtitleEvent(**defaults)


class TestCloneEvent:
    def test_clone_preserves_all_fields(self):
        original = _make_event(
            1, 0.0, 1.0, "hello",
            physical_bin_id="bin-1",
            source_word_ids=["w1", "w2"],
            speaker_status="confirmed",
            revision_trace=[{"op": "init"}],
        )
        clone = clone_event(original)

        assert clone.index == 1
        assert clone.start == 0.0
        assert clone.end == 1.0
        assert clone.text == "hello"
        assert clone.physical_bin_id == "bin-1"
        assert clone.source_word_ids == ["w1", "w2"]
        assert clone.speaker_status == "confirmed"
        assert clone.revision_trace == [{"op": "init"}]

    def test_clone_overrides_fields(self):
        original = _make_event(1, 0.0, 1.0, "hello")
        clone = clone_event(original, text="world", index=2)

        assert clone.text == "world"
        assert clone.index == 2
        # Unchanged
        assert clone.start == 0.0

    def test_clone_is_deep_copy(self):
        original = _make_event(1, 0.0, 1.0, "hello", source_word_ids=["w1"])
        clone = clone_event(original)
        clone.source_word_ids.append("w2")

        assert original.source_word_ids == ["w1"]
        assert clone.source_word_ids == ["w1", "w2"]


class TestShiftEvent:
    def test_shift_moves_all_time_fields(self):
        event = _make_event(
            1, 1.0, 2.0, "test",
            physical_bin_start=1.0, physical_bin_end=2.0,
            physical_spans=[{"start": 1.0, "end": 2.0, "physical_clip_id": "c1"}],
        )
        shifted = shift_event(event, 10.0)

        assert shifted.start == 11.0
        assert shifted.end == 12.0
        assert shifted.physical_start == 11.0
        assert shifted.physical_end == 12.0
        assert shifted.physical_bin_start == 11.0
        assert shifted.physical_bin_end == 12.0
        assert shifted.physical_spans[0]["start"] == 11.0
        assert shifted.physical_spans[0]["end"] == 12.0

    def test_shift_zero_is_noop(self):
        event = _make_event(1, 1.0, 2.0, "test")
        shifted = shift_event(event, 0.0)

        assert shifted.start == event.start
        assert shifted.end == event.end


class TestCanMerge:
    def test_two_adjacent_events_are_mergeable(self):
        left = _make_event(1, 0.0, 0.5, "hello")
        right = _make_event(2, 0.52, 1.0, "world")

        ok, reason = can_merge_events([left, right])
        assert ok is True
        assert reason is None

    def test_hard_split_prevents_merge(self):
        left = _make_event(1, 0.0, 0.5, "hello")
        right = _make_event(2, 0.52, 1.0, "world", hard_split_before=True)

        ok, reason = can_merge_events([left, right])
        assert ok is False
        assert "hard_split_before" in reason.lower()

    def test_physical_bin_conflict_prevents_merge(self):
        left = _make_event(1, 0.0, 0.5, "hello", physical_bin_id="bin-1")
        right = _make_event(2, 0.52, 1.0, "world", physical_bin_id="bin-2")

        ok, reason = can_merge_events([left, right])
        assert ok is False
        assert "physical bin" in reason.lower()

    def test_different_confirmed_speakers_prevent_merge(self):
        left = _make_event(1, 0.0, 0.5, "hello", speaker_id=1, speaker_status="confirmed")
        right = _make_event(2, 0.52, 1.0, "world", speaker_id=2, speaker_status="confirmed")

        ok, reason = can_merge_events([left, right])
        assert ok is False
        assert "speaker" in reason.lower()

    def test_gap_exceeds_max(self):
        left = _make_event(1, 0.0, 0.5, "hello")
        right = _make_event(2, 1.0, 1.5, "world")

        ok, reason = can_merge_events([left, right], max_gap=0.12)
        assert ok is False
        assert "gap" in reason.lower()

    def test_single_event_is_always_mergeable(self):
        ok, reason = can_merge_events([_make_event(1, 0.0, 1.0, "solo")])
        assert ok is True


class TestMergeEventGroup:
    def test_merge_deduplicates_source_word_ids(self):
        left = _make_event(1, 0.0, 0.5, "hello", source_word_ids=["w1"])
        right = _make_event(2, 0.5, 1.0, "world", source_word_ids=["w2"])
        # Simulate overlap in word IDs (should only happen in pathological cases)
        right.source_word_ids = ["w1", "w2"]

        merged = merge_event_group([left, right], reason="test")

        assert merged.source_word_ids == ["w1", "w2"]

    def test_merge_adds_revision_trace(self):
        left = _make_event(1, 0.0, 0.5, "hello", revision_trace=[{"op": "init"}])
        right = _make_event(2, 0.5, 1.0, "world")

        merged = merge_event_group([left, right], reason="semantic")

        assert len(merged.revision_trace) == 2
        assert merged.revision_trace[-1]["reason"] == "semantic"

    def test_merge_uses_custom_text(self):
        left = _make_event(1, 0.0, 0.5, "hello")
        right = _make_event(2, 0.5, 1.0, "world")

        merged = merge_event_group([left, right], text="hello world", reason="join")

        assert merged.text == "hello world"

    def test_merge_preserves_confirmed_speaker(self):
        left = _make_event(1, 0.0, 0.5, "a", speaker_id=1, speaker_status="confirmed", speaker_source="diarization")
        right = _make_event(2, 0.5, 1.0, "b", speaker_id=1, speaker_status="unknown")

        merged = merge_event_group([left, right], reason="speaker")

        assert merged.speaker_id == 1
        assert merged.speaker_status == "confirmed"


class TestSplitEvent:
    def test_split_by_word_ranges(self):
        from vocal_subtitle.asr.base import WordTimestamp

        event = _make_event(1, 0.0, 2.0, "hello world")
        event.words = [
            WordTimestamp("hello", 0.0, 0.8, 1.0),
            WordTimestamp("world", 1.2, 2.0, 1.0),
        ]

        results = split_event_by_word_ranges(event, [(0, 1), (1, 2)], reason="break")

        assert len(results) == 2
        assert results[0].text == "hello"
        assert results[1].text == "world"
        assert results[0].revision_trace[-1]["reason"] == "break"

    def test_split_adds_revision_entries(self):
        from vocal_subtitle.asr.base import WordTimestamp

        event = _make_event(1, 0.0, 2.0, "hello world", revision_trace=[{"op": "init"}])
        event.words = [
            WordTimestamp("hello", 0.0, 0.8, 1.0),
            WordTimestamp("world", 1.2, 2.0, 1.0),
        ]

        results = split_event_by_word_ranges(event, [(0, 1), (1, 2)], reason="overlong")

        for r in results:
            assert r.revision_trace[0] == {"op": "init"}
            assert r.revision_trace[-1]["op"] == "split"
