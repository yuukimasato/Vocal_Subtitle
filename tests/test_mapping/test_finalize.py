"""Tests for subtitle event finalization."""

import pytest

from vocal_subtitle.mapping.display_timeline import DisplayCue
from vocal_subtitle.mapping.finalize import (
    FinalizeConfig,
    FinalizeResult,
    finalize_subtitle_events,
    _validate_input_events,
    _events_to_semantic_groups,
)
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.asr.base import WordTimestamp


def _make_event(index, start, end, text, **kwargs):
    return SubtitleEvent(index=index, start=start, end=end, text=text, **kwargs)


# ── FinalizeConfig ───────────────────────────────────────────────────

def test_config_defaults():
    cfg = FinalizeConfig()
    assert cfg.strict_segmentation_enabled
    assert not cfg.llm_post_enabled
    assert cfg.validate


# ── _validate_input_events ───────────────────────────────────────────

def test_validate_filters_empty_text():
    events = [
        _make_event(1, 0.0, 1.0, ""),
        _make_event(2, 1.0, 2.0, "valid"),
    ]
    diag = {}
    valid = _validate_input_events(events, diag)
    assert len(valid) == 1
    assert diag["skipped_invalid"] == 1


def test_validate_filters_inverted_time():
    events = [
        _make_event(1, 2.0, 1.0, "bad"),
        _make_event(2, 1.0, 2.0, "good"),
    ]
    diag = {}
    valid = _validate_input_events(events, diag)
    assert len(valid) == 1
    assert valid[0].text == "good"


# ── _events_to_semantic_groups ───────────────────────────────────────

def test_events_to_groups_preserves_metadata():
    events = [
        _make_event(1, 1.0, 2.0, "hello",
                    physical_start=0.9, physical_end=2.1,
                    speaker_id=1, speaker_label="Speaker A",
                    source_word_ids=["w1", "w2"],
                    physical_spans=[{"physical_clip_id": "clip-a", "start": 0.9, "end": 2.1}]),
    ]
    groups = _events_to_semantic_groups(events)
    assert len(groups) == 1
    g = groups[0]
    assert g["physical_start"] == 0.9
    assert g["physical_end"] == 2.1
    assert g["speaker_id"] == 1
    assert g["source_word_ids"] == ["w1", "w2"]


# ── finalize_subtitle_events ─────────────────────────────────────────

def test_finalize_preserves_event_count():
    events = [
        _make_event(1, 1.0, 2.5, "first subtitle"),
        _make_event(2, 3.0, 4.5, "second subtitle"),
    ]
    result = finalize_subtitle_events(events)

    assert result.subtitle_count == 2
    assert len(result.display_cues) == 2
    assert result.events[0].index == 1
    assert result.events[1].index == 2


def test_finalize_clips_to_audio_duration():
    """Events with audio_duration clamp should not exceed the duration."""
    events = [
        _make_event(1, 1.0, 2.0, "test",
                    physical_start=1.0, physical_end=2.0),
    ]
    result = finalize_subtitle_events(events, audio_duration=3.0)
    # With sufficient audio_duration, display_end covers physical_end
    assert result.events[0].end <= 3.0
    assert result.events[0].end >= result.events[0].physical_end


def test_finalize_skips_empty_events():
    events = [
        _make_event(1, 0.0, 1.0, ""),
        _make_event(2, 1.0, 2.0, "real text"),
    ]
    result = finalize_subtitle_events(events)

    assert result.subtitle_count == 1
    assert result.events[0].text == "real text"


def test_finalize_result_to_dict():
    events = [_make_event(1, 1.0, 2.0, "hello")]
    result = finalize_subtitle_events(events)
    d = result.to_dict()
    assert len(d["events"]) == 1
    assert d["events"][0]["text"] == "hello"
    assert "display_cues" in d
    assert "diagnostics" in d


def test_finalize_idempotent():
    """Running finalize twice on the same input should produce the same count."""
    events = [
        _make_event(1, 1.0, 2.0, "first"),
        _make_event(2, 3.0, 4.0, "second"),
    ]
    result1 = finalize_subtitle_events(events)
    result2 = finalize_subtitle_events(list(result1.events))

    assert result1.subtitle_count == result2.subtitle_count


def test_finalize_handles_empty_input():
    result = finalize_subtitle_events([])
    assert result.subtitle_count == 0
    assert len(result.events) == 0
    assert len(result.display_cues) == 0


def test_finalize_does_not_mutate_source_events():
    event = _make_event(
        8, 1.0, 1.2, "hello", physical_start=1.0, physical_end=1.2
    )

    result = finalize_subtitle_events([event], audio_duration=3.0)

    assert result.events[0] is not event
    assert event.index == 8
    assert event.start == 1.0
    assert event.end == 1.2


def test_finalize_splits_event_longer_than_max_duration():
    event = _make_event(
        1,
        0.0,
        10.0,
        "这是一个很长的中文字幕句子需要在最终化阶段被拆分为多条字幕事件以便阅读",
    )

    result = finalize_subtitle_events([event])

    assert len(result.events) > 1
    assert all(item.end - item.start <= 5.0 for item in result.events)
    assert result.subtitle_count == len(result.display_cues)
    assert result.diagnostics["split_long_event_count"] >= 1


def test_finalize_restores_sentence_granularity_at_word_boundaries():
    event = _make_event(
        1,
        0.0,
        2.0,
        "Hello world. Next",
        words=[
            WordTimestamp("Hello", 0.1, 0.4),
            WordTimestamp("world.", 0.5, 0.9),
            WordTimestamp("Next", 1.4, 1.8),
        ],
        physical_start=0.1,
        physical_end=1.8,
        source_word_ids=["w1", "w2", "w3"],
    )

    result = finalize_subtitle_events([event])

    assert [item.text for item in result.events] == ["Hello world.", "Next"]
    assert [item.source_word_ids for item in result.events] == [["w1", "w2"], ["w3"]]
    assert result.diagnostics["strict_segmentation"]["sentence_split_count"] >= 1


def test_finalize_does_not_merge_short_events_across_physical_bins():
    events = [
        _make_event(
            1,
            0.0,
            0.6,
            "第一句",
            physical_start=0.0,
            physical_end=0.6,
            physical_bin_id="bin-a",
            physical_bin_start=0.0,
            physical_bin_end=0.6,
            physical_spans=[{"physical_clip_id": "clip-a", "start": 0.0, "end": 0.6}],
        ),
        _make_event(
            2,
            0.6,
            1.0,
            "第二句",
            physical_start=0.6,
            physical_end=1.0,
            physical_bin_id="bin-b",
            physical_bin_start=0.6,
            physical_bin_end=1.0,
            physical_spans=[{"physical_clip_id": "clip-a", "start": 0.6, "end": 1.0}],
        ),
    ]

    result = finalize_subtitle_events(events)

    assert [item.text for item in result.events] == ["第一句", "第二句"]
    assert [item.physical_bin_id for item in result.events] == ["bin-a", "bin-b"]
