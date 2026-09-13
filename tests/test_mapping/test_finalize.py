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
from vocal_subtitle.mapping.strict_segmenter import repair_cross_boundary_fragments
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


# ── repair_cross_boundary_fragments ──────────────────────────────────

def test_repair_moves_trailing_fragment_to_next_event():
    """ASR 段边界切在句中：上一条尾部的残句片段移交给下一条开头。"""
    events = [
        _make_event(1, 10.986, 12.579, "我脚底板现在还在打颤。 得", speaker_id=0),
        _make_event(2, 12.926, 13.126, "了吧。", speaker_id=0),
    ]

    result, diag = repair_cross_boundary_fragments(events)

    assert [item.text for item in result] == [
        "我脚底板现在还在打颤。",
        "得了吧。",
    ]
    assert diag["moved_fragment_count"] == 1
    assert diag["moved_fragments"] == ["得"]


def test_repair_keeps_next_event_speaker_and_timing():
    events = [
        _make_event(1, 10.986, 12.579, "我脚底板现在还在打颤。 得", speaker_id=0),
        _make_event(2, 12.926, 13.126, "了吧。", speaker_id=1),
    ]

    result, _diag = repair_cross_boundary_fragments(events)

    assert result[1].speaker_id == 1
    assert (result[0].start, result[0].end) == (10.986, 12.579)
    assert (result[1].start, result[1].end) == (12.926, 13.126)


def test_repair_is_idempotent():
    events = [
        _make_event(1, 0.0, 2.0, "今天天气不错。 我"),
        _make_event(2, 2.2, 3.5, "们去爬山。"),
    ]

    first, diag = repair_cross_boundary_fragments(events)
    second, diag2 = repair_cross_boundary_fragments(first)

    assert [item.text for item in first] == ["今天天气不错。", "我们去爬山。"]
    assert [item.text for item in second] == [item.text for item in first]
    assert diag2["moved_fragment_count"] == 0


def test_repair_skips_when_no_sentence_mark():
    events = [
        _make_event(1, 0.0, 2.0, "还没有说完的话"),
        _make_event(2, 2.2, 3.5, "下一句。"),
    ]

    _result, diag = repair_cross_boundary_fragments(events)

    assert diag["moved_fragment_count"] == 0


def test_repair_skips_long_fragment():
    events = [
        _make_event(1, 0.0, 2.0, "第一句完了。 这一整句都留在上一条里"),
        _make_event(2, 2.2, 3.5, "下一句。"),
    ]

    result, diag = repair_cross_boundary_fragments(events)

    assert diag["moved_fragment_count"] == 0
    assert result[0].text == "第一句完了。 这一整句都留在上一条里"


def test_repair_skips_self_contained_fragment_with_punctuation():
    """片段自带结束标点（如“谢谢!”）时是完整表达，保留在原条。"""
    events = [
        _make_event(1, 0.0, 2.0, "第一句完了。 谢谢!"),
        _make_event(2, 2.2, 3.5, "明天见。"),
    ]

    result, diag = repair_cross_boundary_fragments(events)

    assert diag["moved_fragment_count"] == 0
    assert result[0].text == "第一句完了。 谢谢!"
    assert result[1].text == "明天见。"


def test_repair_skips_when_gap_too_large():
    events = [
        _make_event(1, 0.0, 2.0, "第一句完了。 得"),
        _make_event(2, 6.0, 7.0, "了吧。"),
    ]

    result, diag = repair_cross_boundary_fragments(events)

    assert diag["moved_fragment_count"] == 0
    assert result[0].text == "第一句完了。 得"


def test_repair_leaves_last_event_fragment_alone():
    events = [
        _make_event(1, 0.0, 2.0, "前面的话。"),
        _make_event(2, 2.2, 3.5, "最后一句。 收"),
    ]

    result, diag = repair_cross_boundary_fragments(events)

    assert diag["moved_fragment_count"] == 0
    assert result[1].text == "最后一句。 收"


def test_finalize_repairs_cross_boundary_spillover():
    """finalize 集成：串句修复后条数不变、文本连贯、诊断有记录。"""
    events = [
        _make_event(1, 10.986, 12.579, "我脚底板现在还在打颤。 得", speaker_id=0),
        _make_event(2, 12.926, 13.126, "了吧。", speaker_id=0),
    ]

    result = finalize_subtitle_events(events)

    assert result.subtitle_count == 2
    assert [item.text for item in result.events] == [
        "我脚底板现在还在打颤。",
        "得了吧。",
    ]
    assert result.diagnostics["cross_boundary_fragments"]["moved_fragment_count"] == 1
    assert "repair_cross_boundary_fragments" in result.diagnostics["step"]
