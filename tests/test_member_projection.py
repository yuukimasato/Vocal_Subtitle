"""member_projection: 聚合窗口事件重投影回物理成员段的单元测试。"""

from __future__ import annotations

import pytest

from vocal_subtitle.asr.base import WordTimestamp
from vocal_subtitle.application.member_projection import (
    reproject_events_to_members,
)
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


def _event(start, end, words_abs, text=None):
    """words_abs: [(text, abs_start, abs_end)]，绝对时间；内部转相对。"""
    words = [
        WordTimestamp(word=word, start=ws - start, end=we - start)
        for word, ws, we in words_abs
    ]
    if text is None:
        text = "".join(word for word, _, _ in words_abs)
    return SubtitleEvent(index=1, start=start, end=end, text=text, words=words)


def test_split_across_member_gap_restores_silence_boundaries():
    members = [(0.0, 1.0), (1.4, 2.5)]
    event = _event(0.0, 2.5, [
        ("你", 0.10, 0.30),
        ("好", 0.40, 0.60),
        ("吗", 1.45, 1.60),
    ])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.split_events == 1
    assert stats.events_out == 2
    first, second = projected
    assert first.start == pytest.approx(0.10)
    assert first.end == pytest.approx(0.60)
    assert first.text == "你好"
    assert first.hard_split_before is False
    assert second.start == pytest.approx(1.45)
    assert second.end == pytest.approx(1.60)
    assert second.text == "吗"
    assert second.hard_split_before is True
    # 词时间保持相对新事件 start 的契约
    assert second.words[0].start == pytest.approx(0.0)


def test_small_member_gap_is_intra_phrase_and_keeps_text_whole():
    # 0.25s 的成员间隙属于句内微停顿（< member_split_min_gap 默认 0.3s）：
    # 文本不拆行，端点钳制到组成员包络，尾部词不被截断。
    members = [(0.0, 1.0), (1.25, 2.5)]
    event = _event(0.0, 2.5, [
        ("你", 0.10, 0.30),
        ("好", 0.40, 0.60),
        ("吗", 1.30, 1.50),
    ])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.split_events == 0
    assert stats.clamped_events == 0  # 事件端点与组合包络一致 → 原样透传
    assert len(projected) == 1
    assert projected[0] is event
    only = projected[0]
    assert only.text == "你好吗"
    assert only.start == pytest.approx(0.0)
    assert only.end == pytest.approx(2.5)
    assert only.hard_split_before is False


def test_over_limit_merged_group_splits_at_member_gaps():
    # 模拟用户实测的 5.7s 列举句：成员间隙全部 < 0.3s（句内微停顿不拆），
    # 但聚合组超过 max_duration 时必须在成员间隙处继续拆分成短行。
    members = [(0.0, 1.4), (1.6, 3.0), (3.15, 4.5), (4.65, 5.8)]
    event = _event(0.0, 5.8, [
        ("a hot dishes section,", 0.10, 1.30),
        ("a pastries section,", 1.70, 2.90),
        ("a fruits section,", 3.20, 4.40),
        ("and a live cooking station.", 4.70, 5.70),
    ])

    projected, stats = reproject_events_to_members(
        [event], members, max_duration=2.0,
    )

    assert stats.split_events == 1
    assert stats.over_limit_splits == 3
    assert [item.text for item in projected] == [
        "a hot dishes section,",
        "a pastries section,",
        "a fruits section,",
        "and a live cooking station.",
    ]
    for piece in projected:
        assert piece.duration <= 2.0
    assert [item.hard_split_before for item in projected] == [
        False, True, True, True,
    ]
    assert all(
        item.revision_trace[-1].get("over_limit") for item in projected
    )


def test_default_max_duration_caps_window_sized_group():
    # 默认 5.0s 上限：整组 5.6s 跨越四个成员 → 在最后一个成员间隙处切开。
    members = [(0.0, 1.4), (1.6, 3.0), (3.15, 4.5), (4.65, 5.8)]
    event = _event(0.0, 5.8, [
        ("a hot dishes section,", 0.10, 1.30),
        ("a pastries section,", 1.70, 2.90),
        ("a fruits section,", 3.20, 4.40),
        ("and a live cooking station.", 4.70, 5.70),
    ])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.over_limit_splits == 1
    assert len(projected) == 2
    assert projected[0].duration <= 5.0
    assert projected[1].text == "and a live cooking station."


def test_over_limit_width_splits_at_member_gap():
    # 显示宽度超限（CJK=2 计，84 上限）同样触发成员间隙拆分。
    members = [(0.0, 1.0), (1.15, 2.2)]
    event = _event(0.0, 2.2, [
        ("a" * 50, 0.10, 0.90),
        ("b" * 50, 1.20, 2.00),
    ])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.over_limit_splits == 1
    assert len(projected) == 2


def test_single_member_over_limit_stays_whole():
    # 单一成员内部没有可用的声学切点：保持整行，交由下游 finalize 兜底。
    members = [(0.0, 6.0)]
    event = _event(0.0, 6.0, [("完整长句无停顿", 0.10, 5.90)])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.over_limit_splits == 0
    assert len(projected) == 1


def test_split_min_gap_zero_restores_legacy_per_member_split():
    members = [(0.0, 1.0), (1.25, 2.5)]
    event = _event(0.0, 2.5, [
        ("你", 0.10, 0.30),
        ("吗", 1.30, 1.50),
    ])

    projected, stats = reproject_events_to_members(
        [event], members, split_min_gap=0.0,
    )

    assert stats.split_events == 1
    assert [item.text for item in projected] == ["你", "吗"]


def test_end_overshoot_clamped_to_member_end():
    members = [(0.0, 1.0)]
    event = _event(0.0, 1.3, [("好", 0.10, 0.60)], text="好")

    projected, stats = reproject_events_to_members([event], members)

    assert stats.clamped_events == 1
    assert len(projected) == 1
    assert projected[0].end == pytest.approx(1.0)
    assert projected[0].text == "好"
    assert projected[0].revision_trace[-1]["stage"] == "member_reprojection"


def test_start_spillover_from_context_padding_clamped():
    members = [(0.5, 1.5)]
    event = _event(-0.2, 0.8, [("好", 0.55, 0.70)], text="好")

    projected, stats = reproject_events_to_members([event], members)

    assert stats.clamped_events == 1
    assert projected[0].start == pytest.approx(0.5)
    assert projected[0].end == pytest.approx(0.8)


def test_word_fully_inside_real_silence_gap_is_dropped():
    # 旧逐段切片行为：跨组间隙（>= split_min_gap，真静音）内的 ASR
    # 幻觉词不属于任何拆分组，钳制后退化即丢弃。
    members = [(0.0, 1.0), (1.4, 2.5)]
    event = _event(1.05, 1.35, [("啊", 1.05, 1.35)], text="啊")

    projected, stats = reproject_events_to_members([event], members)

    assert stats.dropped_events == 1
    assert projected == []


def test_word_in_intra_phrase_micro_gap_is_kept():
    # 合并组内部的微停顿（0.25s 间隙）属于语句区间：词端点保留，
    # 不再按"必须落在成员段内"丢弃（silencedetect 会削掉词边）。
    members = [(0.0, 1.0), (1.25, 2.5)]
    event = _event(1.05, 1.15, [("啊", 1.05, 1.15)], text="啊")

    projected, stats = reproject_events_to_members([event], members)

    assert stats.dropped_events == 0
    assert len(projected) == 1
    assert projected[0].start == pytest.approx(1.05)
    assert projected[0].end == pytest.approx(1.15)


def test_unworded_event_spanning_members_clamped_not_split():
    members = [(0.0, 1.0), (1.25, 2.5)]
    event = _event(0.2, 2.8, [])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.unworded_events == 1
    assert stats.clamped_events == 1
    assert len(projected) == 1
    assert projected[0].start == pytest.approx(0.2)
    assert projected[0].end == pytest.approx(2.5)
    assert projected[0].text == _event(0.2, 2.8, []).text


def test_unworded_event_outside_all_members_dropped():
    members = [(0.5, 1.5)]
    event = _event(0.0, 0.3, [])

    projected, stats = reproject_events_to_members([event], members)

    assert stats.dropped_events == 1
    assert projected == []


def test_english_split_keeps_word_spacing():
    members = [(0.0, 1.0), (1.4, 2.5)]
    event = _event(0.0, 2.5, [
        ("hello", 0.10, 0.40),
        (" world", 1.45, 1.70),
    ])

    projected, _ = reproject_events_to_members([event], members)

    assert [item.text for item in projected] == ["hello", "world"]


def test_input_events_are_not_mutated():
    members = [(0.0, 1.0), (1.25, 2.5)]
    event = _event(0.0, 2.5, [
        ("你", 0.10, 0.30),
        ("吗", 1.30, 1.50),
    ])

    reproject_events_to_members([event], members)

    assert event.start == pytest.approx(0.0)
    assert event.end == pytest.approx(2.5)
    assert len(event.words) == 2
    assert event.revision_trace == []


def test_empty_members_passthrough():
    event = _event(0.0, 1.0, [("好", 0.1, 0.5)])

    projected, stats = reproject_events_to_members([event], [])

    assert stats.events_out == 1
    assert projected == [event]
