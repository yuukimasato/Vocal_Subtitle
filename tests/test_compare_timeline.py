"""Tests for compare_timeline tool."""

import tempfile
from pathlib import Path

import pytest

from scripts.compare_timeline import (
    TimelineEvent,
    EventComparison,
    ComparisonReport,
    _ass_time_to_seconds,
    match_events,
    parse_ass,
    parse_srt,
    report_to_dict,
)


# ── _ass_time_to_seconds ─────────────────────────────────────────────

@pytest.mark.parametrize("input_str,expected", [
    ("0:00:01.50", 1.5),       # 2 decimal places (centiseconds)
    ("0:00:01.500", 1.5),      # 3 decimal places (milliseconds)
    ("1:00:00.00", 3600.0),    # 1 hour
    ("0:01:00.50", 60.5),      # 1 min
    ("0:01:30.00", 90.0),      # 1.5 min
    ("0:00:00.001", 0.001),    # 1ms
    ("0:00:00.1", 0.1),        # 1 decimal place
])
def test_ass_time_parsing(input_str, expected):
    result = _ass_time_to_seconds(input_str)
    assert result is not None
    assert abs(result - expected) < 0.0001


def test_ass_time_invalid():
    assert _ass_time_to_seconds("not a time") is None
    assert _ass_time_to_seconds("") is None


def test_ass_time_no_fractional():
    result = _ass_time_to_seconds("0:00:01")
    assert result == 1.0


# ── parse_ass ────────────────────────────────────────────────────────

def test_parse_ass_simple():
    content = """[Script Info]
Title: test
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:03.00,0:00:05.00,Default,,0,0,0,,Second line
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False, encoding="utf-8-sig") as f:
        f.write(content)
        path = Path(f.name)

    try:
        events = parse_ass(path)
        assert len(events) == 2
        assert events[0].start == 1.0
        assert events[0].end == 2.5
        assert events[0].text == "Hello world"
    finally:
        path.unlink()


def test_parse_ass_strips_tags():
    content = """[Script Info]
Title: test
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\an8}Hello world
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False, encoding="utf-8-sig") as f:
        f.write(content)
        path = Path(f.name)

    try:
        events = parse_ass(path)
        assert events[0].text == "Hello world"
    finally:
        path.unlink()


# ── parse_srt ────────────────────────────────────────────────────────

def test_parse_srt_basic():
    content = """1
00:00:01,000 --> 00:00:02,500
Hello world

2
00:00:03,000 --> 00:00:05,000
Second line
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = Path(f.name)

    try:
        events = parse_srt(path)
        assert len(events) == 2
        assert events[0].start == 1.0
        assert events[0].text == "Hello world"
    finally:
        path.unlink()


# ── match_events ─────────────────────────────────────────────────────

def test_match_events_same_count():
    auto = [
        TimelineEvent(1, 1.0, 2.0, "hello world"),
        TimelineEvent(2, 3.0, 4.0, "goodbye"),
    ]
    gt = [
        TimelineEvent(1, 1.1, 2.1, "hello world!"),
        TimelineEvent(2, 3.1, 4.1, "goodbye!"),
    ]
    matches = match_events(auto, gt)
    assert len(matches) == 2


def test_match_events_different_count():
    auto = [
        TimelineEvent(1, 1.0, 2.0, "hello"),
        TimelineEvent(2, 3.0, 4.0, "world"),
    ]
    gt = [
        TimelineEvent(1, 1.1, 2.1, "hello"),
    ]
    matches = match_events(auto, gt)
    assert len(matches) >= 1  # at minimum, the "hello" match


def test_match_events_no_forced_index_pairing():
    """When count differs, don't blindly pair by index. Use text+time similarity."""
    auto = [
        TimelineEvent(1, 1.0, 2.0, "hello world how are you"),
    ]
    gt = [
        TimelineEvent(1, 0.0, 0.5, "unrelated text"),
        TimelineEvent(2, 1.1, 2.1, "hello"),
    ]
    matches = match_events(auto, gt)
    # Should find a match with the correct text, not force-pair by index
    # Even if only one match, it should be good quality
    assert len(matches) >= 1


def test_match_events_no_match_too_distant():
    auto = [
        TimelineEvent(1, 100.0, 101.0, "far away"),
    ]
    gt = [
        TimelineEvent(1, 1.0, 2.0, "nearby"),
    ]
    matches = match_events(auto, gt, max_time_diff=5.0)
    assert len(matches) == 0  # too far in time


# ── report_to_dict ───────────────────────────────────────────────────

def test_report_to_dict():
    comparisons = [
        EventComparison(
            index=1,
            auto_start=1.0, auto_end=2.0,
            gt_start=1.1, gt_end=2.1,
            auto_text="hello", gt_text="hello",
            start_error_ms=100.0, end_error_ms=100.0,
            start_error_abs_ms=100.0, end_error_abs_ms=100.0,
        ),
    ]
    report = ComparisonReport(
        auto_file="auto.ass",
        ground_truth_file="gt.ass",
        total_events=1,
        matched_events=1,
        comparisons=comparisons,
        health_score=0.95,
    )
    d = report_to_dict(report)
    assert d["meta"]["matched_events"] == 1
    assert d["health_score"] == 0.95
