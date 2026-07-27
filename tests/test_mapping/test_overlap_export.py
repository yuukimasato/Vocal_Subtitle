"""Tests for overlap export module."""

import pytest

from vocal_subtitle.mapping.overlap_export import (
    OverlapTrack,
    OverlapGroup,
    OverlapExportConfig,
    group_overlapping_events,
    render_overlap_srt,
)


# ── OverlapTrack ─────────────────────────────────────────────────────

def test_track_formats_srt_line_name_only():
    track = OverlapTrack(text="Hello world", speaker_id=1, speaker_label="Alice")
    assert "Alice" in track.format_srt_line("name_only")


def test_track_formats_srt_line_colon():
    track = OverlapTrack(text="Hello world", speaker_id=1, speaker_label="Bob")
    line = track.format_srt_line("colon")
    assert line == "Bob: Hello world"


def test_track_formats_srt_line_bracket():
    track = OverlapTrack(text="Hello world", speaker_id=1, speaker_label="Charlie")
    line = track.format_srt_line("bracket")
    assert line == "[Charlie] Hello world"


def test_track_fallback_label():
    track = OverlapTrack(text="Hello", speaker_id=0)
    line = track.format_srt_line("name_only")
    assert "Speaker 0" in line


# ── OverlapGroup ─────────────────────────────────────────────────────

def test_group_srt_text_two_tracks():
    group = OverlapGroup(
        group_id="og-1",
        tracks=[
            OverlapTrack(text="I'm first", speaker_id=1, speaker_label="Alice"),
            OverlapTrack(text="No, I'm second", speaker_id=2, speaker_label="Bob"),
        ],
        start=1.0,
        end=2.5,
    )
    text = group.srt_text()
    assert "Alice" in text
    assert "Bob" in text
    assert "\n" in text
    assert "I'm first" in text


def test_group_srt_limits_to_two_lines():
    group = OverlapGroup(
        group_id="og-1",
        tracks=[
            OverlapTrack(text="A", speaker_id=1, speaker_label="Spk1"),
            OverlapTrack(text="B", speaker_id=2, speaker_label="Spk2"),
            OverlapTrack(text="C", speaker_id=3, speaker_label="Spk3"),
        ],
    )
    text = group.srt_text()
    lines = text.split("\n")
    assert len(lines) <= 2


def test_group_ass_events():
    group = OverlapGroup(
        group_id="og-1",
        tracks=[
            OverlapTrack(text="Track A", speaker_id=1, speaker_label="Alice"),
            OverlapTrack(text="Track B", speaker_id=2, speaker_label="Bob"),
        ],
    )
    events = group.ass_events()
    assert len(events) == 2
    assert events[0]["an_position"] != events[1]["an_position"]
    assert events[0]["overlap_group_id"] == "og-1"


def test_group_to_dict():
    group = OverlapGroup(
        group_id="og-1",
        tracks=[
            OverlapTrack(text="hello", speaker_id=1, speaker_label="Alice",
                         source_word_ids=["w1"], start=1.0, end=2.0),
        ],
        start=1.0,
        end=2.0,
        warnings=["test"],
    )
    d = group.to_dict()
    assert d["group_id"] == "og-1"
    assert len(d["tracks"]) == 1
    assert d["tracks"][0]["source_word_ids"] == ["w1"]


# ── OverlapExportConfig ──────────────────────────────────────────────

def test_config_defaults():
    cfg = OverlapExportConfig()
    assert cfg.max_tracks == 2
    assert len(cfg.ass_track_positions) == 2


# ── group_overlapping_events ─────────────────────────────────────────

class _FakeEvent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_group_returns_empty_for_no_overlap():
    events = [
        _FakeEvent(start=0, end=1, text="a", genuine_overlap=False),
    ]
    groups = group_overlapping_events(events)
    assert groups == []


def test_group_creates_group_for_overlapping_events():
    events = [
        _FakeEvent(start=1.0, end=2.5, text="Speaker A", genuine_overlap=True,
                   overlap_group_id="og-1", speaker_id=1, speaker_label="Alice",
                   physical_start=1.0, physical_end=2.5, speaker_source="diarization"),
        _FakeEvent(start=1.2, end=2.7, text="Speaker B", genuine_overlap=True,
                   overlap_group_id="og-1", speaker_id=2, speaker_label="Bob",
                   physical_start=1.2, physical_end=2.7, speaker_source="diarization"),
    ]
    groups = group_overlapping_events(events)

    assert len(groups) == 1
    assert groups[0].group_id == "og-1"
    assert groups[0].track_count == 2


def test_group_deduplicates_same_speaker():
    events = [
        _FakeEvent(start=1.0, end=2.5, text="A1", genuine_overlap=True,
                   overlap_group_id="og-1", speaker_id=1, speaker_label="Alice",
                   physical_start=1.0, physical_end=2.5, speaker_source="diarization"),
        _FakeEvent(start=1.2, end=2.7, text="A2", genuine_overlap=True,
                   overlap_group_id="og-1", speaker_id=1, speaker_label="Alice",
                   physical_start=1.2, physical_end=2.7, speaker_source="diarization"),
    ]
    groups = group_overlapping_events(events)

    assert groups[0].track_count == 1  # same speaker deduplicated


def test_group_flags_unverified():
    events = [
        _FakeEvent(start=1.0, end=2.5, text="X", genuine_overlap=True,
                   overlap_group_id="og-1", speaker_id=1, speaker_label="Unknown",
                   physical_start=1.0, physical_end=2.5, speaker_source="unknown"),
    ]
    groups = group_overlapping_events(events)

    assert not groups[0].verified
    assert "unverified_speaker_attribution" in groups[0].warnings


# ── render_overlap_srt ───────────────────────────────────────────────

def test_render_srt_empty_groups():
    result = render_overlap_srt([])
    assert result == ""


def test_render_srt_single_group():
    groups = [
        OverlapGroup(
            group_id="og-1",
            tracks=[
                OverlapTrack(text="Hello", speaker_id=1, speaker_label="Alice"),
                OverlapTrack(text="Hi there", speaker_id=2, speaker_label="Bob"),
            ],
            start=1.5,
            end=3.0,
        ),
    ]
    srt = render_overlap_srt(groups)
    assert "00:00:01,500" in srt
    assert "00:00:03,000" in srt
    assert "Alice" in srt
    assert "Bob" in srt


def test_render_srt_multiple_groups():
    groups = [
        OverlapGroup(
            group_id="og-1",
            tracks=[OverlapTrack(text="A", speaker_id=1, speaker_label="S1")],
            start=1.0, end=2.0,
        ),
        OverlapGroup(
            group_id="og-2",
            tracks=[OverlapTrack(text="B", speaker_id=2, speaker_label="S2")],
            start=3.0, end=4.0,
        ),
    ]
    srt = render_overlap_srt(groups)
    assert "1\n" in srt
    assert "2\n" in srt
    assert "S1" in srt
    assert "S2" in srt
