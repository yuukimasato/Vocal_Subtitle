"""Tests for display timeline mapping."""

import pytest

from vocal_subtitle.mapping.display_timeline import (
    DisplayCue,
    DisplayTimelineConfig,
    map_to_display_timeline,
)


# ── DisplayTimelineConfig ────────────────────────────────────────────

def test_config_defaults():
    cfg = DisplayTimelineConfig()
    assert cfg.max_lead_ms > 0
    assert cfg.max_trail_ms > 0
    assert cfg.min_reading_duration_ms > 0
    assert cfg.max_reading_duration_ms >= cfg.min_reading_duration_ms


def test_config_rejects_negative_lead():
    with pytest.raises(ValueError):
        DisplayTimelineConfig(max_lead_ms=-10)


def test_config_rejects_invalid_reading_range():
    with pytest.raises(ValueError):
        DisplayTimelineConfig(min_reading_duration_ms=5000, max_reading_duration_ms=1000)


# ── DisplayCue ───────────────────────────────────────────────────────

def test_cue_validates_physical_envelope():
    cue = DisplayCue(
        index=1,
        physical_start=1.0,
        physical_end=3.0,
        display_start=0.9,
        display_end=3.1,
        text="hello",
    )
    assert cue.duration_ms == 2200.0


def test_cue_display_must_cover_physical():
    with pytest.raises(ValueError, match="display must cover"):
        DisplayCue(
            index=1,
            physical_start=1.0,
            physical_end=3.0,
            display_start=2.0,  # after physical start
            display_end=3.5,
            text="missed start",
        )


def test_cue_enforces_positive_times():
    with pytest.raises(ValueError):
        DisplayCue(
            index=1,
            physical_start=-0.5,
            physical_end=3.0,
            display_start=0.0,
            display_end=3.1,
            text="negative",
        )


def test_cue_requires_end_gt_start():
    with pytest.raises(ValueError):
        DisplayCue(
            index=1,
            physical_start=3.0,
            physical_end=1.0,
            display_start=0.0,
            display_end=3.1,
            text="inverted",
        )


def test_cue_timing_degraded_bypasses_display_coverage():
    """When timing_degraded=True, display doesn't need to cover physical."""
    cue = DisplayCue(
        index=1,
        physical_start=1.0,
        physical_end=3.0,
        display_start=2.0,  # doesn't cover physical start
        display_end=3.5,
        text="degraded event",
        timing_degraded=True,
    )
    assert cue.timing_degraded


def test_cue_to_dict():
    cue = DisplayCue(
        index=1,
        physical_start=1.0,
        physical_end=3.0,
        display_start=0.9,
        display_end=3.1,
        text="hello world",
        speaker_id=1,
        speaker_label="Speaker A",
        source_word_ids=["w1", "w2"],
        physical_spans=[{"physical_clip_id": "clip-a", "start": 0.9, "end": 3.1}],
        warnings=["test_warning"],
    )
    d = cue.to_dict()
    assert d["index"] == 1
    assert d["physical_start"] == 1.0
    assert d["display_start"] == 0.9
    assert d["text"] == "hello world"
    assert d["speaker_id"] == 1
    assert d["source_word_ids"] == ["w1", "w2"]


# ── map_to_display_timeline ──────────────────────────────────────────

def test_map_simple_group():
    groups = [
        {
            "physical_start": 1.0,
            "physical_end": 2.5,
            "text": "simple subtitle",
            "index": 1,
            "projected_start": 0.9,
            "projected_end": 2.6,
        },
    ]

    cues = map_to_display_timeline(groups)

    assert len(cues) == 1
    cue = cues[0]
    assert cue.text == "simple subtitle"
    assert cue.physical_start == 1.0
    assert cue.physical_end == 2.5
    # display snaps to projected boundaries, clamped by max_lead/trail
    assert cue.display_start <= 1.0  # must cover physical start
    assert cue.display_end >= 2.5    # must cover physical end


def test_map_respects_audio_duration():
    groups = [
        {
            "physical_start": 1.0,
            "physical_end": 2.0,
            "text": "clamped",
            "index": 1,
            "projected_start": 0.5,
            "projected_end": 2.5,
        },
    ]

    cues = map_to_display_timeline(groups, audio_duration=2.0)

    assert cues[0].display_end <= 2.0
    assert cues[0].display_start >= 0.0


def test_map_handles_min_reading_duration():
    """Very short events should be extended to min reading duration."""
    groups = [
        {
            "physical_start": 1.0,
            "physical_end": 1.1,  # very short
            "text": "short",
            "index": 1,
            "projected_start": 1.0,
            "projected_end": 1.1,
        },
    ]

    cues = map_to_display_timeline(
        groups,
        config=DisplayTimelineConfig(min_reading_duration_ms=600),
    )

    duration = (cues[0].display_end - cues[0].display_start) * 1000
    assert duration >= 600


def test_map_multiple_groups_renumbered():
    groups = [
        {"physical_start": 1.0, "physical_end": 2.0, "text": "first"},
        {"physical_start": 3.0, "physical_end": 4.0, "text": "second"},
    ]

    cues = map_to_display_timeline(groups)

    assert cues[0].index == 1
    assert cues[1].index == 2


def test_map_timing_degraded_events():
    """timing_degraded events don't force display to cover physical."""
    groups = [
        {
            "physical_start": 2.0,
            "physical_end": 3.0,
            "text": "degraded",
            "index": 1,
            "projected_start": 2.5,
            "projected_end": 3.5,
            "timing_degraded": True,
        },
    ]

    cues = map_to_display_timeline(groups)

    assert cues[0].timing_degraded


def test_map_different_speakers_no_display_overlap():
    """Display times should not cross speaker boundaries."""
    groups = [
        {
            "physical_start": 1.0,
            "physical_end": 2.0,
            "text": "speaker A",
            "index": 1,
            "speaker_id": 1,
            "projected_start": 0.9,
            "projected_end": 2.0,
        },
        {
            "physical_start": 2.2,
            "physical_end": 3.5,
            "text": "speaker B",
            "index": 2,
            "speaker_id": 2,
            "projected_start": 2.2,
            "projected_end": 3.6,
        },
    ]

    cues = map_to_display_timeline(groups)
    assert len(cues) == 2
    # Display times must not overlap
    assert cues[0].display_end <= cues[1].display_start or (
        cues[0].speaker_id == cues[1].speaker_id
    )


def test_map_preserves_physical_spans():
    spans = [{"physical_clip_id": "clip-a", "start": 1.0, "end": 2.5}]
    groups = [
        {
            "physical_start": 1.0,
            "physical_end": 2.5,
            "text": "test",
            "physical_spans": spans,
        },
    ]

    cues = map_to_display_timeline(groups)
    assert cues[0].physical_spans == spans
