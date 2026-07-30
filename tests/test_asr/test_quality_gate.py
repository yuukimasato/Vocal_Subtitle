"""Tests for physical-evidence ASR quality gating."""

from types import SimpleNamespace

from vocal_subtitle.asr.quality_gate import evaluate_asr_quality


def event(start, end, text):
    return SimpleNamespace(
        start=start,
        end=end,
        physical_start=start,
        physical_end=end,
        text=text,
    )


def test_quality_gate_passes_when_events_cover_speech():
    result = evaluate_asr_quality(
        [event(0.0, 1.0, "你好"), event(1.2, 2.0, "世界")],
        [(0.0, 1.0), (1.2, 2.0)],
        10.0,
        min_coverage_ratio=0.9,
        min_text_density=0.5,
    )
    assert result.status == "pass"
    assert result.metrics["coverage_ratio"] == 1.0


def test_quality_gate_rejects_missing_physical_tail():
    result = evaluate_asr_quality(
        [event(0.0, 1.0, "hello")],
        [(0.0, 1.0), (4.0, 5.0)],
        5.0,
        min_coverage_ratio=0.8,
    )
    assert result.status == "failed"
    assert "low_physical_coverage" in result.reasons


def test_quality_gate_rejects_long_audio_single_sparse_event():
    result = evaluate_asr_quality(
        [event(0.0, 61.0, "只有一句")],
        [(0.0, 61.0)],
        61.0,
        max_event_duration=12.0,
        long_audio_seconds=60.0,
        long_audio_min_text_chars=12,
    )
    assert result.status == "failed"
    assert "abnormally_long_events" in result.reasons
    assert "long_audio_low_text_amount" in result.reasons


def test_quality_gate_rejects_overlapping_events():
    result = evaluate_asr_quality(
        [event(0.0, 2.0, "a"), event(1.0, 3.0, "b")],
        [(0.0, 3.0)],
        3.0,
        max_overlap_ratio=0.2,
    )
    assert result.status == "failed"
    assert "excessive_overlap" in result.reasons
