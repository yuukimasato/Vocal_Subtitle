"""Tests for speaker-change detection from acoustic features."""

import numpy as np
import pytest

from vocal_subtitle.diarization.speaker_change import (
    SpeakerChangeSignal,
    SpeakerChangeConfig,
    SpeakerChangeResult,
    detect_speaker_change_from_features,
    detect_volume_surge,
)


# ── SpeakerChangeSignal ──────────────────────────────────────────────

def test_signal_rejects_negative_time():
    with pytest.raises(ValueError, match="start"):
        SpeakerChangeSignal(start=-0.1, end=1.0, signal_type="embedding_distance",
                            confidence=0.8, source="embedding")


def test_signal_rejects_start_gt_end():
    with pytest.raises(ValueError, match="start.*end"):
        SpeakerChangeSignal(start=1.5, end=0.5, signal_type="embedding_distance",
                            confidence=0.8, source="embedding")


def test_signal_clamps_confidence():
    sig = SpeakerChangeSignal(start=0.0, end=1.0, signal_type="rms_surge",
                              confidence=1.5, source="rms")
    assert sig.confidence == 1.0


def test_signal_requires_valid_type():
    with pytest.raises(ValueError, match="signal_type"):
        SpeakerChangeSignal(start=0.0, end=1.0, signal_type="",
                            confidence=0.5, source="rms")


def test_signal_to_dict():
    sig = SpeakerChangeSignal(start=1.0, end=2.5, signal_type="embedding_distance",
                              confidence=0.75, source="embedding", metadata={"dist": 0.95})
    d = sig.to_dict()
    assert d["start"] == 1.0
    assert d["end"] == 2.5
    assert d["signal_type"] == "embedding_distance"
    assert d["confidence"] == 0.75
    assert d["source"] == "embedding"
    assert d["metadata"] == {"dist": 0.95}


# ── SpeakerChangeConfig ──────────────────────────────────────────────

def test_config_defaults():
    cfg = SpeakerChangeConfig()
    assert cfg.rms_surge_ratio > 1.0
    assert cfg.min_signal_duration > 0.0
    assert cfg.hard_boundary_threshold >= 0.5
    assert 0.0 <= cfg.candidate_threshold < cfg.hard_boundary_threshold


def test_config_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        SpeakerChangeConfig(candidate_threshold=0.9, hard_boundary_threshold=0.5)


def test_config_rejects_negative_rms_ratio():
    with pytest.raises(ValueError):
        SpeakerChangeConfig(rms_surge_ratio=0.5)


# ── detect_speaker_change_from_features ──────────────────────────────

def _fake_features(n_frames=10, n_features=87):
    rng = np.random.RandomState(42)
    return rng.randn(n_frames, n_features).astype(np.float64)


def test_detect_no_change_with_similar_features():
    """Identical features should not trigger speaker change."""
    feats1 = _fake_features(5)
    feats2 = feats1.copy()  # identical

    result = detect_speaker_change_from_features(
        feats1, feats2,
        time1=(0.0, 0.5),
        time2=(0.5, 1.0),
    )
    assert not result.is_hard_boundary
    assert result.confidence < 0.5


def test_detect_no_change_with_empty_features():
    """Empty feature arrays should return a clear no-change result."""
    result = detect_speaker_change_from_features(
        np.empty((0, 87)), np.empty((0, 87)),
        time1=(0.0, 0.5),
        time2=(0.5, 1.0),
    )
    assert not result.is_hard_boundary
    assert result.confidence == 0.0
    assert result.signal_type == "insufficient_data"


def test_detect_reports_feature_dimensionality_mismatch():
    result = detect_speaker_change_from_features(
        _fake_features(5, 87), _fake_features(5, 50),
        time1=(0.0, 0.5), time2=(0.5, 1.0),
    )
    assert not result.is_hard_boundary
    assert result.confidence == 0.0


def test_detect_candidate_signal():
    """Moderately different features should produce a candidate signal."""
    feats1 = _fake_features(10, 87)
    feats2 = feats1 + 0.8  # moderate shift

    result = detect_speaker_change_from_features(
        feats1, feats2,
        time1=(0.0, 1.0), time2=(1.0, 2.0),
    )
    assert result.confidence > 0.0
    assert result.signal_type in ("embedding_distance", "feature_divergence",
                                  "insufficient_data", "no_change")


def test_detect_returns_valid_times():
    """The change_time should reflect the boundary between the two segments."""
    result = detect_speaker_change_from_features(
        _fake_features(5), _fake_features(5),
        time1=(1.0, 2.5), time2=(2.5, 4.0),
    )
    assert result.before_start == 1.0
    assert result.after_end == 4.0
    # change_time is the end of the first signal = 2.5 (the boundary)
    assert result.change_time == 2.5


# ── detect_volume_surge ──────────────────────────────────────────────

def test_volume_surge_detected():
    """Rms surge above ratio should be detected."""
    # Use longer audio to meet min_signal_duration
    low = np.ones(3200, dtype=np.float32) * 0.01
    high = np.ones(3200, dtype=np.float32) * 0.1
    signal = detect_volume_surge(low, high, sample_rate=16000, surge_ratio=3.0)
    assert signal.signal_type == "rms_surge"
    assert signal.confidence > 0.0


def test_volume_surge_not_detected_for_equal_energy():
    # Use longer audio to meet min duration
    audio = np.ones(3200, dtype=np.float32) * 0.05
    signal = detect_volume_surge(audio, audio, sample_rate=16000, surge_ratio=3.0)
    assert signal.signal_type == "rms_stable"


def test_volume_surge_handles_silence():
    # Use longer audio to meet min duration
    silent = np.zeros(3200, dtype=np.float32)
    normal = np.ones(3200, dtype=np.float32) * 0.05
    signal = detect_volume_surge(silent, normal, sample_rate=16000, surge_ratio=3.0)
    # From silence to speech — detected but low confidence
    assert signal.signal_type in ("rms_surge", "rms_stable")


def test_volume_surge_rejects_short_audio():
    short = np.ones(100, dtype=np.float32) * 0.01
    signal = detect_volume_surge(short, short, sample_rate=16000, surge_ratio=3.0)
    assert signal.signal_type == "insufficient_data"


# ── SpeakerChangeResult ──────────────────────────────────────────────

def test_result_to_dict():
    signals = [
        SpeakerChangeSignal(start=1.0, end=3.0, signal_type="embedding_distance",
                            confidence=0.85, source="embedding"),
    ]
    result = SpeakerChangeResult(
        signals=tuple(signals),
        is_hard_boundary=True,
        confidence=0.85,
        signal_type="embedding_distance",
        evidence_ids=("diar-turn-1",),
    )
    d = result.to_dict()
    assert d["is_hard_boundary"] is True
    assert d["confidence"] == 0.85
    assert len(d["signals"]) == 1
    assert d["evidence_ids"] == ["diar-turn-1"]


def test_result_empty_signals():
    result = SpeakerChangeResult(
        signals=(),
        is_hard_boundary=False,
        confidence=0.0,
        signal_type="insufficient_data",
    )
    assert not result.is_hard_boundary
    assert result.confidence == 0.0
    assert result.change_time is None
