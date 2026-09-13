"""adaptive_silence_threshold_db: 骨架阈值按音频噪声底自适应推导。"""

from __future__ import annotations

import numpy as np
import pytest

from vocal_subtitle.acoustic.skeleton import adaptive_silence_threshold_db


def _tone_with_floor(amplitude: float, seconds: float = 3.0, sr: int = 16000):
    rng = np.random.default_rng(7)
    t = np.arange(int(sr * seconds)) / sr
    tone = amplitude * np.sin(2 * np.pi * 220.0 * t)
    noise = 0.0001 * rng.standard_normal(len(t))
    return (tone + noise).astype(np.float32)


def test_disabled_or_missing_audio_returns_fallback():
    audio = _tone_with_floor(0.01)
    assert adaptive_silence_threshold_db(audio, 16000, enabled=False) == -40.0
    assert adaptive_silence_threshold_db(None, 16000) == -40.0


def test_digital_silence_converges_to_lower_bound():
    # 全零音频的估计不可靠（util 内部 0.001 下限 → 极低噪声底），
    # 阈值收敛到钳制下界 -45，行为确定且无害。
    audio = np.zeros(16000 * 2, dtype=np.float32)
    threshold = adaptive_silence_threshold_db(audio, 16000)
    assert -45.0 <= threshold <= -30.0


def test_threshold_tracks_noise_floor_within_bounds():
    quiet = adaptive_silence_threshold_db(_tone_with_floor(0.001), 16000)
    loud = adaptive_silence_threshold_db(_tone_with_floor(0.05), 16000)

    # 噪声底越高，阈值越高；最终值始终钳制到 [-45, -30]（noise-shadow 同策略）
    assert loud > quiet
    for threshold in (quiet, loud):
        assert -45.0 <= threshold <= -30.0


def test_margin_configurable():
    audio = _tone_with_floor(0.01)
    low_margin = adaptive_silence_threshold_db(audio, 16000, margin_db=4.0)
    high_margin = adaptive_silence_threshold_db(audio, 16000, margin_db=14.0)
    assert high_margin >= low_margin


@pytest.mark.parametrize("fallback", [-35.0, -40.0, -50.0])
def test_fallback_value_passthrough(fallback):
    assert adaptive_silence_threshold_db(None, 16000, fallback_db=fallback) == fallback
