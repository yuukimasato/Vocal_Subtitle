"""Deterministic tests for task-level ASR routing."""

import numpy as np
import pytest

from vocal_subtitle.asr.base import LanguageDetection
from vocal_subtitle.asr.router import ASRRouter, build_probe_windows
from vocal_subtitle.config import PipelineConfig


class FakeProbe:
    name = "faster-whisper"
    model_name = "tiny"

    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = 0

    def load_model(self):
        return None

    def detect_language_info(self, audio, sample_rate):
        value = self.detections[min(self.calls, len(self.detections) - 1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


def _router(detections, config=None):
    config = config or PipelineConfig()
    config.asr.auto_routing.language_probe_window_seconds = 1.0
    config.asr.auto_routing.language_probe_max_windows = 4
    fake_probe = FakeProbe(detections)
    router = ASRRouter(
        config,
        lambda engine, model=None, probe=False: fake_probe,
    )
    return router, fake_probe


def test_probe_windows_cover_short_audio_head_and_tail():
    windows = build_probe_windows(3.0, [(0.2, 2.8)], 1.0, 4)
    assert len(windows) >= 2
    assert windows[0][0] == 0.0
    assert windows[-1][1] >= 2.8


def test_all_high_probability_chinese_windows_select_funasr():
    router, probe = _router(
        [
            LanguageDetection("zh", 0.98, "tiny"),
            LanguageDetection("zh", 0.91, "tiny"),
            LanguageDetection("zh", 0.96, "tiny"),
        ]
    )
    decision = router.decide(np.zeros(3000, dtype=np.float32), 1000)
    assert decision.selected_engine == "funasr"
    assert decision.detected_language == "zh"
    assert decision.language_probability == 0.91
    assert probe.calls == 3


def test_any_english_window_selects_faster_whisper():
    router, _ = _router(
        [
            LanguageDetection("zh", 0.99, "tiny"),
            LanguageDetection("en", 0.99, "tiny"),
            LanguageDetection("zh", 0.99, "tiny"),
        ]
    )
    decision = router.decide(np.zeros(3000, dtype=np.float32), 1000)
    assert decision.selected_engine == "faster-whisper"
    assert decision.detected_language == "en"


def test_uncertain_or_probe_failure_selects_faster_whisper():
    router, _ = _router(
        [
            LanguageDetection("zh", 0.50, "tiny"),
            RuntimeError("probe failed"),
        ]
    )
    decision = router.decide(np.zeros(3000, dtype=np.float32), 1000)
    assert decision.selected_engine == "faster-whisper"
    assert decision.decision_reason == "language_probe_failed"


def test_low_confidence_non_chinese_probe_keeps_language_unlocked():
    router, _ = _router([LanguageDetection("ru", 0.28, "tiny")])

    decision = router.decide(np.zeros(1000, dtype=np.float32), 1000)

    assert decision.detected_language == "other"
    assert decision.language is None


@pytest.mark.parametrize("engine", ["faster-whisper", "whisper-cpp"])
def test_explicit_engine_is_never_overridden(engine):
    config = PipelineConfig()
    config.asr.engine = engine
    router, probe = _router([], config)
    decision = router.decide(np.zeros(1000, dtype=np.float32), 1000)
    assert decision.selected_engine == engine
    assert probe.calls == 0


def test_explicit_funasr_non_chinese_language_fails_before_model():
    config = PipelineConfig()
    config.asr.engine = "funasr"
    config.asr.language = "en"
    router, probe = _router([], config)
    with pytest.raises(ValueError, match="中文专用"):
        router.decide(np.zeros(1000, dtype=np.float32), 1000)
    assert probe.calls == 0


def test_auto_explicit_language_lock_skips_probe():
    config = PipelineConfig()
    config.asr.engine = "auto"
    config.asr.language = "en"
    router, probe = _router([], config)
    decision = router.decide(np.zeros(1000, dtype=np.float32), 1000)
    assert decision.selected_engine == "faster-whisper"
    assert decision.detected_language == "en"
    assert probe.calls == 0
