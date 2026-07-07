"""测试 TEN VAD 引擎"""

import numpy as np
import pytest

from vocal_subtitle.vad.base import SpeechSegment
from vocal_subtitle.vad.ten_vad import TENVAD


class TestTENVAD:
    """TEN VAD 引擎单元测试"""

    def test_engine_name(self):
        engine = TENVAD()
        assert engine.name == "ten"

    def test_load_model_fallback(self):
        """TEN VAD 在 SDK 不可用时应降级"""
        engine = TENVAD()
        engine.load_model()
        assert engine._is_loaded is True

    def test_detect_on_array_basic(self, sample_audio_mono):
        """使用基于能量的降级检测"""
        engine = TENVAD()
        engine._is_loaded = True  # 模拟已加载

        segments = engine.detect_on_array(
            sample_audio_mono,
            sample_rate=16000,
            threshold=0.5,
        )
        # 能量检测在正弦波上应找到语音段
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, SpeechSegment)
            assert seg.start < seg.end

    def test_detect_on_array_silence(self, sample_audio_silence):
        """静音不应检测到语音"""""
        engine = TENVAD()
        engine._is_loaded = True

        segments = engine.detect_on_array(
            sample_audio_silence,
            sample_rate=16000,
            threshold=0.5,
        )
        assert len(segments) == 0

    def test_detect_on_array_empty(self):
        """空数组应返回空列表"""
        engine = TENVAD()
        engine._is_loaded = True

        segments = engine.detect_on_array(
            np.array([], dtype=np.float32),
            sample_rate=16000,
        )
        assert segments == []
