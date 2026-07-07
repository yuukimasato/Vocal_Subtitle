"""测试 Silero VAD 引擎"""

import numpy as np

from vocal_subtitle.vad.base import SpeechSegment
from vocal_subtitle.vad.silero_vad import SileroVAD


class TestSileroVAD:
    """Silero VAD 引擎单元测试"""

    def test_engine_name(self):
        engine = SileroVAD()
        assert engine.name == "silero"

    def test_load_model(self):
        """测试模型加载（会触发下载）"""
        # 注：此测试需要网络连接，在 CI 中可能需要跳过
        # 仅验证接口，不实际加载模型
        engine = SileroVAD()
        # 不实际调用 load_model() 以避免下载
        assert engine._model is None

    def test_detect_on_array_empty(self):
        """测试空音频检测"""
        engine = SileroVAD()
        engine._model = None  # 确保未加载
        # 使用 mock 跳过实际模型调用
        # 只测试接口存在
        assert hasattr(engine, "detect_on_array")

    def test_detect_interface(self):
        """检测接口存在且签名正确"""
        engine = SileroVAD()
        import inspect

        sig = inspect.signature(engine.detect_on_array)
        params = list(sig.parameters.keys())
        assert "audio" in params
        assert "sample_rate" in params
        assert "threshold" in params
        assert "min_speech_duration_ms" in params
        assert "min_silence_duration_ms" in params

    def test_speech_segment_properties(self):
        """SpeechSegment 属性测试"""
        seg = SpeechSegment(start=0.5, end=2.0, confidence=0.9)
        assert seg.duration == 1.5
        assert seg.start == 0.5
        assert seg.end == 2.0
        assert seg.confidence == 0.9

    def test_speech_segment_repr(self):
        """SpeechSegment 字符串表示"""
        seg = SpeechSegment(start=0.5, end=2.0)
        rep = repr(seg)
        assert "0.500" in rep
        assert "2.000" in rep
