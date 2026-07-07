"""测试 WebRTC VAD 引擎"""

import numpy as np

from vocal_subtitle.vad.webrtc_vad import WebRTCVAD


class TestWebRTCVAD:
    """WebRTC VAD 引擎单元测试"""

    def test_engine_name(self):
        engine = WebRTCVAD()
        assert engine.name == "webrtc"

    def test_valid_sample_rates(self):
        engine = WebRTCVAD()
        assert 16000 in engine.VALID_SAMPLE_RATES
        assert 8000 in engine.VALID_SAMPLE_RATES
        assert 32000 in engine.VALID_SAMPLE_RATES
        assert 48000 in engine.VALID_SAMPLE_RATES

    def test_detect_interface_exists(self):
        engine = WebRTCVAD()
        assert hasattr(engine, "detect")
        assert hasattr(engine, "detect_on_array")
        assert hasattr(engine, "load_model")
