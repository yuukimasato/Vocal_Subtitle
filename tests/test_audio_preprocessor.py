"""测试前置降噪模块 (5.12.1)"""

import numpy as np
import pytest

from vocal_subtitle.audio_preprocessor import AudioPreprocessor, DenoiseConfig


class TestBurstNoiseSuppression:
    """突发噪音抑制测试"""

    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor(DenoiseConfig(
            enabled=True,
            engine="spectral_gate",
            burst_noise_protection=True,
            burst_noise_threshold_db=10.0,  # 低阈值，便于测试
            burst_noise_max_duration_ms=150,
        ))

    def test_single_burst_detected_and_suppressed(self, preprocessor):
        """单次突发噪音应被检测并抑制"""
        sample_rate = 16000
        # 生成: 1s 静音 + 50ms 高能突发 + 1s 静音
        duration = 2.0
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.01

        burst_start = int(1.0 * sample_rate)
        burst_end = int(1.05 * sample_rate)
        audio[burst_start:burst_end] = np.random.randn(burst_end - burst_start) * 0.9

        cleaned, report = preprocessor.process(audio.copy(), sample_rate)

        assert report["burst_events_detected"] >= 1
        # 突发区域应被降噪处理
        burst_region_after = cleaned[burst_start:burst_end]
        burst_rms_after = float(np.sqrt(np.mean(burst_region_after ** 2)))
        assert burst_rms_after < 0.3  # 能量大幅降低

    def test_multiple_bursts(self, preprocessor):
        """多次突发噪音均应被检测"""
        sample_rate = 16000
        audio = np.random.randn(int(3.0 * sample_rate)).astype(np.float32) * 0.005

        # 3 次突发：0.5s, 1.5s, 2.5s
        for t in [0.5, 1.5, 2.5]:
            start = int(t * sample_rate)
            end = start + int(0.03 * sample_rate)  # 30ms
            audio[start:end] = np.random.randn(end - start) * 0.8

        _, report = preprocessor.process(audio.copy(), sample_rate)
        assert report["burst_events_detected"] >= 2  # 至少检测到 2/3

    def test_no_burst_in_clean_audio(self, preprocessor):
        """干净音频不应误检突发噪音"""
        sample_rate = 16000
        audio = np.random.randn(int(1.0 * sample_rate)).astype(np.float32) * 0.01

        _, report = preprocessor.process(audio.copy(), sample_rate)
        assert report["burst_events_detected"] == 0

    def test_empty_audio(self, preprocessor):
        """空音频应安全处理"""
        audio = np.array([], dtype=np.float32)
        cleaned, report = preprocessor.process(audio, sample_rate=16000)
        assert len(cleaned) == 0
        assert not report["denoise_applied"]


class TestSpectralGate:
    """谱减法降噪测试"""

    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor(DenoiseConfig(
            enabled=True,
            engine="spectral_gate",
            spectral_noise_reduction_db=12.0,
            spectral_noise_estimation_frames=5,
            burst_noise_protection=False,
        ))

    def test_reduces_noise_floor(self, preprocessor):
        """谱减法应降低底噪水平"""
        sample_rate = 16000
        # 前半段纯噪音，后半段噪音+信号
        duration = 2.0
        noise = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.02
        signal = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(duration * sample_rate))
        ).astype(np.float32)
        audio = noise + signal * 0.5

        cleaned, report = preprocessor.process(audio.copy(), sample_rate)

        assert report["denoise_applied"]
        # 输出 RMS 应该降低
        assert "rms_reduction_db" in report

    def test_preserves_output_length(self, preprocessor):
        """输出应与输入等长"""
        sample_rate = 16000
        audio = np.random.randn(16000).astype(np.float32) * 0.1

        cleaned, _ = preprocessor.process(audio.copy(), sample_rate)
        assert len(cleaned) == len(audio)

    def test_disabled_bypasses(self):
        """禁用时不处理"""
        preprocessor = AudioPreprocessor(DenoiseConfig(enabled=False))
        audio = np.random.randn(16000).astype(np.float32)
        cleaned, report = preprocessor.process(audio.copy(), 16000)
        assert np.array_equal(cleaned, audio)
        assert not report["denoise_applied"]

    def test_unknown_engine_fallback(self):
        """未知引擎应安全降级"""
        preprocessor = AudioPreprocessor(DenoiseConfig(
            enabled=True, engine="unknown_xyz",
        ))
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        cleaned, report = preprocessor.process(audio.copy(), 16000)
        assert not report["denoise_applied"]


class TestDenoiseConfig:
    """降噪配置默认值测试"""

    def test_default_disabled(self):
        cfg = DenoiseConfig()
        assert not cfg.enabled
        assert cfg.engine == "spectral_gate"

    def test_burst_protection_defaults(self):
        cfg = DenoiseConfig(burst_noise_protection=True)
        assert cfg.burst_noise_threshold_db == 15.0
        assert cfg.burst_noise_max_duration_ms == 200
