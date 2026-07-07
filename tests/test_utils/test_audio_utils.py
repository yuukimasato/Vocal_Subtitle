"""测试 AudioUtils 音频工具模块"""

import numpy as np
import pytest
from pathlib import Path

from vocal_subtitle.utils.audio_utils import AudioUtils


class TestAudioUtils:
    """音频工具单元测试"""

    def test_time_to_sample(self):
        """时间转采样点数"""
        assert AudioUtils.time_to_sample(1.0, 16000) == 16000
        assert AudioUtils.time_to_sample(0.5, 16000) == 8000
        assert AudioUtils.time_to_sample(0.0, 16000) == 0

    def test_sample_to_time(self):
        """采样点数转时间"""
        assert AudioUtils.sample_to_time(16000, 16000) == 1.0
        assert AudioUtils.sample_to_time(8000, 16000) == 0.5
        assert AudioUtils.sample_to_time(0, 16000) == 0.0

    def test_normalize_audio_float32(self):
        """float32 音频标准化（值在 [-1, 1] 内不变）"""
        audio = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        result = AudioUtils.normalize_audio(audio)
        assert result.dtype == np.float32

    def test_normalize_audio_int16(self):
        """int16 音频标准化"""
        audio = np.array([16384, -16384, 0], dtype=np.int16)
        result = AudioUtils.normalize_audio(audio)
        assert result.dtype == np.float32
        assert -1.0 <= result[0] <= 1.0

    def test_normalize_audio_int32(self):
        """int32 音频标准化"""
        audio = np.array([1073741824, -1073741824], dtype=np.int32)
        result = AudioUtils.normalize_audio(audio)
        assert result.dtype == np.float32
        assert -1.0 <= result[0] <= 1.0

    def test_extract_segment(self):
        """音频片段提取"""
        audio = np.arange(100, dtype=np.float32)
        seg = AudioUtils.extract_segment(audio, 10, 30)
        assert len(seg) == 20
        assert seg[0] == 10.0
        assert seg[-1] == 29.0

    def test_extract_segment_boundary_clamp(self):
        """边界裁剪"""
        audio = np.arange(100, dtype=np.float32)
        seg = AudioUtils.extract_segment(audio, -5, 200)
        assert len(seg) == 100  # clamped

    def test_save_and_load_wav(self, temp_dir: Path):
        """保存和加载 WAV 文件"""
        audio = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32) * 0.5
        out_path = temp_dir / "test.wav"
        result = AudioUtils.save_audio(audio, out_path)
        assert result.exists()

        loaded, sr = AudioUtils.load_audio(out_path)
        assert sr == 16000
        assert len(loaded) > 0
        assert loaded.dtype == np.float32

    def test_get_audio_info_wav(self, temp_dir: Path, sample_audio_mono):
        """获取 WAV 文件信息"""
        path = temp_dir / "info_test.wav"
        AudioUtils.save_audio(sample_audio_mono, path)

        info = AudioUtils.get_audio_info(path)
        assert info["channels"] == 1
        assert info["sample_rate"] == 16000
        assert info["sample_width"] == 2
        assert info["duration_seconds"] > 0

    def test_get_duration(self, temp_dir: Path):
        """获取音频时长"""
        audio = np.ones(32000, dtype=np.float32) * 0.1  # 2 秒
        path = temp_dir / "duration_test.wav"
        AudioUtils.save_audio(audio, path)

        duration = AudioUtils.get_duration_seconds(path)
        assert duration == pytest.approx(2.0, abs=0.1)

    def test_resample_linear(self):
        """线性插值重采样"""
        audio = np.sin(np.linspace(0, 10 * np.pi, 16000)).astype(np.float32)
        resampled = AudioUtils._resample(audio, 16000, 8000)
        assert len(resampled) == 8000
        assert abs(len(resampled) - len(audio) / 2) <= 1

    def test_resample_identity(self):
        """同采样率重采样应不变"""
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        resampled = AudioUtils._resample(audio, 100, 100)
        np.testing.assert_array_equal(audio, resampled)

    def test_normalize_audio_peak(self):
        """峰值归一化"""
        audio = np.array([0.1, -0.05, 0.02], dtype=np.float32)
        result = AudioUtils.normalize_audio(audio)
        max_val = np.max(np.abs(result))
        assert max_val == pytest.approx(1.0, abs=0.01)

    def test_normalize_audio_silence(self):
        """静音不应产生 NaN"""
        audio = np.zeros(100, dtype=np.float32)
        result = AudioUtils.normalize_audio(audio)
        assert not np.any(np.isnan(result))
        assert np.all(result == 0.0)

    def test_default_constants(self):
        """默认参数"""
        assert AudioUtils.DEFAULT_SAMPLE_RATE == 16000
        assert AudioUtils.DEFAULT_CHANNELS == 1
        assert AudioUtils.DEFAULT_SAMPLE_WIDTH == 2
