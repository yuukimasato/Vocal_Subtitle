"""测试 ffmpeg silencedetect VAD 引擎 (方案一)"""

import shutil

import numpy as np
import pytest

from vocal_subtitle.vad.base import SpeechSegment
from vocal_subtitle.vad.ffmpeg_vad import FFmpegSilenceVAD, unified_ffmpeg_pass


class TestFFmpegSilenceVAD:
    """ffmpeg VAD 引擎测试"""

    def test_name(self):
        vad = FFmpegSilenceVAD()
        assert vad.name == "ffmpeg-silence"

    def test_load_model_noop(self):
        """load_model 应为空操作（ffmpeg 无需加载模型）"""
        vad = FFmpegSilenceVAD()
        vad.load_model()  # 不应抛异常

    # ----------------------------------------------------------------
    # _invert_intervals (纯逻辑，无外部依赖)
    # ----------------------------------------------------------------

    def test_invert_empty_silence(self):
        """无静音区间 → 整段为语音"""
        result = FFmpegSilenceVAD._invert_intervals([], total_duration=10.0)
        assert result == [(0.0, 10.0)]

    def test_invert_single_silence_middle(self):
        """中间一段静音 → 两段语音"""
        result = FFmpegSilenceVAD._invert_intervals(
            [(3.0, 5.0)], total_duration=10.0,
        )
        assert result == [(0.0, 3.0), (5.0, 10.0)]

    def test_invert_leading_silence(self):
        """开头静音 → 从第一个语音段开始"""
        result = FFmpegSilenceVAD._invert_intervals(
            [(0.0, 2.0)], total_duration=10.0,
        )
        assert result == [(2.0, 10.0)]

    def test_invert_trailing_silence(self):
        """结尾静音 → 最后一段语音到静音开始处"""
        result = FFmpegSilenceVAD._invert_intervals(
            [(7.0, 10.0)], total_duration=10.0,
        )
        assert result == [(0.0, 7.0)]

    def test_invert_min_speech_filter(self):
        """太短的语音段应被过滤"""
        result = FFmpegSilenceVAD._invert_intervals(
            [(0.1, 5.0)], total_duration=10.0,
            min_speech_duration=0.25,
        )
        # 0.0-0.1 只有 100ms < 250ms，应被过滤
        assert len(result) == 1
        assert result[0] == (5.0, 10.0)

    def test_invert_multiple_silences(self):
        """多个静音区间"""
        result = FFmpegSilenceVAD._invert_intervals(
            [(2.0, 3.0), (6.0, 7.0)], total_duration=10.0,
        )
        assert result == [(0.0, 2.0), (3.0, 6.0), (7.0, 10.0)]

    # ----------------------------------------------------------------
    # detect (需要 ffmpeg 在 PATH 中)
    # ----------------------------------------------------------------

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found in PATH",
    )
    def test_detect_on_synthetic_wav(self, sample_wav_file):
        """对合成音频文件检测语音区间"""
        vad = FFmpegSilenceVAD()
        segments = vad.detect(sample_wav_file)
        # 合成音频是 1s 的正弦波，应该检测到语音
        assert len(segments) >= 1
        for seg in segments:
            assert seg.start >= 0.0
            assert seg.end <= 1.0
            assert seg.confidence > 0.0

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found in PATH",
    )
    def test_detect_silence_intervals(self, temp_dir, sample_audio_mono):
        """_detect_silence 应返回静音区间列表"""
        from vocal_subtitle.utils.audio_utils import AudioUtils

        # 创建含明确静音段的音频：0.5s 语音 + 0.5s 静音 + 0.5s 语音
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * 1.5), dtype=np.float32)
        # 语音段 1: 0.0-0.5s
        t1 = np.arange(0, int(0.5 * sample_rate)) / sample_rate
        audio[:len(t1)] = np.sin(2 * np.pi * 440 * t1).astype(np.float32)
        # 语音段 2: 1.0-1.5s
        t2 = np.arange(0, int(0.5 * sample_rate)) / sample_rate
        audio[int(1.0 * sample_rate):] = np.sin(2 * np.pi * 880 * t2).astype(np.float32)

        wav_path = temp_dir / "test_silence.wav"
        AudioUtils.save_audio(audio, wav_path)

        silence_intervals = FFmpegSilenceVAD._detect_silence(
            wav_path, noise_db=-30.0, min_silence_duration=0.3,
        )
        # 中间应有约 0.5s 的静音
        assert len(silence_intervals) >= 1

    # ----------------------------------------------------------------
    # detect_on_array
    # ----------------------------------------------------------------

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found in PATH",
    )
    def test_detect_on_array(self, sample_audio_mono):
        """detect_on_array 应通过临时文件检测"""
        vad = FFmpegSilenceVAD()
        segments = vad.detect_on_array(sample_audio_mono, 16000)
        assert len(segments) >= 1

    # ----------------------------------------------------------------
    # _get_duration
    # ----------------------------------------------------------------

    @pytest.mark.skipif(
        shutil.which("ffprobe") is None,
        reason="ffprobe not found in PATH",
    )
    def test_get_duration(self, sample_wav_file):
        """_get_duration 应返回正确时长"""
        duration = FFmpegSilenceVAD._get_duration(sample_wav_file)
        assert duration == pytest.approx(1.0, abs=0.1)


class TestUnifiedFFmpegPass:
    """统一 ffmpeg 调用测试"""

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found in PATH",
    )
    def test_unified_pass_returns_all_keys(self, sample_wav_file):
        """统一调用应返回 skeleton、coarse_speech、raw_silence_intervals"""
        result = unified_ffmpeg_pass(sample_wav_file)
        assert "skeleton" in result
        assert "coarse_speech" in result
        assert "raw_silence_intervals" in result
        assert "total_duration" in result
        assert result["total_duration"] > 0

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found in PATH",
    )
    def test_coarse_speech_are_speech_segments(self, sample_wav_file):
        """coarse_speech 应返回 SpeechSegment 列表"""
        result = unified_ffmpeg_pass(sample_wav_file)
        for seg in result["coarse_speech"]:
            assert isinstance(seg, SpeechSegment)
            assert seg.confidence > 0
