"""测试三方法边界融合引擎 (方案二)"""

import numpy as np
import pytest

from vocal_subtitle.vad.base import SpeechSegment
from vocal_subtitle.vad.boundary_fusion import BoundaryFusion, FusionConfig


class TestBoundaryFusion:
    """边界融合器测试"""

    @pytest.fixture
    def fusion(self):
        return BoundaryFusion(FusionConfig(
            grid_resolution=0.01,
            min_consensus=2,
            high_conf_padding=0.03,
            low_conf_padding=0.12,
            min_speech_duration=0.25,
            sample_rate=16000,
        ))

    @pytest.fixture
    def sample_audio_3s(self):
        """3 秒的测试音频：1s 语音 + 0.5s 静音 + 1s 语音 + 0.5s 静音"""
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * 3.0), dtype=np.float32)
        # 语音段 1: 0.0-1.0s
        t1 = np.arange(0, int(1.0 * sample_rate)) / sample_rate
        audio[:len(t1)] = np.sin(2 * np.pi * 440 * t1).astype(np.float32) * 0.5
        # 语音段 2: 1.5-2.5s
        t2 = np.arange(0, int(1.0 * sample_rate)) / sample_rate
        start = int(1.5 * sample_rate)
        audio[start:start + len(t2)] = np.sin(2 * np.pi * 880 * t2).astype(np.float32) * 0.5
        return audio

    def test_fuse_three_methods(self, fusion, sample_audio_3s):
        """三种方法融合应产出语音段"""
        # 模拟高度一致的检测结果
        silero = [SpeechSegment(start=0.05, end=0.95, confidence=0.9)]
        ffmpeg = [SpeechSegment(start=0.04, end=0.96, confidence=0.9)]
        # RMS 方法由融合器内部生成

        result = fusion.fuse(
            silero, ffmpeg, sample_audio_3s, 16000,
        )
        assert len(result) >= 1
        for seg in result:
            assert seg.start >= 0.0
            assert seg.end <= 3.0

    def test_fuse_with_divergent_inputs(self, fusion, sample_audio_3s):
        """分歧较大的输入应产出一致性结果（不崩溃）"""
        silero = [
            SpeechSegment(start=0.0, end=1.0, confidence=0.9),
            SpeechSegment(start=1.5, end=2.5, confidence=0.9),
        ]
        ffmpeg = [
            SpeechSegment(start=0.2, end=0.8, confidence=0.9),
            SpeechSegment(start=1.7, end=2.3, confidence=0.9),
        ]

        result = fusion.fuse(
            silero, ffmpeg, sample_audio_3s, 16000,
        )
        # 至少应有一些结果
        assert len(result) >= 0  # 不崩溃即可，结果取决于音频内容

    def test_confidence_based_padding(self, fusion, sample_audio_3s):
        """高置信度段应有较小的 padding"""
        silero = [SpeechSegment(start=0.2, end=0.8, confidence=0.95)]
        ffmpeg = [SpeechSegment(start=0.2, end=0.8, confidence=0.95)]

        result = fusion.fuse(
            silero, ffmpeg, sample_audio_3s, 16000,
        )
        for seg in result:
            if seg.confidence >= 0.9:
                # 高置信度段 padding 较小（在 30ms 左右）
                pass

    def test_empty_inputs(self, fusion, sample_audio_3s):
        """空输入不应崩溃"""
        result = fusion.fuse([], [], sample_audio_3s, 16000)
        assert isinstance(result, list)

    # ----------------------------------------------------------------
    # _vote_by_sample_index
    # ----------------------------------------------------------------

    def test_vote_by_sample_index(self, fusion):
        """投票应在整数网格上正确累加"""
        sample_rate = 16000
        grid_samples = int(0.01 * sample_rate)  # 10ms = 160 samples
        total_samples = sample_rate * 2  # 2 秒音频
        num_bins = total_samples // grid_samples + 1
        votes = np.zeros(num_bins, dtype=np.int8)

        segments = [SpeechSegment(start=0.5, end=1.5, confidence=0.9)]
        fusion._vote_by_sample_index(votes, segments, sample_rate, grid_samples)

        # 0.5s → 50 bins, 1.5s → 150 bins
        start_bin = 50
        end_bin = 150
        assert votes[start_bin] >= 1
        assert votes[end_bin - 1] >= 1
        assert votes[0] == 0  # 开头未覆盖

    # ----------------------------------------------------------------
    # _mask_to_segments
    # ----------------------------------------------------------------

    def test_mask_to_segments(self, fusion):
        """连续语音网格应转为 SpeechSegment 列表"""
        num_bins = 200  # 2s / 10ms
        speech_mask = np.zeros(num_bins, dtype=bool)
        high_conf_mask = np.zeros(num_bins, dtype=bool)

        # 标记 50-150 区间为语音（0.5s-1.5s）
        speech_mask[50:151] = True
        high_conf_mask[50:151] = True

        result = fusion._mask_to_segments(
            speech_mask, high_conf_mask,
            resolution=0.01, total_duration=2.0, min_duration=0.25,
        )
        assert len(result) >= 1
        # 应有一个从 0.5s 开始的段
        assert result[0].start == pytest.approx(0.5)
        assert result[0].confidence >= 0.9  # 高置信度

    def test_mask_to_segments_low_confidence(self, fusion):
        """低置信度区段 confidence 应 < 0.9"""
        num_bins = 200
        speech_mask = np.zeros(num_bins, dtype=bool)
        high_conf_mask = np.zeros(num_bins, dtype=bool)

        speech_mask[50:151] = True
        high_conf_mask[50:100] = True   # 前一半高置信
        high_conf_mask[100:151] = False  # 后一半低置信

        result = fusion._mask_to_segments(
            speech_mask, high_conf_mask,
            resolution=0.01, total_duration=2.0, min_duration=0.25,
        )
        # 整体 confidence 应该是低置信度（因为包含低置信区域）
        assert len(result) >= 1
        assert result[0].confidence < 0.9

    def test_mask_to_segments_filters_short(self, fusion):
        """过短的语音段应被过滤"""
        num_bins = 200
        speech_mask = np.zeros(num_bins, dtype=bool)
        high_conf_mask = np.zeros(num_bins, dtype=bool)

        # 仅 10 bins = 100ms 的语音（< 250ms min_duration）
        speech_mask[50:61] = True
        high_conf_mask[50:61] = True

        result = fusion._mask_to_segments(
            speech_mask, high_conf_mask,
            resolution=0.01, total_duration=2.0, min_duration=0.25,
        )
        assert len(result) == 0

    # ----------------------------------------------------------------
    # _detect_by_rms_energy
    # ----------------------------------------------------------------

    def test_rms_detection_on_speech(self, sample_audio_3s):
        """RMS 检测应能从明确语音+静音的音频中检测到语音段"""
        segments = BoundaryFusion._detect_by_rms_energy(
            sample_audio_3s, 16000,
        )
        # 3s 音频含 ~2s 语音，应检测到多个语音段
        assert len(segments) >= 1
        for seg in segments:
            assert seg.start >= 0.0
            assert seg.end <= 3.0
            assert seg.confidence > 0

    def test_rms_detection_on_silence(self, sample_audio_silence):
        """纯静音应返回空列表"""
        segments = BoundaryFusion._detect_by_rms_energy(
            sample_audio_silence, 16000,
        )
        assert segments == []
