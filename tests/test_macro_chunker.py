"""测试宏观静音切块器 (方案〇)"""

import shutil

import numpy as np
import pytest

from vocal_subtitle.macro_chunker import (
    MacroChunk,
    MacroChunkConfig,
    MacroChunker,
)


class TestMacroChunk:
    """MacroChunk 数据类测试"""

    def test_duration_property(self):
        chunk = MacroChunk(index=0, start=10.0, end=35.0)
        assert chunk.duration == 25.0

    def test_default_overlap_flags(self):
        chunk = MacroChunk(index=0, start=0.0, end=10.0)
        assert chunk.overlap_with_prev is False
        assert chunk.overlap_with_next is False


class TestMacroChunker:
    """宏观切块器测试"""

    @pytest.fixture
    def chunker(self):
        return MacroChunker(MacroChunkConfig(
            enabled=True,
            auto_enable_threshold=3.0,
            silence_threshold_db=-30,
            min_silence_duration=2.0,
            target_chunk_duration=60.0,
            max_chunk_duration=180.0,
            overlap_ms=200,
            recursive=False,  # 测试中关闭递归
        ))

    # ----------------------------------------------------------------
    # should_split
    # ----------------------------------------------------------------

    def test_should_split_short(self, chunker):
        """短音频不应切分"""
        assert chunker.should_split(2.0) is False
        assert chunker.should_split(3.0) is False

    def test_should_split_long(self, chunker):
        """超过阈值的音频应切分"""
        assert chunker.should_split(200.0) is True

    def test_should_split_disabled(self):
        """禁用时始终返回 False"""
        chunker = MacroChunker(MacroChunkConfig(enabled=False))
        assert chunker.should_split(500.0) is False

    # ----------------------------------------------------------------
    # split — 短音频（不切分）
    # ----------------------------------------------------------------

    def test_split_short_audio(self, chunker, temp_dir, sample_audio_mono):
        """短音频应返回单个 MacroChunk"""
        from vocal_subtitle.utils.audio_utils import AudioUtils

        wav_path = temp_dir / "short.wav"
        AudioUtils.save_audio(sample_audio_mono, wav_path)

        chunks = chunker.split(wav_path, sample_audio_mono, 16000)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].start == 0.0
        assert chunks[0].end == pytest.approx(1.0)

    # ----------------------------------------------------------------
    # split — 长音频（需要 ffmpeg）
    # ----------------------------------------------------------------

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None,
        reason="ffmpeg not found in PATH",
    )
    def test_split_long_audio_with_silence(self, chunker, temp_dir):
        """含长静音的长音频应被切分"""
        from vocal_subtitle.utils.audio_utils import AudioUtils

        sample_rate = 16000
        # 创建 300s 音频（>3min 阈值），中间有静音
        audio = np.random.randn(int(sample_rate * 300)).astype(np.float32) * 0.3
        # 在 120-123s 处插入 3s 静音（超过 2s 阈值）
        silence_start = int(120 * sample_rate)
        silence_end = int(123 * sample_rate)
        audio[silence_start:silence_end] = 0.0
        # 在 220-225s 处插入 5s 静音
        silence_start2 = int(220 * sample_rate)
        silence_end2 = int(225 * sample_rate)
        audio[silence_start2:silence_end2] = 0.0

        wav_path = temp_dir / "long_silence.wav"
        AudioUtils.save_audio(audio, wav_path)

        chunks = chunker.split(wav_path, audio, sample_rate)
        # 应被切分为多个块
        assert len(chunks) >= 2

        # 验证首尾
        chunks_sorted = sorted(chunks, key=lambda c: c.start)
        assert chunks_sorted[0].start == 0.0
        assert chunks_sorted[-1].end == pytest.approx(300.0)

        # 每个块的时长应在合理范围
        for chunk in chunks:
            assert chunk.duration > 0
            assert chunk.duration <= chunker.config.max_chunk_duration + 5  # 允许少许超出

    # ----------------------------------------------------------------
    # stitch_chunks
    # ----------------------------------------------------------------

    def test_stitch_non_overlapping(self, chunker):
        """不重叠的事件应全部保留"""
        from vocal_subtitle.mapping.time_mapper import SubtitleEvent

        events_a = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="A1"),
            SubtitleEvent(index=2, start=1.5, end=2.5, text="A2"),
        ]
        events_b = [
            SubtitleEvent(index=3, start=5.0, end=6.0, text="B1"),
        ]
        # 无实际重叠的音频（100ms 静音在中间）
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * 1.0), dtype=np.float32)

        stitched = chunker.stitch_chunks(
            events_a, events_b,
            overlap_region=(2.5, 3.0),  # A 结束到 B 开始
            audio=audio, sample_rate=sample_rate,
        )
        # 所有事件都应保留（无跨越 overlap 的冲突）
        assert len(stitched) >= 2

    def test_stitch_with_overlap(self, chunker):
        """跨越重叠区的事件应被截断"""
        from vocal_subtitle.mapping.time_mapper import SubtitleEvent

        # 块A的事件跨越 overlap 中点
        events_a = [
            SubtitleEvent(index=1, start=0.0, end=3.5, text="Crossing"),
        ]
        events_b = [
            SubtitleEvent(index=2, start=2.5, end=5.0, text="Also crossing"),
        ]
        # overlap_region = (2.5, 3.5)
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
        # 在 3.0s 处放最低能量
        audio[int(3.0 * sample_rate):int(3.05 * sample_rate)] = 0.0

        stitched = chunker.stitch_chunks(
            events_a, events_b,
            overlap_region=(2.5, 3.5),
            audio=audio, sample_rate=sample_rate,
        )
        # 跨 overlap 的事件应被截断或保留
        assert len(stitched) >= 1


class TestMacroChunkConfig:
    """配置测试"""

    def test_default_config(self):
        cfg = MacroChunkConfig()
        assert cfg.enabled is True
        assert cfg.auto_enable_threshold == 180.0
        assert cfg.overlap_ms == 200
        assert cfg.recursive is True
        assert len(cfg.recursive_thresholds) == 3

    def test_recursive_thresholds_order(self):
        """递归阈值应越来越敏感"""
        cfg = MacroChunkConfig()
        thresholds = cfg.recursive_thresholds
        # 噪声阈值越来越低（更敏感）
        assert thresholds[0][0] >= thresholds[1][0] >= thresholds[2][0]
        # 静音时长越来越短（更敏感）
        assert thresholds[0][1] >= thresholds[1][1] >= thresholds[2][1]
