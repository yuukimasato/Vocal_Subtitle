"""测试片段合并策略"""

import numpy as np
import pytest

from vocal_subtitle.merging.merge_strategy import MergeConfig, MergeStrategy
from vocal_subtitle.vad.base import SpeechSegment


class TestMergeStrategy:
    """合并策略测试"""

    def test_empty_segments(self):
        strategy = MergeStrategy()
        result = strategy.merge([])
        assert result == []

    def test_single_segment(self):
        strategy = MergeStrategy()
        seg = SpeechSegment(start=0.0, end=3.0)
        result = strategy.merge([seg])
        assert len(result) == 1
        assert result[0].start == 0.0 - MergeConfig().padding
        # 尾部 padding 降低到 ≤80ms（end 锚定到声学边界）
        assert result[0].end == pytest.approx(3.08, abs=0.01)

    def test_merge_adjacent_within_gap(self):
        """间隔小于 min_silence_gap 的片段应合并"""
        strategy = MergeStrategy(
            MergeConfig(min_silence_gap=0.5, padding=0.0)
        )
        segments = [
            SpeechSegment(start=0.0, end=2.0),
            SpeechSegment(start=2.2, end=4.0),  # gap = 0.2 < 0.5
        ]
        result = strategy.merge(segments)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 4.0

    def test_keep_adjacent_beyond_gap(self):
        """间隔大于 min_silence_gap 的片段应保留"""
        strategy = MergeStrategy(
            MergeConfig(min_silence_gap=0.5, padding=0.0)
        )
        segments = [
            SpeechSegment(start=0.0, end=2.0),
            SpeechSegment(start=3.0, end=5.0),  # gap = 1.0 > 0.5
        ]
        result = strategy.merge(segments)
        assert len(result) == 2

    def test_padding_applied(self):
        """padding 应正确添加（尾部 padding 降低到 ≤30ms）"""
        strategy = MergeStrategy(MergeConfig(padding=0.1))
        seg = SpeechSegment(start=1.0, end=3.0)
        result = strategy.merge([seg])
        assert result[0].start == pytest.approx(0.9)
        # 尾部 padding 降低到 ≤80ms（end 锚定到声学边界）
        assert result[0].end == pytest.approx(3.08, abs=0.01)

    def test_clip_boundaries(self):
        """不应超出音频总时长"""
        strategy = MergeStrategy(MergeConfig(padding=0.5))
        seg = SpeechSegment(start=0.0, end=3.0)
        result = strategy.merge([seg], total_duration=3.0)
        assert result[0].start == 0.0
        assert result[0].end == 3.0

    def test_filter_short_segments(self):
        """过短片段应被过滤"""
        strategy = MergeStrategy(
            MergeConfig(min_segment_length=1.0, padding=0.0)
        )
        segments = [
            SpeechSegment(start=0.0, end=0.3),  # 太短，丢弃
            SpeechSegment(start=1.0, end=3.0),  # 保留
        ]
        result = strategy.merge(segments)
        assert len(result) == 1
        assert result[0].start == 1.0

    def test_sort_disordered_segments(self):
        """乱序片段应被正确排序"""
        strategy = MergeStrategy(MergeConfig(padding=0.0))
        segments = [
            SpeechSegment(start=3.0, end=5.0),
            SpeechSegment(start=0.0, end=2.0),
            SpeechSegment(start=2.0, end=3.0),
        ]
        result = strategy.merge(segments)
        # 0-2 和 2-3 的 gap=0 < 0.4，合并为 0-3，再和 3-5 的 gap=0 < 0.4，合并为 0-5
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 5.0

    def test_split_long_segment(self):
        """超长片段应在静音处切分"""
        sample_rate = 16000
        # 创建一个 60s 的伪音频，中间有静音段
        audio = np.ones(60 * sample_rate, dtype=np.float32) * 0.01
        # 在 30s 处插入一段明显的静音
        silence_mid = int(29.5 * sample_rate)
        silence_end = int(30.5 * sample_rate)
        audio[silence_mid:silence_end] = 0.0

        strategy = MergeStrategy(
            MergeConfig(
                max_segment_length=25.0,
                padding=0.0,
                pre_split_silence=False,   # 禁用预切分，专注测试超长段切分
                adaptive_padding=False,
            )
        )
        seg = SpeechSegment(start=0.0, end=60.0)
        result = strategy.merge([seg], audio=audio, sample_rate=sample_rate)

        # 应该被切分成多段
        assert len(result) >= 2
        for s in result:
            assert s.duration <= 25.0
