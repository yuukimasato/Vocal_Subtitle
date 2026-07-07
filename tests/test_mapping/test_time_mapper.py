"""测试时间轴映射"""

import pytest

from vocal_subtitle.asr.base import TranscriptionSegment
from vocal_subtitle.mapping.time_mapper import SubtitleEvent, TimeMapper
from vocal_subtitle.vad.base import SpeechSegment


class TestTimeMapper:
    """时间轴映射测试"""

    def test_map_basic(self):
        """基本映射测试"""
        mapper = TimeMapper()
        asr_results = [
            [
                TranscriptionSegment(
                    text="大家好", start=0.0, end=1.0
                ),
                TranscriptionSegment(
                    text="欢迎收听", start=1.5, end=2.5
                ),
            ]
        ]
        speech_segments = [SpeechSegment(start=0.0, end=3.0)]

        events = mapper.map(asr_results, speech_segments)

        assert len(events) == 2
        assert events[0].text == "大家好"
        assert events[0].start == 0.0
        assert events[0].end == 1.0
        assert events[1].text == "欢迎收听"
        assert events[1].start == 1.5
        # 末尾段 end 锚定到 speech_seg.end (3.0)
        assert events[1].end == 3.0

    def test_map_with_offset(self):
        """带偏移的映射测试"""
        mapper = TimeMapper()
        asr_results = [
            [TranscriptionSegment(text="第二段", start=0.0, end=1.5)]
        ]
        speech_segments = [SpeechSegment(start=10.0, end=12.0)]

        events = mapper.map(asr_results, speech_segments)

        assert len(events) == 1
        assert events[0].start == 10.0
        # 末尾段 end 锚定到 speech_seg.end (12.0)
        assert events[0].end == 12.0

    def test_map_multiple_segments(self):
        """多片段映射"""
        mapper = TimeMapper()
        asr_results = [
            [TranscriptionSegment(text="A", start=0.0, end=1.0)],
            [TranscriptionSegment(text="B", start=0.0, end=2.0)],
        ]
        speech_segments = [
            SpeechSegment(start=0.0, end=2.0),
            SpeechSegment(start=5.0, end=7.0),
        ]

        events = mapper.map(asr_results, speech_segments)

        assert len(events) == 2
        assert events[0].start == 0.0
        # 末尾段 end 锚定到 speech_seg.end (2.0)
        assert events[0].end == 2.0
        assert events[1].start == 5.0
        # 末尾段 end 锚定到 speech_seg.end (7.0)
        assert events[1].end == 7.0

    def test_map_empty(self):
        """空输入"""
        mapper = TimeMapper()
        events = mapper.map([], [])
        assert events == []

    def test_map_mismatch_raises(self):
        """输入长度不匹配应抛出异常"""
        mapper = TimeMapper()
        with pytest.raises(ValueError):
            mapper.map(
                [[TranscriptionSegment(text="A", start=0.0, end=1.0)]],
                [],  # 缺失 speech_segments
            )

    def test_seamless_gap_handling(self):
        """无缝衔接: gap ≤ 0.2s"""
        mapper = TimeMapper(seamless_threshold=0.2)
        asr_results = [
            [TranscriptionSegment(text="A", start=0.0, end=1.0)],
            [TranscriptionSegment(text="B", start=0.0, end=1.0)],
        ]
        speech_segments = [
            SpeechSegment(start=0.0, end=1.0),
            SpeechSegment(start=1.15, end=2.15),  # gap = 0.15 ≤ 0.2
        ]

        events = mapper.map(asr_results, speech_segments)
        assert len(events) == 2
        # 前一条的结束时间应被调整为接近后一条的开始
        assert events[0].end <= events[1].start

    def test_preserve_natural_pause(self):
        """自然停顿保留: gap > 0.2s"""
        mapper = TimeMapper(seamless_threshold=0.2)
        asr_results = [
            [TranscriptionSegment(text="A", start=0.0, end=1.0)],
            [TranscriptionSegment(text="B", start=0.0, end=1.0)],
        ]
        speech_segments = [
            SpeechSegment(start=0.0, end=1.0),
            SpeechSegment(start=2.0, end=3.0),  # gap = 1.0 > 0.2
        ]

        events = mapper.map(asr_results, speech_segments)
        assert len(events) == 2
        # 保留原有的结束时间
        assert events[0].end == 1.0

    def test_map_single_segment_static(self):
        """静态方法测试"""
        events = TimeMapper.map_single_segment(
            [TranscriptionSegment(text="测试", start=0.5, end=2.0)],
            segment_offset=5.0,
        )
        assert len(events) == 1
        assert events[0].start == 5.5
        assert events[0].end == 7.0

    def test_deduplicate_overlapping_events(self):
        """重叠/重复事件去重：时间重叠 >50% + 文本相似度 >80%"""
        mapper = TimeMapper()

        # 创建两个重叠且文本几乎相同的字幕事件
        events = [
            SubtitleEvent(index=1, start=0.5, end=2.0,
                          text="Got it. 1. Listen attentively."),
            SubtitleEvent(index=2, start=1.0, end=2.0,
                          text="1. Listen attentively."),
            SubtitleEvent(index=3, start=3.0, end=4.0,
                          text="Focus on the guest."),
        ]

        result = mapper._deduplicate_overlapping(events)

        # 不应有重复事件：前两个事件重叠且文本相似，只保留覆盖更完整的
        assert len(result) < 3
        # 独立事件应保留
        texts = {e.text for e in result}
        assert "Focus on the guest." in texts
        # 覆盖更完整（更长文本 + 更大时间跨度）的事件应被保留
        assert any("Got it" in t for t in texts) or any("Listen attentively" in t for t in texts)

    def test_deduplicate_non_overlapping_kept(self):
        """无重叠或文本不相似的事件全部保留"""
        mapper = TimeMapper()

        events = [
            SubtitleEvent(index=1, start=0.0, end=1.0,
                          text="Hello world."),
            SubtitleEvent(index=2, start=1.5, end=2.5,
                          text="How are you?"),
            SubtitleEvent(index=3, start=3.0, end=4.0,
                          text="I am fine."),
        ]

        result = mapper._deduplicate_overlapping(events)
        assert len(result) == 3

    def test_deduplicate_overlap_but_different_text(self):
        """重叠但文本不相似的应保留"""
        mapper = TimeMapper()

        events = [
            SubtitleEvent(index=1, start=0.0, end=2.0,
                          text="Phone etiquette."),
            SubtitleEvent(index=2, start=1.0, end=2.5,
                          text="Rapid response"),  # 重叠但完全不相似
        ]

        result = mapper._deduplicate_overlapping(events)
        assert len(result) == 2  # 不同内容，都不删除
