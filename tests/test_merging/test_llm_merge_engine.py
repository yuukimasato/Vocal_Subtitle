"""测试 LLM 语义合并引擎 (方案五) 和帧级无缝衔接 (3.7)"""

import numpy as np
import pytest

from vocal_subtitle.merging.llm_merge_engine import (
    LLMMergeEngine,
    MergeDecisionConfig,
    apply_frame_seamless_stitching,
)
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


class TestLLMMergeEngineFastPath:
    """快路径（规则强制合并）测试 — 纯逻辑，无外部依赖"""

    @pytest.fixture
    def engine(self):
        return LLMMergeEngine(MergeDecisionConfig(
            fast_merge_max_gap=0.20,
            llm_decision_min_gap=0.20,
            llm_decision_max_gap=1.20,
            hard_split_min_gap=1.20,
            max_combined_duration=5.0,
            llm_tier="rule_only",  # 纯规则，不调 LLM
        ))

    def test_fast_merge_short_gap_same_speaker(self, engine):
        """间隔 < 200ms 且同一说话人 → 应合并"""
        fragments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Hello,", "speaker": "A",
             "gap_to_next_sec": 0.15, "gap_is_silent": True},
            {"id": 2, "start": 1.15, "end": 2.0, "text": "world", "speaker": "A",
             "gap_to_next_sec": None, "gap_is_silent": None},
        ]
        # 补齐间隙信息
        fragments = engine._ensure_gap_info(fragments)
        result = engine._apply_fast_merges(fragments)
        # 间隔 0.15 < 0.20，同一说话人，应合并
        assert len(result) == 1
        assert result[0].get("_fast_merged") is True

    def test_no_merge_large_gap(self, engine):
        """间隔 > 200ms → 不在快路径处理范围"""
        fragments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Hello.", "speaker": "A",
             "gap_to_next_sec": 0.50, "gap_is_silent": True},
            {"id": 2, "start": 1.50, "end": 2.5, "text": "World.", "speaker": "A",
             "gap_to_next_sec": None, "gap_is_silent": None},
        ]
        fragments = engine._ensure_gap_info(fragments)
        result = engine._apply_fast_merges(fragments)
        # 间隔 0.50 > 0.20，不触发快路径
        assert len(result) == 2

    def test_no_merge_different_speaker(self, engine):
        """不同说话人不合并（即使间隔小）"""
        fragments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Hello?", "speaker": "A",
             "gap_to_next_sec": 0.10, "gap_is_silent": True},
            {"id": 2, "start": 1.10, "end": 2.0, "text": "Yes.", "speaker": "B",
             "gap_to_next_sec": None, "gap_is_silent": None},
        ]
        fragments = engine._ensure_gap_info(fragments)
        result = engine._apply_fast_merges(fragments)
        # speaker A != B → 不合并
        assert len(result) == 2

    def test_no_merge_sentence_ending(self, engine):
        """以句尾标点结束的片段不触发快路径合并"""
        fragments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Done.", "speaker": "A",
             "gap_to_next_sec": 0.10, "gap_is_silent": True},
            {"id": 2, "start": 1.10, "end": 2.0, "text": "Next.", "speaker": "A",
             "gap_to_next_sec": None, "gap_is_silent": None},
        ]
        fragments = engine._ensure_gap_info(fragments)
        result = engine._apply_fast_merges(fragments)
        # "Done." 以句号结尾，不应触发快路径
        assert len(result) == 2

    def test_fast_merge_preserves_start_end(self, engine):
        """快路径合并后 start 取第一个片段，end 取最后一个"""
        fragments = [
            {"id": 1, "start": 10.0, "end": 12.0, "text": "Part one,", "speaker": "A",
             "gap_to_next_sec": 0.10, "gap_is_silent": True},
            {"id": 2, "start": 12.10, "end": 14.0, "text": "part two", "speaker": "A",
             "gap_to_next_sec": None, "gap_is_silent": None},
        ]
        fragments = engine._ensure_gap_info(fragments)
        result = engine._apply_fast_merges(fragments)
        assert len(result) == 1
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 14.0

    def test_single_fragment_no_merge(self, engine):
        """单片段不触发合并"""
        fragments = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Solo.", "speaker": "A"},
        ]
        result = engine.merge(fragments)
        assert len(result) == 1


class TestFallbackRuleDecisions:
    """降级规则决策测试"""

    @pytest.fixture
    def engine(self):
        return LLMMergeEngine(MergeDecisionConfig(llm_tier="rule_only"))

    def test_comma_ending_merge(self, engine):
        """逗号结尾 + 中短间隙 → 应合并"""
        candidates = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Before ending,",
             "gap_to_next_sec": 0.30, "gap_is_silent": True},
        ]
        groups = engine._fallback_rule_decisions(candidates)
        assert len(groups) >= 1
        assert groups[0]["ids"] == [1, 2]

    def test_sentence_ending_no_merge(self, engine):
        """句尾标点 + 长间隙 → 不应产生合并组"""
        candidates = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Done.",
             "gap_to_next_sec": 0.80, "gap_is_silent": True},
        ]
        groups = engine._fallback_rule_decisions(candidates)
        # 句尾标点 + gap > 0.4 → 不合并
        assert len(groups) == 0

    def test_short_gap_merge(self, engine):
        """短间隙 (<400ms) → 倾向合并"""
        candidates = [
            {"id": 1, "start": 0.0, "end": 0.8, "text": "Quick",
             "gap_to_next_sec": 0.25, "gap_is_silent": True},
        ]
        groups = engine._fallback_rule_decisions(candidates)
        assert len(groups) >= 1

    def test_chinese_punctuation(self, engine):
        """中文标点也正确识别"""
        candidates = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "你好，",
             "gap_to_next_sec": 0.30, "gap_is_silent": True},
        ]
        groups = engine._fallback_rule_decisions(candidates)
        assert len(groups) >= 1  # 逗号结尾 → 合并

        candidates2 = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "你好。",
             "gap_to_next_sec": 0.80, "gap_is_silent": True},
        ]
        groups2 = engine._fallback_rule_decisions(candidates2)
        assert len(groups2) == 0  # 句号结尾 + 长间隙 → 不合并


class TestBuildMergeInput:
    """构建 LLM 合并输入测试"""

    def test_build_input_structure(self):
        """构建的输入应有正确结构"""
        engine = LLMMergeEngine()
        fragments = [
            {"start": 0.0, "end": 1.5, "text": "Hello", "speaker": "A"},
            {"start": 1.8, "end": 3.0, "text": "World", "speaker": "A"},
        ]
        result = engine.build_merge_input(fragments)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["gap_to_next_sec"] == pytest.approx(0.3)
        assert result[1]["gap_to_next_sec"] is None  # 最后一个

    def test_build_input_with_audio(self, sample_audio_speech_silence_mix):
        """有音频时应计算 gap_is_silent"""
        engine = LLMMergeEngine()
        fragments = [
            {"start": 0.0, "end": 0.5, "text": "First", "speaker": "A"},
            {"start": 0.7, "end": 1.0, "text": "Second", "speaker": "A"},
        ]
        result = engine.build_merge_input(
            fragments, audio=sample_audio_speech_silence_mix, sample_rate=16000,
        )
        assert len(result) == 2
        assert "gap_is_silent" in result[0]
        assert "gap_energy_ratio" in result[0]

    def test_ensure_gap_info_fills_missing(self):
        """_ensure_gap_info 应补充缺失的间隙信息"""
        engine = LLMMergeEngine()
        fragments = [
            {"start": 0.0, "end": 1.0, "text": "A", "speaker": "A"},
            {"start": 2.0, "end": 3.0, "text": "B", "speaker": "A"},
        ]
        result = engine._ensure_gap_info(fragments)
        assert result[0]["gap_to_next_sec"] == pytest.approx(1.0)


class TestFrameSeamlessStitching:
    """帧级无缝衔接测试 (3.7)"""

    def test_non_terminal_stitched(self):
        """非句尾字幕应衔接到下一句"""
        events = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="Hello,"),
            SubtitleEvent(index=2, start=1.05, end=2.5, text="world."),
        ]
        result = apply_frame_seamless_stitching(events, max_stitch_gap=0.12)
        # gap = 0.05 < 0.12, "Hello," 非句尾，应衔接
        assert result[0].end == pytest.approx(1.05)

    def test_terminal_not_stitched(self):
        """句尾标点字幕不应衔接"""
        events = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="Done."),
            SubtitleEvent(index=2, start=1.05, end=2.5, text="Next."),
        ]
        original_end = events[0].end
        result = apply_frame_seamless_stitching(events, max_stitch_gap=0.12)
        # "Done." 以句号结尾 → 不衔接
        assert result[0].end == original_end

    def test_gap_too_large(self):
        """间隙超过 max_stitch_gap → 不衔接"""
        events = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="Hello,"),
            SubtitleEvent(index=2, start=1.50, end=2.5, text="world."),
        ]
        result = apply_frame_seamless_stitching(events, max_stitch_gap=0.12)
        # gap = 0.50 > 0.12 → 不衔接
        assert result[0].end == 1.0

    def test_chinese_sentence_endings(self):
        """中文句尾标点也正确识别"""
        events = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="你好，"),
            SubtitleEvent(index=2, start=1.05, end=2.5, text="世界。"),
        ]
        result = apply_frame_seamless_stitching(events, max_stitch_gap=0.12)
        # "你好，" 逗号非句尾 → 衔接
        assert result[0].end == pytest.approx(1.05)

        events2 = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="你好。"),
            SubtitleEvent(index=2, start=1.05, end=2.5, text="世界。"),
        ]
        result2 = apply_frame_seamless_stitching(events2, max_stitch_gap=0.12)
        # "你好。" 句号结尾 → 不衔接
        assert result2[0].end == 1.0

    def test_empty_events(self):
        """空列表不崩溃"""
        result = apply_frame_seamless_stitching([], max_stitch_gap=0.12)
        assert result == []

    def test_single_event(self):
        """单条字幕不崩溃"""
        events = [SubtitleEvent(index=1, start=0.0, end=1.0, text="Solo.")]
        result = apply_frame_seamless_stitching(events, max_stitch_gap=0.12)
        assert len(result) == 1


# ------------------------------------------------------------------
# 5.12.4 字幕断句排版
# ------------------------------------------------------------------


class TestAutoLineBreak:
    """自动断行测试"""

    def test_short_text_no_break(self):
        """短文本不需要断行"""
        from vocal_subtitle.merging.llm_merge_engine import auto_line_break_fallback

        text = "Hello world"
        result = auto_line_break_fallback(text, max_chars_per_line=20)
        assert "\\N" not in result
        assert result == text

    def test_long_text_break_at_comma(self):
        """长文本应在逗号处断行"""
        from vocal_subtitle.merging.llm_merge_engine import auto_line_break_fallback

        text = "So I have you down for a non-smoking king room, is that right?"
        result = auto_line_break_fallback(text, max_chars_per_line=20)
        assert "\\N" in result
        line1, line2 = result.split("\\N")
        assert "room," in line1 or line1.endswith("room")
        # 逗号在位置 46 时 line1 为 47 字符，在 2.5× 扩展窗口内是合理的
        assert len(line1) <= 50

    def test_break_at_conjunction(self):
        """无逗号时应在连词处断行"""
        from vocal_subtitle.merging.llm_merge_engine import auto_line_break_fallback

        text = "Please confirm your reservation and we will send you a confirmation email"
        result = auto_line_break_fallback(text, max_chars_per_line=20)
        assert "\\N" in result

    def test_force_break_at_midpoint(self):
        """完全无自然断点时强制在中点附近空格处断行"""
        from vocal_subtitle.merging.llm_merge_engine import auto_line_break_fallback

        text = "ThisIsAVeryLongStringThatCannotBeBroken" * 2
        result = auto_line_break_fallback(text, max_chars_per_line=10)
        assert "\\N" in result

    def test_already_has_break_skipped(self):
        """已有换行标记的文本不应被破坏"""
        from vocal_subtitle.merging.llm_merge_engine import auto_line_break_fallback

        text = "Line1\\NLine2"
        result = auto_line_break_fallback(text)
        assert "\\N" in result


class TestAutoLayoutEvents:
    """字幕事件自动排版测试"""

    def test_long_event_gets_layout(self):
        """长字幕应自动断行"""
        from vocal_subtitle.merging.llm_merge_engine import auto_layout_events

        events = [
            type("FakeEvent", (), {
                "start": 0.0, "end": 2.0,
                "text": "Please confirm your reservation details and check the special requests before proceeding",
            })(),
        ]

        result = auto_layout_events(events, max_chars_cjk=20, max_chars_latin=40)
        assert "\\N" in result[0].text

    def test_cjk_text_uses_cjk_threshold(self):
        """中文文本应使用 CJK 字符阈值"""
        from vocal_subtitle.merging.llm_merge_engine import auto_layout_events

        long_cjk = "一二三四五六七八九十一二三四五六七八九十一二三四五"
        events = [
            type("FakeEvent", (), {
                "start": 0.0, "end": 2.0, "text": long_cjk,
            })(),
        ]

        result = auto_layout_events(events, max_chars_cjk=20, max_chars_latin=40)
        assert "\\N" in result[0].text

    def test_short_event_no_change(self):
        """短字幕不应被修改"""
        from vocal_subtitle.merging.llm_merge_engine import auto_layout_events

        events = [
            type("FakeEvent", (), {
                "start": 0.0, "end": 1.0, "text": "Hello",
            })(),
        ]

        result = auto_layout_events(events)
        assert result[0].text == "Hello"


class TestLayoutSuggestions:
    """LLM 断行建议应用测试"""

    def test_apply_valid_suggestions(self):
        """有效的断行建议应被应用"""
        from vocal_subtitle.merging.llm_merge_engine import apply_layout_suggestions

        events = [
            type("FakeEvent", (), {
                "start": 0.0, "end": 2.0,
                "text": "Original long text here",
            })(),
            type("FakeEvent", (), {
                "start": 2.5, "end": 4.0,
                "text": "Another text",
            })(),
        ]

        suggestions = [{
            "group_id": 0,
            "line1": "Original long",
            "line2": "text here",
        }]

        result = apply_layout_suggestions(events, suggestions)
        assert "\\N" in result[0].text
        assert "Original long" in result[0].text
        assert result[1].text == "Another text"

    def test_invalid_group_id_ignored(self):
        """无效的 group_id 不应崩溃"""
        from vocal_subtitle.merging.llm_merge_engine import apply_layout_suggestions

        events = [
            type("FakeEvent", (), {
                "start": 0.0, "end": 2.0, "text": "Test text",
            })(),
        ]

        suggestions = [{"group_id": 99, "line1": "X", "line2": "Y"}]
        result = apply_layout_suggestions(events, suggestions)
        assert result[0].text == "Test text"
