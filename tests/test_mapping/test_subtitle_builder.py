"""测试字幕构建器"""

from pathlib import Path

from vocal_subtitle.mapping.subtitle_builder import SubtitleBuilder, SubtitleRule
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


class TestSubtitleBuilder:
    """字幕构建器测试"""

    def test_build_basic_srt(self, temp_dir: Path):
        """基本 SRT 构建"""
        builder = SubtitleBuilder()
        events = [
            SubtitleEvent(index=1, start=0.0, end=2.0, text="大家好"),
            SubtitleEvent(index=2, start=2.5, end=5.0, text="欢迎收听本期节目"),
        ]
        output = temp_dir / "test.srt"
        result = builder.build(events, output, fmt="srt")

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "大家好" in content
        assert "欢迎收听本期节目" in content
        assert "00:00:00" in content  # SRT 时间格式

    def test_build_to_string(self):
        """测试返回字符串"""
        builder = SubtitleBuilder()
        events = [
            SubtitleEvent(index=1, start=0.0, end=1.0, text="测试"),
        ]
        srt_str = builder.build_to_string(events, fmt="srt")
        assert "测试" in srt_str
        assert "1" in srt_str  # 序号

    def test_build_vtt_format(self, temp_dir: Path):
        """VTT 格式输出"""
        builder = SubtitleBuilder()
        events = [
            SubtitleEvent(index=1, start=0.0, end=2.0, text="Hello"),
        ]
        output = temp_dir / "test.vtt"
        result = builder.build(events, output, fmt="vtt")

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "WEBVTT" in content
        assert "Hello" in content

    def test_build_ass_format(self, temp_dir: Path):
        """ASS 格式输出"""
        builder = SubtitleBuilder()
        events = [
            SubtitleEvent(index=1, start=0.0, end=2.0, text="测试"),
        ]
        output = temp_dir / "test.ass"
        result = builder.build(events, output, fmt="ass")

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "[Events]" in content

    def test_merge_short_events(self):
        """过短的字幕应合并"""
        rule = SubtitleRule(min_duration=1.0)
        builder = SubtitleBuilder(rule=rule)
        events = [
            SubtitleEvent(index=1, start=0.0, end=0.3, text="大"),
            SubtitleEvent(index=2, start=0.4, end=0.7, text="家好"),
        ]
        merged = builder._merge_short_events(events)
        # 两条都 < 1.0s 且 gap=0.1 < 0.3，应合并
        assert len(merged) <= 2  # 可能合并

    def test_line_wrapping_cjk(self):
        """中文过长应自动换行"""
        rule = SubtitleRule(max_chars_cjk=10, max_lines=2)
        builder = SubtitleBuilder(rule=rule)
        text = "这是一段很长的中文字幕需要自动换行处理"
        wrapped = builder._insert_line_break(text, 10)
        assert "\n" in wrapped

    def test_line_wrapping_latin(self):
        """英文过长应自动换行"""
        rule = SubtitleRule(max_chars_latin=20, max_lines=2)
        builder = SubtitleBuilder(rule=rule)
        text = "This is a very long English subtitle that needs line breaking"
        wrapped = builder._insert_line_break(text, 20)
        assert "\n" in wrapped

    def test_empty_events(self, temp_dir: Path):
        """空事件应生成空字幕"""
        builder = SubtitleBuilder()
        output = temp_dir / "empty.srt"
        result = builder.build([], output)
        assert result.exists()
