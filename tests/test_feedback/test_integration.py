"""反馈学习模块 — 单元与集成测试

覆盖范围:
  - aligner: 1:1/N:1/1:N 匹配, 锚点切分, 对齐质量门控
  - diff_analyzer: 修改分类, 归因映射, 参数解耦
  - param_learner: EMA 更新, 学习率分级, 异常值过滤, 硬边界
  - user_profile: CRUD, 备份轮转, 回滚, 分级衰减
  - few_shot_builder: LRU 淘汰, 重复检测, 衰减淘汰
  - health_scorer: 4维加权, 自动回滚判断
  - impact_estimator: 影响预估方向正确性
  - conflict_detector: 震汤检测, 非震汤不误报
  - audio_fingerprint: 向量转换, 马氏距离, KNN 动态阈值
  - shadow_mode: 升级/丢弃决策

设计原则: 所有测试使用合成数据，不加载真实模型或音频文件。
"""

import json
import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 测试 fixtures — 合成 SubtitleEvent
# ---------------------------------------------------------------------------

# 使用简单的 dict-as-event 或动态创建 SubtitleEvent
# SubtitleEvent 来自 vocal_subtitle.mapping.time_mapper


def _make_event(index, start, end, text, speaker_label=None):
    """创建合成的 SubtitleEvent"""
    from vocal_subtitle.mapping.time_mapper import SubtitleEvent

    return SubtitleEvent(
        index=index,
        start=start,
        end=end,
        text=text,
        speaker_label=speaker_label,
    )


def _make_events(times_texts):
    """从 (start, end, text) 列表批量创建 SubtitleEvent"""
    events = []
    for i, (start, end, text) in enumerate(times_texts, 1):
        events.append(_make_event(i, start, end, text))
    return events


# ============================================================================
#  1. Aligner 测试
# ============================================================================


class TestFeedbackIntegration:
    """反馈学习全链路集成测试"""

    def test_full_align_analyze_learn_cycle(self):
        """对齐 → 分析 → 学习 → 保存 → 加载 全链路"""
        from vocal_subtitle.feedback import (
            DiffAnalyzer,
            ParamLearner,
            SubtitleAligner,
            UserProfileManager,
        )

        # 1. 合成数据 — 足够多的 1:1 事件以确保 coverage > 70%
        auto = _make_events([
            (0.0, 2.0, "今天天气不错"),
            (2.5, 5.0, "我们去看电影"),
            (5.5, 8.0, "你觉得怎么样"),
            (8.5, 10.5, "我觉得很不错"),
            (11.0, 13.0, "那就这样决定了"),
        ])
        # 修订版：结束时间后移 + 最后两句合并
        manual = _make_events([
            (0.0, 2.15, "今天天气不错"),
            (2.5, 5.15, "我们去看电影"),
            (5.5, 8.15, "你觉得怎么样"),
            (8.5, 13.0, "我觉得很不错那就这样决定了"),
        ])

        # 2. 对齐
        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner.align(auto, manual)
        matched = [p for p in pairs if p.is_matched]
        assert len(matched) > 0

        # 3. 差异分析
        analyzer = DiffAnalyzer(param_isolation_enabled=True)
        report = analyzer.analyze(pairs)
        assert report.alignment_coverage > 0

        # 4. 参数学习
        mgr = UserProfileManager()
        learner = ParamLearner(mgr)

        # 模拟已有 2 次反馈（本次是第 3 次，触发学习）
        profile = mgr.load("__test_integration__")
        profile["feedback_count"] = 2
        profile["history"] = [
            {"timestamp": "2026-07-01T10:00:00", "diff_report_summary": "增大padding",
             "alignment_coverage": 0.9, "median_semantic_similarity": 0.8, "adjustments": {}},
            {"timestamp": "2026-07-02T10:00:00", "diff_report_summary": "减小合并",
             "alignment_coverage": 0.92, "median_semantic_similarity": 0.85, "adjustments": {}},
        ]
        mgr.save(profile)

        updated = learner.learn_from_diff(
            diff_report=report,
            current_config_overrides={},
            profile_name="__test_integration__",
        )

        # 5. 验证
        loaded = mgr.load("__test_integration__")
        assert loaded["feedback_count"] == 3
        assert len(loaded["history"]) >= 3

        # Cleanup
        mgr.delete("__test_integration__")

    def test_subtitle_parsing_srt(self):
        """SRT 文件解析"""
        import tempfile

        from vocal_subtitle.feedback.aligner import parse_subtitle_file

        srt_content = """1
00:00:00,000 --> 00:00:02,000
今天天气不错

2
00:00:02,500 --> 00:00:05,000
我们去看电影吧

3
00:00:05,500 --> 00:00:08,000
你觉得怎么样
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, encoding="utf-8",
        ) as f:
            f.write(srt_content)
            srt_path = Path(f.name)

        try:
            events = parse_subtitle_file(srt_path)
            assert len(events) == 3
            assert events[0].text == "今天天气不错"
            assert events[0].start == pytest.approx(0.0)
            assert events[0].end == pytest.approx(2.0)
            assert events[1].text == "我们去看电影吧"
            assert events[2].text == "你觉得怎么样"
        finally:
            srt_path.unlink(missing_ok=True)

    def test_subtitle_parsing_ass(self):
        """ASS 文件解析"""
        import tempfile

        from vocal_subtitle.feedback.aligner import parse_subtitle_file

        ass_content = """[Script Info]
Title: Test

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,说话人A,0,0,0,,今天天气不错
Dialogue: 0,0:00:02.50,0:00:05.00,Default,说话人B,0,0,0,,我们去看电影吧
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ass", delete=False, encoding="utf-8",
        ) as f:
            f.write(ass_content)
            ass_path = Path(f.name)

        try:
            events = parse_subtitle_file(ass_path)
            assert len(events) == 2
            assert events[0].text == "今天天气不错"
            assert events[0].speaker_label == "说话人A"
            assert events[1].speaker_label == "说话人B"
        finally:
            ass_path.unlink(missing_ok=True)

    def test_unsupported_format_raises(self):
        """不支持的格式应抛出异常"""
        from vocal_subtitle.feedback.aligner import parse_subtitle_file

        with pytest.raises(ValueError, match="Unsupported"):
            parse_subtitle_file(Path("/tmp/test.txt"))

