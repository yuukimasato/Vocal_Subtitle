"""Feedback tests: test_integration."""

from pathlib import Path

import pytest

from .common import _make_events


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
        auto = _make_events(
            [
                (0.0, 2.0, "今天天气不错"),
                (2.5, 5.0, "我们去看电影"),
                (5.5, 8.0, "你觉得怎么样"),
                (8.5, 10.5, "我觉得很不错"),
                (11.0, 13.0, "那就这样决定了"),
            ]
        )
        # 修订版：结束时间后移 + 最后两句合并
        manual = _make_events(
            [
                (0.0, 2.15, "今天天气不错"),
                (2.5, 5.15, "我们去看电影"),
                (5.5, 8.15, "你觉得怎么样"),
                (8.5, 13.0, "我觉得很不错那就这样决定了"),
            ]
        )

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
            {
                "timestamp": "2026-07-01T10:00:00",
                "diff_report_summary": "增大padding",
                "alignment_coverage": 0.9,
                "median_semantic_similarity": 0.8,
                "adjustments": {},
            },
            {
                "timestamp": "2026-07-02T10:00:00",
                "diff_report_summary": "减小合并",
                "alignment_coverage": 0.92,
                "median_semantic_similarity": 0.85,
                "adjustments": {},
            },
        ]
        mgr.save(profile)

        learner.learn_from_diff(
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

    def test_learn_cold_start_persists_feedback_count(self):
        """冷启动回归：不预置 feedback_count，从 0 开始连续学习

        回归背景：learn_from_diff 的 observation-only 分支（feedback_count ≤ 2）
        此前只改内存不落盘，导致 feedback_count 永远停在 0，每次学习都被当作
        第 1 次反馈，参数覆盖永远无法跨过预热期写入。
        """
        from vocal_subtitle.feedback import (
            DiffAnalyzer,
            ParamLearner,
            SubtitleAligner,
            UserProfileManager,
        )

        auto = _make_events(
            [
                (0.0, 2.0, "今天天气不错"),
                (2.5, 5.0, "我们去看电影"),
                (5.5, 8.0, "你觉得怎么样"),
                (8.5, 10.5, "我觉得很不错"),
                (11.0, 13.0, "那就这样决定了"),
            ]
        )
        manual = _make_events(
            [
                (0.0, 2.15, "今天天气不错"),
                (2.5, 5.15, "我们去看电影"),
                (5.5, 8.15, "你觉得怎么样"),
                (8.5, 13.0, "我觉得很不错那就这样决定了"),
            ]
        )

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner.align(auto, manual)
        report = DiffAnalyzer(param_isolation_enabled=True).analyze(pairs)

        mgr = UserProfileManager()
        learner = ParamLearner(mgr)
        name = "__test_cold_start__"
        try:
            assert not mgr._profile_path(name).exists()

            overrides = {}
            for expected_count in (1, 2):
                overrides = learner.learn_from_diff(
                    diff_report=report,
                    current_config_overrides=overrides,
                    profile_name=name,
                )
                # 关键断言：观测分支必须落盘，计数必须累加（修复前两次均停在 0）
                loaded = mgr.load(name)
                assert loaded["feedback_count"] == expected_count
                assert len(loaded["history"]) == expected_count

            # 第 3 次跨过预热期：若本轮确有归因，应开始写入参数覆盖
            overrides = learner.learn_from_diff(
                diff_report=report,
                current_config_overrides=overrides,
                profile_name=name,
            )
            loaded = mgr.load(name)
            assert loaded["feedback_count"] == 3
            if report.attribution:
                assert loaded["overrides"], "跨过预热期后应产生参数覆盖"
                assert overrides == loaded["overrides"]
        finally:
            mgr.delete(name)
            assert not mgr._profile_path(name).exists()

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
            mode="w",
            suffix=".srt",
            delete=False,
            encoding="utf-8",
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
            mode="w",
            suffix=".ass",
            delete=False,
            encoding="utf-8",
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
