"""Feedback tests: test_quality."""

from .common import *

class TestHealthScorer:
    """健康度评分器测试"""

    def test_perfect_health_score(self):
        """完全对齐 → 健康度接近 100"""
        from vocal_subtitle.feedback.aligner import AlignmentPair, _time_iou, _levenshtein_similarity
        from vocal_subtitle.feedback.health_scorer import compute_health_score_from_pairs

        auto = _make_events([
            (0.0, 2.0, "今天天气真不错"),
            (2.5, 5.0, "我们去看电影吧"),
            (5.5, 8.0, "你觉得怎么样"),
        ])
        manual = _make_events([
            (0.0, 2.0, "今天天气真不错"),
            (2.5, 5.0, "我们去看电影吧"),
            (5.5, 8.0, "你觉得怎么样"),
        ])

        pairs = [
            AlignmentPair(
                auto_events=[a], manual_events=[m], match_type="1:1",
                time_iou=_time_iou(a.start, a.end, m.start, m.end),
                text_similarity=_levenshtein_similarity(a.text, m.text),
                semantic_similarity=1.0, composite_score=1.0,
            )
            for a, m in zip(auto, manual)
        ]

        overall, detail = compute_health_score_from_pairs(pairs)
        assert overall > 85
        assert detail["alignment_coverage"] > 85
        assert detail["structure_consistency"] > 85

    def test_auto_rollback_triggered(self):
        """健康度下降 30% → 触发回滚"""
        from vocal_subtitle.feedback.health_scorer import should_auto_rollback

        should, reason = should_auto_rollback(80.0, 50.0, drop_threshold=0.3)
        assert should  # 下降 37.5% > 30%
        assert "37%" in reason or "38%" in reason

    def test_auto_rollback_not_triggered(self):
        """健康度下降 < 30% → 不触发回滚"""
        from vocal_subtitle.feedback.health_scorer import should_auto_rollback

        should, reason = should_auto_rollback(80.0, 70.0, drop_threshold=0.3)
        assert not should  # 下降 12.5% < 30%

    def test_zero_baseline_no_rollback(self):
        """无基线健康度 → 不触发回滚"""
        from vocal_subtitle.feedback.health_scorer import should_auto_rollback

        should, reason = should_auto_rollback(0.0, 50.0, drop_threshold=0.3)
        assert not should


# ============================================================================
#  7. ImpactEstimator 测试
# ============================================================================

class TestImpactEstimator:
    """参数变更影响预估器测试"""

    def test_padding_increase_impact(self):
        """增大 padding → 时长增加、截断降低、行数减少"""
        from vocal_subtitle.feedback.diff_analyzer import ParamAdjustment
        from vocal_subtitle.feedback.impact_estimator import ImpactEstimator

        estimator = ImpactEstimator()
        adj = ParamAdjustment(
            param_path="merging.padding",
            param_tier="short_term",
            observed_value=0.04,
            confidence=0.8,
            learn_weight=1.0,
            direction="increase",
            reason="结束时间后移",
        )

        impacts = estimator.estimate({"merging.padding": adj}, {"merging": {"padding": 0.10}})
        assert len(impacts) == 1
        impact = impacts[0]

        # padding 增大 → 时长增加
        assert impact.avg_duration_change_pct is not None
        assert impact.avg_duration_change_pct > 0
        # padding 增大 → 截断降低
        assert impact.end_truncation_change_pct is not None
        assert impact.end_truncation_change_pct < 0

    def test_fast_merge_increase_impact(self):
        """增大 fast_merge_max_gap → 合并频次增加"""
        from vocal_subtitle.feedback.diff_analyzer import ParamAdjustment
        from vocal_subtitle.feedback.impact_estimator import ImpactEstimator

        estimator = ImpactEstimator()
        adj = ParamAdjustment(
            param_path="merge_decision.fast_merge_max_gap",
            param_tier="medium_term",
            observed_value=0.05,
            confidence=0.7,
            learn_weight=1.0,
            direction="increase",
            reason="用户合并了多句",
        )

        impacts = estimator.estimate(
            {"merge_decision.fast_merge_max_gap": adj},
            {"merge_decision": {"fast_merge_max_gap": 0.20}},
        )
        assert len(impacts) == 1
        assert impacts[0].merge_frequency_change_pct is not None
        assert impacts[0].merge_frequency_change_pct > 0

    def test_summary_is_readable(self):
        """影响预估摘要应为可读中文"""
        from vocal_subtitle.feedback.diff_analyzer import ParamAdjustment
        from vocal_subtitle.feedback.impact_estimator import ImpactEstimator

        estimator = ImpactEstimator()
        adj = ParamAdjustment(
            param_path="subtitle.max_duration",
            param_tier="long_term",
            observed_value=1.0,
            confidence=0.6,
            learn_weight=0.7,
            direction="increase",
            reason="字幕整体偏短",
        )

        impacts = estimator.estimate(
            {"subtitle.max_duration": adj},
            {"subtitle": {"max_duration": 5.0}},
        )
        assert len(impacts) == 1
        assert len(impacts[0].summary) > 10  # 非空摘要


# ============================================================================
#  8. ConflictDetector 测试
# ============================================================================

class TestConflictDetector:
    """参数冲突检测器测试"""

    def test_oscillation_detected_3_flips(self):
        """3 次方向翻转 → 检测到震汤"""
        from vocal_subtitle.feedback.conflict_detector import ConflictDetector

        detector = ConflictDetector(window=5)
        history = [
            {"adjustments": {"merging.padding": [0.10, 0.14]},
             "timestamp": "2026-07-01T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merging.padding": [0.14, 0.09]},
             "timestamp": "2026-07-02T10:00:00", "diff_report_summary": "减小"},
            {"adjustments": {"merging.padding": [0.09, 0.13]},
             "timestamp": "2026-07-03T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merging.padding": [0.13, 0.08]},
             "timestamp": "2026-07-04T10:00:00", "diff_report_summary": "减小"},
        ]

        report = detector.detect_oscillation("merging.padding", history)
        assert report is not None
        assert report.is_oscillating
        assert report.oscillation_count >= 3
        assert report.severity == "high"
        assert report.recommended_action == "lock"

    def test_no_oscillation_on_consistent_direction(self):
        """同方向调整 → 不检测为震汤"""
        from vocal_subtitle.feedback.conflict_detector import ConflictDetector

        detector = ConflictDetector(window=5)
        history = [
            {"adjustments": {"merging.padding": [0.10, 0.12]},
             "timestamp": "2026-07-01T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merging.padding": [0.12, 0.15]},
             "timestamp": "2026-07-02T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merging.padding": [0.15, 0.18]},
             "timestamp": "2026-07-03T10:00:00", "diff_report_summary": "增大"},
        ]

        report = detector.detect_oscillation("merging.padding", history)
        # 无震荡或仅 mild
        if report:
            assert report.oscillation_count < 3

    def test_detect_all_oscillations(self):
        """检测所有参数的震汤"""
        from vocal_subtitle.feedback.conflict_detector import ConflictDetector

        detector = ConflictDetector(window=5)
        history = [
            {"adjustments": {"merging.padding": [0.10, 0.14]},
             "timestamp": "2026-07-01T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merging.padding": [0.14, 0.09]},
             "timestamp": "2026-07-02T10:00:00", "diff_report_summary": "减小"},
            {"adjustments": {"merging.padding": [0.09, 0.13]},
             "timestamp": "2026-07-03T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merging.padding": [0.13, 0.08]},
             "timestamp": "2026-07-04T10:00:00", "diff_report_summary": "减小"},
            # 另一个参数稳定
            {"adjustments": {"merge_decision.fast_merge_max_gap": [0.20, 0.22]},
             "timestamp": "2026-07-01T10:00:00", "diff_report_summary": "增大"},
            {"adjustments": {"merge_decision.fast_merge_max_gap": [0.22, 0.24]},
             "timestamp": "2026-07-02T10:00:00", "diff_report_summary": "增大"},
        ]

        reports = detector.detect_all_oscillations(history)
        # padding 应被检测到震汤
        padding_reports = [r for r in reports if r.param_path == "merging.padding"]
        assert len(padding_reports) >= 1
        assert padding_reports[0].is_oscillating

    def test_resolve_lock_action(self):
        """选择 lock → 参数被冻结"""
        from vocal_subtitle.feedback.conflict_detector import ConflictDetector, ConflictReport

        detector = ConflictDetector()
        report = ConflictReport(
            param_path="merging.padding",
            is_oscillating=True,
            oscillation_count=3,
        )
        result = detector.resolve(report, "lock")
        assert result["status"] == "locked"
        assert "锁定" in result["message"]


# ============================================================================
#  9. AudioFingerprint 测试
# ============================================================================


