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

