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

