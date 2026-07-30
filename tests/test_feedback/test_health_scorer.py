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

