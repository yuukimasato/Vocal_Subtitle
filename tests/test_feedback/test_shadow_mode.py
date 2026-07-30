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


class TestShadowMode:
    """影子模式评估器测试"""

    def test_should_upgrade_all_conditions_met(self):
        """全部条件满足 → 建议升级"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(
            min_shadow_runs=10,
            upgrade_threshold=0.05,
        )

        for i in range(10):
            evaluator.add_run(ShadowRunResult(
                timestamp=f"2026-07-{(i + 1):02d}T10:00:00",
                health_current=70.0,
                health_shadow=75.0,  # 平均好 5 分 (+7.1%)
                health_detail_current={"alignment_coverage": 70.0, "semantic_similarity": 70.0,
                                        "time_iou": 70.0, "structure_consistency": 70.0},
                health_detail_shadow={"alignment_coverage": 75.0, "semantic_similarity": 75.0,
                                      "time_iou": 75.0, "structure_consistency": 75.0},
            ))

        result = evaluator.should_upgrade()
        assert result.should_upgrade
        assert result.recommendation == "upgrade"

    def test_insufficient_runs(self):
        """运行次数不足 → 继续收集"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(min_shadow_runs=10)
        evaluator.add_run(ShadowRunResult(
            health_current=70.0, health_shadow=80.0,
        ))

        result = evaluator.should_upgrade()
        assert not result.should_upgrade
        assert result.recommendation == "continue"

    def test_insufficient_improvement(self):
        """提升不足阈值 → 丢弃"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(
            min_shadow_runs=3,
            upgrade_threshold=0.05,
        )

        for _ in range(3):
            evaluator.add_run(ShadowRunResult(
                health_current=70.0,
                health_shadow=70.5,  # 仅好 0.5 (+0.7%)
                health_detail_current={"alignment_coverage": 70.0, "semantic_similarity": 70.0,
                                        "time_iou": 70.0, "structure_consistency": 70.0},
                health_detail_shadow={"alignment_coverage": 70.5, "semantic_similarity": 70.5,
                                      "time_iou": 70.5, "structure_consistency": 70.5},
            ))

        result = evaluator.should_upgrade()
        assert result.recommendation == "discard"

    def test_dimension_degradation(self):
        """有子项退化 → 丢弃"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(
            min_shadow_runs=3,
            upgrade_threshold=0.05,
            max_dim_degradation=0.10,
        )

        for _ in range(3):
            evaluator.add_run(ShadowRunResult(
                health_current=70.0,
                health_shadow=80.0,  # 整体高 14%
                health_detail_current={"alignment_coverage": 70.0, "semantic_similarity": 70.0,
                                        "time_iou": 70.0, "structure_consistency": 70.0},
                health_detail_shadow={"alignment_coverage": 95.0, "semantic_similarity": 95.0,
                                      "time_iou": 30.0,   # ★ 严重退化 -57%
                                      "structure_consistency": 95.0},
            ))

        result = evaluator.should_upgrade()
        assert result.recommendation == "discard"
        assert len(result.degraded_dims) >= 1


# ============================================================================
# 11. 集成测试
# ============================================================================

