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


class TestParamLearner:
    """参数学习器测试"""

    def test_learn_rate_tiers(self):
        """学习率分级: ≤2→0%, 3-5→15%, 6-15→25%, >15→35%"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        assert ParamLearner.compute_learn_rate(0) == 0.0
        assert ParamLearner.compute_learn_rate(2) == 0.0
        assert ParamLearner.compute_learn_rate(3) == 0.15
        assert ParamLearner.compute_learn_rate(5) == 0.15
        assert ParamLearner.compute_learn_rate(6) == 0.25
        assert ParamLearner.compute_learn_rate(15) == 0.25
        assert ParamLearner.compute_learn_rate(16) == 0.35
        assert ParamLearner.compute_learn_rate(100) == 0.35

    def test_ema_convergence(self):
        """EMA 逐步收敛到目标值"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        current = 0.10
        target = 0.20
        lr = 0.25

        # 一次更新
        new1 = ParamLearner.ema_update(current, target, lr, "merging.padding")
        assert current < new1 < target  # 朝目标移动但不过头

        # 多次更新应接近目标
        for _ in range(20):
            current = ParamLearner.ema_update(current, target, lr, "merging.padding")
        assert abs(current - target) < 0.01

    def test_hard_bounds_clamping(self):
        """参数值不能超出硬边界"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        # padding 上界 0.30
        clamped = ParamLearner.ema_update(0.29, 0.50, 0.5, "merging.padding")
        assert clamped <= 0.30

        # padding 下界 0.02
        clamped = ParamLearner.ema_update(0.03, 0.001, 0.5, "merging.padding")
        assert clamped >= 0.02

        # max_chars_cjk 范围 [10, 40]
        clamped = ParamLearner.ema_update(39, 50, 0.5, "subtitle.max_chars_cjk")
        assert clamped <= 40

    def test_iqr_outlier_filter(self):
        """IQR 过滤应移除极端值"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        # 大部分值在 0.10-0.15，加入一个极端值 1.0
        values = [0.10, 0.11, 0.12, 0.13, 0.15, 1.0]
        filtered = ParamLearner.filter_outliers(values)
        assert 1.0 not in filtered
        assert len(filtered) < len(values)

    def test_iqr_no_filter_on_uniform(self):
        """均匀分布的值不应被过滤"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        filtered = ParamLearner.filter_outliers(values)
        assert len(filtered) == len(values)


# ============================================================================
#  4. UserProfileManager 测试
# ============================================================================

