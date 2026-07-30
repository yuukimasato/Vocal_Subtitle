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


class TestUserProfileManager:
    """用户配置文件管理测试"""

    def test_create_and_load_default_profile(self):
        """创建并加载默认配置"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        profile = mgr.load("__test_temp__")

        assert profile["profile_id"] == "__test_temp__"
        assert profile["feedback_count"] == 0
        assert profile["overrides"] == {}
        assert profile["history"] == []

    def test_save_and_load_preserves_overrides(self):
        """保存并重新加载 overrides"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        profile = mgr.load("__test_save__")
        profile["overrides"] = {"merging": {"padding": 0.14}}
        profile["feedback_count"] = 5
        mgr.save(profile)

        loaded = mgr.load("__test_save__")
        assert loaded["overrides"]["merging"]["padding"] == 0.14
        assert loaded["feedback_count"] == 5

    def test_rollback_restores_previous(self):
        """回滚到上一备份版本"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()

        # 初始版本
        profile = mgr.load("__test_rollback__")
        profile["overrides"] = {"merging": {"padding": 0.10}}
        mgr.save(profile)

        # 修改版本（触发备份）
        profile["overrides"]["merging"]["padding"] = 0.20
        mgr.save(profile)

        # 回滚
        rolled = mgr.rollback("__test_rollback__")
        assert rolled["overrides"]["merging"]["padding"] == 0.10

    def test_reset_clears_all(self):
        """重置为系统默认"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        profile = mgr.load("__test_reset__")
        profile["overrides"] = {"merging": {"padding": 0.30}}
        profile["feedback_count"] = 10
        mgr.save(profile)

        reset = mgr.reset("__test_reset__")
        assert reset["feedback_count"] == 0
        assert reset["overrides"] == {}

    def test_backup_rotation_limit(self):
        """备份轮转最多保留 MAX_BACKUPS 个"""
        from vocal_subtitle.feedback.user_profile import MAX_BACKUPS, UserProfileManager

        mgr = UserProfileManager()

        for i in range(MAX_BACKUPS + 2):
            profile = mgr.load("__test_backup_rotation__")
            profile["feedback_count"] = i
            mgr.save(profile)

        # 备份数不应超过 MAX_BACKUPS
        bak_dir = mgr._profile_dir
        backups = list(bak_dir.glob("__test_backup_rotation__*.bak.*"))
        assert len(backups) <= MAX_BACKUPS

    def test_decay_weight_long_term(self):
        """长期偏好 180 天半衰期：30 天前权重 ≈ 0.85"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        ts = (datetime.now() - timedelta(days=30)).isoformat()
        w = UserProfileManager.decay_weight(ts, half_life_days=180)
        assert 0.80 < w < 0.90, f"Expected ~0.85, got {w:.3f}"

    def test_decay_weight_short_term(self):
        """短期环境 60 天半衰期：60 天前权重 = exp(-1) ≈ 0.368"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        ts = (datetime.now() - timedelta(days=60)).isoformat()
        w = UserProfileManager.decay_weight(ts, half_life_days=60)
        # exp(-60/60) = exp(-1) ≈ 0.368
        assert 0.35 < w < 0.39, f"Expected ~0.368, got {w:.3f}"

    def test_decay_weight_no_decay(self):
        """half_life_days=None → 不衰减，权重始终为 1.0"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        ts = (datetime.now() - timedelta(days=365)).isoformat()
        w = UserProfileManager.decay_weight(ts, half_life_days=None)
        assert w == 1.0

    def test_cleanup_temp_profiles(self):
        """清理本测试类创建的临时配置文件"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        for name in ["__test_temp__", "__test_save__", "__test_rollback__",
                      "__test_reset__", "__test_backup_rotation__"]:
            try:
                mgr.delete(name)
            except Exception:
                pass


# ============================================================================
#  5. FewShotBuilder / FewShotCacheManager 测试
# ============================================================================

