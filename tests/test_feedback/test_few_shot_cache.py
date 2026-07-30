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


class TestFewShotCache:
    """Few-shot 示例缓存管理测试"""

    def test_add_and_retrieve(self):
        """添加并获取活跃示例"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager(max_capacity=20)
        cache.add(FewShotExample(
            example_type="merge",
            fragments=["对", "就是说"],
            decision="MERGE",
            reason="填充词合并",
        ))

        active = cache.get_active_examples(max_count=5, min_weight=0.1)
        assert len(active) == 1
        assert active[0].example_type == "merge"

    def test_lru_eviction(self):
        """超过 max_capacity 时 LRU 淘汰最久未命中项"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager(max_capacity=5)
        for i in range(7):
            cache.add(FewShotExample(
                example_type="merge",
                fragments=[f"片段{i}"],
                decision="MERGE",
            ))

        assert cache.size() == 5  # max_capacity

    def test_duplicate_detection(self):
        """重复示例应更新已有项的权重而非新增"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager(max_capacity=20)
        cache.add(FewShotExample(
            example_type="merge",
            fragments=["同一个片段"],
            decision="MERGE",
            weight=0.5,
        ))
        cache.add(FewShotExample(
            example_type="merge",
            fragments=["同一个片段"],
            decision="MERGE",
            weight=0.8,
        ))

        assert cache.size() == 1
        active = cache.get_active_examples(min_weight=0.1)
        assert active[0].weight == 0.8  # 权重更新为较大值

    def test_prompt_injection_format(self):
        """Prompt 注入格式正确"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotBuilder,
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager()
        cache.add(FewShotExample(
            example_type="merge",
            fragments=["对", "就是说"],
            decision="MERGE",
            reason="填充词合并",
        ))
        cache.add(FewShotExample(
            example_type="format",
            rule="句末统一使用中文句号。",
            reason="标点偏好",
        ))

        builder = FewShotBuilder(cache_manager=cache, max_examples=3)
        base_prompt = "你是一个字幕优化助手。"
        injected = builder.inject_into_prompt(base_prompt, max_examples=3, min_weight=0.1)

        assert "User Preference Examples" in injected
        assert "对" in injected
        assert "就是说" in injected
        assert "中文句号" in injected
        assert base_prompt in injected  # 基础 prompt 保留


# ============================================================================
#  6. HealthScorer 测试
# ============================================================================

