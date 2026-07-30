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


class TestSubtitleAligner:
    """智能字幕对齐器测试"""

    def test_perfect_1_to_1_match(self):
        """完全匹配: 自动版与修订版完全相同 → 全部 1:1"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        events = _make_events([
            (0.0, 2.0, "今天天气真不错"),
            (2.5, 5.0, "我们去看电影吧"),
            (5.5, 8.0, "你觉得怎么样"),
        ])

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner.align(events, events)

        assert len(pairs) == 3
        assert all(p.match_type == "1:1" for p in pairs)
        assert all(p.time_iou > 0.9 for p in pairs)
        assert all(p.text_similarity > 0.95 for p in pairs)

    def test_n_to_1_merge_detection(self):
        """N:1 检测: 自动版短句 → 修订版合并（通过 _dtw_align 直接测试）"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        # 使用 _dtw_align 绕过 coverage 门控，直接测试 DTW 匹配逻辑
        auto_sub = _make_events([
            (0.0, 1.5, "所以我们需要"),
            (1.5, 3.5, "考虑这个方案的可行性"),
        ])
        manual_sub = _make_events([
            (0.0, 3.5, "所以我们需要考虑这个方案的可行性"),
        ])

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner._dtw_align(auto_sub, manual_sub)

        merge_pairs = [p for p in pairs if p.match_type in ("N:1", "N:M")]
        assert len(merge_pairs) >= 1, f"Expected N:1 merge, got types: {[p.match_type for p in pairs]}"

    def test_1_to_n_split_detection(self):
        """1:N 检测: 自动版 1 条长句 → 修订版拆分为 2 条（通过 _dtw_align 测试）"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        auto_sub = _make_events([
            (0.0, 5.0, "那么接下来我们看第二个问题也就是说如何优化性能"),
        ])
        manual_sub = _make_events([
            (0.0, 2.5, "那么接下来我们看第二个问题"),
            (2.5, 5.0, "也就是说如何优化性能"),
        ])

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner._dtw_align(auto_sub, manual_sub)

        split_pairs = [p for p in pairs if p.match_type in ("1:N", "N:M")]
        assert len(split_pairs) >= 1, f"Expected 1:N split, got types: {[p.match_type for p in pairs]}"

    def test_insert_and_delete_detection(self):
        """INSERT/DELETE: 修订版增加/删除了字幕行（通过 _residual_match 测试）"""
        from vocal_subtitle.feedback.aligner import AlignmentPair, SubtitleAligner

        # 构造 3 对 1:1 匹配 + 1 个 auto 未匹配 + 1 个 manual 未匹配
        auto_events = _make_events([
            (0.0, 2.0, "第一句"),
            (3.0, 5.0, "第二句"),   # manual 删除此句
            (6.0, 8.0, "第三句"),
        ])
        manual_events = _make_events([
            (0.0, 2.0, "第一句"),
            (6.0, 8.0, "第三句"),
            (9.0, 11.0, "新句子"),  # auto 没有此句
        ])

        # 只提供已匹配的 2 对
        from vocal_subtitle.feedback.aligner import _time_iou, _levenshtein_similarity
        pairs = [
            AlignmentPair(
                auto_events=[auto_events[0]], manual_events=[manual_events[0]],
                match_type="1:1",
                time_iou=_time_iou(0,2,0,2),
                text_similarity=_levenshtein_similarity("第一句", "第一句"),
            ),
            AlignmentPair(
                auto_events=[auto_events[2]], manual_events=[manual_events[1]],
                match_type="1:1",
                time_iou=_time_iou(6,8,6,8),
                text_similarity=_levenshtein_similarity("第三句", "第三句"),
            ),
        ]

        aligner = SubtitleAligner(semantic_enabled=False)
        result = aligner._residual_match(pairs, auto_events, manual_events)

        insert_types = [p.match_type for p in result if p.match_type == "INSERT"]
        delete_types = [p.match_type for p in result if p.match_type == "DELETE"]
        assert len(insert_types) + len(delete_types) >= 2, (
            f"Expected INSERT+DELETE types, got: {[p.match_type for p in result]}"
        )

    def test_alignment_coverage_gating(self):
        """对齐覆盖率 < 70% 时应抛出 AlignmentError"""
        from vocal_subtitle.feedback.aligner import AlignmentError, SubtitleAligner

        # 自动版 3 条 vs 手动版 15 条（完全不同的内容 + 数量悬殊）
        # DTW 只能匹配 ~3 条，其余为 INSERT/DELETE（不算 matched），
        # coverage = 3 / max(3, 15) = 20% < 70% → 应抛出异常
        auto = _make_events([
            (0.0, 2.0, "完全不同的一句"),
            (3.0, 5.0, "完全不相关的另一句"),
            (6.0, 8.0, "第三句也不像"),
        ])
        manual = _make_events([
            (20.0, 22.0, f"完全不同的内容A{i}")
            for i in range(15)
        ])

        aligner = SubtitleAligner(semantic_enabled=False)
        with pytest.raises(AlignmentError, match="coverage too low"):
            aligner.align(auto, manual)

    def test_empty_events_raises(self):
        """空事件列表应抛出异常"""
        from vocal_subtitle.feedback.aligner import AlignmentError, SubtitleAligner

        events = _make_events([(0.0, 1.0, "test")])
        aligner = SubtitleAligner()

        with pytest.raises(AlignmentError):
            aligner.align([], events)
        with pytest.raises(AlignmentError):
            aligner.align(events, [])

    def test_anchor_segmentation(self):
        """长停顿 (>2s) 能正确触发段落切分"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        auto = _make_events([
            (0.0, 2.0, "段落A的第一句"),
            (2.5, 4.0, "段落A的第二句"),
            (7.0, 9.0, "段落B的第一句"),   # 间隔 3s > 2s
            (9.5, 12.0, "段落B的第二句"),
        ])
        manual = _make_events([
            (0.0, 2.2, "段落A的第一句修改"),
            (2.5, 4.2, "段落A的第二句修改"),
            (7.0, 9.2, "段落B的第一句修改"),
            (9.5, 12.2, "段落B的第二句修改"),
        ])

        aligner = SubtitleAligner(semantic_enabled=False)
        anchors = aligner._find_global_anchors(auto, manual)
        # 应该有至少 2 个锚点（首尾或段落边界）
        assert len(anchors) >= 2

    def test_time_iou_symmetric(self):
        """时间 IoU 应是对称的"""
        from vocal_subtitle.feedback.aligner import _time_iou

        iou1 = _time_iou(0, 10, 2, 8)
        iou2 = _time_iou(2, 8, 0, 10)
        assert iou1 == pytest.approx(iou2)

    def test_time_iou_non_overlapping(self):
        """完全不重叠 → IoU = 0"""
        from vocal_subtitle.feedback.aligner import _time_iou

        assert _time_iou(0, 5, 10, 15) == 0.0
        assert _time_iou(10, 15, 0, 5) == 0.0

    def test_levenshtein_identical(self):
        """相同文本 → 相似度 1.0"""
        from vocal_subtitle.feedback.aligner import _levenshtein_similarity

        assert _levenshtein_similarity("hello", "hello") == 1.0

    def test_levenshtein_completely_different(self):
        """完全不同文本 → 低相似度"""
        from vocal_subtitle.feedback.aligner import _levenshtein_similarity

        sim = _levenshtein_similarity("abc", "xyz")
        assert sim < 0.5


# ============================================================================
#  2. DiffAnalyzer 测试
# ============================================================================

