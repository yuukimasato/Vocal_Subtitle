"""健康度评分器 (Phase 5.4)

标准化字幕质量综合评分，用于自动回滚、影子模式对比。
评分维度：对齐覆盖率、语义相似度、时间IoU、结构一致性。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aligner import AlignmentPair, SubtitleAligner

logger = logging.getLogger(__name__)


@dataclass
class HealthScoreResult:
    """健康度评分结果"""

    overall: float  # 0-100 的综合评分
    alignment_coverage: float  # 对齐覆盖率得分
    semantic_similarity: float  # 语义相似度均值得分
    time_iou: float  # 时间 IoU 均值得分
    structure_consistency: float  # 结构一致性得分（1:1 占比）
    total_pairs: int = 0
    detail_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.overall >= 50.0

    @property
    def grade(self) -> str:
        if self.overall >= 85:
            return "excellent"
        elif self.overall >= 70:
            return "good"
        elif self.overall >= 50:
            return "fair"
        else:
            return "poor"


# 子项权重
WEIGHT_COVERAGE = 0.35
WEIGHT_SEMANTIC = 0.35
WEIGHT_TIME_IOU = 0.20
WEIGHT_STRUCTURE = 0.10


def compute_health_score(
    auto_events: List,
    manual_events: List,
    aligner: Optional[SubtitleAligner] = None,
) -> Tuple[float, Dict[str, float]]:
    """计算字幕质量的综合健康度评分

    基于自动版和修订版字幕之间的对齐结果，
    计算 4 项子指标并按权重汇总。

    Args:
        auto_events: 自动生成的字幕事件列表
        manual_events: 用户修订的字幕事件列表
        aligner: 对齐器实例（可选，内部创建）

    Returns:
        (overall: 0-100, detail: 各子项得分字典)

    Raises:
        ValueError: 事件列表为空时
    """
    if not auto_events or not manual_events:
        raise ValueError("Cannot compute health score with empty events")

    if aligner is None:
        aligner = SubtitleAligner()

    pairs = aligner.align(auto_events, manual_events)

    return _compute_from_pairs(pairs, len(auto_events), len(manual_events))


def compute_health_score_from_pairs(
    pairs: List[AlignmentPair],
    n_auto: Optional[int] = None,
    n_manual: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """直接从已对齐的 pairs 计算健康度评分

    Args:
        pairs: 对齐后的 AlignmentPair 列表
        n_auto: 自动版事件总数（可选，从 pairs 推导）
        n_manual: 修订版事件总数（可选，从 pairs 推导）

    Returns:
        (overall: 0-100, detail: 各子项得分字典)
    """
    if n_auto is None:
        n_auto = max(
            sum(len(p.auto_events) for p in pairs),
            len(pairs),
        )
    if n_manual is None:
        n_manual = max(
            sum(len(p.manual_events) for p in pairs),
            len(pairs),
        )

    return _compute_from_pairs(pairs, n_auto, n_manual)


def _compute_from_pairs(
    pairs: List[AlignmentPair],
    n_auto: int,
    n_manual: int,
) -> Tuple[float, Dict[str, float]]:
    """核心计算逻辑"""
    if not pairs:
        return 0.0, {
            "alignment_coverage": 0.0,
            "semantic_similarity": 0.0,
            "time_iou": 0.0,
            "structure_consistency": 0.0,
        }

    # 子项 1: 对齐覆盖率 (权重 35%)
    alignment_coverage = len(pairs) / max(n_auto, n_manual, 1)
    score_coverage = alignment_coverage * 100

    # 子项 2: 文本语义相似度均值 (权重 35%)
    # 仅在"有效匹配"(1:1, 1:N, N:1) 上计算
    semantic_sims = [
        p.semantic_similarity
        for p in pairs
        if p.is_matched and p.semantic_similarity > 0
    ]
    score_semantic = (float(np.mean(semantic_sims)) * 100) if semantic_sims else 0.0

    # 子项 3: 时间 IoU 均值 (权重 20%)
    time_ious = [
        p.time_iou
        for p in pairs
        if p.is_matched and p.time_iou > 0
    ]
    score_time = (float(np.mean(time_ious)) * 100) if time_ious else 0.0

    # 子项 4: 结构一致性 (权重 10%)
    # 1:1 匹配占比越高越好
    one_to_one_count = sum(1 for p in pairs if p.match_type == "1:1")
    score_structure = (one_to_one_count / max(len(pairs), 1)) * 100

    # 综合评分
    overall = (
        WEIGHT_COVERAGE * score_coverage
        + WEIGHT_SEMANTIC * score_semantic
        + WEIGHT_TIME_IOU * score_time
        + WEIGHT_STRUCTURE * score_structure
    )

    detail = {
        "alignment_coverage": round(score_coverage, 2),
        "semantic_similarity": round(score_semantic, 2),
        "time_iou": round(score_time, 2),
        "structure_consistency": round(score_structure, 2),
    }

    return round(overall, 2), detail


def should_auto_rollback(
    health_before: float,
    health_after: float,
    drop_threshold: float = 0.3,
) -> Tuple[bool, str]:
    """判断是否应自动触发回滚

    条件: health_after < health_before * (1 - drop_threshold)

    Args:
        health_before: 调整前的健康度
        health_after: 调整后的健康度
        drop_threshold: 断崖式下降阈值 (默认 0.3 = 30%)

    Returns:
        (should_rollback: bool, reason: str)
    """
    if health_before <= 0:
        return False, "no baseline health score available"

    relative_change = (health_after - health_before) / health_before

    if relative_change <= -drop_threshold:
        return True, (
            f"Health score dropped {abs(relative_change)*100:.0f}% "
            f"({health_before:.1f} → {health_after:.1f}), "
            f"exceeding threshold of {drop_threshold*100:.0f}%"
        )

    return False, (
        f"Health change: {relative_change*100:+.1f}% "
        f"({health_before:.1f} → {health_after:.1f}) — within acceptable range"
    )


def health_score_result(
    auto_events: List,
    manual_events: List,
    aligner: Optional[SubtitleAligner] = None,
) -> HealthScoreResult:
    """计算并返回结构化的健康度结果"""
    overall, detail = compute_health_score(auto_events, manual_events, aligner)

    return HealthScoreResult(
        overall=overall,
        alignment_coverage=detail["alignment_coverage"],
        semantic_similarity=detail["semantic_similarity"],
        time_iou=detail["time_iou"],
        structure_consistency=detail["structure_consistency"],
        total_pairs=len(auto_events),
        detail_breakdown=detail,
    )
