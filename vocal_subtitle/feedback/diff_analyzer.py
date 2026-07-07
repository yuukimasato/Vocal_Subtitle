"""差异分类与归因分析器

对齐后分析用户"为什么这么改"，将修改行为映射到 PipelineConfig 的具体参数。

归因表:
    时间轴整体平移 → merging.padding (补偿平移量)
    句尾被截断     → merging.padding / padding_max (增大 end_padding)
    句首被截断     → merging.padding / padding_min (增大 start_padding)
    本该合并却被拆分 → fast_merge_max_gap (调大合并间隙)
    本该拆分却被合并 → llm_decision_max_gap (调小拆分阈值)
    句末标点修改   → LLM Few-shot 示例
    说话人标签修正 → SpeakerRoleConfig (记录偏好)
    整体偏长/偏短 → max_duration (调整最大时长约束)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .aligner import AlignmentPair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TimeShift:
    """时间偏移"""
    auto_start: float
    auto_end: float
    manual_start: float
    manual_end: float
    delta_start: float  # manual.start - auto.start
    delta_end: float    # manual.end - auto.end
    match_type: str


@dataclass
class MergeAction:
    """合并/拆分动作"""
    action_type: str  # "merge" | "split"
    auto_count: int
    manual_count: int
    gap_between: float  # 自动版中相邻片段间的间隙
    text_auto: str
    text_manual: str


@dataclass
class TextEdit:
    """文本修改"""
    auto_text: str
    manual_text: str
    edit_type: str  # "punctuation" | "rewording" | "typo_fix" | "other"
    edit_distance: int


@dataclass
class ParamAdjustment:
    """参数调整建议"""
    param_path: str       # e.g. "merging.padding"
    param_tier: str       # "long_term" | "medium_term" | "short_term"
    observed_value: float # 从用户修改中观测到的目标值
    confidence: float     # 调整置信度 [0, 1]
    learn_weight: float   # 学习权重 [0, 1]（考虑 ASR 置信度等）
    direction: str        # "increase" | "decrease"
    reason: str           # 人类可读的原因


@dataclass
class DiffReport:
    """单次反馈的差异分析报告"""
    total_pairs: int
    alignment_coverage: float       # 对齐覆盖率
    time_shifts: List[TimeShift] = field(default_factory=list)
    merge_actions: List[MergeAction] = field(default_factory=list)
    text_edits: List[TextEdit] = field(default_factory=list)
    attribution: Dict[str, ParamAdjustment] = field(default_factory=dict)
    structural_revision: bool = False
    median_semantic_similarity: float = 0.0


# ---------------------------------------------------------------------------
# 差异分析器
# ---------------------------------------------------------------------------


class DiffAnalyzer:
    """差异分类与归因分析器

    对每个 AlignmentPair 分类修改行为，汇总为参数调整建议。
    """

    # 归因阈值常量
    TIME_SHIFT_THRESHOLD_MS = 50      # 时间偏移 > 50ms 才归因
    MERGE_RATIO_THRESHOLD = 0.15      # 合并/拆分比例 > 15% 才触发调整
    PUNCTUATION_SIM_THRESHOLD = 0.9   # 文本相似度 > 0.9 且仅标点不同 → 标点编辑
    ASR_LOW_CONFIDENCE_THRESHOLD = -1.5  # avg_logprob < 此值 → ASR 低置信度

    def __init__(self, param_isolation_enabled: bool = True):
        self._param_isolation_enabled = param_isolation_enabled

    def analyze(self, pairs: List[AlignmentPair]) -> DiffReport:
        """分析对齐对，生成差异报告

        Args:
            pairs: 对齐器输出的 AlignmentPair 列表

        Returns:
            DiffReport 含归因建议
        """
        matched = [p for p in pairs if p.is_matched]
        total = len(pairs)

        coverage = len(matched) / max(total, 1)

        report = DiffReport(
            total_pairs=total,
            alignment_coverage=coverage,
        )

        # 语义相似度中位数
        semantic_sims = [p.semantic_similarity for p in matched if p.semantic_similarity > 0]
        report.median_semantic_similarity = float(np.median(semantic_sims)) if semantic_sims else 0.0

        # 1. 分析时间偏移
        report.time_shifts = self._analyze_time_shifts(matched)
        # 2. 分析合并/拆分
        report.merge_actions = self._analyze_merge_actions(pairs)
        # 3. 分析文本修改
        report.text_edits = self._analyze_text_edits(matched)

        # 4. 归因 → 参数调整建议
        report.attribution = self._attribute(report, pairs)

        # 5. 检测结构性修订
        insert_count = sum(1 for p in pairs if p.match_type == "INSERT")
        delete_count = sum(1 for p in pairs if p.match_type == "DELETE")
        if insert_count + delete_count > len(pairs) * 0.3:
            report.structural_revision = True

        return report

    # ------------------------------------------------------------------
    # 分析子方法
    # ------------------------------------------------------------------

    def _analyze_time_shifts(self, matched: List[AlignmentPair]) -> List[TimeShift]:
        """分析每个对齐对的时间偏移"""
        shifts = []
        for p in matched:
            if p.match_type not in ("1:1",):
                continue
            if not p.auto_events or not p.manual_events:
                continue

            ae = p.auto_events[0]
            me = p.manual_events[0]

            delta_start = me.start - ae.start
            delta_end = me.end - ae.end

            if abs(delta_start) > 0.01 or abs(delta_end) > 0.01:
                shifts.append(TimeShift(
                    auto_start=ae.start,
                    auto_end=ae.end,
                    manual_start=me.start,
                    manual_end=me.end,
                    delta_start=delta_start,
                    delta_end=delta_end,
                    match_type=p.match_type,
                ))

        return shifts

    def _analyze_merge_actions(self, pairs: List[AlignmentPair]) -> List[MergeAction]:
        """分析合并/拆分动作"""
        actions = []
        for p in pairs:
            n_auto = len(p.auto_events)
            n_manual = len(p.manual_events)

            if n_auto == 1 and n_manual > 1:
                # 拆分: 1 auto → N manual
                gap = 0.0
                if len(p.manual_events) > 1:
                    gap = max(
                        p.manual_events[i + 1].start - p.manual_events[i].end
                        for i in range(len(p.manual_events) - 1)
                        if p.manual_events[i + 1].start > p.manual_events[i].end
                    ) if all(e.end <= p.manual_events[min(i + 1, len(p.manual_events) - 1)].start for i, e in enumerate(p.manual_events[:-1])) else 0.0

                actions.append(MergeAction(
                    action_type="split",
                    auto_count=n_auto,
                    manual_count=n_manual,
                    gap_between=gap,
                    text_auto=" ".join(e.text for e in p.auto_events),
                    text_manual=" ".join(e.text for e in p.manual_events),
                ))
            elif n_auto > 1 and n_manual == 1:
                # 合并: N auto → 1 manual
                gap = max(
                    p.auto_events[i + 1].start - p.auto_events[i].end
                    for i in range(len(p.auto_events) - 1)
                )

                actions.append(MergeAction(
                    action_type="merge",
                    auto_count=n_auto,
                    manual_count=n_manual,
                    gap_between=gap,
                    text_auto=" ".join(e.text for e in p.auto_events),
                    text_manual=" ".join(e.text for e in p.manual_events),
                ))

        return actions

    def _analyze_text_edits(self, matched: List[AlignmentPair]) -> List[TextEdit]:
        """分析文本级别的修改"""
        edits = []
        for p in matched:
            if p.match_type != "1:1":
                continue
            if not p.auto_events or not p.manual_events:
                continue

            a_text = p.auto_events[0].text
            m_text = p.manual_events[0].text

            if a_text == m_text:
                continue

            # 判断编辑类型
            edit_type = self._classify_text_edit(a_text, m_text, p.text_similarity)
            edit_distance = self._compute_edit_distance(a_text, m_text)

            edits.append(TextEdit(
                auto_text=a_text,
                manual_text=m_text,
                edit_type=edit_type,
                edit_distance=edit_distance,
            ))

        return edits

    @staticmethod
    def _classify_text_edit(auto_text: str, manual_text: str, similarity: float) -> str:
        """分类文本修改类型"""
        if similarity > 0.9:
            # 去除标点后检查是否相同
            import re
            a_no_punct = re.sub(r"[，。！？、,\.\!\?\s]", "", auto_text)
            m_no_punct = re.sub(r"[，。！？、,\.\!\?\s]", "", manual_text)
            if a_no_punct == m_no_punct:
                return "punctuation"
            return "rewording"
        elif similarity > 0.7:
            return "rewording"
        else:
            return "other"

    @staticmethod
    def _compute_edit_distance(a: str, b: str) -> int:
        """计算编辑距离"""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    # ------------------------------------------------------------------
    # 归因
    # ------------------------------------------------------------------

    def _attribute(
        self,
        report: DiffReport,
        pairs: List[AlignmentPair],
    ) -> Dict[str, ParamAdjustment]:
        """汇总归因为参数调整建议"""
        attributions: Dict[str, ParamAdjustment] = {}

        # ---- 1. 时间偏移归因 ----
        if report.time_shifts:
            delta_ends = [s.delta_end for s in report.time_shifts]
            delta_starts = [s.delta_start for s in report.time_shifts]

            median_delta_end = float(np.median(delta_ends))
            median_delta_start = float(np.median(delta_starts))

            # 整体后移（end 系统性偏早）
            if median_delta_end > self.TIME_SHIFT_THRESHOLD_MS / 1000:
                attributions["merging.padding"] = ParamAdjustment(
                    param_path="merging.padding",
                    param_tier="short_term",
                    observed_value=median_delta_end,
                    confidence=min(1.0, abs(median_delta_end) / 0.2),
                    learn_weight=0.8,
                    direction="increase",
                    reason=f"结束时间系统性后移 {median_delta_end*1000:.0f}ms → 增大 padding",
                )

            # 整体前移（start 系统性偏晚）
            if median_delta_start < -self.TIME_SHIFT_THRESHOLD_MS / 1000:
                attributions["merging.padding"] = ParamAdjustment(
                    param_path="merging.padding",
                    param_tier="short_term",
                    observed_value=abs(median_delta_start),
                    confidence=min(1.0, abs(median_delta_start) / 0.15),
                    learn_weight=0.8,
                    direction="increase",
                    reason=f"开始时间系统性前移 {abs(median_delta_start)*1000:.0f}ms → 增大 start padding",
                )

        # ---- 2. 合并/拆分归因 ----
        if report.merge_actions:
            merge_count = sum(1 for a in report.merge_actions if a.action_type == "merge")
            split_count = sum(1 for a in report.merge_actions if a.action_type == "split")
            total_matched = max(len([p for p in pairs if p.is_matched]), 1)

            merge_ratio = merge_count / total_matched
            split_ratio = split_count / total_matched

            # 用户倾向于更多合并 → 调大合并间隙
            if merge_ratio > self.MERGE_RATIO_THRESHOLD:
                attributions["merge_decision.fast_merge_max_gap"] = ParamAdjustment(
                    param_path="merge_decision.fast_merge_max_gap",
                    param_tier="medium_term",
                    observed_value=merge_ratio,
                    confidence=min(1.0, merge_ratio / 0.3),
                    learn_weight=1.0,
                    direction="increase",
                    reason=f"用户合并了 {merge_count} 处相邻句 ({merge_ratio:.0%}) → 增大合并间隙",
                )

            # 用户倾向于更多拆分 → 调小合并间隙
            if split_ratio > self.MERGE_RATIO_THRESHOLD:
                attributions["merge_decision.llm_decision_max_gap"] = ParamAdjustment(
                    param_path="merge_decision.llm_decision_max_gap",
                    param_tier="medium_term",
                    observed_value=split_ratio,
                    confidence=min(1.0, split_ratio / 0.3),
                    learn_weight=1.0,
                    direction="decrease",
                    reason=f"用户拆分了 {split_count} 处合并 ({split_ratio:.0%}) → 减小合并阈值",
                )

            # 参数解耦：如果同时有合并和拆分需求，选置信度更高的
            if self._param_isolation_enabled:
                if ("merge_decision.fast_merge_max_gap" in attributions
                        and "merge_decision.llm_decision_max_gap" in attributions):
                    a1 = attributions["merge_decision.fast_merge_max_gap"]
                    a2 = attributions["merge_decision.llm_decision_max_gap"]
                    if a1.confidence >= a2.confidence:
                        del attributions["merge_decision.llm_decision_max_gap"]
                        logger.info("Param isolation: keeping fast_merge_max_gap (higher confidence)")
                    else:
                        del attributions["merge_decision.fast_merge_max_gap"]
                        logger.info("Param isolation: keeping llm_decision_max_gap (higher confidence)")

        # ---- 3. 文本修改归因 ----
        punctuation_edits = [e for e in report.text_edits if e.edit_type == "punctuation"]
        if punctuation_edits:
            # 句末标点偏好 → 记录为 Few-shot 示例，不直接调整参数
            logger.info(
                "Detected %d punctuation edits — will be recorded as few-shot examples",
                len(punctuation_edits),
            )

        # ---- 4. 整体时长偏差归因 ----
        matched_pairs = [p for p in pairs if p.match_type == "1:1"]
        if matched_pairs:
            auto_durations = [(e.end - e.start) for p in matched_pairs for e in p.auto_events]
            manual_durations = [(e.end - e.start) for p in matched_pairs for e in p.manual_events]
            if auto_durations and manual_durations:
                median_auto = float(np.median(auto_durations))
                median_manual = float(np.median(manual_durations))
                duration_diff = median_manual - median_auto
                if abs(duration_diff) > 0.5:  # > 500ms
                    direction = "decrease" if duration_diff < 0 else "increase"
                    attributions["subtitle.max_duration"] = ParamAdjustment(
                        param_path="subtitle.max_duration",
                        param_tier="long_term",
                        observed_value=abs(duration_diff),
                        confidence=min(1.0, abs(duration_diff) / 2.0),
                        learn_weight=0.7,
                        direction=direction,
                        reason=f"字幕整体{'变短' if duration_diff < 0 else '变长'} {abs(duration_diff)*1000:.0f}ms",
                    )

        return attributions
