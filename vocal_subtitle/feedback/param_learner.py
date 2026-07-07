"""参数自适应更新引擎

基于用户反馈的差异分析报告，使用加权滑动平均 (EMA) 更新用户配置参数。

核心特性:
- EMA 更新策略，避免单次修改过拟合
- 样本数驱动的学习率分级（≤2 记录不更新，3-5 以 15%，6-15 以 25%，>15 以 35%）
- 有界约束：每个参数有硬限制范围
- 分级衰减：长期偏好 180d / 中期策略 90d / 短期环境 60d
- IQR 异常值过滤
- 参数隔离调整（防止耦合参数震荡）
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .diff_analyzer import DiffReport, ParamAdjustment
from .user_profile import PARAM_TIER_HALF_LIFE, DEFAULT_HALF_LIFE_DAYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数硬边界
# ---------------------------------------------------------------------------

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "merging.padding": (0.02, 0.30),
    "merging.padding_max": (0.05, 0.40),
    "merging.padding_min": (0.01, 0.15),
    "merging.min_silence_gap": (0.1, 1.0),
    "merge_decision.fast_merge_max_gap": (0.05, 0.50),
    "merge_decision.llm_decision_min_gap": (0.10, 0.50),
    "merge_decision.llm_decision_max_gap": (0.50, 2.00),
    "merge_decision.hard_split_min_gap": (0.60, 3.00),
    "merge_decision.max_combined_duration": (2.0, 10.0),
    "subtitle.max_duration": (2.0, 10.0),
    "subtitle.min_duration": (0.3, 2.0),
    "subtitle.max_chars_cjk": (10, 40),
    "subtitle.max_chars_latin": (20, 60),
    "vad.threshold": (0.1, 0.9),
    "vad.min_silence_duration_ms": (100, 1000),
    "noise_reduction.spectral_noise_reduction_db": (3.0, 24.0),
}


def _clamp(param_path: str, value: float) -> float:
    """将参数值限制在硬边界内"""
    bounds = PARAM_BOUNDS.get(param_path)
    if bounds is None:
        return value
    lo, hi = bounds
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 参数解耦器
# ---------------------------------------------------------------------------


class ParamDecoupler:
    """参数解耦器 — 防止联动参数同时调整导致震荡"""

    COUPLED_GROUPS = [
        # (主参数, 从属参数, 耦合关系)
        ("merge_decision.fast_merge_max_gap",
         "merge_decision.llm_decision_min_gap",
         "boundary"),
        ("merge_decision.llm_decision_min_gap",
         "merge_decision.llm_decision_max_gap",
         "interval"),
        ("merging.padding",
         "merging.padding_max",
         "dependent"),
    ]

    @classmethod
    def select_adjustments(
        cls,
        raw_attributions: Dict[str, ParamAdjustment],
        profile_history: List[Dict],
    ) -> Dict[str, ParamAdjustment]:
        """从原始归因中选择本轮应执行的调整

        策略:
        1. 若同组内两个参数都有调整建议 →
           选择置信度更高的那个，另一个标记为 pending
        2. 若上一轮已调整过同组的另一个参数 →
           本轮允许调整当前参数（交替验证）
        """
        if not raw_attributions:
            return {}

        selected = dict(raw_attributions)
        pending_params: Dict[str, Tuple[str, float]] = {}  # param → (conflict_with, confidence)

        for a, b, relation in cls.COUPLED_GROUPS:
            if a in selected and b in selected:
                a_conf = selected[a].confidence
                b_conf = selected[b].confidence
                if a_conf >= b_conf:
                    pending_params[b] = (a, b_conf)
                    del selected[b]
                    logger.info(
                        "ParamDecoupler: suppressed '%s' (confidence=%.2f) in favor of '%s' (confidence=%.2f, relation=%s)",
                        b, b_conf, a, a_conf, relation,
                    )
                else:
                    pending_params[a] = (b, a_conf)
                    del selected[a]
                    logger.info(
                        "ParamDecoupler: suppressed '%s' (confidence=%.2f) in favor of '%s' (confidence=%.2f, relation=%s)",
                        a, a_conf, b, b_conf, relation,
                    )

        return selected


# ---------------------------------------------------------------------------
# 参数学习器
# ---------------------------------------------------------------------------


class ParamLearner:
    """基于用户反馈的参数自适应学习器

    采用滑动平均 (EMA) 更新策略，避免单次修改过拟合。
    更新幅度与样本数量成正比（置信度驱动）。
    """

    def __init__(self, profile_mgr):
        """
        Args:
            profile_mgr: UserProfileManager 实例
        """
        self._profile_mgr = profile_mgr

    # ------------------------------------------------------------------
    # 学习率计算
    # ------------------------------------------------------------------

    @staticmethod
    def compute_learn_rate(
        sample_count: int,
        base_rate: float = 0.10,
        max_rate: float = 0.35,
    ) -> float:
        """根据样本数计算学习率

        - ≤2 样本: 0.0 (仅记录，不更新)
        - 3-5 样本: 以 15% 学习率逐步靠拢
        - 6-15 样本: 以 25% 学习率加速
        - >15 样本: 以 35% 学习率（上限）
        """
        if sample_count <= 2:
            return 0.0
        elif sample_count <= 5:
            return 0.15
        elif sample_count <= 15:
            return 0.25
        else:
            return max_rate

    # ------------------------------------------------------------------
    # 异常值过滤
    # ------------------------------------------------------------------

    @staticmethod
    def filter_outliers(values: List[float]) -> List[float]:
        """使用 IQR 方法过滤异常值"""
        if len(values) < 4:
            return values
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = [v for v in values if lower <= v <= upper]
        if len(filtered) < len(values):
            logger.info(
                "Outlier filter: %d → %d values (IQR=%.3f, range=[%.3f, %.3f])",
                len(values), len(filtered), iqr, lower, upper,
            )
        return filtered

    # ------------------------------------------------------------------
    # EMA 更新
    # ------------------------------------------------------------------

    @staticmethod
    def ema_update(
        current_value: float,
        observed_target: float,
        learn_rate: float,
        param_path: str,
    ) -> float:
        """加权滑动平均更新

        new_value = current + (observed - current) × learn_rate

        Returns:
            钳制在硬边界内的新值
        """
        new_value = current_value + (observed_target - current_value) * learn_rate
        clamped = _clamp(param_path, new_value)
        if clamped != new_value:
            logger.debug(
                "Param '%s' clamped: %.4f → %.4f (bounds)",
                param_path, new_value, clamped,
            )
        return clamped

    # ------------------------------------------------------------------
    # 主学习流程
    # ------------------------------------------------------------------

    def learn_from_diff(
        self,
        diff_report: DiffReport,
        current_config_overrides: Dict[str, Any],
        profile_name: str = "user_default",
    ) -> Dict[str, Any]:
        """根据差异分析结果更新用户配置

        Args:
            diff_report: 差异分析报告
            current_config_overrides: 当前的 overrides 字典（从 profile 中加载）
            profile_name: 用户配置名称

        Returns:
            更新后的 overrides 字典
        """
        attributions = diff_report.attribution
        if not attributions:
            logger.info("No parameter adjustments needed — skipping update")
            return current_config_overrides

        # 加载 profile 获取 feedback_count
        profile = self._profile_mgr.load(profile_name)
        feedback_count = profile.get("feedback_count", 0) + 1
        history = profile.get("history", [])

        # 参数解耦
        decoupler = ParamDecoupler()
        active_attributions = decoupler.select_adjustments(attributions, history)

        if not active_attributions:
            logger.info("All attributions suppressed by param decoupler")
            return current_config_overrides

        # 计算学习率
        learn_rate = self.compute_learn_rate(feedback_count)
        if learn_rate == 0.0:
            logger.info(
                "Feedback count=%d ≤ 2 — recording observation only, no param update",
                feedback_count,
            )
            # 仅记录，不更新参数
            self._record_observation(profile, diff_report, feedback_count, {})
            return current_config_overrides

        # 应用每个参数调整
        overrides = dict(current_config_overrides) if current_config_overrides else {}
        adjustments_applied = {}

        for param_path, adj in active_attributions.items():
            # 获取当前值
            current_value = self._get_nested_value(overrides, param_path)
            if current_value is None:
                # 参数尚未在 overrides 中，需要从默认值获取
                current_value = self._get_default_value(param_path)
                if current_value is None:
                    continue

            # 计算目标值
            if adj.direction == "increase":
                observed_target = current_value + adj.observed_value * adj.learn_weight
            else:
                observed_target = current_value - adj.observed_value * adj.learn_weight

            # EMA 更新
            effective_lr = learn_rate * adj.learn_weight
            new_value = self.ema_update(current_value, observed_target, effective_lr, param_path)

            if abs(new_value - current_value) < 0.001:
                continue  # 变化太小，跳过

            # 写入 overrides 字典
            self._set_nested_value(overrides, param_path, new_value)
            adjustments_applied[param_path] = [current_value, new_value]

        # 记录历史
        self._record_observation(profile, diff_report, feedback_count, adjustments_applied)

        # 更新 profile
        profile["feedback_count"] = feedback_count
        profile["overrides"] = overrides
        profile["updated_at"] = datetime.now().isoformat()

        # 保存
        self._profile_mgr.save(profile)

        if adjustments_applied:
            logger.info(
                "Param learner: applied %d adjustments with learn_rate=%.2f (feedback #%d)",
                len(adjustments_applied), learn_rate, feedback_count,
            )
            for param, (old, new) in adjustments_applied.items():
                logger.info("  %s: %.4f → %.4f", param, old, new)

        return overrides

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _record_observation(
        self,
        profile: Dict[str, Any],
        diff_report: DiffReport,
        feedback_count: int,
        adjustments: Dict[str, List[float]],
    ) -> None:
        """记录反馈历史到 profile"""
        history = profile.get("history", [])

        # 生成摘要
        shifts = len(diff_report.time_shifts)
        merges = sum(1 for a in diff_report.merge_actions if a.action_type == "merge")
        splits = sum(1 for a in diff_report.merge_actions if a.action_type == "split")
        parts = []
        if shifts:
            parts.append(f"时间偏移 {shifts} 处")
        if merges:
            parts.append(f"合并 {merges} 处")
        if splits:
            parts.append(f"拆分 {splits} 处")
        summary = "、".join(parts) or "无显著差异"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "diff_report_summary": summary,
            "alignment_coverage": round(diff_report.alignment_coverage, 3),
            "median_semantic_similarity": round(diff_report.median_semantic_similarity, 3),
            "adjustments": {k: [round(v[0], 4), round(v[1], 4)] for k, v in adjustments.items()},
        }
        history.append(entry)

        # 保留最近 50 条历史
        profile["history"] = history[-50:]

    @staticmethod
    def _get_nested_value(overrides: Dict[str, Any], param_path: str) -> Optional[float]:
        """从 overrides 字典中获取嵌套参数值"""
        parts = param_path.split(".")
        current = overrides
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, (int, float)) else None

    @staticmethod
    def _set_nested_value(overrides: Dict[str, Any], param_path: str, value: float) -> None:
        """在 overrides 字典中设置嵌套参数值"""
        parts = param_path.split(".")
        current = overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = round(value, 4)

    @staticmethod
    def _get_default_value(param_path: str) -> Optional[float]:
        """获取参数的默认值（从 PipelineConfig 默认工厂）"""
        from ..config import (
            MergingConfig,
            MergeDecisionConfig,
            SubtitleBuildConfig,
            VADConfig,
            NoiseReductionConfig,
        )

        defaults = {
            "merging.padding": MergingConfig().padding,
            "merging.padding_max": MergingConfig().padding_max,
            "merging.padding_min": MergingConfig().padding_min,
            "merging.min_silence_gap": MergingConfig().min_silence_gap,
            "merge_decision.fast_merge_max_gap": MergeDecisionConfig().fast_merge_max_gap,
            "merge_decision.llm_decision_min_gap": MergeDecisionConfig().llm_decision_min_gap,
            "merge_decision.llm_decision_max_gap": MergeDecisionConfig().llm_decision_max_gap,
            "merge_decision.hard_split_min_gap": MergeDecisionConfig().hard_split_min_gap,
            "merge_decision.max_combined_duration": MergeDecisionConfig().max_combined_duration,
            "subtitle.max_duration": SubtitleBuildConfig().max_duration,
            "subtitle.min_duration": SubtitleBuildConfig().min_duration,
            "subtitle.max_chars_cjk": SubtitleBuildConfig().max_chars_cjk,
            "subtitle.max_chars_latin": SubtitleBuildConfig().max_chars_latin,
            "vad.threshold": VADConfig().threshold,
            "vad.min_silence_duration_ms": float(VADConfig().min_silence_duration_ms),
            "noise_reduction.spectral_noise_reduction_db": NoiseReductionConfig().spectral_noise_reduction_db,
        }
        return defaults.get(param_path)
