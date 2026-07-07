"""参数变更影响预估器 (Phase 5.4)

基于历史反馈数据中的参数-效果映射关系，
对新参数变更进行效果预估，增强可解释性。

预估维度:
  - 单行字幕平均时长变化 (%)
  - 合并频次变化 (%)
  - 拆分频次变化 (%)
  - 句尾截断概率变化 (%)
  - 字幕总行数变化 (%)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .diff_analyzer import ParamAdjustment
from .param_learner import PARAM_BOUNDS

logger = logging.getLogger(__name__)


@dataclass
class ImpactPrediction:
    """单个参数变更的影响预测"""

    param_path: str
    current_value: float
    new_value: float
    delta: float
    delta_pct: float  # 百分比变化

    # 预估影响（每项含方向 + 幅度 + 置信区间）
    avg_duration_change_pct: Optional[float] = None   # 单行字幕平均时长变化
    merge_frequency_change_pct: Optional[float] = None  # 合并频次变化
    split_frequency_change_pct: Optional[float] = None  # 拆分频次变化
    end_truncation_change_pct: Optional[float] = None   # 句尾截断概率变化
    total_line_count_change_pct: Optional[float] = None  # 字幕总行数变化

    confidence_low: float = 0.0
    confidence_high: float = 0.0
    summary: str = ""


# 参数-影响映射系数（基于经验数据）
# 单位：参数值每变化 1%，对应指标变化 X%
_PARAM_IMPACT_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "merging.padding": {
        "avg_duration": 0.8,       # padding +1% → 时长 +0.8%
        "end_truncation": -1.2,    # padding +1% → 截断 -1.2%
        "total_lines": -0.3,       # padding +1% → 行数 -0.3%
    },
    "merging.padding_max": {
        "avg_duration": 0.4,
        "end_truncation": -0.6,
    },
    "merge_decision.fast_merge_max_gap": {
        "merge_frequency": 1.5,    # gap +1% → 合并 +1.5%
        "avg_duration": 0.6,       # 合并更多 → 行更长
        "total_lines": -0.5,       # 合并更多 → 行数减少
    },
    "merge_decision.llm_decision_max_gap": {
        "merge_frequency": 0.8,
        "split_frequency": -1.0,
        "avg_duration": 0.4,
    },
    "merge_decision.llm_decision_min_gap": {
        "merge_frequency": 0.6,
        "split_frequency": -0.5,
    },
    "merge_decision.hard_split_min_gap": {
        "split_frequency": 0.4,
        "merge_frequency": -0.3,
    },
    "subtitle.max_duration": {
        "avg_duration": 0.9,
        "total_lines": -0.4,
        "merge_frequency": 0.3,
    },
    "subtitle.max_chars_cjk": {
        "avg_duration": 0.5,
        "total_lines": -0.3,
    },
}


class ImpactEstimator:
    """参数变更影响预估器

    基于历史反馈中的参数-效果映射关系，
    对新参数变更进行效果预估和可读性说明。
    """

    def estimate(
        self,
        adjustments: Dict[str, ParamAdjustment],
        current_overrides: Dict[str, Any],
    ) -> List[ImpactPrediction]:
        """对每个参数变更生成影响预估

        Args:
            adjustments: 要应用的参数调整建议
            current_overrides: 当前的参数覆盖值

        Returns:
            ImpactPrediction 列表
        """
        predictions = []

        for param_path, adj in adjustments.items():
            current_val = self._get_nested(current_overrides, param_path)
            if current_val is None:
                current_val = self._get_default(param_path)

            # 计算新值（EMA 近似）
            if adj.direction == "increase":
                new_val = current_val + adj.observed_value * adj.learn_weight
            else:
                new_val = current_val - adj.observed_value * adj.learn_weight

            # 钳制
            bounds = PARAM_BOUNDS.get(param_path)
            if bounds:
                new_val = max(bounds[0], min(bounds[1], new_val))

            delta = new_val - current_val
            delta_pct = (delta / current_val * 100) if current_val > 0 else 0.0

            pred = ImpactPrediction(
                param_path=param_path,
                current_value=round(current_val, 4),
                new_value=round(new_val, 4),
                delta=round(delta, 4),
                delta_pct=round(delta_pct, 1),
                confidence_low=round(max(0, adj.confidence - 0.15), 2),
                confidence_high=round(min(1.0, adj.confidence + 0.15), 2),
            )

            # 应用映射系数
            coeffs = _PARAM_IMPACT_COEFFICIENTS.get(param_path, {})
            if "avg_duration" in coeffs:
                pred.avg_duration_change_pct = round(delta_pct * coeffs["avg_duration"], 1)
            if "merge_frequency" in coeffs:
                pred.merge_frequency_change_pct = round(delta_pct * coeffs["merge_frequency"], 1)
            if "split_frequency" in coeffs:
                pred.split_frequency_change_pct = round(delta_pct * coeffs["split_frequency"], 1)
            if "end_truncation" in coeffs:
                pred.end_truncation_change_pct = round(delta_pct * coeffs["end_truncation"], 1)
            if "total_lines" in coeffs:
                pred.total_line_count_change_pct = round(delta_pct * coeffs["total_lines"], 1)

            # 生成人类可读摘要
            pred.summary = self._make_summary(pred)

            predictions.append(pred)

        return predictions

    def estimate_single(
        self,
        param_path: str,
        current_value: float,
        adj: ParamAdjustment,
    ) -> ImpactPrediction:
        """预估单个参数变更的影响"""
        return self.estimate({param_path: adj}, {param_path: current_value})[0]

    @staticmethod
    def _make_summary(pred: ImpactPrediction) -> str:
        """生成人类可读的影响摘要"""
        parts = []
        param_name = pred.param_path.split(".")[-1]

        if pred.avg_duration_change_pct is not None:
            direction = "增加" if pred.avg_duration_change_pct > 0 else "减少"
            parts.append(f"单行字幕平均时长{direction}{abs(pred.avg_duration_change_pct):.0f}%")

        if pred.merge_frequency_change_pct is not None:
            direction = "增加" if pred.merge_frequency_change_pct > 0 else "减少"
            parts.append(f"合并频次{direction}{abs(pred.merge_frequency_change_pct):.0f}%")

        if pred.split_frequency_change_pct is not None:
            direction = "增加" if pred.split_frequency_change_pct > 0 else "减少"
            parts.append(f"拆分频次{direction}{abs(pred.split_frequency_change_pct):.0f}%")

        if pred.end_truncation_change_pct is not None:
            direction = "降低" if pred.end_truncation_change_pct < 0 else "增加"
            parts.append(f"句尾截断概率{direction}{abs(pred.end_truncation_change_pct):.0f}%")

        if pred.total_line_count_change_pct is not None:
            direction = "减少" if pred.total_line_count_change_pct < 0 else "增加"
            parts.append(f"字幕总行数{direction}{abs(pred.total_line_count_change_pct):.0f}%")

        if not parts:
            return f"变更 {param_name}: {pred.current_value} → {pred.new_value}"

        return f"变更 {param_name} ({pred.current_value} → {pred.new_value}): " + "，".join(parts)

    @staticmethod
    def _get_nested(overrides: Dict[str, Any], param_path: str) -> Optional[float]:
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
    def _get_default(param_path: str) -> float:
        """获取参数默认值"""
        # 复用 ParamLearner 的默认值获取逻辑
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
        return defaults.get(param_path, 0.0)
