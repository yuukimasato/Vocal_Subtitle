"""影子模式评估器 (Phase 5.4)

在后台异步验证新参数效果，积累足够对比数据后
自动决定是否将新参数升级为正式参数。

工程价值：
  - 零风险验证：新参数不影响用户获得的字幕质量
  - 数据驱动升级：基于 health_score 差值自动决策
  - 快速回退：评估不通过时透明丢弃新参数
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShadowRunResult:
    """单次影子运行的结果"""

    timestamp: str = ""
    health_current: float = 0.0  # 旧参数健康度
    health_shadow: float = 0.0   # 新参数健康度
    health_detail_current: Dict[str, float] = field(default_factory=dict)
    health_detail_shadow: Dict[str, float] = field(default_factory=dict)
    alignment_coverage: float = 0.0
    diff_summary: str = ""


@dataclass
class ShadowEvaluation:
    """影子模式评估结论"""

    should_upgrade: bool = False
    reason: str = ""
    current_mean_health: float = 0.0
    shadow_mean_health: float = 0.0
    health_delta: float = 0.0  # shadow - current
    total_runs: int = 0
    degraded_dims: List[str] = field(default_factory=list)  # 有退化的子维度
    recommendation: str = ""  # "upgrade" | "discard" | "continue" | "timeout_upgrade" | "timeout_discard"


class ShadowModeEvaluator:
    """影子模式评估器 — 决定是否将新参数升级为正式参数

    升级条件（全部满足）:
    1. shadow_runs 数量 >= min_shadow_runs（默认 10 次）
    2. 新参数 health_score 均值 > 旧参数 + upgrade_threshold（默认 5%）
    3. 新参数在各项子指标上均无显著退化（无子项下降 > 10%）
    4. 运行天数 <= max_shadow_duration_days（超期自动对比并决定）
    """

    def __init__(
        self,
        min_shadow_runs: int = 10,
        upgrade_threshold: float = 0.05,
        max_shadow_duration_days: int = 14,
        max_dim_degradation: float = 0.10,  # 子项最大允许退化
    ):
        self.min_shadow_runs = min_shadow_runs
        self.upgrade_threshold = upgrade_threshold
        self.max_shadow_duration_days = max_shadow_duration_days
        self.max_dim_degradation = max_dim_degradation
        self._runs: List[ShadowRunResult] = []

    @property
    def run_count(self) -> int:
        return len(self._runs)

    def add_run(self, result: ShadowRunResult):
        """添加一次影子运行结果"""
        if not result.timestamp:
            result.timestamp = datetime.now().isoformat()
        self._runs.append(result)
        logger.info(
            "Shadow run #%d: current=%.1f, shadow=%.1f, delta=%+.1f",
            len(self._runs),
            result.health_current,
            result.health_shadow,
            result.health_shadow - result.health_current,
        )

    def should_upgrade(self) -> ShadowEvaluation:
        """评估是否应升级为新参数

        Returns:
            ShadowEvaluation 包含升级决策和原因
        """
        n = len(self._runs)
        if n == 0:
            return ShadowEvaluation(
                should_upgrade=False,
                reason="No shadow runs collected yet",
                recommendation="continue",
            )

        # 收集指标
        current_healths = [r.health_current for r in self._runs if r.health_current > 0]
        shadow_healths = [r.health_shadow for r in self._runs if r.health_shadow > 0]

        if not current_healths or not shadow_healths:
            return ShadowEvaluation(
                should_upgrade=False,
                reason="Missing health score data",
                total_runs=n,
                recommendation="continue",
            )

        current_mean = float(np.mean(current_healths))
        shadow_mean = float(np.mean(shadow_healths))
        delta = shadow_mean - current_mean

        eval_result = ShadowEvaluation(
            current_mean_health=round(current_mean, 2),
            shadow_mean_health=round(shadow_mean, 2),
            health_delta=round(delta, 2),
            total_runs=n,
        )

        # ---- 检查运行天数 ----
        if n >= 2:
            try:
                first_ts = datetime.fromisoformat(self._runs[0].timestamp)
                last_ts = datetime.fromisoformat(self._runs[-1].timestamp)
                elapsed_days = (last_ts - first_ts).total_seconds() / 86400

                if elapsed_days > self.max_shadow_duration_days:
                    logger.info(
                        "Shadow mode exceeded max duration (%.1f days > %d days)",
                        elapsed_days, self.max_shadow_duration_days,
                    )
                    # 超期：强行决策
                    if delta > 0:
                        eval_result.should_upgrade = True
                        eval_result.reason = (
                            f"Shadow period expired ({elapsed_days:.1f}d > {self.max_shadow_duration_days}d), "
                            f"shadow is better by {delta:+.1f}"
                        )
                        eval_result.recommendation = "timeout_upgrade"
                    else:
                        eval_result.reason = (
                            f"Shadow period expired ({elapsed_days:.1f}d > {self.max_shadow_duration_days}d), "
                            f"shadow is worse by {abs(delta):.1f}"
                        )
                        eval_result.recommendation = "timeout_discard"
                    return eval_result
            except (ValueError, TypeError):
                pass

        # ---- 条件 1: 最少运行次数 ----
        if n < self.min_shadow_runs:
            eval_result.reason = (
                f"Insufficient runs: {n}/{self.min_shadow_runs} "
                f"(current_mean={current_mean:.1f}, shadow_mean={shadow_mean:.1f})"
            )
            eval_result.recommendation = "continue"
            return eval_result

        # ---- 条件 2: 健康度提升阈值 ----
        relative_improvement = delta / max(current_mean, 1.0)
        if relative_improvement < self.upgrade_threshold:
            eval_result.reason = (
                f"Insufficient improvement: {relative_improvement*100:+.1f}% "
                f"< {self.upgrade_threshold*100:.0f}% threshold "
                f"(current={current_mean:.1f}, shadow={shadow_mean:.1f})"
            )
            eval_result.recommendation = "discard"
            return eval_result

        # ---- 条件 3: 子项无显著退化 ----
        degraded_dims = self._check_dim_degradation()
        if degraded_dims:
            eval_result.degraded_dims = degraded_dims
            eval_result.reason = (
                f"Shadow has degraded dimensions: {', '.join(degraded_dims)}"
            )
            eval_result.recommendation = "discard"
            return eval_result

        # ---- 全部通过 ----
        eval_result.should_upgrade = True
        eval_result.reason = (
            f"Shadow parameters improve health by {relative_improvement*100:+.1f}% "
            f"({current_mean:.1f} → {shadow_mean:.1f}) over {n} runs"
        )
        eval_result.recommendation = "upgrade"
        return eval_result

    def _check_dim_degradation(self) -> List[str]:
        """检查子维度是否有显著退化

        Returns:
            退化的子维度名称列表
        """
        if not self._runs:
            return []

        dims = ["alignment_coverage", "semantic_similarity", "time_iou", "structure_consistency"]
        degraded = []

        for dim in dims:
            current_vals = []
            shadow_vals = []
            for r in self._runs:
                c_val = r.health_detail_current.get(dim, 0)
                s_val = r.health_detail_shadow.get(dim, 0)
                if c_val > 0 and s_val > 0:
                    current_vals.append(c_val)
                    shadow_vals.append(s_val)

            if not current_vals or not shadow_vals:
                continue

            current_mean = np.mean(current_vals)
            shadow_mean = np.mean(shadow_vals)

            if current_mean > 0:
                relative_change = (shadow_mean - current_mean) / current_mean
                if relative_change < -self.max_dim_degradation:
                    degraded.append(f"{dim} ({relative_change*100:+.1f}%)")

        return degraded

    def reset(self):
        """重置所有影子运行数据"""
        self._runs.clear()
        logger.info("Shadow mode data reset")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "total_runs": len(self._runs),
            "runs": [
                {
                    "timestamp": r.timestamp,
                    "health_current": r.health_current,
                    "health_shadow": r.health_shadow,
                    "health_delta": round(r.health_shadow - r.health_current, 2),
                    "detail_current": r.health_detail_current,
                    "detail_shadow": r.health_detail_shadow,
                }
                for r in self._runs[-20:]  # 最近 20 条
            ],
        }
