"""问题优先级计算器

基于 QUALITY_OPERATIONS.md §4 的四因子优先级公式:
  优先级 = 影响范围 × 用户严重度 × 可复现性 × 修复置信度

阈值:
  - ≥ 30: 立即修复 (IMMEDIATE)
  - 15-29: 本版本修复 (THIS_VERSION)
  - ≤ 14: 排期 (SCHEDULED)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PriorityLevel(str, Enum):
    """优先级等级"""

    IMMEDIATE = "immediate"  # ≥ 30 立即修复
    THIS_VERSION = "this_version"  # 15-29 本版本修复
    SCHEDULED = "scheduled"  # ≤ 14 排期


# 各因子的有效范围
FACTOR_RANGES: dict[str, tuple[int, int]] = {
    "impact_range": (1, 5),
    "user_severity": (1, 5),
    "reproducibility": (1, 3),
    "fix_confidence": (1, 3),
}

# 各因子的评分标准说明（供 CLI/WebUI 帮助文本使用）
FACTOR_GUIDANCE: dict[str, dict[int, str]] = {
    "impact_range": {
        1: "单一场景",
        3: "多场景",
        5: "所有用户",
    },
    "user_severity": {
        1: "格式偏好",
        3: "漏句",
        5: "输出损坏",
    },
    "reproducibility": {
        1: "偶发",
        2: "条件触发",
        3: "必然复现",
    },
    "fix_confidence": {
        1: "不确定",
        2: "有方案",
        3: "确定性修复",
    },
}

# 优先级阈值与等级映射
PRIORITY_THRESHOLDS: list[tuple[int, PriorityLevel]] = [
    (30, PriorityLevel.IMMEDIATE),
    (15, PriorityLevel.THIS_VERSION),
    (0, PriorityLevel.SCHEDULED),
]


@dataclass
class PriorityFactors:
    """优先级四因子"""

    impact_range: int = 1  # 1-5 影响范围
    user_severity: int = 1  # 1-5 用户严重度
    reproducibility: int = 1  # 1-3 可复现性
    fix_confidence: int = 1  # 1-3 修复置信度

    def validate(self) -> None:
        """验证各因子是否在有效范围内。

        Raises:
            ValueError: 任一因子超出范围
        """
        for name, (lo, hi) in FACTOR_RANGES.items():
            value = getattr(self, name)
            if not (lo <= value <= hi):
                raise ValueError(f"{name} 必须在 [{lo}, {hi}] 范围内，当前值: {value}")

    def to_dict(self) -> dict:
        return {
            "impact_range": self.impact_range,
            "user_severity": self.user_severity,
            "reproducibility": self.reproducibility,
            "fix_confidence": self.fix_confidence,
        }


@dataclass
class PriorityResult:
    """优先级计算结果"""

    score: int
    level: PriorityLevel
    factors: PriorityFactors
    recommendation: str = ""

    def __post_init__(self) -> None:
        if not self.recommendation:
            self.recommendation = _level_recommendation(self.level)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "level": self.level.value,
            "recommendation": self.recommendation,
            "factors": self.factors.to_dict(),
        }


def _level_recommendation(level: PriorityLevel) -> str:
    """获取优先级等级的中文建议"""
    _map = {
        PriorityLevel.IMMEDIATE: "立即修复",
        PriorityLevel.THIS_VERSION: "本版本修复",
        PriorityLevel.SCHEDULED: "排期",
    }
    return _map.get(level, "排期")


def _score_to_level(score: int) -> PriorityLevel:
    """将数值分数映射到优先级等级"""
    for threshold, level in PRIORITY_THRESHOLDS:
        if score >= threshold:
            return level
    return PriorityLevel.SCHEDULED


class IssuePriorityCalculator:
    """问题优先级计算器。

    使用示例:
        calc = IssuePriorityCalculator()
        factors = PriorityFactors(impact_range=3, user_severity=4,
                                  reproducibility=2, fix_confidence=3)
        result = calc.calculate(factors)
        # result.score = 72, result.level = IMMEDIATE
    """

    @staticmethod
    def calculate(factors: PriorityFactors) -> PriorityResult:
        """计算优先级分数和等级。

        Args:
            factors: 四因子评分

        Returns:
            PriorityResult 包含分数、等级、建议

        Raises:
            ValueError: 因子超出有效范围
        """
        factors.validate()

        score = (
            factors.impact_range
            * factors.user_severity
            * factors.reproducibility
            * factors.fix_confidence
        )
        level = _score_to_level(score)

        return PriorityResult(
            score=score,
            level=level,
            factors=factors,
        )

    @staticmethod
    def calculate_from_issue(
        issue: dict[str, Any],
        factors: PriorityFactors,
    ) -> PriorityResult:
        """从 IssueRecord dict 和因子计算优先级。

        Args:
            issue: IssueRecord.to_dict() 的结果
            factors: 四因子评分

        Returns:
            PriorityResult
        """
        return IssuePriorityCalculator.calculate(factors)

    @staticmethod
    def batch_calculate(
        issues: list[dict[str, Any]],
    ) -> list[PriorityResult]:
        """批量计算优先级。

        每个 issue dict 需包含 impact_range, user_severity,
        reproducibility, fix_confidence 字段。

        Args:
            issues: 问题列表，每项包含四因子字段

        Returns:
            按分数降序排列的 PriorityResult 列表
        """
        results = []
        for issue in issues:
            factors = PriorityFactors(
                impact_range=issue.get("impact_range", 1),
                user_severity=issue.get("user_severity", 1),
                reproducibility=issue.get("reproducibility", 1),
                fix_confidence=issue.get("fix_confidence", 1),
            )
            results.append(IssuePriorityCalculator.calculate(factors))

        return sorted(results, key=lambda r: r.score, reverse=True)


__all__ = [
    "PriorityLevel",
    "PriorityFactors",
    "PriorityResult",
    "IssuePriorityCalculator",
    "FACTOR_RANGES",
    "FACTOR_GUIDANCE",
    "PRIORITY_THRESHOLDS",
]
