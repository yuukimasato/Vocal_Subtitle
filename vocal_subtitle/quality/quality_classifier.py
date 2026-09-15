"""质量问题分类器

基于 QUALITY_OPERATIONS.md §1 定义的 9 类质量问题进行自动分类和严重度评估。

类别:
  - usability (可用性): 系统能否正常运行和产出
  - completeness (完整性): 是否遗漏语音内容
  - text_accuracy (文本准确性): 识别文本是否准确
  - time_accuracy (时间准确性): 时间轴是否精确
  - speaker (说话人): 说话人标注是否正确
  - readability (格式可读性): 断句、标点、长度是否合理
  - performance (性能): 处理速度、资源消耗
  - cost (成本): LLM API 调用、计算资源
  - explainability (可解释性): 输出能否被理解和审计
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IssueCategory(str, Enum):
    """质量问题类别"""

    USABILITY = "usability"
    COMPLETENESS = "completeness"
    TEXT_ACCURACY = "text_accuracy"
    TIME_ACCURACY = "time_accuracy"
    SPEAKER = "speaker"
    READABILITY = "readability"
    PERFORMANCE = "performance"
    COST = "cost"
    EXPLAINABILITY = "explainability"
    UNKNOWN = "unknown"


class IssueSeverity(str, Enum):
    """问题严重度"""

    CRITICAL = "critical"  # 系统不可用、输出损坏
    HIGH = "high"  # 严重影响使用
    MEDIUM = "medium"  # 一定影响
    LOW = "low"  # 轻微影响


# 各类别的严重度范围 (§1 表格)
CATEGORY_SEVERITY_RANGE: dict[IssueCategory, tuple[IssueSeverity, IssueSeverity]] = {
    IssueCategory.USABILITY: (IssueSeverity.CRITICAL, IssueSeverity.HIGH),
    IssueCategory.COMPLETENESS: (IssueSeverity.HIGH, IssueSeverity.MEDIUM),
    IssueCategory.TEXT_ACCURACY: (IssueSeverity.MEDIUM, IssueSeverity.LOW),
    IssueCategory.TIME_ACCURACY: (IssueSeverity.MEDIUM, IssueSeverity.LOW),
    IssueCategory.SPEAKER: (IssueSeverity.MEDIUM, IssueSeverity.LOW),
    IssueCategory.READABILITY: (IssueSeverity.LOW, IssueSeverity.LOW),
    IssueCategory.PERFORMANCE: (IssueSeverity.MEDIUM, IssueSeverity.MEDIUM),
    IssueCategory.COST: (IssueSeverity.LOW, IssueSeverity.LOW),
    IssueCategory.EXPLAINABILITY: (IssueSeverity.MEDIUM, IssueSeverity.MEDIUM),
    IssueCategory.UNKNOWN: (IssueSeverity.MEDIUM, IssueSeverity.LOW),
}

# 分类关键词映射（中英文混合，覆盖 §1 示例）
CATEGORY_KEYWORDS: dict[IssueCategory, list[str]] = {
    IssueCategory.USABILITY: [
        "崩溃",
        "crash",
        "无法导出",
        "输出损坏",
        "启动失败",
        "不能运行",
        "死机",
        "无响应",
        "闪退",
        "报错退出",
        "can't export",
        "output corrupted",
        "won't start",
    ],
    IssueCategory.COMPLETENESS: [
        "漏句",
        "漏词",
        "跳段",
        "漏识",
        "缺失",
        "遗漏",
        "少字幕",
        "缺字幕",
        "不完整",
        "空洞",
        "空白字幕",
        "missing",
        "dropped",
        "skipped",
        "incomplete",
        "gap",
    ],
    IssueCategory.TEXT_ACCURACY: [
        "错字",
        "同音字",
        "多字",
        "少字",
        "错词",
        "误识",
        "识别错误",
        "文本错误",
        "误判",
        "替换错误",
        "wrong word",
        "misrecognition",
        "incorrect text",
        "hallucination",
        "幻觉",
    ],
    IssueCategory.TIME_ACCURACY: [
        "偏移",
        "跨静音",
        "拉伸",
        "时间轴",
        "时间不准",
        "提前",
        "滞后",
        "对不上",
        "不同步",
        "延迟",
        "offset",
        "misaligned",
        "out of sync",
        "latency",
        "boundary",
        "边界",
    ],
    IssueCategory.SPEAKER: [
        "错标",
        "漏标",
        "说话人",
        "角色",
        "speaker",
        "标错人",
        "混标",
        "重复标注",
        "人标",
        "speaker label",
        "mislabel",
        "角色标注",
    ],
    IssueCategory.READABILITY: [
        "过长",
        "断句",
        "标点",
        "可读",
        "显示",
        "太长",
        "拆分",
        "合并",
        "排版",
        "换行",
        "readability",
        "formatting",
        "too long",
        "punctuation",
    ],
    IssueCategory.PERFORMANCE: [
        "超时",
        "OOM",
        "内存",
        "慢",
        "卡顿",
        "速度",
        "耗时",
        "资源",
        "显存",
        "GPU",
        "CPU占用",
        "timeout",
        "out of memory",
        "slow",
        "performance",
    ],
    IssueCategory.COST: [
        "费用",
        "成本",
        "API",
        "token",
        "调用量",
        "计费",
        "花费",
        "价格",
        "cost",
        "expensive",
        "billing",
    ],
    IssueCategory.EXPLAINABILITY: [
        "无 trace",
        "来源不明",
        "无法解释",
        "不知道哪来的",
        "无来源",
        "无法追溯",
        "黑盒",
        "不可审计",
        "untraceable",
        "unknown source",
        "black box",
    ],
}

# 严重度降级指示器
SEVERITY_DOWNGRADE_INDICATORS: dict[str, dict[str, Any]] = {
    "workaround_available": {
        "description": "存在可用的绕过方案",
        "effect": "high→medium, medium→low",
    },
    "user_blocked": {
        "description": "用户被完全阻塞",
        "effect": "upgrade to critical",
    },
    "data_corruption": {
        "description": "数据损坏",
        "effect": "→ critical",
    },
    "intermittent": {
        "description": "间歇性发生",
        "effect": "downgrade one level",
    },
}


@dataclass
class CategoryDefinition:
    """类别定义"""

    category: IssueCategory
    description: str
    severity_max: IssueSeverity
    severity_min: IssueSeverity
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "description": self.description,
            "severity_max": self.severity_max.value,
            "severity_min": self.severity_min.value,
            "examples": self.examples,
        }


@dataclass
class QualityIssue:
    """质量问题记录"""

    issue_id: str = ""
    title: str = ""
    description: str = ""
    category: IssueCategory = IssueCategory.UNKNOWN
    severity: IssueSeverity = IssueSeverity.MEDIUM
    detected_by: str = ""  # D0_engineering | D1_reference | D2_feedback | D3_feedback | D4_challenge | user_report
    impacted_scenes: list[str] = field(default_factory=list)
    source: str = ""  # run_id / sample_id / report path
    first_seen: str = ""
    impact_range: int = 1  # 1-5
    reproducibility: int = 1  # 1-3
    fix_confidence: int = 1  # 1-3

    def to_dict(self) -> dict:
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "detected_by": self.detected_by,
            "impacted_scenes": self.impacted_scenes,
            "source": self.source,
            "first_seen": self.first_seen,
            "impact_range": self.impact_range,
            "reproducibility": self.reproducibility,
            "fix_confidence": self.fix_confidence,
        }


def _build_category_definitions() -> dict[IssueCategory, CategoryDefinition]:
    """构建类别定义索引。"""
    descriptions = {
        IssueCategory.USABILITY: "系统能否正常运行和产出",
        IssueCategory.COMPLETENESS: "是否遗漏语音内容",
        IssueCategory.TEXT_ACCURACY: "识别文本是否准确",
        IssueCategory.TIME_ACCURACY: "时间轴是否精确",
        IssueCategory.SPEAKER: "说话人标注是否正确",
        IssueCategory.READABILITY: "断句、标点、长度是否合理",
        IssueCategory.PERFORMANCE: "处理速度、资源消耗",
        IssueCategory.COST: "LLM API 调用、计算资源",
        IssueCategory.EXPLAINABILITY: "输出能否被理解和审计",
        IssueCategory.UNKNOWN: "未知类别",
    }
    examples = {
        IssueCategory.USABILITY: ["崩溃", "无法导出", "输出损坏"],
        IssueCategory.COMPLETENESS: ["漏句", "漏词", "跳段"],
        IssueCategory.TEXT_ACCURACY: ["错字", "同音字", "多字少字"],
        IssueCategory.TIME_ACCURACY: ["偏移", "跨静音", "过度拉伸"],
        IssueCategory.SPEAKER: ["错标", "漏标", "重复"],
        IssueCategory.READABILITY: ["过长字幕", "不当断句"],
        IssueCategory.PERFORMANCE: ["超时", "OOM", "4h+ 处理"],
        IssueCategory.COST: ["高 API 费用"],
        IssueCategory.EXPLAINABILITY: ["无 trace", "来源不明"],
        IssueCategory.UNKNOWN: [],
    }
    defs = {}
    for cat in IssueCategory:
        sev_range = CATEGORY_SEVERITY_RANGE.get(
            cat, (IssueSeverity.MEDIUM, IssueSeverity.LOW)
        )
        defs[cat] = CategoryDefinition(
            category=cat,
            description=descriptions.get(cat, ""),
            severity_max=sev_range[0],
            severity_min=sev_range[1],
            examples=examples.get(cat, []),
        )
    return defs


class IssueClassifier:
    """质量问题分类器。

    使用示例:
        classifier = IssueClassifier()
        issue = classifier.analyze({
            "title": "长音频尾部漏句",
            "description": ">30min 音频最后一段语音缺失字幕",
            "indicators": {"workaround_available": False}
        })
        # issue.category = COMPLETENESS, issue.severity = HIGH
    """

    CATEGORY_DEFINITIONS: dict[IssueCategory, CategoryDefinition] = (
        _build_category_definitions()
    )

    @staticmethod
    def classify(text: str) -> IssueCategory:
        """基于关键词对问题描述进行分类。

        Args:
            text: 问题标题或描述

        Returns:
            匹配的类别，无匹配返回 UNKNOWN
        """
        text_lower = text.lower()
        best_category = IssueCategory.UNKNOWN
        best_count = 0

        for category, keywords in CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_category = category

        if best_count == 0:
            logger.debug("Unable to classify issue text: %s", text[:100])

        return best_category

    @staticmethod
    def assess_severity(
        category: IssueCategory,
        indicators: Mapping[str, Any] | None = None,
    ) -> IssueSeverity:
        """评估问题严重度。

        从类别的最高严重度开始，根据 indicators 进行升降。

        Args:
            category: 问题类别
            indicators: 严重度指示器，如:
                {"workaround_available": True, "user_blocked": False,
                 "data_corruption": False, "intermittent": False}

        Returns:
            IssueSeverity 级别
        """
        sev_range = CATEGORY_SEVERITY_RANGE.get(
            category, (IssueSeverity.MEDIUM, IssueSeverity.LOW)
        )
        severity = sev_range[0]  # 从最高开始

        if indicators is None:
            indicators = {}

        # 数据损坏 → 直接升级到 critical
        if indicators.get("data_corruption"):
            return IssueSeverity.CRITICAL

        # 用户完全被阻塞 → 升级
        if indicators.get("user_blocked"):
            if severity != IssueSeverity.CRITICAL:
                return IssueSeverity.CRITICAL

        # 有绕过方案 → 降一级
        if indicators.get("workaround_available"):
            downgrade_map = {
                IssueSeverity.CRITICAL: IssueSeverity.HIGH,
                IssueSeverity.HIGH: IssueSeverity.MEDIUM,
                IssueSeverity.MEDIUM: IssueSeverity.LOW,
                IssueSeverity.LOW: IssueSeverity.LOW,
            }
            severity = downgrade_map.get(severity, severity)

        # 间歇性 → 降一级
        if indicators.get("intermittent"):
            downgrade_map = {
                IssueSeverity.CRITICAL: IssueSeverity.HIGH,
                IssueSeverity.HIGH: IssueSeverity.MEDIUM,
                IssueSeverity.MEDIUM: IssueSeverity.LOW,
                IssueSeverity.LOW: IssueSeverity.LOW,
            }
            severity = downgrade_map.get(severity, severity)

        # 不降到类别最低严重度以下
        sev_order = [
            IssueSeverity.CRITICAL,
            IssueSeverity.HIGH,
            IssueSeverity.MEDIUM,
            IssueSeverity.LOW,
        ]
        min_idx = sev_order.index(sev_range[1])
        current_idx = sev_order.index(severity)
        if current_idx > min_idx:
            severity = sev_order[min_idx]

        return severity

    @classmethod
    def analyze(
        cls,
        raw: dict[str, Any],
        *,
        source: str = "",
        detected_by: str = "",
    ) -> QualityIssue:
        """对原始问题描述进行全面分析。

        Args:
            raw: {"title": ..., "description": ..., "indicators": {...}}
            source: 问题来源 (run_id / sample_id)
            detected_by: 检测来源 (D0_engineering, user_report, etc.)

        Returns:
            包含分类和严重度的 QualityIssue
        """
        title = str(raw.get("title", ""))
        description = str(raw.get("description", ""))
        combined_text = f"{title} {description}"

        category = cls.classify(combined_text)
        indicators = raw.get("indicators", {})
        severity = cls.assess_severity(category, indicators)

        return QualityIssue(
            title=title,
            description=description,
            category=category,
            severity=severity,
            detected_by=detected_by or raw.get("detected_by", ""),
            impacted_scenes=list(raw.get("impacted_scenes", [])),
            source=source or raw.get("source", ""),
            first_seen=raw.get("first_seen", ""),
            impact_range=raw.get("impact_range", 1),
            reproducibility=raw.get("reproducibility", 1),
            fix_confidence=raw.get("fix_confidence", 1),
        )

    @classmethod
    def batch_analyze(
        cls,
        issues: list[dict[str, Any]],
        *,
        source: str = "",
        detected_by: str = "",
    ) -> list[QualityIssue]:
        """批量分析问题列表。

        Args:
            issues: 原始问题列表
            source: 问题来源
            detected_by: 检测来源

        Returns:
            QualityIssue 列表
        """
        return [
            cls.analyze(issue, source=source, detected_by=detected_by)
            for issue in issues
        ]


__all__ = [
    "IssueCategory",
    "IssueSeverity",
    "CategoryDefinition",
    "QualityIssue",
    "IssueClassifier",
    "CATEGORY_SEVERITY_RANGE",
    "CATEGORY_KEYWORDS",
    "SEVERITY_DOWNGRADE_INDICATORS",
]
