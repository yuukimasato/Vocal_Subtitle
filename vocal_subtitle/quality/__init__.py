"""Quality gates for reproducible offline production validation.

质量运营模块 — 对应 QUALITY_OPERATIONS.md (quality-ops-v1)。

组件:
  1. golden_gate — 黄金质量门禁（确定性评测）
  2. quality_classifier — 9 类质量问题自动分类与严重度评估
  3. scene_slicer — 7 维场景切片与平衡检查
  4. priority_calculator — 四因子优先级计算（影响×严重度×可复现性×置信度）
  5. trend_reporter — 版本趋势报告生成（YAML/Markdown）
  6. controlled_runner — 对照运行管理（baseline/candidate/shadow）
  7. data_version_manager — D0-D4 数据集版本管理
"""

from .controlled_runner import (
    ComparisonRecommendation,
    ControlledRunManager,
    ControlledRunSpec,
    RunKind,
)
from .data_version_manager import (
    DatasetTier,
    DatasetVersion,
    DatasetVersionStatus,
    DataVersionManager,
    RETENTION_POLICIES,
)
from .golden_gate import GoldenQualityThresholds, evaluate_golden_set
from .priority_calculator import (
    FACTOR_GUIDANCE,
    FACTOR_RANGES,
    PRIORITY_THRESHOLDS,
    IssuePriorityCalculator,
    PriorityFactors,
    PriorityLevel,
    PriorityResult,
)
from .quality_classifier import (
    CATEGORY_KEYWORDS,
    CATEGORY_SEVERITY_RANGE,
    SEVERITY_DOWNGRADE_INDICATORS,
    CategoryDefinition,
    IssueCategory,
    IssueClassifier,
    IssueSeverity,
    QualityIssue,
)
from .scene_slicer import (
    AUDIO_LENGTHS,
    BACKGROUND_NOISES,
    DEVICES,
    DIMENSION_NAMES,
    LANGUAGES,
    MAX_SINGLE_DIMENSION_RATIO,
    SCENE_TYPES,
    SPEAKER_COUNTS,
    SPEECH_RATES,
    SceneSlicer,
    SceneTag,
)
from .trend_reporter import (
    TREND_METRIC_DIRECTIONS,
    MetricChange,
    TrendDiff,
    TrendReporter,
    VersionMetrics,
)

__all__ = [
    # golden_gate
    "GoldenQualityThresholds",
    "evaluate_golden_set",
    # quality_classifier
    "IssueCategory",
    "IssueSeverity",
    "CategoryDefinition",
    "QualityIssue",
    "IssueClassifier",
    "CATEGORY_SEVERITY_RANGE",
    "CATEGORY_KEYWORDS",
    "SEVERITY_DOWNGRADE_INDICATORS",
    # scene_slicer
    "SceneTag",
    "SceneSlicer",
    "DIMENSION_NAMES",
    "LANGUAGES",
    "SPEAKER_COUNTS",
    "BACKGROUND_NOISES",
    "SPEECH_RATES",
    "AUDIO_LENGTHS",
    "DEVICES",
    "SCENE_TYPES",
    "MAX_SINGLE_DIMENSION_RATIO",
    # priority_calculator
    "PriorityLevel",
    "PriorityFactors",
    "PriorityResult",
    "IssuePriorityCalculator",
    "FACTOR_RANGES",
    "FACTOR_GUIDANCE",
    "PRIORITY_THRESHOLDS",
    # trend_reporter
    "VersionMetrics",
    "MetricChange",
    "TrendDiff",
    "TrendReporter",
    "TREND_METRIC_DIRECTIONS",
    # controlled_runner
    "RunKind",
    "ControlledRunSpec",
    "ComparisonRecommendation",
    "ControlledRunManager",
    # data_version_manager
    "DatasetTier",
    "DatasetVersion",
    "DatasetVersionStatus",
    "DataVersionManager",
    "RETENTION_POLICIES",
]
