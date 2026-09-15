"""Stage 5: 时间轴映射与字幕输出模块

负责将 ASR 分段识别结果映射回全局时间轴，并构建标准格式字幕。
"""

from .boundary_projection import (
    BoundaryCandidate,
    ProjectedBoundary,
    ProjectionResult,
    project_boundaries,
    project_with_repair,
)
from .end_time_validator import EndTimePostValidator
from .event_ops import clone_event, merge_event_group, shift_event
from .finalize import FinalizeConfig, FinalizeResult, finalize_subtitle_events
from .overlap_export import OverlapExportConfig, OverlapGroup, OverlapTrack
from .quality_report import DimensionScore, QualityReport, build_quality_report
from .strict_segmenter import (
    SegmentationResult,
    StrictSegmentationConfig,
    segment_events,
)
from .subtitle_builder import SubtitleBuilder, SubtitleRule
from .time_mapper import TimeMapper

__all__ = [
    "TimeMapper",
    "SubtitleBuilder",
    "SubtitleRule",
    "StrictSegmentationConfig",
    "SegmentationResult",
    "segment_events",
    "BoundaryCandidate",
    "ProjectedBoundary",
    "ProjectionResult",
    "project_boundaries",
    "project_with_repair",
    "FinalizeConfig",
    "FinalizeResult",
    "finalize_subtitle_events",
    "DimensionScore",
    "QualityReport",
    "build_quality_report",
    "clone_event",
    "merge_event_group",
    "shift_event",
    "OverlapExportConfig",
    "OverlapGroup",
    "OverlapTrack",
    "EndTimePostValidator",
]
