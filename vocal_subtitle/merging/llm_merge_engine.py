"""Backward-compatible LLM merge imports.

The implementation lives in :mod:`vocal_subtitle.merging.merge_engine`.
"""

from .merge_engine import (
    _SECTION_END_MARKERS,
    _SECTION_START_PATTERNS,
    MERGE_DECISION_PROMPT,
    SUBTITLE_LAYOUT_RULES,
    LLMMergeEngine,
    MergeDecisionConfig,
    _detect_semantic_boundary,
    _physical_owner_compatible,
    _physical_owner_compatible_for_events,
    apply_frame_seamless_stitching,
    apply_layout_suggestions,
    auto_layout_events,
    auto_line_break_fallback,
)

__all__ = [
    "LLMMergeEngine",
    "MergeDecisionConfig",
    "MERGE_DECISION_PROMPT",
    "_SECTION_END_MARKERS",
    "_SECTION_START_PATTERNS",
    "_detect_semantic_boundary",
    "_physical_owner_compatible",
    "_physical_owner_compatible_for_events",
    "SUBTITLE_LAYOUT_RULES",
    "apply_frame_seamless_stitching",
    "apply_layout_suggestions",
    "auto_layout_events",
    "auto_line_break_fallback",
]
