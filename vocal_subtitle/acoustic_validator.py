"""Backward-compatible acoustic validator imports.

The implementation lives in :mod:`vocal_subtitle.acoustic.validator`.
"""

import warnings

warnings.warn(
    "vocal_subtitle.acoustic_validator is deprecated; import from "
    "vocal_subtitle.acoustic instead. This shim will be removed in a "
    "future release.",
    DeprecationWarning,
    stacklevel=2,
)

# 本模块的唯一职责是兼容再导出（含测试依赖的私有名），导入未直接
# 使用是预期行为，不做 F401 清理。
from .acoustic.diagnostics import generate_diagnostic_report  # noqa: F401
from .acoustic.validator import (  # noqa: F401
    AcousticValidationConfig,
    AcousticValidator,
    _boundary_confidence,
    _classify_energy_type,
    _compute_vad_overlap,
    _find_boundary_in_skeleton,
    _find_directional_boundary,
    _has_speech_in_range,
    _is_time_in_speech,
    _rms_energy_check,
    _silence_confirmed,
    classify_acoustic_events,
    export_skeleton_segments,
)

__all__ = [
    "AcousticValidationConfig",
    "AcousticValidator",
    "classify_acoustic_events",
    "export_skeleton_segments",
    "generate_diagnostic_report",
]
