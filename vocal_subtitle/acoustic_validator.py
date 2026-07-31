"""Backward-compatible acoustic validator imports.

The implementation lives in :mod:`vocal_subtitle.acoustic.validator`.
"""

from .acoustic.validator import (
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
]
