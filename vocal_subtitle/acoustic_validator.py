"""全局声学标尺校验 (方案七) — backward-compat re-export hub.

This module used to contain the entire validator, skeleton, boundary snap,
diagnostics, and export logic. Those live in:

- vocal_subtitle/acoustic/validator.py  — AcousticValidator class
- vocal_subtitle/acoustic/skeleton.py   — skeleton query helpers
- vocal_subtitle/acoustic/boundary.py   — boundary snap & confidence
- vocal_subtitle/acoustic/event_checks.py — RMS, VAD overlap, energy classification
- vocal_subtitle/acoustic/diagnostics.py — generate_diagnostic_report
- vocal_subtitle/acoustic/export.py     — export_skeleton_segments

Importing from here still works for backward compatibility.
"""

# Re-export all public symbols
from .acoustic.boundary import (  # noqa: F401
    _boundary_confidence,
    _find_boundary_in_skeleton,
    _find_directional_boundary,
    _preserve_reliable_asr_boundary,
    _record_boundary_diagnostic,
    _silence_confirmed,
)
from .acoustic.diagnostics import generate_diagnostic_report  # noqa: F401
from .acoustic.event_checks import (  # noqa: F401
    _classify_energy_type,
    _compute_vad_overlap,
    _rms_energy_check,
    classify_acoustic_events,
)
from .acoustic.export import export_skeleton_segments  # noqa: F401
from .acoustic.skeleton import (  # noqa: F401
    _has_speech_in_range,
    _is_time_in_speech,
)
from .acoustic.validator import AcousticValidator  # noqa: F401

# AcousticValidationConfig is imported from config for backward compatibility.
from .config_loader import AcousticValidationConfig  # noqa: F401
