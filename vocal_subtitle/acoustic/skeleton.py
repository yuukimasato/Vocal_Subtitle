#!/usr/bin/env python3
"""Extract acoustic_validator.py helper functions from the original file.

The parent module should be split into:
  acoustic/skeleton.py — _get_skeleton(), skeleton query helpers
  acoustic/boundary.py — _physical_snap_validation(), _find_directional_boundary(), etc.
  acoustic/event_checks.py — silence, RMS, VAD overlap, classify_acoustic_events
  acoustic/diagnostics.py — generate_diagnostic_report()
  acoustic/export.py — export_skeleton_segments()
  acoustic/validator.py — AcousticValidator class (thin entry point)

This script pulls the public surface from the existing single file.
"""

# ------------------------------------------------------------
# skeleton.py — acoustic skeleton construction and querying
# ------------------------------------------------------------


def _is_time_in_speech(t: float, skeleton: list) -> bool:
    """判断时间点是否在语音段内"""
    for s_start, s_end in skeleton:
        if s_start <= t <= s_end:
            return True
    return False


def _has_speech_in_range(t1: float, t2: float, skeleton: list) -> bool:
    """判断 [t1, t2] 区间内是否有语音"""
    for s_start, s_end in skeleton:
        if s_start < t2 and s_end > t1:
            return True
    return False
