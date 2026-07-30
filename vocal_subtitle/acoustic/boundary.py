"""Boundary search, snap validation, and confidence helpers for acoustic validation.

Uses absolute-second coordinates. No display-time or frame-time conversions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .skeleton import _is_time_in_speech


def _find_boundary_in_skeleton(
    t: float, skeleton: List[Tuple[float, float]],
) -> Tuple[bool, float]:
    """判断时间点 t 是否在语音段内，并返回最近的边界

    Returns:
        (is_in_speech, nearest_boundary)
    """
    for s_start, s_end in skeleton:
        if s_start <= t <= s_end:
            return True, t
        if t < s_start:
            return False, s_start

    # t 在所有语音段之后
    return False, skeleton[-1][1] if skeleton else t


def _find_directional_boundary(
    t: float,
    skeleton: List[Tuple[float, float]],
    boundary_type: str,
) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]]]:
    """Find the boundary appropriate for a start or end endpoint.

    The legacy helper above returns the next boundary in a gap for both
    endpoint types. That is valid for a start, but an end must use the
    previous speech end or it can be extended across an entire silence gap.

    Returns ``(inside_speech, candidate_boundary, containing_speech)``.
    ``candidate_boundary`` is ``None`` when no boundary exists in the
    direction that is safe for this endpoint.
    """
    if boundary_type not in {"start", "end"}:
        raise ValueError("boundary_type must be 'start' or 'end'")

    previous_end: Optional[float] = None
    for speech_start, speech_end in skeleton:
        if speech_start <= t <= speech_end:
            return True, t, (speech_start, speech_end)
        if t < speech_start:
            if boundary_type == "start":
                return False, speech_start, None
            return False, previous_end, None
        previous_end = speech_end

    if boundary_type == "end":
        return False, previous_end, None
    return False, None, None


def _boundary_confidence(event: object, boundary_type: str) -> Optional[float]:
    """Return confidence for the word anchoring one event endpoint."""
    words = list(getattr(event, "words", []) or [])
    if words:
        word = words[0] if boundary_type == "start" else words[-1]
        value = getattr(word, "confidence", None)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass

    for name in (f"{boundary_type}_confidence", "boundary_confidence"):
        value = getattr(event, name, None)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass
    return None


def _preserve_reliable_asr_boundary(
    event: object,
    boundary_type: str,
    config,
) -> bool:
    """Keep a reliable word-level endpoint ahead of physical snapping."""
    words = list(getattr(event, "words", []) or [])
    confidence = _boundary_confidence(event, boundary_type)
    return bool(
        words
        and confidence is not None
        and confidence >= config.confidence_threshold
    )


def _silence_confirmed(
    audio: Optional[np.ndarray],
    sample_rate: int,
    time_point: float,
    *,
    distance: float,
) -> bool:
    """Confirm a gap is silent, with a conservative no-audio fallback."""
    if audio is None:
        # Without samples, only a tiny structural correction is safe.
        return distance <= 0.03
    return not _rms_energy_check(
        audio, sample_rate, time_point,
        window_ms=50, threshold_ratio=2.0,
    )


def _record_boundary_diagnostic(
    report: Dict,
    event: object,
    boundary_type: str,
    action: str,
    *,
    reason: str,
    original_time: float,
    candidate_time: Optional[float],
    distance: Optional[float],
) -> None:
    """Append a compact, auditable endpoint decision."""
    report.setdefault("boundary_diagnostics", []).append({
        "id": getattr(event, "index", 0),
        "boundary": boundary_type,
        "action": action,
        "reason": reason,
        "original_time": round(float(original_time), 6),
        "candidate_time": (
            round(float(candidate_time), 6)
            if candidate_time is not None else None
        ),
        "distance_ms": (
            round(float(distance) * 1000, 3)
            if distance is not None else None
        ),
    })


# Import at bottom to avoid circular import
from .event_checks import _rms_energy_check
