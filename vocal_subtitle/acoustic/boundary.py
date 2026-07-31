"""Boundary candidates and auditable endpoint decisions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def find_boundary_in_skeleton(t: float, skeleton: List[Tuple[float, float]]) -> Tuple[bool, float]:
    for start, end in skeleton:
        if start <= t <= end:
            return True, t
        if t < start:
            return False, start
    return False, skeleton[-1][1] if skeleton else t


def find_directional_boundary(t: float, skeleton: List[Tuple[float, float]], boundary_type: str) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]]]:
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


def boundary_confidence(event: object, boundary_type: str) -> Optional[float]:
    words = list(getattr(event, "words", []) or [])
    if words:
        value = getattr(words[0] if boundary_type == "start" else words[-1], "confidence", None)
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


def preserve_reliable_asr_boundary(event: object, boundary_type: str, config: object) -> bool:
    words = list(getattr(event, "words", []) or [])
    confidence = boundary_confidence(event, boundary_type)
    return bool(words and confidence is not None and confidence >= config.confidence_threshold)


def record_boundary_diagnostic(report: Dict, event: object, boundary_type: str, action: str, *, reason: str, original_time: float, candidate_time: Optional[float], distance: Optional[float]) -> None:
    report.setdefault("boundary_diagnostics", []).append({
        "id": getattr(event, "index", 0),
        "boundary": boundary_type,
        "action": action,
        "reason": reason,
        "original_time": round(float(original_time), 6),
        "candidate_time": round(float(candidate_time), 6) if candidate_time is not None else None,
        "distance_ms": round(float(distance) * 1000, 3) if distance is not None else None,
    })


_find_boundary_in_skeleton = find_boundary_in_skeleton
_find_directional_boundary = find_directional_boundary
_boundary_confidence = boundary_confidence
_preserve_reliable_asr_boundary = preserve_reliable_asr_boundary
_record_boundary_diagnostic = record_boundary_diagnostic
