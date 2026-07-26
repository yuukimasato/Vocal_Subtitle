"""Context window construction for physical clips."""

from __future__ import annotations

import math
from typing import List

from .timeline import ContextWindow, PhysicalTimeline


def _context_value(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def build_context_windows(
    timeline: PhysicalTimeline,
    left_context: float,
    right_context: float,
    id_prefix: str = "ctx",
) -> List[ContextWindow]:
    """Build one independently owned context window per physical clip."""
    if not isinstance(timeline, PhysicalTimeline):
        raise ValueError("timeline must be a PhysicalTimeline")
    if not isinstance(id_prefix, str) or not id_prefix.strip():
        raise ValueError("id_prefix must be a non-empty string")
    left = _context_value(left_context, "left_context")
    right = _context_value(right_context, "right_context")
    errors = timeline.validate()
    if errors:
        raise ValueError("invalid physical timeline: " + "; ".join(errors))

    windows: List[ContextWindow] = []
    left_ms = round(left * 1000)
    right_ms = round(right * 1000)
    for clip in timeline.physical_clips:
        start = max(0.0, clip.start - left)
        end = min(timeline.duration, clip.end + right)
        if end <= start:
            raise ValueError(f"clip {clip.id} cannot produce a context window")
        window_id = f"{id_prefix}:{clip.id}:l{left_ms:04d}:r{right_ms:04d}"
        metadata = {}
        if start != clip.start - left or end != clip.end + right:
            metadata["clamped_to_duration"] = True
        windows.append(ContextWindow(
            id=window_id,
            start=start,
            end=end,
            physical_clip_id=clip.id,
            left_context=left,
            right_context=right,
            metadata=metadata,
        ))
    return sorted(windows, key=lambda item: (item.start, item.end, item.id))
