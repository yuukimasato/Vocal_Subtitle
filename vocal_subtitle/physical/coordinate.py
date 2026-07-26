"""Explicit local/global coordinate conversion helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _span(start: Any, end: Any, name: str) -> Tuple[float, float]:
    first = _number(start, f"{name}_start")
    last = _number(end, f"{name}_end")
    if first < 0 or last <= first:
        raise ValueError(f"{name} must satisfy 0 <= start < end")
    return first, last


@dataclass(frozen=True)
class CoordinateRange:
    """A range carrying its coordinate space to prevent double conversion."""

    start: float
    end: float
    coordinate_space: str

    def __post_init__(self) -> None:
        start, end = _span(self.start, self.end, "range")
        if self.coordinate_space not in {"local", "global"}:
            raise ValueError("coordinate_space must be 'local' or 'global'")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class MacroChunkCoordinate:
    """Read-only local/global view of a macro chunk."""

    source_id: str
    index: int
    local_start: float
    local_end: float
    global_start: float
    global_end: float
    overlap_with_prev: bool = False
    overlap_with_next: bool = False


@dataclass(frozen=True)
class CoordinateMapper:
    """Map one explicitly local window onto the absolute audio timeline."""

    origin_offset: float
    duration: Optional[float]
    source_id: str

    def __post_init__(self) -> None:
        offset = _number(self.origin_offset, "origin_offset")
        if offset < 0:
            raise ValueError("origin_offset must be non-negative")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if self.duration is not None:
            duration = _number(self.duration, "duration")
            if duration <= 0:
                raise ValueError("duration must be greater than zero")
            object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "origin_offset", offset)

    def to_global(self, local_start: Any, local_end: Any = None) -> Tuple[float, float]:
        if isinstance(local_start, CoordinateRange):
            if local_end is not None and local_start is not local_end:
                raise ValueError("coordinate range arguments must be one matching object")
            if local_start.coordinate_space != "local":
                raise ValueError("to_global received an already global range")
            local_start, local_end = local_start.start, local_start.end
        elif isinstance(local_end, CoordinateRange):
            raise ValueError("coordinate range arguments must be one matching object")
        elif local_end is None:
            raise ValueError("to_global requires a local start and end")
        start, end = _span(local_start, local_end, "local range")
        global_start = start + self.origin_offset
        global_end = end + self.origin_offset
        if self.duration is not None and global_end > self.duration:
            raise ValueError("global range exceeds duration")
        return global_start, global_end

    def to_local(self, global_start: Any, global_end: Any = None) -> Tuple[float, float]:
        if isinstance(global_start, CoordinateRange):
            if global_end is not None and global_start is not global_end:
                raise ValueError("coordinate range arguments must be one matching object")
            if global_start.coordinate_space != "global":
                raise ValueError("to_local received an already local range")
            global_start, global_end = global_start.start, global_start.end
        elif isinstance(global_end, CoordinateRange):
            raise ValueError("coordinate range arguments must be one matching object")
        elif global_end is None:
            raise ValueError("to_local requires a global start and end")
        start, end = _span(global_start, global_end, "global range")
        if self.duration is not None and end > self.duration:
            raise ValueError("global range exceeds duration")
        local_start = start - self.origin_offset
        local_end = end - self.origin_offset
        if local_start < 0 or local_end <= local_start:
            raise ValueError("global range is before mapper origin")
        return local_start, local_end

    def clamp_global(self, start: Any, end: Any) -> Tuple[float, float]:
        first = _number(start, "global_start")
        last = _number(end, "global_end")
        if last <= first:
            raise ValueError("global range must satisfy start < end")
        if self.duration is not None:
            first = max(0.0, min(first, self.duration))
            last = max(0.0, min(last, self.duration))
        elif first < 0:
            raise ValueError("global range must not start before zero")
        if last <= first:
            raise ValueError("clamped global range is empty")
        return first, last

    def map_segment(self, segment: Any) -> MacroChunkCoordinate:
        """Return a new coordinate view without mutating ``segment``."""
        if getattr(segment, "coordinate_space", "local") != "local":
            raise ValueError("map_segment requires a local-coordinate segment")
        local_start, local_end = _span(segment.start, segment.end, "segment")
        global_start, global_end = self.to_global(local_start, local_end)
        return MacroChunkCoordinate(
            source_id=self.source_id,
            index=int(getattr(segment, "index", 0)),
            local_start=local_start,
            local_end=local_end,
            global_start=global_start,
            global_end=global_end,
            overlap_with_prev=bool(getattr(segment, "overlap_with_prev", False)),
            overlap_with_next=bool(getattr(segment, "overlap_with_next", False)),
        )
