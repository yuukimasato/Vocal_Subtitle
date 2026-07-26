from dataclasses import dataclass

import pytest

from vocal_subtitle.physical.coordinate import (
    CoordinateMapper,
    CoordinateRange,
)


@dataclass
class LocalChunk:
    index: int
    start: float
    end: float
    overlap_with_prev: bool = False
    overlap_with_next: bool = False


def test_mapper_requires_explicit_local_global_conversion():
    mapper = CoordinateMapper(origin_offset=10.0, duration=30.0, source_id="chunk-1")
    assert mapper.to_global(2.0, 4.0) == (12.0, 14.0)
    assert mapper.to_local(12.0, 14.0) == (2.0, 4.0)
    assert mapper.clamp_global(-1.0, 31.0) == (0.0, 30.0)


def test_coordinate_range_prevents_double_conversion():
    mapper = CoordinateMapper(origin_offset=5.0, duration=20.0, source_id="window")
    local = CoordinateRange(1.0, 2.0, "local")
    global_range = CoordinateRange(6.0, 7.0, "global")

    assert mapper.to_global(local, local) == (6.0, 7.0)
    assert mapper.to_local(global_range, global_range) == (1.0, 2.0)
    with pytest.raises(ValueError):
        mapper.to_global(global_range, global_range)
    with pytest.raises(ValueError):
        mapper.to_local(local, local)


def test_map_segment_returns_read_only_view_and_preserves_overlap_flags():
    mapper = CoordinateMapper(origin_offset=10.0, duration=30.0, source_id="chunk-1")
    chunk = LocalChunk(2, 1.0, 4.0, overlap_with_prev=True, overlap_with_next=True)
    mapped = mapper.map_segment(chunk)

    assert mapped.local_start == 1.0
    assert mapped.global_start == 11.0
    assert mapped.overlap_with_prev is True
    assert mapped.overlap_with_next is True
    assert chunk.start == 1.0
    with pytest.raises((AttributeError, TypeError)):
        mapped.global_start = 12.0


def test_invalid_and_repeated_coordinate_ranges_are_rejected():
    with pytest.raises(ValueError):
        CoordinateMapper(origin_offset=-1.0, duration=10.0, source_id="x")
    mapper = CoordinateMapper(origin_offset=3.0, duration=10.0, source_id="x")
    with pytest.raises(ValueError):
        mapper.to_global(8.0, 9.0)
    with pytest.raises(ValueError):
        mapper.to_local(1.0, 2.0)
