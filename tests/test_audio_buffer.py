"""只读音频缓冲契约(2026-09-15 重构计划 Task 7)。"""

import numpy as np
import pytest

from vocal_subtitle.utils.audio_buffer import (
    audio_buffer_from_array,
)


def test_duration_and_slice_view():
    data = np.arange(100, dtype=np.float32)
    buffer = audio_buffer_from_array(data, sample_rate=100)

    assert buffer.duration_seconds == 1.0
    view = buffer.slice_seconds(0.1, 0.3)
    assert view.shape == (20,)
    assert view.base is not None or not view.flags.writeable


def test_slice_is_read_only():
    buffer = audio_buffer_from_array(np.ones(50, dtype=np.float32), 100)

    view = buffer.slice_seconds(0.0, 0.5)

    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = 0.0


def test_slice_clamps_out_of_range():
    buffer = audio_buffer_from_array(np.ones(50, dtype=np.float32), 100)

    assert buffer.slice_seconds(-1.0, 0.2).shape == (20,)
    assert buffer.slice_seconds(0.4, 9.9).shape == (10,)
    assert buffer.slice_seconds(0.8, 0.2).shape == (0,)


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        audio_buffer_from_array(np.ones(10), sample_rate=0)
    with pytest.raises(ValueError):
        audio_buffer_from_array(np.ones((2, 10)), sample_rate=16)


def test_derived_buffer_keeps_immutability():
    buffer = audio_buffer_from_array(np.zeros(10, dtype=np.float32), 10)
    derived = buffer.with_sample_rate(20, np.ones(20, dtype=np.float32))

    assert derived.sample_rate == 20
    assert derived.duration_seconds == 1.0
    with pytest.raises(ValueError):
        buffer.with_sample_rate(-1, np.ones(4))
