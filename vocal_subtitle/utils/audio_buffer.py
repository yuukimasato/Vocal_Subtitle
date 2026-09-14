"""只读音频缓冲(2026-09-15 重构计划 Task 7)。

包装一次解码结果,跨阶段复用采样率与切片视图,避免重复加载;
切片为只读视图(带额外校验),不允许越过音频边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AudioBuffer:
    """一次解码、多阶段复用的只读音频缓冲。"""

    data: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.data) / self.sample_rate

    def slice_seconds(self, start: float, end: float) -> np.ndarray:
        """按秒切片,返回只读视图;越界会被钳制到有效范围。"""
        rate = self.sample_rate if self.sample_rate > 0 else 1
        start_sample = max(0, int(round(start * rate)))
        end_sample = min(len(self.data), max(start_sample, int(round(end * rate))))
        view = self.data[start_sample:end_sample]
        view.setflags(write=False)
        return view

    def with_sample_rate(self, sample_rate: int, data: np.ndarray) -> "AudioBuffer":
        """重采样结果的派生缓冲(显式传入新数据,不隐式转换)。"""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return AudioBuffer(data=np.asarray(data), sample_rate=int(sample_rate))


def audio_buffer_from_array(data: np.ndarray, sample_rate: int) -> AudioBuffer:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    array = np.asarray(data)
    if array.ndim != 1:
        raise ValueError("AudioBuffer expects a mono 1-D array")
    return AudioBuffer(data=array, sample_rate=int(sample_rate))


__all__ = ["AudioBuffer", "audio_buffer_from_array"]
