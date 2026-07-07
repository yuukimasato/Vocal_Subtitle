"""VAD 引擎抽象基类与数据结构"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class SpeechSegment:
    """语音片段

    Attributes:
        start: 起始时间（秒）
        end: 结束时间（秒）
        confidence: 置信度 (0.0–1.0)
        audio: 对应的音频数据（可选，延迟加载）
    """

    start: float
    end: float
    confidence: float = 1.0
    audio: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"SpeechSegment(start={self.start:.3f}, end={self.end:.3f}, "
            f"duration={self.duration:.3f}s, confidence={self.confidence:.2f})"
        )


class VADEngine(ABC):
    """VAD 引擎抽象基类

    所有语音活动检测引擎必须实现此接口。

    使用示例:
        engine = SileroVAD()
        engine.load_model()
        segments = engine.detect(audio_path, threshold=0.5)
    """

    @abstractmethod
    def detect(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """检测语音区间

        Args:
            audio_path: 音频文件路径
            threshold: 语音概率阈值 (0.0–1.0)
            min_speech_duration_ms: 最小语音时长 (ms)
            min_silence_duration_ms: 最小静音时长 (ms)

        Returns:
            语音片段列表
        """
        ...

    @abstractmethod
    def detect_on_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """在 numpy 数组上检测（避免重复 I/O）

        Args:
            audio: 音频数据，float32 归一化到 [-1, 1]
            sample_rate: 采样率
            threshold: 语音概率阈值
            min_speech_duration_ms: 最小语音时长 (ms)
            min_silence_duration_ms: 最小静音时长 (ms)

        Returns:
            语音片段列表
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """加载 VAD 模型（延迟加载）"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
