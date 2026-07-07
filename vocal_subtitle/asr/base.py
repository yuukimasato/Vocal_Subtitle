"""ASR 引擎抽象基类与数据结构"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class WordTimestamp:
    """词级时间戳"""

    word: str
    start: float
    end: float
    confidence: float = 1.0

    def __repr__(self) -> str:
        return (
            f"WordTimestamp(word='{self.word}', "
            f"start={self.start:.3f}, end={self.end:.3f})"
        )


@dataclass
class TranscriptionSegment:
    """转录结果段

    Attributes:
        text: 转录文本
        start: 段内起始时间（秒）
        end: 段内结束时间（秒）
        words: 词级时间戳列表
        avg_logprob: 平均对数概率
    """

    text: str
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)
    avg_logprob: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"TranscriptionSegment(text='{self.text[:30]}...', "
            f"start={self.start:.3f}, end={self.end:.3f})"
        )


class ASREngine(ABC):
    """ASR 引擎抽象基类

    所有语音识别引擎必须实现此接口。

    使用示例:
        engine = FasterWhisperEngine(model="large-v3", device="cuda")
        engine.load_model()
        result = engine.transcribe(audio_array, language="zh")
    """

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        """识别音频片段

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率
            language: 语言代码 (zh/en/ja/...)，None 表示自动检测

        Returns:
            转录结果列表，每项包含 text, start, end, words(可选)
        """
        ...

    def detect_language(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[str]:
        """检测音频语言（默认实现：返回 None，即不检测）

        子类可覆盖此方法，利用底层模型的音频编码器做语言检测，
        避免对极短片段做不可靠的逐段自动检测。

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            检测到的语言代码 (zh/en/ja/...) 或 None（未知/不支持）
        """
        return None

    @abstractmethod
    def load_model(self) -> None:
        """加载 ASR 模型（延迟加载）"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前模型名称"""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"model={self.model_name})"
        )
