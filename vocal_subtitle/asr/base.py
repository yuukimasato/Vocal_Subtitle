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
    speaker_id: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"WordTimestamp(word='{self.word}', "
            f"start={self.start:.3f}, end={self.end:.3f})"
        )


@dataclass
class LanguageDetection:
    """语言检测结果

    Attributes:
        language: 语言代码 (zh/en/ja/...)
        probability: 检测置信度 (0-1)
        source: 检测来源 (模型名称或 "heuristic")
    """

    language: str
    probability: float
    source: str

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "probability": self.probability,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LanguageDetection":
        return cls(
            language=payload["language"],
            probability=payload["probability"],
            source=payload.get("source", "unknown"),
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
        language: 检测到的语言代码
        language_probability: 语言检测置信度
        no_speech_prob: 无语音概率
        compression_ratio: 压缩比
        speaker_id: 说话人 ID（可选）
    """

    text: str
    start: float
    end: float
    words: List[WordTimestamp] = field(default_factory=list)
    avg_logprob: float = 0.0
    language: Optional[str] = None
    language_probability: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    speaker_id: Optional[int] = None

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
