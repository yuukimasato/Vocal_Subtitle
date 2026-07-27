"""说话人分离与角色标注 — 数据结构与抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ClusteredSegment:
    """带说话人编号的语音片段

    独立于 SpeechSegment 以避免破坏现有 VAD/merging 接口契约。
    """

    start: float
    end: float
    confidence: float = 1.0
    speaker_id: int = -1  # -1 = 未分配
    audio: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerTurn:
    """单个说话人轮次

    Attributes:
        start: 起始时间（秒）
        end: 结束时间（秒）
        speaker_id: 说话人编号 (0-indexed)
        speaker: 说话人标签 (可选，如 "SPEAKER_00")
        confidence: 置信度 (0-1)
        overlapped: 是否与其他说话人重叠
    """

    start: float
    end: float
    speaker_id: int
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    overlapped: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """说话人分离结果

    Attributes:
        turns: 所有 speaker turns
        exclusive_turns: 无重叠的 speaker turns
        speaker_count: 识别的说话人数量
        backend: 使用的后端名称
        status: 处理状态 ("ok", "degraded", "failed")
        diagnostics: 诊断信息
    """

    turns: List[SpeakerTurn] = field(default_factory=list)
    exclusive_turns: List[SpeakerTurn] = field(default_factory=list)
    speaker_count: int = 0
    backend: str = "unknown"
    status: str = "unknown"
    diagnostics: dict = field(default_factory=dict)
    overlap_duration: float = 0.0


@dataclass
class SpeakerRole:
    """说话人角色标注结果

    三级标注策略:
    - identity: 从上下文挖掘到名字 + 推断角色 → "张三(嘉宾)"
    - role:      仅推断到角色类型 → "主持人"
    - fallback:  无法识别 → "说话人A"
    """

    speaker_id: int
    name: Optional[str] = None       # 从上下文挖掘的名字: "张三"
    role: Optional[str] = None       # 推断的角色类型: "嘉宾"
    label: str = ""                  # 最终显示标签: "张三(嘉宾)"
    confidence: str = "fallback"     # "identity" | "role" | "fallback"


@dataclass
class AtomicSpeechSpan:
    """原子语音跨度 — 物理边界与 diarization turns 融合后的最小单元

    Attributes:
        start: 起始时间（秒）
        end: 结束时间（秒）
        speaker_id: 说话人编号 (None 表示 UNKNOWN)
        physical_source: 物理来源标识
        speaker_source: 说话人来源标识
        overlapped: 是否重叠
    """

    start: float
    end: float
    speaker_id: Optional[int] = None
    physical_source: str = ""
    speaker_source: str = "unknown"
    overlapped: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiarizationEngine(ABC):
    """说话人分离引擎抽象基类

    遵循 SeparationEngine / VADEngine / ASREngine 的惯例:
    - load_model() 延迟加载
    - name 属性标识引擎类型
    """

    @abstractmethod
    def diarize(
        self,
        segments: List,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[int]:
        """为每个语音片段分配 speaker_id

        Args:
            segments: SpeechSegment 列表
            audio: 完整音频数据 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            与 segments 等长的 speaker_id 列表 (int: 0, 1, 2, ...)
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """加载模型（延迟加载）"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
