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
