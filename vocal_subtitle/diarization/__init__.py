"""Stage 3.5 & 4.5: 说话人分离与角色标注模块

提供基于音色特征的说话人聚类和基于 LLM 的说话人角色标注。

组件:
- SpeakerDiarizer: 音色特征提取 + 凝聚聚类，将语音片段按说话人分组
- RoleLabeler: 利用 LLM 从对话上下文中挖掘说话人名字和角色
- SpeakerEmbeddingEngine: 说话人嵌入引擎抽象基类
- PyannoteEmbeddingEngine: pyannote.audio ECAPA-TDNN 实现
- ClusteredSegment: 带说话人编号的语音片段数据结构
- SpeakerRole: 说话人角色标注结果
"""

from .base import ClusteredSegment, DiarizationEngine, SpeakerRole
from .role_labeler import RoleLabeler
from .speaker_clusterer import SpeakerDiarizer
from .speaker_embedding import (
    DummyEmbeddingEngine,
    PyannoteEmbeddingEngine,
    SpeakerEmbeddingEngine,
    create_embedding_engine,
)

__all__ = [
    "ClusteredSegment",
    "DiarizationEngine",
    "DummyEmbeddingEngine",
    "PyannoteEmbeddingEngine",
    "RoleLabeler",
    "SpeakerDiarizer",
    "SpeakerEmbeddingEngine",
    "SpeakerRole",
    "create_embedding_engine",
]
