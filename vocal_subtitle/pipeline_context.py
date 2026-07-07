"""Pipeline 统一数据流上下文 (文档 5.1.2)

定义 Pipeline 全生命周期中的统一中间数据结构。
所有方案通过此结构交换数据，避免模块间隐式耦合。

使用示例:
    ctx = PipelineContext(audio_path=audio_path, audio=audio, sample_rate=16000)
    # 方案〇
    ctx.macro_chunks = chunker.split(...)
    # 方案七
    ctx.acoustic_skeleton = build_skeleton(...)
    validator.validate(ctx.seamless_events, ctx.acoustic_skeleton)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# 子结构
# ------------------------------------------------------------------


@dataclass
class NoiseProfile:
    """环境底噪档案 (3.8)"""

    noise_rms: float
    speech_threshold: float
    is_noisy_environment: bool


@dataclass
class ASRFragment:
    """ASR 识别片段（方案三预切分后的最小单元）

    包含精确时间戳、文本、词级时间戳和合并标记。
    """

    index: int
    start: float
    end: float
    text: str
    words: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    speaker_label: Optional[str] = None
    gap_to_next: Optional[float] = None
    gap_is_silent: Optional[bool] = None
    # 方案四精修标记
    start_refined: bool = False
    end_refined: bool = False
    # 方案五合并标记
    merge_decision: Optional[str] = None  # "fast" | "llm" | "hard_split"

    @property
    def duration(self) -> float:
        return self.end - self.start


# ------------------------------------------------------------------
# PipelineContext
# ------------------------------------------------------------------


@dataclass
class PipelineContext:
    """Pipeline 全局上下文 —— 所有方案通过此结构交换数据

    每个方案只读取已由上游方案写入的字段，写入自己的产出字段。
    """

    # === 输入 ===
    audio_path: Path
    audio: np.ndarray
    sample_rate: int = 16000

    # === 方案〇产出 ===
    macro_chunks: List[Any] = field(default_factory=list)

    # === 方案一+二产出 ===
    silero_segments: List[Any] = field(default_factory=list)
    ffmpeg_segments: List[Any] = field(default_factory=list)
    fused_segments: List[Any] = field(default_factory=list)

    # === 方案三产出 ===
    pre_split_segments: List[Any] = field(default_factory=list)
    noise_profile: Optional[NoiseProfile] = None

    # === ASR 产出 ===
    asr_fragments: List[ASRFragment] = field(default_factory=list)

    # === 方案四产出 ===
    refined_fragments: List[ASRFragment] = field(default_factory=list)

    # === 方案五产出 ===
    merged_events: List[Any] = field(default_factory=list)

    # === 方案六产出 ===
    seamless_events: List[Any] = field(default_factory=list)

    # === 方案七产出 ===
    acoustic_skeleton: List[Tuple[float, float]] = field(default_factory=list)
    validation_report: dict = field(default_factory=dict)

    # === 元信息 ===
    ffmpeg_unified_result: Optional[dict] = None
    diagnostics: List[str] = field(default_factory=list)

    # ==================================================================
    # 便捷方法
    # ==================================================================

    def add_diagnostic(self, message: str) -> None:
        """添加诊断消息"""
        self.diagnostics.append(message)

    def has_ffmpeg_skeleton(self) -> bool:
        """是否已构建 ffmpeg 声学骨架"""
        return len(self.acoustic_skeleton) > 0

    def is_multi_chunk(self) -> bool:
        """是否启用了宏观切块"""
        return len(self.macro_chunks) > 1

    def total_duration(self) -> float:
        """音频总时长（秒）"""
        return len(self.audio) / max(self.sample_rate, 1)

    def get_health_score(self) -> Optional[float]:
        """获取声学校验健康度评分"""
        return self.validation_report.get("health_score")
