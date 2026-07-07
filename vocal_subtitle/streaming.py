"""流式处理架构 (文档 5.12.5)

为 Pipeline 提供离线/流式双模式支持。

流式模式下自动降级不可用模块：
- ❌ 方案〇 宏观切块（无全局视角）
- ⚠️ 方案一 ffmpeg VAD（窗口内运行）
- ❌ 方案二 三方法融合（简化，仅 Silero VAD）
- ✅ 方案三 段内预切分（窗口内）
- ✅ 方案四 边界精修（窗口内）
- ⚠️ 方案五 LLM 合并（降级：仅本地 NLP，无 Batch LLM）
- ✅ 方案六 帧无缝衔接（窗口内）
- ❌ 方案七 声学标尺（无全局标尺）

核心组件:
- PipelineMode: 运行模式配置
- StreamingBuffer: 滑动窗口缓冲区（含重叠）
- StreamingMergeEngine: 流式合并引擎（本地 NLP + 规则）
- resolve_streaming_modules(): 流式降级模块映射
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------


@dataclass
class PipelineMode:
    """Pipeline 运行模式"""

    mode: str = "offline"                     # "offline" | "streaming"

    # 流式模式参数
    streaming_chunk_duration: float = 2.0     # 每次处理的音频窗口（秒）
    streaming_overlap_duration: float = 0.5   # 窗口重叠（秒）
    streaming_max_latency: float = 3.0        # 最大允许延迟（秒）

    def is_streaming(self) -> bool:
        return self.mode == "streaming"


def resolve_streaming_modules() -> Dict[str, bool]:
    """根据流式模式决定启用哪些模块

    流式模式下自动降级依赖全局视角的模块。

    Returns:
        dict: 模块名 → 是否启用
    """
    return {
        "macro_chunk": False,            # ❌ 无全局视角
        "ffmpeg_vad": True,              # ✅ 可在窗口内运行
        "boundary_fusion": False,         # ❌ 简化（仅 Silero VAD）
        "pre_split": True,               # ✅ 窗口内预切分
        "asr_refine": True,              # ✅ 窗口内精修
        "llm_merge": "local_only",       # ⚠️ 降级：仅本地 NLP
        "frame_seamless": True,          # ✅ 窗口内无缝衔接
        "acoustic_validation": False,     # ❌ 无全局标尺
        "diarization": True,             # ✅ 窗口内说话人分离
        "speaker_role": "local_only",    # ⚠️ 仅本地规则推断
        "llm_optimize": False,           # ❌ 不可用
    }


# ------------------------------------------------------------------
# 滑动窗口缓冲区
# ------------------------------------------------------------------


class StreamingBuffer:
    """滑动窗口音频缓冲区

    维护一个固定长度 + 重叠的滑动窗口，供流式 Pipeline 使用。

    使用示例:
        buffer = StreamingBuffer(chunk_duration=2.0, overlap_duration=0.5)
        for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            if buffer.ready():
                window = buffer.get_window()
                # ... 处理窗口 ...
                yield events
                buffer.advance()
    """

    CHUNK_SAMPLES = 16000  # 固定 16kHz 单声道

    def __init__(
        self,
        chunk_duration: float = 2.0,
        overlap_duration: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate

        self._chunk_samples = int(chunk_duration * sample_rate)
        self._overlap_samples = int(overlap_duration * sample_rate)
        self._advance_samples = self._chunk_samples - self._overlap_samples

        self._buffer = np.array([], dtype=np.float32)
        self._total_processed = 0  # 已输出的采样点数（全局偏移）

    def append(self, audio: np.ndarray) -> None:
        """追加音频数据到缓冲区

        Args:
            audio: 新到达的音频帧 (float32)
        """
        self._buffer = np.concatenate([self._buffer, audio])

    def ready(self) -> bool:
        """缓冲区是否有足够的样本形成一个处理窗口"""
        return len(self._buffer) >= self._chunk_samples

    def get_window(self) -> np.ndarray:
        """获取当前处理窗口（不消费缓冲区）

        Returns:
            当前窗口的音频数组
        """
        window_end = min(self._chunk_samples, len(self._buffer))
        return self._buffer[:window_end].copy()

    def advance(self) -> None:
        """消费缓冲区的前 advance_samples 个样本"""
        consume = min(self._advance_samples, len(self._buffer))
        self._buffer = self._buffer[consume:]
        self._total_processed += consume

    def global_offset(self) -> float:
        """当前窗口起始的全局时间偏移（秒）"""
        return self._total_processed / self.sample_rate

    def remaining(self) -> int:
        """缓冲区剩余样本数"""
        return len(self._buffer)

    def flush(self) -> Optional[np.ndarray]:
        """处理剩余不足一个窗口的尾部数据

        Returns:
            尾部窗口，若无剩余返回 None
        """
        if len(self._buffer) < 160:  # < 10ms, 忽略
            return None
        tail = self._buffer.copy()
        self._buffer = np.array([], dtype=np.float32)
        return tail


# ------------------------------------------------------------------
# 流式合并引擎（替代离线方案五的 Batch LLM）
# ------------------------------------------------------------------


class StreamingMergeEngine:
    """流式合并引擎（文档 5.12.5）

    替代离线方案五的 Batch LLM 调用。
    使用滑动窗口 + 本地 NLP 模型 + 简单规则，
    每帧决策必须在 <50ms 内完成。

    使用示例:
        engine = StreamingMergeEngine()
        should_merge = engine.decide_merge_streaming(current, previous)
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.similarity_model = self._load_local_model(model_name)
        self.sentence_buffer: List = []  # 保留最近 N 条未决字幕

    def decide_merge_streaming(
        self,
        current: Dict,
        previous: Dict,
    ) -> bool:
        """实时合并决策（必须 <50ms 内完成）

        Args:
            current: 当前片段 {"start", "end", "text", "speaker"}
            previous: 前一片段

        Returns:
            True=合并, False=不合并
        """
        gap = current.get("start", 0) - previous.get("end", 0)

        # ---- 快速路径 (<5ms) ----
        # 极短间隙 → 一定合并
        if gap < 0.12:
            return True

        # 长间隙 → 一定不合并
        if gap > 0.80:
            return False

        # 说话人不同 → 不合并
        if (
            current.get("speaker") != previous.get("speaker")
            and current.get("speaker") != "unknown"
            and previous.get("speaker") != "unknown"
        ):
            return False

        # ---- 标点规则 (<1ms) ----
        prev_text = previous.get("text", "").rstrip()
        if prev_text and prev_text[-1] in {",", "，", ";", "；"}:
            return True  # 标点未完 → 合并
        if prev_text and prev_text[-1] in {".", "!", "?", "。", "！", "？"}:
            return False  # 句尾标点 → 不合并

        # ---- 本地语义模型 (~30ms) ----
        if 0.15 <= gap <= 0.60:
            curr_text = current.get("text", "")
            similarity = self._compute_similarity(prev_text, curr_text)
            # 语义高度相似 → 合并；不相似 → 不合并
            return similarity > 0.55

        # 默认：短间隙合并
        return gap < 0.30

    def buffer_decision(
        self,
        fragment: Dict,
    ) -> Optional[List[Dict]]:
        """缓冲决策模式：维护最近 3 条未决字幕

        当缓冲区满时（>=3条），对最早的两条做最终决策并输出。

        Args:
            fragment: 新到达的片段

        Returns:
            None（仍在缓冲）或已决策的字幕列表
        """
        self.sentence_buffer.append(fragment)

        if len(self.sentence_buffer) >= 3:
            # 对前两条做最终决策
            a = self.sentence_buffer[0]
            b = self.sentence_buffer[1]

            if self.decide_merge_streaming(b, a):
                # 合并 a + b
                merged = {
                    "start": a["start"],
                    "end": b["end"],
                    "text": f"{a.get('text', '')} {b.get('text', '')}".strip(),
                    "speaker": a.get("speaker", "unknown"),
                }
                self.sentence_buffer = [merged] + self.sentence_buffer[2:]
                return None  # 继续缓冲
            else:
                # 输出 a
                self.sentence_buffer = self.sentence_buffer[1:]
                return [a]

        return None

    def flush_buffer(self) -> List[Dict]:
        """输出缓冲区中所有剩余字幕"""
        result = list(self.sentence_buffer)
        self.sentence_buffer = []
        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _load_local_model(model_name: str):
        """加载轻量本地语义模型（单次加载，全局复用）

        委托给统一的 model_loader 工具，确保离线优先策略：
        1. 本地缓存 → 即时返回（零网络）
        2. 本地无缓存 → 限时下载 + 镜像站回退
        3. 全部失败 → 返回 None（优雅降级到规则模式）
        """
        from .utils.model_loader import load_sentence_transformer
        return load_sentence_transformer(model_name)

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的语义相似度

        Returns:
            0.0 ~ 1.0, 模型不可用时返回 0.5（中性值）
        """
        if self.similarity_model is None:
            return 0.5  # 无模型时返回中性值

        if not text_a or not text_b:
            return 0.5

        try:
            embeddings = self.similarity_model.encode(
                [text_a, text_b], convert_to_numpy=True,
            )
            dot = float(np.dot(embeddings[0], embeddings[1]))
            norm_a = float(np.linalg.norm(embeddings[0]))
            norm_b = float(np.linalg.norm(embeddings[1]))
            return dot / max(norm_a * norm_b, 1e-8)
        except Exception as e:
            logger.debug("Similarity computation failed: %s", e)
            return 0.5
