"""宏观静音切块 (方案〇)

在长音频中检测 >2s 的静音区间，在这些位置将音频切分为
独立的大块。每个大块作为一个独立的 Pipeline 运行。

核心功能:
1. ffmpeg silencedetect 检测长静音
2. 带重叠的切分（±200ms 回卷）
3. 重叠区能量最低点缝合
4. 递归切分（对仍然太长的块用更敏感的阈值）

适用场景:
- < 3 分钟: 不切分（直接跳过）
- 3-10 分钟: 可选
- 10-30 分钟: 推荐
- > 30 分钟: 必须
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MacroChunkConfig:
    """宏观切块配置"""

    enabled: bool = True
    auto_enable_threshold: float = 180.0  # 音频 > 3分钟自动启用
    silence_threshold_db: float = -30
    min_silence_duration: float = 2.0
    target_chunk_duration: float = 60.0
    max_chunk_duration: float = 180.0
    overlap_ms: int = 200

    # 递归切分
    recursive: bool = True
    recursive_thresholds: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-30, 3.0),   # 第一轮
        (-35, 1.5),   # 第二轮
        (-40, 0.8),   # 第三轮
    ])


@dataclass
class MacroChunk:
    """宏观切块"""

    index: int
    start: float
    end: float
    audio: Optional[np.ndarray] = None  # 延迟加载
    overlap_with_prev: bool = False
    overlap_with_next: bool = False

    @property
    def duration(self) -> float:
        """块时长（秒）"""
        return self.end - self.start


class MacroChunker:
    """宏观静音切块器

    使用示例:
        chunker = MacroChunker(MacroChunkConfig())
        chunks = chunker.split(audio_path, audio, sample_rate)
    """

    def __init__(self, config: Optional[MacroChunkConfig] = None):
        self.config = config or MacroChunkConfig()

    def should_split(
        self, total_duration: float,
    ) -> bool:
        """判断是否需要进行宏观切块"""
        if not self.config.enabled:
            return False
        return total_duration > self.config.auto_enable_threshold

    def split(
        self,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[MacroChunk]:
        """执行宏观切块

        Returns:
            MacroChunk 列表（按时间排序）
        """
        cfg = self.config
        total_duration = len(audio) / sample_rate

        if not self.should_split(total_duration):
            # 短音频：作为单个大块
            logger.info(
                "Audio duration %.1fs < threshold %.0fs, skipping macro chunking",
                total_duration, cfg.auto_enable_threshold,
            )
            return [MacroChunk(
                index=0, start=0.0, end=total_duration,
                audio=audio, overlap_with_prev=False, overlap_with_next=False,
            )]

        # Step 1: 检测长静音
        silence_intervals = self._detect_long_silences(audio_path)

        if not silence_intervals:
            logger.info("No long silences found, treating as single chunk")
            return [MacroChunk(
                index=0, start=0.0, end=total_duration,
                audio=audio, overlap_with_prev=False, overlap_with_next=False,
            )]

        # Step 2: 带重叠切分
        chunks = self._split_with_overlap(
            silence_intervals, total_duration, audio, sample_rate,
        )

        # Step 3: 递归切分仍然太长的块
        if cfg.recursive:
            chunks = self._recursive_split(
                chunks, audio_path, audio, sample_rate,
            )

        logger.info(
            "Macro chunking: %.1fs → %d chunks (avg %.1fs)",
            total_duration, len(chunks),
            sum(c.duration for c in chunks) / max(len(chunks), 1),
        )
        return chunks

    def stitch_chunks(
        self,
        chunk_a_events: List,
        chunk_b_events: List,
        overlap_region: Tuple[float, float],
        audio: np.ndarray,
        sample_rate: int,
    ) -> List:
        """缝合两个有重叠的大块的字幕结果

        在重叠区内找 RMS 能量最低点作为最终缝合线。
        """
        overlap_start, overlap_end = overlap_region

        # 在重叠区内以 10ms 步长扫描，找能量最低点
        frame_size = int(0.01 * sample_rate)
        hop = frame_size // 2
        start_sample = int(overlap_start * sample_rate)
        end_sample = int(overlap_end * sample_rate)

        min_rms = float("inf")
        stitch_point = (overlap_start + overlap_end) / 2  # 默认中点

        for i in range(start_sample, end_sample - frame_size + 1, hop):
            frame = audio[i: i + frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < min_rms:
                min_rms = rms
                stitch_point = i / sample_rate

        # 缝合：stitch_point 之前归块A，之后归块B
        stitched = []
        for event in chunk_a_events:
            if event.end <= stitch_point:
                stitched.append(event)
            elif event.start < stitch_point:
                event.end = stitch_point
                stitched.append(event)

        for event in chunk_b_events:
            if event.start >= stitch_point:
                stitched.append(event)
            elif event.end > stitch_point:
                event.start = stitch_point
                stitched.append(event)

        return stitched

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _detect_long_silences(
        self, audio_path: Path,
    ) -> List[Tuple[float, float]]:
        """使用 ffmpeg silencedetect 检测长静音"""
        from .vad.ffmpeg_vad import FFmpegSilenceVAD

        cfg = self.config
        return FFmpegSilenceVAD._detect_silence(
            audio_path,
            noise_db=cfg.silence_threshold_db,
            min_silence_duration=cfg.min_silence_duration,
        )

    def _split_with_overlap(
        self,
        silence_intervals: List[Tuple[float, float]],
        total_duration: float,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[MacroChunk]:
        """带重叠的宏观切块

        每个切分点前后各保留 overlap_ms 重叠区。
        """
        cfg = self.config
        overlap_sec = cfg.overlap_ms / 1000.0

        chunks = []
        prev_end = 0.0

        for silence_start, silence_end in silence_intervals:
            # 切分点选在静音中点
            split_point = (silence_start + silence_end) / 2.0

            # 滤除开头的静音（split_point 太靠前）
            if split_point < 1.0:
                prev_end = max(prev_end, silence_end)
                continue

            # 回卷：块的结束点 = 切分点 + overlap
            chunk_end = min(split_point + overlap_sec, total_duration)
            chunk_start = prev_end

            # 提取音频
            start_sample = int(chunk_start * sample_rate)
            end_sample = int(chunk_end * sample_rate)
            chunk_audio = audio[start_sample:end_sample].copy()

            chunks.append(MacroChunk(
                index=len(chunks),
                start=chunk_start,
                end=chunk_end,
                audio=chunk_audio,
                overlap_with_prev=len(chunks) > 0,
                overlap_with_next=True,
            ))

            # 下一块的起始点 = 切分点 - overlap（回卷）
            prev_end = max(0.0, split_point - overlap_sec)

        # 最后一块（到音频末尾）
        if prev_end < total_duration - 0.5:  # 至少 0.5s 剩余
            start_sample = int(prev_end * sample_rate)
            chunk_audio = audio[start_sample:].copy()
            chunks.append(MacroChunk(
                index=len(chunks),
                start=prev_end,
                end=total_duration,
                audio=chunk_audio,
                overlap_with_prev=len(chunks) > 0,
                overlap_with_next=False,
            ))

        # 修正重叠标记
        if chunks:
            chunks[0].overlap_with_prev = False
            chunks[-1].overlap_with_next = False

        return chunks

    def _recursive_split(
        self,
        chunks: List[MacroChunk],
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        depth: int = 0,
    ) -> List[MacroChunk]:
        """递归切分仍然太长的块"""
        cfg = self.config

        if depth >= len(cfg.recursive_thresholds):
            return chunks

        threshold_db, min_silence = cfg.recursive_thresholds[depth]

        result = []
        for chunk in chunks:
            duration = chunk.end - chunk.start
            if duration <= cfg.max_chunk_duration:
                result.append(chunk)
                continue

            # 对太长的块，用更敏感的阈值找静音
            from .vad.ffmpeg_vad import FFmpegSilenceVAD

            # 需要提取该块的音频到临时文件
            import tempfile
            from .utils.audio_utils import AudioUtils

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)

            try:
                chunk_audio = audio[
                    int(chunk.start * sample_rate):
                    int(chunk.end * sample_rate)
                ]
                AudioUtils.save_audio(chunk_audio, tmp_path, sample_rate)

                sub_silences = FFmpegSilenceVAD._detect_silence(
                    tmp_path,
                    noise_db=threshold_db,
                    min_silence_duration=min_silence,
                )

                if not sub_silences:
                    result.append(chunk)
                    continue

                # 偏移回全局时间
                for silence_start, silence_end in sub_silences:
                    split_point = chunk.start + (silence_start + silence_end) / 2.0

                    # 递归创建子块
                    sub_chunks = self._split_with_overlap(
                        [(silence_start, silence_end)],
                        duration,
                        chunk_audio,
                        sample_rate,
                    )
                    for sc in sub_chunks:
                        sc.start += chunk.start
                        sc.end += chunk.start
                        sc.index = len(result) + len(sub_chunks)
                        result.append(sc)

            finally:
                tmp_path.unlink(missing_ok=True)

        logger.info(
            "Recursive split (depth=%d, db=%.0f, min_s=%.1f): %d → %d chunks",
            depth, threshold_db, min_silence, len(chunks), len(result),
        )
        return result
