"""滑动窗口冗余 ASR 识别

对低置信度边界，在偏移窗口中重新执行 ASR，捕获被边界截断的单词。

核心思路：
  原始分段: [---Seg N---][---Seg N+1---]
                          ^ 低置信边界

  窗口 A（左扩展）：Seg N + forward_overlap → 尾词完整
  窗口 B（右扩展）：Seg N+1 - backward_overlap → 首词完整
  窗口 C（融合窗）：边界前后各 1s → 跨边界完整时间戳

三个窗口的 ASR 结果汇总为词级共识格，供 LLM 仲裁使用。

与现有 pipeline 集成：
  - 复用 _get_asr_engine() 获取的 ASR 引擎
  - 复用现有的缓存机制（按 audio_hash + time_range 存储）
  - 窗口按 ThreadPoolExecutor 并行执行
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..asr.base import TranscriptionSegment, WordTimestamp
from ..vad.base import SpeechSegment
from ..utils.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


@dataclass
class SlidingWindow:
    """滑动窗口定义"""

    window_type: str         # "left_expand" | "right_expand" | "fusion"
    start_time: float        # 窗口起始时间（全局坐标）
    end_time: float          # 窗口结束时间（全局坐标）
    boundary_index: int      # 关联的边界索引


@dataclass
class WindowASRResult:
    """单个窗口的 ASR 结果"""

    window: SlidingWindow
    segments: List[TranscriptionSegment]  # ASR 转录结果
    success: bool = True
    error: Optional[str] = None


@dataclass
class BoundaryReASRResult:
    """单个边界的所有冗余 ASR 结果"""

    boundary_index: int
    original_left_text: str          # 原始 Seg N 文本
    original_right_text: str         # 原始 Seg N+1 文本
    original_left_words: List[WordTimestamp]   # 原始 Seg N 词级时间戳
    original_right_words: List[WordTimestamp]  # 原始 Seg N+1 词级时间戳
    windows: List[WindowASRResult] = field(default_factory=list)

    @property
    def all_words_global(self) -> Dict[str, List[Tuple[float, float, str]]]:
        """返回所有窗口的词级时间戳（全局坐标）

        Returns:
            {window_type: [(start_global, end_global, word), ...]}
        """
        result = {}
        for wr in self.windows:
            if not wr.success:
                continue
            words = []
            for seg in wr.segments:
                if seg.words:
                    for w in seg.words:
                        global_start = wr.window.start_time + w.start
                        global_end = wr.window.start_time + w.end
                        words.append((global_start, global_end, w.word))
            result[wr.window.window_type] = words
        return result


@dataclass
class SlidingWindowConfig:
    """滑动窗口配置"""

    base_overlap_ms: int = 500            # 基础重叠量
    fast_speech_wps: float = 4.0          # 快速语速阈值（词/秒）
    fast_overlap_ms: int = 750            # 快速语速重叠量
    very_fast_overlap_ms: int = 1000      # 极快语速重叠量
    fusion_window_sec: float = 1.0        # 融合窗（边界前后各取N秒）
    max_workers: int = 3                  # 并行 ASR 线程数


class SlidingWindowReASR:
    """滑动窗口冗余 ASR 识别器

    使用示例:
        reasr = SlidingWindowReASR(config, asr_engine, cache)
        results = reasr.process_boundaries(
            low_conf_indices=[3, 7],
            segments=merged_segments,
            asr_results=asr_results,
            audio=audio, sample_rate=16000,
        )
        # results[3] → BoundaryReASRResult
    """

    def __init__(
        self,
        config: Optional[SlidingWindowConfig] = None,
        asr_engine=None,
        cache=None,
        language: Optional[str] = None,
        cache_params: Optional[Dict] = None,
    ):
        self.config = config or SlidingWindowConfig()
        self._asr_engine = asr_engine
        self._cache = cache
        self._language = language  # 预检测的语言代码，避免短窗口语言误判
        self._cache_params = cache_params or {}

    def _cache_key(self, window: SlidingWindow) -> str:
        """Compute a deterministic cache key for a sliding window.

        Incorporates audio identity and decode policy from ``cache_params``
        so that different inputs or decode settings never collide.
        """
        if self._cache is None:
            import hashlib
            return hashlib.md5(
                f"{window.boundary_index}_{window.window_type}_{window.start_time}_{window.end_time}".encode()
            ).hexdigest()
        from pathlib import Path
        params = dict(self._cache_params)
        params["boundary_index"] = window.boundary_index
        params["window_type"] = window.window_type
        params["start"] = window.start_time
        params["end"] = window.end_time
        return self._cache.make_key(
            Path(f"boundary_window_{window.boundary_index}_{window.window_type}"),
            **params,
        )

    def set_asr_engine(self, engine) -> None:
        """延迟设置 ASR 引擎（pipeline 中惰性初始化后调用）"""
        self._asr_engine = engine

    def process_boundaries(
        self,
        low_conf_indices: List[int],
        segments: List[SpeechSegment],
        asr_results: List[List[TranscriptionSegment]],
        audio: np.ndarray,
        sample_rate: int,
        total_duration: float = 0.0,
    ) -> Dict[int, BoundaryReASRResult]:
        """对低置信度边界执行滑动窗口冗余 ASR

        Args:
            low_conf_indices: 需要重识别的边界索引列表
            segments: 语音段列表
            asr_results: 各段的 ASR 结果
            audio: 原始音频
            sample_rate: 采样率
            total_duration: 音频总时长（用于裁剪）

        Returns:
            {boundary_index: BoundaryReASRResult}
        """
        cfg = self.config
        results: Dict[int, BoundaryReASRResult] = {}

        if not low_conf_indices or self._asr_engine is None:
            return results

        if total_duration <= 0:
            total_duration = len(audio) / sample_rate

        # 为每个低置信边界创建窗口
        all_windows: List[SlidingWindow] = []
        for idx in low_conf_indices:
            if idx >= len(segments) - 1:
                continue
            windows = self._create_windows(
                idx, segments[idx], segments[idx + 1],
                asr_results[idx], total_duration,
            )
            all_windows.extend(windows)

        if not all_windows:
            return results

        logger.info(
            "Sliding-window re-ASR: %d boundaries → %d windows",
            len(low_conf_indices), len(all_windows),
        )

        # 并行执行所有窗口的 ASR
        window_results: Dict[int, List[WindowASRResult]] = {}
        for w in all_windows:
            window_results.setdefault(w.boundary_index, [])

        with ThreadPoolExecutor(max_workers=min(cfg.max_workers, len(all_windows))) as executor:
            futures = {
                executor.submit(
                    self._transcribe_window,
                    w, audio, sample_rate,
                ): w
                for w in all_windows
            }
            for future in as_completed(futures):
                w = futures[future]
                try:
                    wr = future.result()
                except Exception as e:
                    logger.error("Window ASR failed for boundary %d: %s", w.boundary_index, e)
                    wr = WindowASRResult(window=w, segments=[], success=False, error=str(e))
                window_results[w.boundary_index].append(wr)

        # 汇总每个边界的结果
        for idx in low_conf_indices:
            if idx >= len(segments) - 1:
                continue

            # 原始段的词级时间戳（转换为全局坐标）
            left_words_global = self._words_to_global(
                asr_results[idx], segments[idx].start,
            )
            right_words_global = self._words_to_global(
                asr_results[idx + 1], segments[idx + 1].start,
            )

            results[idx] = BoundaryReASRResult(
                boundary_index=idx,
                original_left_text=self._join_text(asr_results[idx]),
                original_right_text=self._join_text(asr_results[idx + 1]),
                original_left_words=left_words_global,
                original_right_words=right_words_global,
                windows=window_results.get(idx, []),
            )

        success_count = sum(
            1 for r in results.values()
            if any(w.success for w in r.windows)
        )
        logger.info(
            "Re-ASR complete: %d/%d boundaries have usable window data",
            success_count, len(results),
        )

        return results

    # ------------------------------------------------------------------
    # 窗口创建
    # ------------------------------------------------------------------

    def _create_windows(
        self,
        boundary_index: int,
        seg_left: SpeechSegment,
        seg_right: SpeechSegment,
        asr_left: List[TranscriptionSegment],
        total_duration: float,
    ) -> List[SlidingWindow]:
        """为单个边界创建 3 个滑动窗口"""
        cfg = self.config

        # 根据语速自适应重叠量
        overlap_ms = self._estimate_overlap(asr_left)

        windows = []

        # 窗口 A：左扩展（Seg N + forward_overlap）
        left_expand_end = min(seg_left.end + overlap_ms / 1000.0, total_duration)
        # 不能侵入 Seg N+1 后半段太深，最多 2× overlap
        max_left_end = seg_right.start + overlap_ms / 1000.0
        left_expand_end = min(left_expand_end, max_left_end)

        if left_expand_end > seg_left.end + 0.05:  # 至少有 50ms 扩展才值得
            windows.append(SlidingWindow(
                window_type="left_expand",
                start_time=seg_left.start,
                end_time=left_expand_end,
                boundary_index=boundary_index,
            ))

        # 窗口 B：右扩展（Seg N+1 - backward_overlap）
        right_expand_start = max(seg_right.start - overlap_ms / 1000.0, 0.0)
        max_right_start = seg_left.end - overlap_ms / 1000.0
        right_expand_start = max(right_expand_start, max_right_start)

        if right_expand_start < seg_right.end - 0.05:
            windows.append(SlidingWindow(
                window_type="right_expand",
                start_time=right_expand_start,
                end_time=seg_right.end,
                boundary_index=boundary_index,
            ))

        # 窗口 C：融合窗（边界前后各 fusion_window_sec）
        fusion_half = cfg.fusion_window_sec
        fusion_start = max(seg_left.end - fusion_half, 0.0)
        fusion_end = min(seg_right.start + fusion_half, total_duration)
        # 确保覆盖边界两侧
        if fusion_start < seg_left.end and fusion_end > seg_right.start:
            windows.append(SlidingWindow(
                window_type="fusion",
                start_time=fusion_start,
                end_time=fusion_end,
                boundary_index=boundary_index,
            ))

        return windows

    def _estimate_overlap(
        self,
        asr_segs: List[TranscriptionSegment],
    ) -> int:
        """根据语速估算重叠窗口大小

        语速 = 总词数 / 段时长
        """
        cfg = self.config
        total_words = sum(len(seg.words) for seg in asr_segs if seg.words)
        if total_words == 0:
            return cfg.base_overlap_ms

        # 估算段时长
        if asr_segs:
            duration = asr_segs[-1].end - asr_segs[0].start
        else:
            duration = 1.0

        if duration <= 0:
            return cfg.base_overlap_ms

        wps = total_words / duration

        if wps > 5.0:
            return cfg.very_fast_overlap_ms
        elif wps > cfg.fast_speech_wps:
            return cfg.fast_overlap_ms
        else:
            return cfg.base_overlap_ms

    # ------------------------------------------------------------------
    # ASR 转录
    # ------------------------------------------------------------------

    def _transcribe_window(
        self,
        window: SlidingWindow,
        audio: np.ndarray,
        sample_rate: int,
    ) -> WindowASRResult:
        """对单个窗口执行 ASR 转录"""
        try:
            start_sample = AudioUtils.time_to_sample(window.start_time, sample_rate)
            end_sample = AudioUtils.time_to_sample(window.end_time, sample_rate)
            segment_audio = AudioUtils.extract_segment(audio, start_sample, end_sample)

            if len(segment_audio) < sample_rate * 0.1:  # < 100ms 太短
                return WindowASRResult(
                    window=window, segments=[], success=False,
                    error="audio too short",
                )

            # 检查缓存
            if self._cache is not None:
                from pathlib import Path
                cache_key = self._cache.make_key(
                    Path(f"boundary_window_{window.boundary_index}_{window.window_type}"),
                    start=window.start_time,
                    end=window.end_time,
                )
                cached = self._cache.get("transcription", cache_key)
                if cached is not None:
                    return WindowASRResult(window=window, segments=cached)

            # 调用 ASR 引擎（使用预检测的语言，避免短窗口语言误判）
            segments = self._asr_engine.transcribe(
                segment_audio, sample_rate,
                language=self._language,
            )

            # 写入缓存
            if self._cache is not None:
                from pathlib import Path
                cache_key = self._cache.make_key(
                    Path(f"boundary_window_{window.boundary_index}_{window.window_type}"),
                    start=window.start_time,
                    end=window.end_time,
                )
                self._cache.set("transcription", cache_key, segments)

            return WindowASRResult(window=window, segments=segments)

        except Exception as e:
            logger.debug(
                "Window ASR failed [%s, boundary=%d]: %s",
                window.window_type, window.boundary_index, e,
            )
            return WindowASRResult(
                window=window, segments=[], success=False, error=str(e),
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _words_to_global(
        asr_segs: List[TranscriptionSegment],
        segment_offset: float,
    ) -> List[WordTimestamp]:
        """将段内词级时间戳转换为全局坐标"""
        global_words = []
        for seg in asr_segs:
            if seg.words:
                for w in seg.words:
                    global_words.append(WordTimestamp(
                        word=w.word,
                        start=segment_offset + w.start,
                        end=segment_offset + w.end,
                        confidence=getattr(w, "confidence", 1.0),
                    ))
        return global_words

    @staticmethod
    def _join_text(asr_segs: List[TranscriptionSegment]) -> str:
        if not asr_segs:
            return ""
        return " ".join(ts.text for ts in asr_segs).strip()

    @staticmethod
    def extract_window_texts(
        reasr_result: BoundaryReASRResult,
    ) -> Dict[str, str]:
        """提取各窗口的文本（供 LLM 仲裁使用）

        Returns:
            {"original_left": "...", "original_right": "...",
             "left_expand": "...", "right_expand": "...", "fusion": "..."}
        """
        texts = {
            "original_left": reasr_result.original_left_text,
            "original_right": reasr_result.original_right_text,
        }
        for wr in reasr_result.windows:
            if wr.success and wr.segments:
                window_text = " ".join(ts.text for ts in wr.segments).strip()
                texts[wr.window.window_type] = window_text
            else:
                texts[wr.window.window_type] = ""
        return texts

    @staticmethod
    def extract_fusion_words_global(
        reasr_result: BoundaryReASRResult,
    ) -> List[Tuple[float, float, str, float]]:
        """从融合窗提取词级时间戳（全局坐标 + 置信度）

        Returns:
            [(global_start, global_end, word, confidence), ...]
        """
        words = []
        for wr in reasr_result.windows:
            if not wr.success or wr.window.window_type != "fusion":
                continue
            for seg in wr.segments:
                if seg.words:
                    for w in seg.words:
                        gs = wr.window.start_time + w.start
                        ge = wr.window.start_time + w.end
                        conf = getattr(w, "confidence", 1.0)
                        words.append((gs, ge, w.word, conf))
        return words
