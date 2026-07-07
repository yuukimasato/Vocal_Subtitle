"""片段合并策略

对 VAD 检测结果进行合并、切分和填充处理。

算法:
1. 按 start 时间排序
2. (可选) RMS 边界精修 — 将 VAD 帧级边界精修到 ~5ms 精度
3. (可选) 段内静音预切分 — 在段内长静音处切分，避免 ASR 时间戳漂移
4. 合并相邻间隔 < min_silence_gap 且间隙为真实静音的片段
5. 对超出 max_segment_length 的片段在段内最长静音处切分
6. (可选) 自适应 padding — 根据边界能量梯度调整 padding 大小
7. 最小片段保护 — 过滤/合并过短片段
8. 边界裁剪
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """合并策略参数

    Attributes:
        min_silence_gap: 合并阈值（秒），相邻间隔小于此值则合并
        max_segment_length: 最大段长（秒），超出则切分
        padding: 基础两端填充（秒）
        adaptive_padding: 是否根据边界能量梯度自适应调整 padding
        padding_min: 自适应最小 padding（能量清晰边界）
        padding_max: 自适应最大 padding（能量模糊边界）
        pre_split_silence: 是否在段内静音处预切分
        pre_split_threshold: 内部静音 > 此值则切分（秒）
        min_fragment_duration: 最小语音片段（秒），短于此值强制合并
        min_segment_length: 最小段长（秒），过短则丢弃
        refine_boundaries: 是否用 RMS 能量精修 VAD 边界
        boundary_window: 边界精修搜索窗口（秒）
    """

    min_silence_gap: float = 0.4
    max_segment_length: float = 20.0
    padding: float = 0.10
    adaptive_padding: bool = True
    padding_min: float = 0.05
    padding_max: float = 0.20
    pre_split_silence: bool = True
    pre_split_threshold: float = 0.5
    min_fragment_duration: float = 0.15
    min_segment_length: float = 0.5  # 最小段长（秒），与 MergingConfig 保持一致
    refine_boundaries: bool = True
    boundary_window: float = 0.15
    protect_single_word: bool = True      # 禁止在单词中间切分
    min_word_gap_ms: int = 80             # 单词内部允许的最大"静音"（清辅音间隔）


class MergeStrategy:
    """片段合并策略

    使用示例:
        strategy = MergeStrategy(MergeConfig(
            min_silence_gap=0.4,
            max_segment_length=20.0,
            adaptive_padding=True,
        ))
        merged = strategy.merge(vad_segments, audio, sample_rate)
    """

    def __init__(self, config: Optional[MergeConfig] = None):
        self.config = config or MergeConfig()

    def merge(
        self,
        segments: List[SpeechSegment],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        total_duration: Optional[float] = None,
    ) -> List[SpeechSegment]:
        """执行合并策略

        Args:
            segments: VAD 检测结果
            audio: 原始音频数据（用于边界精修和静音验证）
            sample_rate: 采样率
            total_duration: 音频总时长（秒），用于边界裁剪

        Returns:
            合并/切分/填充后的语音片段列表
        """
        if not segments:
            return []

        cfg = self.config
        input_count = len(segments)

        # Step 1: 按 start 排序
        segments = sorted(segments, key=lambda s: s.start)

        # Step 1.5: RMS 边界精修（提升 VAD 边界精度）
        if cfg.refine_boundaries and audio is not None:
            from ..utils.audio_utils import AudioUtils

            AudioUtils.refine_speech_boundaries(
                segments, audio, sample_rate,
                window=cfg.boundary_window,
            )

        # Step 1.6: 段内静音预切分（Phase 1 新增）
        # 在 VAD 段内部的长静音处切分，避免 ASR 时间戳漂移
        if cfg.pre_split_silence and audio is not None:
            segments = self._pre_split_at_silence_gaps(
                segments, audio, sample_rate,
            )

        # Step 2: 合并相邻片段（含 RMS 静音验证）
        merged = self._merge_adjacent(segments, audio, sample_rate)

        # Step 3: 切分超长片段
        split = self._split_long_segments(merged, audio, sample_rate)

        # Step 4: 自适应 padding（Phase 1 新增）
        padded = self._apply_adaptive_padding(split, audio, sample_rate)

        # Step 5: 边界裁剪
        if total_duration is not None:
            padded = self._clip_boundaries(padded, total_duration)

        # Step 6: 最小片段保护（Phase 1 新增）
        protected = self._apply_minimum_fragment_protection(padded)

        # Step 7: 过滤过短片段
        filtered = [
            seg
            for seg in protected
            if seg.duration >= cfg.min_segment_length
        ]

        logger.info(
            "Merge: %d→%d→%d→%d segments "
            "(cfg: gap<%.2fs, len<%.1fs, pad=%.2f, adaptive=%s, pre_split=%s)",
            input_count,
            len(merged),
            len(padded),
            len(filtered),
            cfg.min_silence_gap,
            cfg.max_segment_length,
            cfg.padding,
            cfg.adaptive_padding,
            cfg.pre_split_silence,
        )

        return filtered

    # ------------------------------------------------------------------
    # 段内静音预切分
    # ------------------------------------------------------------------

    def _pre_split_at_silence_gaps(
        self,
        segments: List[SpeechSegment],
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[SpeechSegment]:
        """在 VAD 段内部的静音间隙处预切分

        目的：避免段内长静音导致 ASR 时间戳漂移。

        使用动态 RMS 阈值（相对于当前段的局部能量基准），
        而非全局绝对阈值，避免语尾渐弱被误判为静音。
        """
        cfg = self.config
        frame_ms = 10
        frame_size = int(frame_ms / 1000 * sample_rate)
        hop_size = max(1, frame_size // 2)  # 5ms hop

        if frame_size < 1:
            return segments

        result = []
        for seg in segments:
            seg_samples = int((seg.end - seg.start) * sample_rate)
            # 短片段不需要预切分
            if seg.duration < cfg.pre_split_threshold * 2:
                result.append(seg)
                continue

            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)

            # ---- 动态 RMS 阈值：基于当前段前部能量 ----
            # 取该段前 500ms（排除开头 20ms）作为能量基准
            ref_start = start_sample + int(0.02 * sample_rate)
            ref_end = min(
                end_sample,
                start_sample + int(0.5 * sample_rate),
            )
            baseline_rms = self._compute_baseline_rms(
                audio, ref_start, ref_end, frame_size,
            )
            if baseline_rms is None:
                result.append(seg)
                continue

            # 动态阈值 = 基准能量的 25%
            # 比全局阈值更适应说话者音量变化
            dynamic_threshold = baseline_rms * 0.25

            # 同时计算全局静音阈值作为绝对下限
            from ..utils.audio_utils import AudioUtils
            global_silence = AudioUtils.estimate_silence_rms(
                audio, sample_rate, percentile=20,
            )
            effective_threshold = max(dynamic_threshold, global_silence * 1.5)

            # ---- 逐帧扫描段内静音区间 ----
            silence_gaps = []  # [(gap_start_time, gap_end_time)]
            in_silence = False
            silence_begin_sample = 0

            for i in range(start_sample, end_sample - frame_size + 1, hop_size):
                frame = audio[i : i + frame_size]
                rms = float(np.sqrt(np.mean(frame ** 2)))

                if rms < effective_threshold:
                    if not in_silence:
                        silence_begin_sample = i
                        in_silence = True
                else:
                    if in_silence:
                        gap_duration = (i - silence_begin_sample) / sample_rate
                        if gap_duration >= cfg.pre_split_threshold:
                            silence_gaps.append((
                                silence_begin_sample / sample_rate,
                                i / sample_rate,
                            ))
                        in_silence = False

            # 处理尾部静音
            if in_silence:
                gap_duration = (end_sample - silence_begin_sample) / sample_rate
                if gap_duration >= cfg.pre_split_threshold:
                    silence_gaps.append((
                        silence_begin_sample / sample_rate,
                        end_sample / sample_rate,
                    ))

            # ---- 在静音间隙处切分 ----
            if not silence_gaps:
                result.append(seg)
            else:
                # 防护：如果单个静音间隙覆盖了 > 80% 的段长，
                # 说明整个段被判定为静音（可能是极低能量信号），
                # 不应切分，保留原始段。
                total_gap_duration = sum(
                    gap_e - gap_s for gap_s, gap_e in silence_gaps
                )
                if total_gap_duration > seg.duration * 0.8 and len(silence_gaps) == 1:
                    result.append(seg)
                    continue

                # 检查切分后的子段是否都足够长
                sub_boundaries = [seg.start]
                for gap_s, gap_e in silence_gaps:
                    sub_boundaries.append(gap_s)
                    sub_boundaries.append(gap_e)
                sub_boundaries.append(seg.end)

                for i in range(0, len(sub_boundaries) - 1, 2):
                    sub_s = sub_boundaries[i]
                    sub_e = sub_boundaries[i + 1]
                    if sub_e - sub_s >= cfg.min_fragment_duration:
                        result.append(SpeechSegment(
                            start=sub_s, end=sub_e,
                            confidence=seg.confidence,
                        ))
                    else:
                        # 子段太短（如单字），保留在结果中
                        # 最小片段保护会在后续步骤处理
                        result.append(SpeechSegment(
                            start=sub_s, end=sub_e,
                            confidence=seg.confidence,
                        ))

        # 重新按 start 排序
        result.sort(key=lambda s: s.start)

        logger.info(
            "Pre-split: %d → %d segments (silence gap > %.1fs, "
            "dynamic threshold)",
            len(segments), len(result), cfg.pre_split_threshold,
        )
        return result

    @staticmethod
    def _compute_baseline_rms(
        audio: np.ndarray,
        ref_start: int,
        ref_end: int,
        frame_size: int,
    ) -> Optional[float]:
        """计算参考段的 RMS 能量基准（鲁棒中位数估计）"""
        if ref_end <= ref_start + frame_size:
            chunk = audio[ref_start:ref_end]
            return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else None

        rms_vals = []
        for i in range(ref_start, ref_end - frame_size + 1, frame_size):
            frame = audio[i : i + frame_size]
            rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))

        if not rms_vals:
            return None

        rms_vals.sort()
        # 取 40%-60% 分位数区间的中位数（鲁棒估计）
        lo = max(0, int(len(rms_vals) * 0.4))
        hi = min(len(rms_vals), int(len(rms_vals) * 0.6) + 1)
        robust_vals = rms_vals[lo:hi]
        return float(np.median(robust_vals)) if robust_vals else rms_vals[len(rms_vals)//2]

    # ------------------------------------------------------------------
    # 自适应 Padding
    # ------------------------------------------------------------------

    def _apply_adaptive_padding(
        self,
        segments: List[SpeechSegment],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[SpeechSegment]:
        """根据边界能量梯度自适应调整 padding

        原理：
        - 边界处能量急剧下降（清晰边界）→ 小 padding
        - 边界处能量缓慢下降（模糊边界）→ 大 padding
        - 无 audio 数据时回退到固定 padding
        """
        cfg = self.config
        if not cfg.adaptive_padding or audio is None:
            # 回退到固定 padding
            return self._add_padding(segments)

        for seg in segments:
            # 计算 onset（段首）的能量梯度
            onset_slope = self._energy_slope_at(
                audio, sample_rate, seg.start, direction="forward",
            )
            # 计算 offset（段尾）的能量梯度
            offset_slope = self._energy_slope_at(
                audio, sample_rate, seg.end, direction="backward",
            )

            onset_pad = self._slope_to_padding(onset_slope, cfg)
            offset_pad = self._slope_to_padding(offset_slope, cfg)

            seg.start = max(0.0, seg.start - onset_pad)
            # 尾部 padding: end 锚定到声学边界，保留 ≤80ms 保护语尾拖音
            # （能量扫描对中文鼻韵尾/气声不可靠，padding 是主要保护手段）
            tail_pad = min(offset_pad, 0.08)
            seg.end = min(
                len(audio) / sample_rate if sample_rate > 0 else float("inf"),
                seg.end + tail_pad,
            )

        return segments

    def _energy_slope_at(
        self,
        audio: np.ndarray,
        sample_rate: int,
        boundary_time: float,
        direction: str,
        window_ms: int = 60,
        frame_ms: int = 10,
    ) -> float:
        """在边界处测量能量变化斜率

        取边界前后各 window_ms 范围，比较能量变化幅度。

        Args:
            direction: "forward" 检查 onset（静音→语音），
                       "backward" 检查 offset（语音→静音）

        Returns:
            能量比值（越大 = 边界越清晰）
        """
        center_sample = int(boundary_time * sample_rate)
        window_samples = int(window_ms / 1000 * sample_rate)
        frame_size = int(frame_ms / 1000 * sample_rate)
        if frame_size < 1:
            return 1.0

        total_samples = len(audio)

        # 采集边界前后的能量
        pre_energies = []
        post_energies = []

        for offset in range(-window_samples, window_samples, frame_size):
            frame_start = center_sample + offset
            frame_end = frame_start + frame_size
            if 0 <= frame_start < total_samples and frame_end <= total_samples:
                frame = audio[frame_start:frame_end]
                rms = float(np.sqrt(np.mean(frame ** 2)))
                if offset < 0:
                    pre_energies.append(rms)
                else:
                    post_energies.append(rms)

        if not pre_energies or not post_energies:
            return 1.0

        pre_avg = np.mean(pre_energies)
        post_avg = np.mean(post_energies)

        if direction == "forward":
            # onset: 前(静音) → 后(语音)，比值越大越清晰
            return post_avg / max(pre_avg, 1e-8)
        else:
            # offset: 前(语音) → 后(静音)，比值越大越清晰
            return pre_avg / max(post_avg, 1e-8)

    @staticmethod
    def _slope_to_padding(slope: float, cfg: MergeConfig) -> float:
        """能量斜率 → padding 大小映射

        设计意图：此阶段的 start padding 宁可偏大（保护语首辅音），不可偏小。
        BoundaryRefiner 会在后续阶段基于 ASR 词级时间戳做精确收缩。
        两者协作形成 "padding 给足 → 精确裁切" 的两阶段策略。

        尾部 padding 在 _apply_adaptive_padding 中被限制到 30ms 以内，
        因为 end 已锚定到声学边界，不再依赖 ASR last-word 时间戳。

        - slope > 8:   阶跃跳变，边界非常清晰 → 最小 padding (50ms)
        - slope > 3:   中等变化 → 基础 padding (100ms)
        - slope > 1.5: 缓慢变化 → 较大 padding (150ms)
        - slope ≤ 1.5: 能量模糊（语尾渐弱） → 最大 padding (200ms)
        """
        if slope > 8.0:
            return cfg.padding_min   # 50ms
        elif slope > 3.0:
            return cfg.padding       # 100ms
        elif slope > 1.5:
            return (cfg.padding + cfg.padding_max) / 2  # 150ms
        else:
            return cfg.padding_max   # 200ms

    # ------------------------------------------------------------------
    # 最小片段保护
    # ------------------------------------------------------------------

    def _apply_minimum_fragment_protection(
        self,
        segments: List[SpeechSegment],
    ) -> List[SpeechSegment]:
        """防止过短片段（<150ms）造成字幕碎片化

        处理策略：
        - 片段 >= min_fragment_duration → 保留
        - 片段 < min_fragment_duration → 尝试与相邻片段合并
        - 优先与间距较小的一方合并
        """
        cfg = self.config
        if len(segments) <= 1:
            return segments

        result: List[SpeechSegment] = []
        i = 0
        while i < len(segments):
            seg = segments[i]

            if seg.duration >= cfg.min_fragment_duration:
                result.append(seg)
                i += 1
                continue

            # 片段过短：决定合并方向
            prev_seg = result[-1] if result else None
            next_seg = segments[i + 1] if i + 1 < len(segments) else None

            prev_gap = (seg.start - prev_seg.end) if prev_seg else float("inf")
            next_gap = (next_seg.start - seg.end) if next_seg else float("inf")

            # 阈值：只合并间隔 < 0.5s 的相邻片段
            max_merge_gap = 0.5

            if next_gap <= prev_gap and next_gap < max_merge_gap and next_seg:
                # 与下一片段合并
                merged = SpeechSegment(
                    start=seg.start,
                    end=next_seg.end,
                    confidence=min(seg.confidence, next_seg.confidence),
                )
                result.append(merged)
                i += 2
            elif prev_gap < max_merge_gap and prev_seg:
                # 与前一片段合并
                prev_seg.end = max(prev_seg.end, seg.end)
                prev_seg.confidence = min(prev_seg.confidence, seg.confidence)
                i += 1
            else:
                # 无法合并（间隙太大），但仍保留（宁可碎片化也不丢内容）
                result.append(seg)
                i += 1

        if len(result) != len(segments):
            logger.info(
                "Fragment protection: %d → %d segments (min=%.0fms)",
                len(segments), len(result), cfg.min_fragment_duration * 1000,
            )
        return result

    # ------------------------------------------------------------------
    # 原有方法（保持兼容）
    # ------------------------------------------------------------------

    def _merge_adjacent(
        self,
        segments: List[SpeechSegment],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[SpeechSegment]:
        """合并间隔小于阈值且间隙为真实静音的相邻片段。

        使用 RMS 能量验证间隙是否真的是静音：
        - 间隙 RMS < 2×静音阈值 → 静音间隙 → 合并
        - 间隙 RMS ≥ 2×静音阈值 → 有语音能量 → 保留独立段
        """
        if len(segments) <= 1:
            return segments

        cfg = self.config
        result = [segments[0]]

        # 估算静音阈值（用于间隙验证）
        silence_rms = None
        if audio is not None:
            from ..utils.audio_utils import AudioUtils
            silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)

        for seg in segments[1:]:
            gap = seg.start - result[-1].end

            if gap < cfg.min_silence_gap:
                # 验证间隙是否为真实静音
                is_silent_gap = True
                if audio is not None and silence_rms is not None and gap > 0.01:
                    from ..utils.audio_utils import AudioUtils
                    gap_rms = AudioUtils.get_segment_rms(
                        audio, result[-1].end, seg.start, sample_rate,
                    )
                    # 间隙有语音能量 → 不应合并
                    if gap_rms > silence_rms * 2.0:
                        is_silent_gap = False
                        logger.debug(
                            "Gap %.3f-%.3f (%.2fs) has energy (rms=%.6f), "
                            "keeping segments separate",
                            result[-1].end, seg.start, gap, gap_rms,
                        )

                if is_silent_gap:
                    # 合并到前一个段
                    result[-1].end = max(result[-1].end, seg.end)
                    result[-1].confidence = max(
                        result[-1].confidence, seg.confidence
                    )
                    continue

            result.append(seg)

        return result

    def _split_long_segments(
        self,
        segments: List[SpeechSegment],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[SpeechSegment]:
        """切分超长片段"""
        cfg = self.config
        result = []

        for seg in segments:
            if seg.duration <= cfg.max_segment_length:
                result.append(seg)
                continue

            # 需要在段内最长静音处切分
            splits = self._find_split_points(seg, audio, sample_rate)
            if len(splits) <= 1:
                result.append(seg)
                continue

            # 创建子片段
            prev_time = seg.start
            for split_time in splits[1:]:
                result.append(
                    SpeechSegment(
                        start=prev_time,
                        end=split_time,
                        confidence=seg.confidence,
                    )
                )
                prev_time = split_time

            # 最后一段
            result.append(
                SpeechSegment(
                    start=prev_time,
                    end=seg.end,
                    confidence=seg.confidence,
                )
            )

        return result

    def _find_split_points(
        self,
        segment: SpeechSegment,
        audio: Optional[np.ndarray],
        sample_rate: int,
    ) -> List[float]:
        """在长段内寻找合适的切分点

        策略：每隔 max_segment_length 找一个附近 RMS 最小的位置。
        """
        cfg = self.config
        duration = segment.duration

        if duration <= cfg.max_segment_length:
            return [segment.start]

        # 计算需要的切分点数量
        num_chunks = int(np.ceil(duration / cfg.max_segment_length))
        chunk_dur = duration / num_chunks

        split_times = [segment.start]

        for i in range(1, num_chunks):
            ideal_time = segment.start + i * chunk_dur

            # 在 ideal_time ± window 内寻找最安静的位置
            window = min(1.0, chunk_dur * 0.2)  # 搜索窗口
            search_start = max(segment.start, ideal_time - window)
            search_end = min(segment.end, ideal_time + window)

            split_times.append(
                self._find_quietest_point(
                    search_start, search_end, audio, sample_rate
                )
            )

        split_times.append(segment.end)
        return split_times

    def _find_quietest_point(
        self,
        start: float,
        end: float,
        audio: Optional[np.ndarray],
        sample_rate: int,
    ) -> float:
        """在时间范围内寻找 RMS 最小的采样点"""
        if audio is None or sample_rate <= 0:
            return (start + end) / 2

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if end_sample <= start_sample:
            return (start + end) / 2

        # 使用滑动窗口计算 RMS，寻找最小能量位置
        window_samples = int(0.1 * sample_rate)  # 100ms 窗口
        if window_samples == 0:
            return (start + end) / 2

        min_rms = float("inf")
        min_pos = start_sample

        step = max(1, window_samples // 2)
        for i in range(start_sample, end_sample - window_samples + 1, step):
            chunk = audio[i : i + window_samples]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < min_rms:
                min_rms = rms
                min_pos = i + window_samples // 2

        return min_pos / sample_rate

    def _add_padding(
        self, segments: List[SpeechSegment]
    ) -> List[SpeechSegment]:
        """为每个片段添加固定 padding（非自适应回退方案）"""
        if self.config.padding <= 0:
            return segments

        for seg in segments:
            seg.start -= self.config.padding
            # 尾部 padding: 锚定到声学边界，保留 ≤80ms
            tail_pad = min(self.config.padding, 0.08)
            seg.end += tail_pad

        return segments

    def _clip_boundaries(
        self,
        segments: List[SpeechSegment],
        total_duration: float,
    ) -> List[SpeechSegment]:
        """确保片段不超出 [0, total_duration] 范围"""
        result = []
        for seg in segments:
            start = max(0.0, seg.start)
            end = min(total_duration, seg.end)

            if end > start:
                seg.start = start
                seg.end = end
                result.append(seg)
        return result
