"""ASR 边界双向精修

用 ASR 词级时间戳和三帧能量斜率对语音段边界做双向精修。

两个子功能：
1. refine_with_word_timestamps(): 用 ASR 首/末词时间戳收缩段边界
2. refine_boundary_bidirectional(): 三帧能量斜率校验法，保护辅音 Attack/Release

与方案七冲突时的裁决规则：
- 优先级1: 三帧斜率检测到阶跃跳变 (energy_ratio > 5.0) → 锁定方案四结果
- 优先级2: 方案七检测到端点落在绝对静音 → 方案七覆盖
- 优先级3: 取两者中更保守的修正
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..asr.base import TranscriptionSegment
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


@dataclass
class BoundaryRefinementConfig:
    """边界精修配置"""

    enabled: bool = True
    max_shrink_ms: float = 200       # 最多向内收缩 200ms
    max_extend_ms: float = 100       # 最多向外扩展 100ms
    check_frames: int = 3            # 三帧能量斜率校验
    frame_ms: int = 10               # 每帧 10ms
    min_boundary_confidence: float = 0.3  # ASR 词级时间戳最低置信度
    shrink_end_enabled: bool = False  # 能量扫描对语尾低能量（鼻韵尾、气声）不可靠，默认关闭


class BoundaryRefiner:
    """ASR 边界双向精修器

    使用示例:
        refiner = BoundaryRefiner()
        refined_segments, refined_asr = refiner.refine_all(
            segments, asr_results, audio, sample_rate
        )
    """

    def __init__(self, config: Optional[BoundaryRefinementConfig] = None):
        self.config = config or BoundaryRefinementConfig()

    def refine_all(
        self,
        segments: List[SpeechSegment],
        asr_results: List[List[TranscriptionSegment]],
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[List[SpeechSegment], List[List[TranscriptionSegment]]]:
        """对所有段执行边界精修

        Returns:
            (refined_segments, refined_asr_results)
        """
        cfg = self.config
        if not cfg.enabled:
            return segments, asr_results

        refined_segments = []
        for seg, asr_segs in zip(segments, asr_results):
            refined = self._refine_single_segment(
                seg, asr_segs, audio, sample_rate,
            )
            refined_segments.append(refined)

        return refined_segments, asr_results

    def _refine_single_segment(
        self,
        seg: SpeechSegment,
        asr_segs: List[TranscriptionSegment],
        audio: np.ndarray,
        sample_rate: int,
    ) -> SpeechSegment:
        """对单个段执行边界精修"""
        cfg = self.config

        # 收集所有词级时间戳
        all_words = []
        for asr in asr_segs:
            if asr.words:
                all_words.extend(asr.words)

        if not all_words:
            # 无词级时间戳 → 仅用三帧能量斜率
            seg_start = self.refine_boundary_bidirectional(
                audio, sample_rate, seg.start, "onset",
            )
            seg_end = self.refine_boundary_bidirectional(
                audio, sample_rate, seg.end, "offset",
            )
            seg.start = seg_start
            seg.end = seg_end
            return seg

        # ---- 段首精修：用第一个可靠词 ----
        first_word = all_words[0]
        if getattr(first_word, "start", None) is not None:
            first_word_start = first_word.start  # 段内偏移
            first_word_conf = getattr(first_word, "confidence", 1.0)

            if first_word_start > 0.05 and first_word_conf >= cfg.min_boundary_confidence:
                # 段首有 >50ms 静音，适当收缩
                # 区分 "padding 前导" 和 "真实前导静音"（同段尾策略）
                if first_word_start > 0.10:
                    shrink = min(first_word_start - 0.05, cfg.max_shrink_ms / 1000)
                    seg.start += shrink

        # ---- 段尾精修：反向 RMS 能量扫描 ----
        # 不再使用 ASR words[-1].end（Whisper 对结束位置估计天生不准）。
        # 改为从 seg.end 向段内反向扫描，找到能量下降到 silence 水平的精确点。
        if cfg.shrink_end_enabled:
            energy_end = self._find_energy_end(
                audio, seg.start, seg.end, sample_rate,
                frame_ms=5,
            )
            if energy_end is not None and energy_end < seg.end:
                # 收缩到能量结束点 + 30ms 语尾拖音余量
                shrink_amount = seg.end - (energy_end + 0.03)
                if shrink_amount > 0:
                    actual_shrink = min(shrink_amount, cfg.max_shrink_ms / 1000)
                    seg.end -= actual_shrink

        # ---- 双向三帧能量斜率精修 ----
        seg.start = self.refine_boundary_bidirectional(
            audio, sample_rate, seg.start, "onset",
        )
        seg.end = self.refine_boundary_bidirectional(
            audio, sample_rate, seg.end, "offset",
        )

        return seg

    def refine_boundary_bidirectional(
        self,
        audio: np.ndarray,
        sample_rate: int,
        boundary_time: float,
        direction: str,
    ) -> float:
        """ASR边界双向精修：三帧能量斜率校验

        原理：
        - 在边界前后各取 check_frames 帧，计算能量变化斜率
        - 阶跃跳变（Step Jump）：能量在1-2帧内急剧变化 → 边界精确，锁定
        - 缓慢爬升/下降：宁向外扩，不向内缩（保护辅音的 Attack/Release）

        Args:
            directional: "onset" 检查段首，"offset" 检查段尾

        Returns:
            精修后的边界时间
        """
        cfg = self.config
        center_sample = int(boundary_time * sample_rate)
        frame_size = int(cfg.frame_ms / 1000 * sample_rate)
        if frame_size < 1:
            return boundary_time

        check_frames = cfg.check_frames

        # 采集边界前后的能量值
        energies = []
        for i in range(-check_frames, check_frames):
            frame_start = center_sample + i * frame_size
            frame_end = frame_start + frame_size
            if 0 <= frame_start < len(audio) and frame_end <= len(audio):
                frame = audio[frame_start:frame_end]
                energies.append(float(np.sqrt(np.mean(frame ** 2))))
            else:
                energies.append(0.0)

        if len(energies) < check_frames * 2:
            return boundary_time

        # 计算前后能量比
        if direction == "onset":
            pre_energy = max(energies[:check_frames])
            post_energy = max(energies[check_frames:])
        else:  # offset
            pre_energy = max(energies[:check_frames])
            post_energy = max(energies[check_frames:])

        energy_ratio = (
            post_energy / max(pre_energy, 1e-8)
            if direction == "onset"
            else pre_energy / max(post_energy, 1e-8)
        )

        # 决策
        if energy_ratio > 5.0:
            # 阶跃跳变：边界非常清晰 → 锁定
            return boundary_time
        elif energy_ratio > 2.0:
            # 中等变化：轻微外扩
            adjustment = 0.02
            if direction == "onset":
                return boundary_time - adjustment
            else:
                return boundary_time + adjustment
        else:
            # 缓慢变化：较大外扩（保护辅音Attack/Release）
            adjustment = 0.05
            if direction == "onset":
                return max(0.0, boundary_time - adjustment)
            else:
                return min(len(audio) / sample_rate, boundary_time + adjustment)

    def _find_energy_end(
        self,
        audio: np.ndarray,
        seg_start: float,
        seg_end: float,
        sample_rate: int,
        frame_ms: int = 5,
    ) -> Optional[float]:
        """反向 RMS 能量扫描：从 seg_end 向段内扫描，找语音→静音的转折点。

        与 TimeMapper._find_speech_end_backward 不同，此方法在段内部扫描
        （从段尾向内），用于收缩 MergeStrategy 添加的尾部 padding。

        原理：
        - 从段尾区域（最后 200ms）估算静音基准 RMS
        - 从 seg_end 向 seg_start 逐 5ms 帧扫描
        - 找到第一帧 RMS > silence_rms * 2.5 → 语音在此结束
        - 如果全程能量都很低 → 返回 None

        Returns:
            语音结束时间（秒），或 None（不收缩）
        """
        frame_samples = int(frame_ms / 1000.0 * sample_rate)
        if frame_samples < 1:
            return None

        hop_sec = frame_ms / 1000.0
        total_samples = len(audio)
        seg_end_sample = min(int(seg_end * sample_rate), total_samples)

        # 从段尾区域估算静音基准（尾部 padding 区最可能是纯静音）
        tail_duration = min(0.2, seg_end - seg_start)
        tail_samples = int(tail_duration * sample_rate)
        tail_start_sample = max(0, seg_end_sample - tail_samples)
        tail_audio = audio[tail_start_sample:seg_end_sample]

        if len(tail_audio) > 0:
            tail_rms = float(np.sqrt(np.mean(
                np.asarray(tail_audio, dtype=np.float64) ** 2
            )))
        else:
            tail_rms = 0.0

        # 也取扫描区间的最小 RMS 作为补充（防止段尾仍有语音拖音）
        scan_start_sample = max(0, int(seg_start * sample_rate))
        scan_region = audio[scan_start_sample:seg_end_sample]
        if len(scan_region) > 0:
            # 采样计算：每隔 50ms 取一帧的 RMS，取最小值
            min_rms = float('inf')
            step_samples = int(0.05 * sample_rate)
            for s in range(0, len(scan_region) - frame_samples, step_samples):
                frame = scan_region[s:s + frame_samples]
                rms = float(np.sqrt(np.mean(
                    np.asarray(frame, dtype=np.float64) ** 2
                )))
                if rms < min_rms:
                    min_rms = rms
            if min_rms == float('inf'):
                min_rms = 0.0
        else:
            min_rms = 0.0

        # 静音基准 = min(段尾RMS, 扫描区间最小RMS)
        silence_rms = min(tail_rms, min_rms)
        energy_threshold = max(silence_rms * 2.5, 0.0005)

        # 从 seg_end 向 seg_start 扫描
        t = seg_end
        while t > seg_start + hop_sec:
            t -= hop_sec
            frame_start = int(t * sample_rate)
            frame_end = frame_start + frame_samples
            if frame_end > total_samples:
                continue
            if frame_end <= frame_start:
                continue

            frame = audio[frame_start:frame_end]
            rms = float(np.sqrt(np.mean(np.asarray(frame, dtype=np.float64) ** 2)))

            if rms > energy_threshold:
                # 找到语音能量 — 语音在此帧结束
                return t + hop_sec

        return None  # 整个段尾区域都是静音，不收缩

    def get_energy_ratio_at(
        self,
        audio: np.ndarray,
        sample_rate: int,
        boundary_time: float,
        direction: str,
    ) -> float:
        """获取边界处的能量比值（用于方案四vs方案七冲突裁决）

        Returns:
            energy_ratio: 越大表示边界越清晰
        """
        cfg = self.config
        center_sample = int(boundary_time * sample_rate)
        frame_size = int(cfg.frame_ms / 1000 * sample_rate)
        if frame_size < 1:
            return 1.0

        check_frames = cfg.check_frames
        energies = []
        for i in range(-check_frames, check_frames):
            frame_start = center_sample + i * frame_size
            frame_end = frame_start + frame_size
            if 0 <= frame_start < len(audio) and frame_end <= len(audio):
                frame = audio[frame_start:frame_end]
                energies.append(float(np.sqrt(np.mean(frame ** 2))))
            else:
                energies.append(0.0)

        if len(energies) < check_frames * 2:
            return 1.0

        if direction == "onset":
            pre_energy = max(energies[:check_frames])
            post_energy = max(energies[check_frames:])
            return post_energy / max(pre_energy, 1e-8)
        else:
            pre_energy = max(energies[:check_frames])
            post_energy = max(energies[check_frames:])
            return pre_energy / max(post_energy, 1e-8)


def resolve_boundary_conflict(
    asr_refined_time: float,
    asr_energy_ratio: float,
    physical_nearest: float,
    physical_is_silence: bool,
    original_time: float,
) -> float:
    """方案四 vs 方案七 边界冲突裁决

    按优先级：
    1. 阶跃跳变 → ASR结果锁定
    2. 绝对静音 → 物理标尺覆盖
    3. 取两者中更保守的

    Returns:
        最终边界时间
    """
    # 优先级1: 阶跃跳变 → ASR结果锁定
    if asr_energy_ratio > 5.0:
        return asr_refined_time

    # 优先级2: 绝对静音 → 物理标尺覆盖
    if physical_is_silence:
        return physical_nearest

    # 优先级3: 取保守值（修正量较小的）
    asr_delta = abs(asr_refined_time - original_time)
    physical_delta = abs(physical_nearest - original_time)

    if asr_delta <= physical_delta:
        return asr_refined_time
    else:
        return physical_nearest


# ------------------------------------------------------------------
# 5.12.2 混合语种切换处理
# ------------------------------------------------------------------


def detect_language_switches(
    words: List,
    confidence_drop_threshold: float = 0.3,
    min_switch_length: int = 2,
) -> List[int]:
    """检测词级时间戳中的语种切换点

    语种切换的特征（faster-whisper 在中英夹杂场景下表现）：
    1. 连续 2+ 个词的置信度骤降 >30%
    2. 词间出现异常长间隙（模型在"思考"语言切换）
    3. 降信词之后恢复正常（切换完成）

    Args:
        words: 词级时间戳列表，每项含 confidence/start/end
        confidence_drop_threshold: 置信度下降阈值
        min_switch_length: 最少连续降信词数才判定为切换

    Returns:
        语种切换位置索引列表（词索引）
    """
    if len(words) < 4:
        return []

    # 提取置信度序列
    confidences = [
        getattr(w, "confidence", 1.0)
        if hasattr(w, "confidence") else w.get("confidence", 1.0)
        for w in words
    ]

    switch_points = []
    in_low_confidence = False
    low_start = 0

    for i in range(1, len(confidences)):
        # 置信度骤降检测
        avg_before = (
            sum(confidences[max(0, i - 2):i])
            / max(1, min(i, 2))
        )
        avg_after = (
            sum(confidences[i:min(len(confidences), i + 3)])
            / min(3, len(confidences) - i)
        )

        drop = avg_before - avg_after

        if drop > confidence_drop_threshold and not in_low_confidence:
            low_start = i
            in_low_confidence = True
        elif drop <= confidence_drop_threshold * 0.5 and in_low_confidence:
            # 置信度恢复
            low_duration = i - low_start
            if low_duration >= min_switch_length:
                switch_points.append(low_start)
            in_low_confidence = False

    # 末尾仍处于低置信度
    if in_low_confidence:
        low_duration = len(confidences) - low_start
        if low_duration >= min_switch_length:
            switch_points.append(low_start)

    return switch_points


def smooth_multilingual_timestamps(
    words: List,
    switch_points: List[int],
) -> List:
    """对语种切换点附近的时间戳做平滑处理

    语种切换边界的词级时间戳不可靠（置信度低），
    用切换区前后的可靠词时间戳做线性插值替代。

    Args:
        words: 词级时间戳列表（会被浅拷贝）
        switch_points: detect_language_switches() 的输出

    Returns:
        平滑后的词列表（标记 _timestamp_smoothed=True）
    """
    if not switch_points:
        return list(words)

    smoothed = list(words)  # 浅拷贝

    for switch_idx in switch_points:
        if switch_idx >= len(words):
            continue

        # 取切换点前后各 3 个可靠词的时间戳做线性插值
        pre_idx = max(0, switch_idx - 3)
        post_idx = min(len(words) - 1, switch_idx + 3)

        if post_idx <= pre_idx:
            continue

        # 获取边界时间
        pre_word = words[pre_idx]
        post_word = words[post_idx]

        t_start = (
            pre_word.start if hasattr(pre_word, "start")
            else pre_word.get("start", 0)
        )
        t_end = (
            post_word.end if hasattr(post_word, "end")
            else post_word.get("end", 0)
        )

        if t_end <= t_start:
            continue

        n_words_in_switch = post_idx - pre_idx + 1

        for j, idx in enumerate(range(pre_idx, post_idx + 1)):
            fraction = j / max(n_words_in_switch - 1, 1)

            original = words[idx]
            if hasattr(original, "start"):
                # TranscriptionWord 对象 → 创建 dict 替代
                new_word = dict(
                    start=t_start + fraction * (t_end - t_start) * 0.9,
                    end=t_start + (fraction + 1.0 / n_words_in_switch)
                              * (t_end - t_start) * 0.9,
                    word=getattr(original, "word", ""),
                    confidence=getattr(original, "confidence", 1.0),
                    _timestamp_smoothed=True,
                )
            else:
                # 已经是 dict
                new_word = {
                    **original,
                    "start": t_start + fraction * (t_end - t_start) * 0.9,
                    "end": t_start + (fraction + 1.0 / n_words_in_switch)
                                * (t_end - t_start) * 0.9,
                    "_timestamp_smoothed": True,
                }
            smoothed[idx] = new_word

    return smoothed
