"""Pure acoustic skeleton queries using absolute-second coordinates."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np

from ..utils.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

DEFAULT_HARD_SILENCE_SECONDS = 0.4


def adaptive_silence_threshold_db(
    audio: np.ndarray | None,
    sample_rate: int,
    *,
    enabled: bool = True,
    fallback_db: float = -40.0,
    margin_db: float = 10.0,
    lower_bound_db: float = -45.0,
    upper_bound_db: float = -30.0,
) -> float:
    """按音频噪声底自适应推导 silencedetect 阈值（dBFS）。

    噪声底取底部 20% 帧 RMS 的中位数（AudioUtils.estimate_silence_rms），
    阈值 = 噪声底 dB + margin_db，并钳制到 [lower_bound_db, upper_bound_db]
    —— 与 reporting/noise_shadow 的建议策略保持一致（建议值同样只钳不放大）。
    关闭、音频缺失或估计失败时返回 fallback_db 固定值。
    """
    if not enabled or audio is None:
        return float(fallback_db)
    try:
        silence_rms = float(AudioUtils.estimate_silence_rms(audio, sample_rate))
        if not math.isfinite(silence_rms) or silence_rms <= 1e-6:
            return float(fallback_db)
        floor_db = 20.0 * math.log10(min(1.0, silence_rms))
        threshold = max(
            float(lower_bound_db), min(float(upper_bound_db), floor_db + margin_db)
        )
        logger.info(
            "Adaptive skeleton threshold: noise_floor=%.1fdB -> threshold=%.1fdB "
            "(margin=%.1fdB, fallback=%.1fdB)",
            floor_db,
            threshold,
            margin_db,
            fallback_db,
        )
        return round(threshold, 1)
    except Exception as exc:  # noqa: BLE001 - 阈值估计任何失败都回退固定值
        logger.warning(
            "Adaptive skeleton threshold failed, using %sdB: %s", fallback_db, exc
        )
        return float(fallback_db)


def group_speech_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    max_gap: float = DEFAULT_HARD_SILENCE_SECONDS,
) -> list[tuple[float, float]]:
    """Group intervals for ASR context without crossing hard silence.

    The result is an ASR input policy only. Callers must retain the original
    intervals for physical projection and coverage auditing.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    grouped: list[tuple[float, float]] = []
    for start, end in ordered:
        if not grouped or start - grouped[-1][1] > max_gap:
            grouped.append((start, end))
            continue
        grouped[-1] = (grouped[-1][0], max(grouped[-1][1], end))
    return grouped


def is_time_in_speech(t: float, skeleton: list[tuple[float, float]]) -> bool:
    return any(start <= t <= end for start, end in skeleton)


def has_speech_in_range(
    start: float, end: float, skeleton: list[tuple[float, float]]
) -> bool:
    return any(
        speech_start < end and speech_end > start
        for speech_start, speech_end in skeleton
    )


def rms_energy_check(
    audio: np.ndarray,
    sample_rate: int,
    time_point: float,
    window_ms: int = 50,
    threshold_ratio: float = 2.0,
) -> bool:
    silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)
    half_window = window_ms / 2000.0
    start = max(0.0, time_point - half_window)
    end = min(len(audio) / sample_rate, time_point + half_window)
    gap_rms = AudioUtils.get_segment_rms(audio, start, end, sample_rate)
    return gap_rms > silence_rms * threshold_ratio


def silence_confirmed(
    audio: np.ndarray | None, sample_rate: int, time_point: float, *, distance: float
) -> bool:
    if audio is None:
        return distance <= 0.03
    return not rms_energy_check(
        audio, sample_rate, time_point, window_ms=50, threshold_ratio=2.0
    )


def compute_vad_overlap(start: float, end: float, vad_segments: list) -> float:
    duration = end - start
    if duration <= 0:
        return 0.0
    overlap_total = 0.0
    for segment in vad_segments:
        segment_start = segment.start if hasattr(segment, "start") else segment[0]
        segment_end = segment.end if hasattr(segment, "end") else segment[1]
        overlap_start = max(start, segment_start)
        overlap_end = min(end, segment_end)
        if overlap_start < overlap_end:
            overlap_total += overlap_end - overlap_start
    return min(1.0, overlap_total / duration)


def next_speech_onset_after(
    t: float,
    skeleton: list[tuple[float, float]],
) -> float | None:
    """返回严格晚于 t 的下一个骨架段起点（跨过当前所在语音段）。"""
    best: float | None = None
    for s_start, _s_end in skeleton:
        if s_start > t + 1e-6 and (best is None or s_start < best):
            best = s_start
    return best


def speech_frame_mask(
    audio: np.ndarray,
    sample_rate: int,
    t0: float,
    t1: float,
) -> np.ndarray | None:
    """返回 [t0,t1] 内 20ms 帧是否含语音能量的布尔掩码。

    判据 = 帧峰值 > max(4% 全局峰值, 2× 底噪 RMS)。只用底噪倍数会把
    TTS/录音残留噪声（底噪的 2~2.5 倍）误判为语音；4% 峰值门限与
    噪声影子线一致，在干净素材上能干净地区分句内停顿与语音。
    """
    win = max(1, int(0.02 * sample_rate))
    i0 = int(t0 * sample_rate)
    i1 = min(len(audio), int(t1 * sample_rate))
    seg = audio[i0:i1].astype(np.float32)
    if seg.size < win:
        return None
    frames = seg[: seg.size // win * win].reshape(-1, win)
    frame_peak = np.abs(frames).max(axis=1)
    threshold = max(
        0.04 * float(np.abs(audio).max()) + 1e-9,
        2.0 * AudioUtils.estimate_silence_rms(audio, sample_rate),
    )
    return frame_peak > threshold


def next_energy_onset_after(
    audio: np.ndarray,
    sample_rate: int,
    t: float,
    horizon: float,
) -> float | None:
    """在 [t, t+horizon] 内扫描第一段持续语音爆发（≥2 帧超噪声底）。

    骨架段会把 <min_silence 的停顿并进同一段连续语音，句内停顿后的
    重新开口在骨架上没有边界；这里直接看能量，找到局部真实起点。
    """
    mask = speech_frame_mask(audio, sample_rate, t, t + horizon)
    if mask is None or mask.size < 2:
        return None
    for i in range(len(mask) - 1):
        if mask[i] and mask[i + 1]:
            return t + i * 0.02
    return None


def leading_silence_ahead(
    audio: np.ndarray | None,
    sample_rate: int,
    t: float,
    window_sec: float = 0.12,
    speech_frame_ratio: float = 0.2,
) -> bool:
    """判断 [t, t+window] 是否基本无语音能量（用于 start 后向吸附）。

    skeleton 的"点是否在语音段内"对帧级无缝衔接产生的共享边界会误判
    （上一句尾巴把下一句 start 顶进了语音段里），这里直接看能量：
    窗口内含语音帧占比 ≤ 阈值视为静音。
    """
    if audio is None:
        return False
    mask = speech_frame_mask(audio, sample_rate, t, t + window_sec)
    if mask is None:
        return True
    return float(mask.mean()) <= speech_frame_ratio


_is_time_in_speech = is_time_in_speech
_has_speech_in_range = has_speech_in_range
_rms_energy_check = rms_energy_check
_silence_confirmed = silence_confirmed
_compute_vad_overlap = compute_vad_overlap
_next_speech_onset_after = next_speech_onset_after
_next_energy_onset_after = next_energy_onset_after
_speech_frame_mask = speech_frame_mask
_leading_silence_ahead = leading_silence_ahead
