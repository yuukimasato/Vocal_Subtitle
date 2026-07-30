"""RMS energy check, VAD overlap, and acoustic event classification.

All functions use absolute-second coordinates. No display-time conversions.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _rms_energy_check(
    audio: np.ndarray,
    sample_rate: int,
    time_point: float,
    window_ms: int = 50,
    threshold_ratio: float = 2.0,
) -> bool:
    """在时间点附近做 RMS 能量确认

    Returns:
        True 如果检测到语音能量
    """
    from ..utils.audio_utils import AudioUtils

    silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)
    half_window = window_ms / 2000.0  # 转秒再折半

    t1 = max(0, time_point - half_window)
    t2 = min(len(audio) / sample_rate, time_point + half_window)

    gap_rms = AudioUtils.get_segment_rms(audio, t1, t2, sample_rate)
    return gap_rms > silence_rms * threshold_ratio


def _compute_vad_overlap(
    start: float,
    end: float,
    vad_segments: List,
) -> float:
    """计算区间与 VAD 检测结果的重叠比例

    Returns:
        0.0 ~ 1.0，重叠比例
    """
    duration = end - start
    if duration <= 0:
        return 0.0

    overlap_total = 0.0
    for seg in vad_segments:
        seg_start = seg.start if hasattr(seg, "start") else seg[0]
        seg_end = seg.end if hasattr(seg, "end") else seg[1]
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)
        if overlap_start < overlap_end:
            overlap_total += overlap_end - overlap_start

    return min(1.0, overlap_total / duration)


def _classify_energy_type(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> str:
    """基于频谱特征区分噪音类型

    使用自相关法检测谐波结构：
    - 有谐波结构 → "music_or_tonal"（音乐、警报等）
    - 无谐波结构 → "transient_noise"（拍桌子、关门等）

    Returns:
        "transient_noise" | "music_or_tonal" | "unknown"
    """
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = audio[start_sample:end_sample]

    if len(segment) < 256:
        return "unknown"

    try:
        # 自相关
        autocorr = np.correlate(segment, segment, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)

        # 找前几个峰值（基频和谐波）
        peaks = []
        for i in range(1, min(len(autocorr) - 1, sample_rate // 50)):  # 50Hz 下限
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > 0.15:  # 显著的峰值
                    peaks.append((i, autocorr[i]))

        if not peaks:
            return "transient_noise"  # 无谐波结构 → 瞬态噪音

        # 谐波比 = 峰值平均
        peak_vals = [p[1] for p in peaks[:10]]
        harmonics_ratio = sum(peak_vals) / len(peak_vals)

        if harmonics_ratio > 0.3:
            return "music_or_tonal"
        else:
            return "transient_noise"
    except Exception:
        return "unknown"


def classify_acoustic_events(
    skeleton: List[tuple],
    silero_segments: List,
    audio: np.ndarray,
    sample_rate: int,
) -> List[Dict]:
    """将声学骨架中的事件分为"人声"和"非人声高能事件"（文档 5.12.3）

    判定逻辑:
    - ffmpeg 标记为语音 + Silero 也标记为语音 → 人声（可信）
    - ffmpeg 标记为语音 + Silero 未标记 → 非人声高能事件（跳过吸附）
    - ffmpeg 标记为静音 + Silero 标记为语音 → 低音量人声（Silero优先）

    Args:
        skeleton: ffmpeg 声学骨架 [(start, end), ...]
        silero_segments: Silero VAD 检测结果
        audio: 音频数组
        sample_rate: 采样率

    Returns:
        分类后的事件列表 [{"start", "end", "type", "confidence", ...}, ...]
    """
    classified = []

    for sk_start, sk_end in skeleton:
        # 检查该区间是否被 Silero VAD 确认
        silero_overlap = _compute_vad_overlap(
            sk_start, sk_end, silero_segments,
        )

        if silero_overlap > 0.5:
            event_type = "human_speech"
            confidence = "high"
        elif silero_overlap > 0.1:
            event_type = "human_speech"
            confidence = "low"  # 边缘情况，可能是语尾渐弱
        else:
            # ffmpeg 检测到能量但 Silero 不认为是人声
            event_type = "non_human_energy"
            confidence = "high"

            # 进一步分类：音乐 vs 瞬态噪音
            energy_subtype = _classify_energy_type(
                audio, sample_rate, sk_start, sk_end,
            )

        entry = {
            "start": sk_start,
            "end": sk_end,
            "type": event_type,
            "confidence": confidence,
            "silero_overlap_ratio": round(silero_overlap, 2),
        }

        if event_type == "non_human_energy":
            entry["energy_subtype"] = energy_subtype

        classified.append(entry)

    non_human_count = sum(1 for c in classified if c["type"] == "non_human_energy")
    if non_human_count > 0:
        logger.info(
            "Acoustic event classification: %d total, %d non-human energy "
            "(%.0f%%), %d human speech",
            len(classified),
            non_human_count,
            non_human_count / max(len(classified), 1) * 100,
            sum(1 for c in classified if c["type"] == "human_speech"),
        )
    return classified
