"""边界融合引擎

融合 Silero VAD、ffmpeg silencedetect、RMS Energy Scan
三种方法的检测结果，输出高置信度的语音段边界。

算法:
1. 时间轴离散化（10ms 网格，整数采样点索引）
2. 每种方法投票：该网格是否为语音
3. 2/3 多数决 → 高置信度；1/3 → 低置信度
4. 连续语音网格 → 语音段
5. 根据置信度应用不同的 padding
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .base import SpeechSegment

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """融合配置"""

    grid_resolution: float = 0.01       # 时间网格精度（秒），10ms
    min_consensus: int = 2              # 最少共识方法数（共3种方法）
    high_conf_padding: float = 0.03     # 高置信边界的 padding
    low_conf_padding: float = 0.12      # 低置信边界的 padding
    min_speech_duration: float = 0.25   # 最小语音段（秒）
    sample_rate: int = 16000            # 统一采样率


class BoundaryFusion:
    """三方法边界融合器

    使用示例:
        fusion = BoundaryFusion(FusionConfig())
        fused = fusion.fuse(
            silero_segments, ffmpeg_segments, audio, sample_rate
        )
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._sample_rate = self.config.sample_rate

    def fuse(
        self,
        silero_segments: List[SpeechSegment],
        ffmpeg_segments: List[SpeechSegment],
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[SpeechSegment]:
        """融合三种检测结果

        Args:
            silero_segments: Silero VAD 检测结果
            ffmpeg_segments: ffmpeg silencedetect 检测结果
            audio: 音频数组
            sample_rate: 采样率

        Returns:
            融合后的高置信度语音段列表
        """
        cfg = self.config
        total_duration = len(audio) / sample_rate

        # Step 1: 生成 RMS 能量检测结果（第三种方法）
        rms_segments = self._detect_by_rms_energy(
            audio, sample_rate,
            min_speech_duration=cfg.min_speech_duration,
        )

        # Step 2: 时间轴离散化 + 投票（整数采样点索引）
        grid_samples = int(cfg.grid_resolution * sample_rate)
        num_bins = len(audio) // grid_samples + 1
        votes = np.zeros(num_bins, dtype=np.int8)

        self._vote_by_sample_index(
            votes, silero_segments, sample_rate, grid_samples,
        )
        self._vote_by_sample_index(
            votes, ffmpeg_segments, sample_rate, grid_samples,
        )
        self._vote_by_sample_index(
            votes, rms_segments, sample_rate, grid_samples,
        )

        # Step 3: 多数决 → 高/低置信度语音网格
        speech_mask = votes >= cfg.min_consensus
        high_conf_mask = votes >= 3  # 三方法全票

        # Step 4: 连续语音网格 → 语音段
        fused = self._mask_to_segments(
            speech_mask, high_conf_mask,
            cfg.grid_resolution, total_duration,
            cfg.min_speech_duration,
        )

        # Step 5: 根据置信度应用不同的 padding
        fused = self._apply_adaptive_padding(
            fused, high_conf_padding=cfg.high_conf_padding,
            low_conf_padding=cfg.low_conf_padding,
        )

        logger.info(
            "Fusion: Silero=%d + FFmpeg=%d + RMS=%d → %d segments",
            len(silero_segments), len(ffmpeg_segments),
            len(rms_segments), len(fused),
        )
        return fused

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _to_sample_index(self, time_sec: float) -> int:
        """时间 → 采样点索引（整数）"""
        return int(time_sec * self._sample_rate)

    def _to_time(self, sample_idx: int) -> float:
        """采样点索引 → 时间"""
        return sample_idx / self._sample_rate

    def _vote_by_sample_index(
        self,
        votes: np.ndarray,
        segments: List[SpeechSegment],
        sample_rate: int,
        grid_samples: int,
    ) -> None:
        """使用整数采样点索引投票，规避浮点精度问题"""
        for seg in segments:
            start_idx = self._to_sample_index(seg.start) // grid_samples
            end_idx = (
                self._to_sample_index(seg.end) + grid_samples - 1
            ) // grid_samples
            start_idx = max(0, start_idx)
            end_idx = min(len(votes), end_idx)
            votes[start_idx:end_idx] += 1

    @staticmethod
    def _detect_by_rms_energy(
        audio: np.ndarray,
        sample_rate: int,
        speech_threshold_ratio: float = 3.0,
        min_speech_duration: float = 0.25,
        frame_ms: int = 10,
        hop_ms: int = 5,
    ) -> List[SpeechSegment]:
        """基于 RMS 能量的纯能量语音检测

        比 ffmpeg silencedetect 更细粒度（10ms 帧 + 5ms hop）。
        """
        from ..utils.audio_utils import AudioUtils

        total_samples = len(audio)
        frame_size = int(frame_ms / 1000 * sample_rate)
        hop_size = int(hop_ms / 1000 * sample_rate)
        if frame_size < 1:
            return []

        # 估计静音 RMS 阈值
        silence_rms = AudioUtils.estimate_silence_rms(
            audio, sample_rate, percentile=20,
        )
        speech_threshold = max(silence_rms * speech_threshold_ratio, 1e-6)

        # 逐帧计算 RMS
        frame_times = []
        is_speech = []
        for i in range(0, total_samples - frame_size + 1, hop_size):
            frame = audio[i: i + frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            frame_times.append(i / sample_rate)
            is_speech.append(rms > speech_threshold)

        if not is_speech:
            return []

        # 连续语音帧 → 语音段
        segments = []
        in_speech = False
        seg_start = 0.0

        for i, speech in enumerate(is_speech):
            t = frame_times[i]
            if speech and not in_speech:
                seg_start = t
                in_speech = True
            elif not speech and in_speech:
                if t - seg_start >= min_speech_duration:
                    segments.append(SpeechSegment(
                        start=seg_start, end=t, confidence=0.7,
                    ))
                in_speech = False

        if in_speech:
            final_t = total_samples / sample_rate
            if final_t - seg_start >= min_speech_duration:
                segments.append(SpeechSegment(
                    start=seg_start, end=final_t, confidence=0.7,
                ))

        return segments

    def _mask_to_segments(
        self,
        speech_mask: np.ndarray,
        high_conf_mask: np.ndarray,
        resolution: float,
        total_duration: float,
        min_duration: float,
    ) -> List[SpeechSegment]:
        """连续语音网格 → SpeechSegment 列表"""
        segments = []
        in_speech = False
        seg_start = 0.0
        seg_high_conf = True

        for i, is_speech in enumerate(speech_mask):
            t = i * resolution
            if is_speech and not in_speech:
                seg_start = t
                in_speech = True
                seg_high_conf = True
            elif is_speech and in_speech:
                if not high_conf_mask[i]:
                    seg_high_conf = False
            elif not is_speech and in_speech:
                duration = t - seg_start
                if duration >= min_duration:
                    confidence = 0.95 if seg_high_conf else 0.65
                    segments.append(SpeechSegment(
                        start=seg_start, end=t,
                        confidence=confidence,
                    ))
                in_speech = False

        if in_speech:
            duration = total_duration - seg_start
            if duration >= min_duration:
                segments.append(SpeechSegment(
                    start=seg_start, end=total_duration,
                    confidence=0.95 if seg_high_conf else 0.65,
                ))

        return segments

    def _apply_adaptive_padding(
        self,
        segments: List[SpeechSegment],
        high_conf_padding: float,
        low_conf_padding: float,
    ) -> List[SpeechSegment]:
        """根据置信度应用不同 padding"""
        for seg in segments:
            is_high_conf = seg.confidence >= 0.9
            pad = high_conf_padding if is_high_conf else low_conf_padding

            seg.start = max(0.0, seg.start - pad)
            # end 的边界由调用者控制（需要知道总时长）

        return segments
