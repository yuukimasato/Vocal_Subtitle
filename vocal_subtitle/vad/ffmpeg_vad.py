"""ffmpeg silencedetect VAD 引擎

基于 ffmpeg silencedetect 滤波器的语音活动检测。
纯能量阈值判断，与神经网络 VAD 互补。

优势：
- 采样级精度 (~1ms)，远超神经网络帧级精度 (~32ms)
- 零模型加载开销
- C 实现，极低计算成本
- 不受 SOCKS 代理影响

劣势：
- 纯能量判断，不区分人声与乐器残留
- 低音量语音可能丢失

推荐用法：与 Silero VAD 并行使用，通过 BoundaryFusion 取优。

协议: 通过 CLI 子进程调用 ffmpeg（GPL），不构成衍生作品。
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import SpeechSegment, VADEngine

logger = logging.getLogger(__name__)


class FFmpegSilenceVAD(VADEngine):
    """ffmpeg silencedetect 引擎

    使用 ffmpeg 内置 silencedetect 滤波器检测语音区间。
    """

    def __init__(self):
        self._sample_rate = 16000

    @property
    def name(self) -> str:
        return "ffmpeg-silence"

    def load_model(self) -> None:
        """ffmpeg 无需加载模型"""
        pass

    def detect(
        self,
        audio_path: Path,
        threshold: float = -35.0,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """检测音频文件中的语音区间

        Args:
            audio_path: 音频文件路径
            threshold: 静音阈值 (dB)，默认 -35dB
            min_speech_duration_ms: 最小语音时长
            min_silence_duration_ms: 最小静音时长

        Returns:
            SpeechSegment 列表
        """
        silence_intervals = self._detect_silence(
            audio_path,
            noise_db=threshold,
            min_silence_duration=min_silence_duration_ms / 1000.0,
        )

        total_duration = self._get_duration(audio_path)
        speech_segments = self._invert_intervals(
            silence_intervals, total_duration,
            min_speech_duration=min_speech_duration_ms / 1000.0,
        )

        return [
            SpeechSegment(start=s, end=e, confidence=0.9)
            for s, e in speech_segments
        ]

    def detect_on_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = -35.0,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """在 numpy 数组上检测（通过临时 WAV 文件）

        为避免重复写文件，生产环境建议直接调用 detect()。
        """
        import tempfile

        from ..utils.audio_utils import AudioUtils

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            AudioUtils.save_audio(audio, tmp_path, sample_rate)
            return self.detect(
                tmp_path,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_silence(
        audio_path: Path,
        noise_db: float = -35.0,
        min_silence_duration: float = 0.4,
    ) -> List[Tuple[float, float]]:
        """调用 ffmpeg silencedetect 检测静音区间

        Returns:
            [(silence_start, silence_end), ...] 单位：秒
        """
        cmd = [
            "ffmpeg", "-y", "-loglevel", "info",
            "-i", str(audio_path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_silence_duration}",
            "-f", "null", "-",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg silencedetect timed out (>120s)")
            return []
        except FileNotFoundError:
            logger.error("ffmpeg not found in PATH")
            return []

        # 解析 stderr 中的 silencedetect 输出
        # silence_start: 12.5
        # silence_end: 15.2 | silence_duration: 2.7
        intervals = []
        current_start = None

        for line in result.stderr.split("\n"):
            start_match = re.search(r"silence_start:\s*([\d.]+)", line)
            end_match = re.search(r"silence_end:\s*([\d.]+)", line)

            if start_match:
                current_start = float(start_match.group(1))
            elif end_match and current_start is not None:
                end_time = float(end_match.group(1))
                intervals.append((current_start, end_time))
                current_start = None

        # 处理末尾未闭合的静音（一直静音到文件结尾）
        if current_start is not None:
            total_dur = FFmpegSilenceVAD._get_duration(audio_path)
            intervals.append((current_start, total_dur))

        logger.info(
            "ffmpeg silencedetect (%.0fdB, min=%.2fs): %d silence intervals",
            noise_db, min_silence_duration, len(intervals),
        )
        return intervals

    @staticmethod
    def _invert_intervals(
        silence_intervals: List[Tuple[float, float]],
        total_duration: float,
        min_speech_duration: float = 0.25,
    ) -> List[Tuple[float, float]]:
        """反转静音区间为语音区间

        Args:
            silence_intervals: [(start, end), ...] 静音区间
            total_duration: 音频总时长
            min_speech_duration: 最小语音段时长（过滤过短段）

        Returns:
            [(speech_start, speech_end), ...]
        """
        if not silence_intervals:
            return [(0.0, total_duration)]

        speech = []
        prev_end = 0.0

        for s_start, s_end in silence_intervals:
            if s_start - prev_end >= min_speech_duration:
                speech.append((prev_end, s_start))
            prev_end = s_end

        # 最后一段（最后一段静音结束后）
        if total_duration - prev_end >= min_speech_duration:
            speech.append((prev_end, total_duration))

        return speech

    @staticmethod
    def _get_duration(audio_path: Path) -> float:
        """获取音频时长（秒）"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            # 回退：通过 WAV 头解析
            import wave
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    return wf.getnframes() / wf.getframerate()
            except Exception:
                return 0.0


def unified_ffmpeg_pass(
    audio_path: Path,
    noise_db: float = -40.0,
    min_silence_duration: float = 0.1,
) -> Dict:
    """统一的 ffmpeg silencedetect 调用

    跑一次 ffmpeg（敏感模式），产出两种粒度的结果：
    - skeleton: 方案七用的完整声学骨架（所有 >0.1s 静音反转）
    - coarse_speech: 方案一用的粗粒度语音段（过滤后 >0.4s 静音反转）
    - raw_silence_intervals: 原始静音区间

    避免方案一和方案七各自调用一次 ffmpeg。
    """
    raw_intervals = FFmpegSilenceVAD._detect_silence(
        audio_path, noise_db=noise_db,
        min_silence_duration=min_silence_duration,
    )
    total_duration = FFmpegSilenceVAD._get_duration(audio_path)

    return {
        # 方案七用：所有 >0.1s 的静音 → 反转得到完整声学骨架
        "skeleton": FFmpegSilenceVAD._invert_intervals(
            raw_intervals, total_duration, min_speech_duration=0.05,
        ),
        # 方案一用：过滤后 >0.4s 的静音 → 反转得到 VAD 级语音段
        "coarse_speech": [
            SpeechSegment(start=s, end=e, confidence=0.9)
            for s, e in FFmpegSilenceVAD._invert_intervals(
                [(s, e) for s, e in raw_intervals if e - s >= 0.4],
                total_duration, min_speech_duration=0.25,
            )
        ],
        # 原始静音区间
        "raw_silence_intervals": raw_intervals,
        "total_duration": total_duration,
    }
