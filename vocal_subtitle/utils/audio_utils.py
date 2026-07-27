"""音频工具模块

提供音频格式转换、标准化、加载/保存等基础操作。
基于 pydub + ffmpeg 实现（pydub 为可选依赖，仅在需要加载非 WAV 格式时使用）。
"""

import logging
import struct
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioUtils:
    """音频处理工具集

    所有处理方法的输出统一为标准格式：
        采样率：16000 Hz
        声道：mono
        格式：WAV (PCM 16-bit)

    pydub 仅在加载非 WAV 格式或导出非 WAV 格式时延迟导入。
    """

    # 标准输出参数
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
    DEFAULT_FORMAT = "wav"

    @staticmethod
    def _get_pydub():
        """延迟导入 pydub"""
        import pydub
        return pydub

    @classmethod
    def load_audio(
        cls, audio_path: Path, target_sr: int = DEFAULT_SAMPLE_RATE
    ) -> Tuple[np.ndarray, int]:
        """加载音频文件并返回 numpy 数组

        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率，默认 16000

        Returns:
            (audio_array, sample_rate) — audio_array 为 float32 归一化到 [-1, 1]
        """
        suffix = audio_path.suffix.lower()

        # WAV 文件直接使用标准库读取
        if suffix == ".wav":
            return cls._load_wav(audio_path, target_sr)

        # 其他格式使用 pydub
        pydub = cls._get_pydub()
        audio = pydub.AudioSegment.from_file(str(audio_path))
        audio = audio.set_channels(cls.DEFAULT_CHANNELS)
        audio = audio.set_frame_rate(target_sr)
        audio = audio.set_sample_width(cls.DEFAULT_SAMPLE_WIDTH)

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (cls.DEFAULT_SAMPLE_WIDTH * 8 - 1)

        return samples, target_sr

    @classmethod
    def _load_wav(
        cls, wav_path: Path, target_sr: int
    ) -> Tuple[np.ndarray, int]:
        """使用标准库加载 WAV 文件（无需 pydub）

        如果 wave.open 失败（例如扩展名为 .wav 但实际是其他格式，
        如 MP4 视频），自动回退到 pydub（内部使用 ffmpeg）。
        """
        try:
            with wave.open(str(wav_path), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
        except wave.Error:
            logger.warning(
                "File %s has .wav extension but is not a valid WAV file; "
                "falling back to pydub/ffmpeg for format detection.",
                wav_path,
            )
            return cls._load_with_pydub(wav_path, target_sr)

        # 解析样本
        if sample_width == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32)
        elif sample_width == 4:
            fmt = f"<{n_frames * n_channels}i"
            samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32)
        else:
            # 回退到 pydub
            return cls._load_with_pydub(wav_path, target_sr)

        samples = samples / (2 ** (sample_width * 8 - 1))

        # 转为单声道
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # 重采样（简单线性插值）
        if framerate != target_sr:
            samples = cls._resample(samples, framerate, target_sr)

        return samples.astype(np.float32), target_sr

    @classmethod
    def _load_with_pydub(
        cls, audio_path: Path, target_sr: int
    ) -> Tuple[np.ndarray, int]:
        """使用 pydub 加载（回退方案）"""
        pydub = cls._get_pydub()
        audio = pydub.AudioSegment.from_file(str(audio_path))
        audio = audio.set_channels(cls.DEFAULT_CHANNELS)
        audio = audio.set_frame_rate(target_sr)
        audio = audio.set_sample_width(cls.DEFAULT_SAMPLE_WIDTH)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= 32768.0
        return samples, target_sr

    @staticmethod
    def _resample(
        audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """高质量重采样（优先 scipy，回退线性插值）。

        使用 scipy.signal.resample_poly 进行多相滤波重采样，
        避免线性插值的混叠伪影和相位偏移。
        """
        if orig_sr == target_sr:
            return audio

        # 优先使用 scipy 高质量重采样
        try:
            from scipy.signal import resample_poly

            # 计算互质的重采样因子
            from math import gcd

            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g

            # 限制因子上限防止内存爆炸（scipy 内部处理大因子较慢）
            max_factor = 500
            if up > max_factor or down > max_factor:
                # 分两步重采样：先到公约数倍数，再到目标
                result = audio.astype(np.float64)
                result = resample_poly(result, up=target_sr, down=orig_sr,
                                       window=("kaiser", 5.0))
                return result.astype(np.float32)

            result = resample_poly(
                audio.astype(np.float64), up=up, down=down,
                window=("kaiser", 5.0),
            )
            return result.astype(np.float32)

        except ImportError:
            pass

        # 回退：简单线性插值
        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_out)
        idx_lo = np.floor(indices).astype(int)
        idx_hi = np.minimum(idx_lo + 1, len(audio) - 1)
        frac = indices - idx_lo
        return (audio[idx_lo] * (1 - frac) + audio[idx_hi] * frac).astype(np.float32)

    @classmethod
    def save_audio(
        cls,
        audio: np.ndarray,
        output_path: Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> Path:
        """保存 numpy 数组为 WAV 文件

        Args:
            audio: float32 音频数组，归一化到 [-1, 1]
            output_path: 输出路径
            sample_rate: 采样率

        Returns:
            输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换回 int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(cls.DEFAULT_CHANNELS)
            wf.setsampwidth(cls.DEFAULT_SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return output_path

    @classmethod
    def normalize_audio(
        cls,
        audio: np.ndarray,
        target_sr: int = DEFAULT_SAMPLE_RATE,
        target_channels: int = DEFAULT_CHANNELS,
    ) -> np.ndarray:
        """标准化音频数组

        统一采样率、声道数，并将值域归一化到 [-1, 1]。

        Args:
            audio: 输入音频数组
            target_sr: 目标采样率（暂未实现重采样，保留参数）
            target_channels: 目标声道数

        Returns:
            标准化后的音频数组
        """
        # 确保是 float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)

        # 如果是立体声，转单声道
        if audio.ndim > 1 and audio.shape[0] == 2:
            audio = audio.mean(axis=0)

        # 峰值归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio.astype(np.float32)

    @classmethod
    def get_duration_seconds(cls, audio_path: Path) -> float:
        """获取音频时长（秒）"""
        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    return wf.getnframes() / wf.getframerate()
            except wave.Error:
                logger.warning(
                    "File %s has .wav extension but is not a valid WAV; "
                    "falling back to pydub.",
                    audio_path,
                )
        pydub = cls._get_pydub()
        audio = pydub.AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0

    @classmethod
    def get_audio_info(cls, audio_path: Path) -> dict:
        """获取音频文件元信息

        Returns:
            dict: channels, sample_rate, sample_width, duration_seconds, format
        """
        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    return {
                        "channels": wf.getnchannels(),
                        "sample_rate": wf.getframerate(),
                        "sample_width": wf.getsampwidth(),
                        "duration_seconds": wf.getnframes() / wf.getframerate(),
                        "format": suffix.lstrip("."),
                    }
            except wave.Error:
                logger.warning(
                    "File %s has .wav extension but is not a valid WAV; "
                    "falling back to pydub.",
                    audio_path,
                )
        pydub = cls._get_pydub()
        audio = pydub.AudioSegment.from_file(str(audio_path))
        return {
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
            "duration_seconds": len(audio) / 1000.0,
            "format": suffix.lstrip("."),
        }

    @classmethod
    def extract_segment(
        cls,
        audio: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> np.ndarray:
        """从音频数组中提取片段

        Args:
            audio: 音频数组
            start_sample: 起始采样点
            end_sample: 结束采样点

        Returns:
            片段音频数组
        """
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        return audio[start_sample:end_sample]

    @classmethod
    def time_to_sample(
        cls, time_seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> int:
        """将时间（秒）转换为采样点数"""
        return int(time_seconds * sample_rate)

    @classmethod
    def sample_to_time(
        cls, num_samples: int, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> float:
        """将采样点数转换为时间（秒）"""
        return num_samples / sample_rate

    # ------------------------------------------------------------------
    # 语音边界精修（提升字幕时间轴精度）
    # ------------------------------------------------------------------

    @classmethod
    def refine_speech_boundaries(
        cls,
        segments: list,  # List[SpeechSegment]
        audio: np.ndarray,
        sample_rate: int,
        window: float = 0.15,
        speech_silence_ratio: float = 3.0,
    ) -> list:
        """在 VAD 边界附近用 RMS 能量扫描精修语音起止点。

        原理：VAD 以 32ms 帧为单位，边界精度 ±16ms；且模型需要
        若干帧才能触发，实际 onset 可能偏晚。本方法在 VAD 边界
        附近 ±window 范围内用 10ms 帧长扫描 RMS 能量，找到实际
        语音能量跃迁点，将边界精度提升到 ~5ms 级别。

        Args:
            segments: VAD 检测出的 SpeechSegment 列表
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率
            window: 扫描窗口大小（秒），默认 0.15s
            speech_silence_ratio: 语音/静音 RMS 比值阈值，默认 3.0

        Returns:
            精修后的 SpeechSegment 列表（原地修改 + 返回引用）
        """
        if not segments or audio is None:
            return segments

        total_samples = len(audio)
        if total_samples == 0:
            return segments

        frame_size = int(0.01 * sample_rate)  # 10ms 帧
        if frame_size < 1:
            return segments
        hop = max(1, frame_size // 2)  # 5ms hop

        # 采样全局 RMS 分布（每 100ms 取一帧），估计静音阈值
        global_rms_samples = []
        for i in range(0, total_samples - frame_size, int(0.1 * sample_rate)):
            frame = audio[i : i + frame_size]
            global_rms_samples.append(float(np.sqrt(np.mean(frame ** 2))))
        if not global_rms_samples:
            return segments
        global_rms_samples.sort()
        silence_rms = global_rms_samples[max(0, int(len(global_rms_samples) * 0.2))]
        speech_threshold = max(silence_rms * speech_silence_ratio, 1e-6)

        window_samples = int(window * sample_rate)
        half_window = window / 2.0

        for seg in segments:
            # ---- 精修 onset（语音开始） ----
            center_sample = int(seg.start * sample_rate)
            search_start = max(0, center_sample - window_samples)
            search_end = min(total_samples, center_sample + window_samples)

            refined_onset = cls._find_energy_transition(
                audio, search_start, search_end, frame_size, hop,
                speech_threshold, sample_rate, direction="onset",
            )
            if refined_onset is not None:
                if abs(refined_onset - seg.start) <= half_window:
                    seg.start = refined_onset

            # ---- 精修 offset（语音结束） ----
            center_sample = int(seg.end * sample_rate)
            search_start = max(0, center_sample - window_samples)
            search_end = min(total_samples, center_sample + window_samples)

            refined_offset = cls._find_energy_transition(
                audio, search_start, search_end, frame_size, hop,
                speech_threshold, sample_rate, direction="offset",
            )
            if refined_offset is not None:
                if abs(refined_offset - seg.end) <= half_window:
                    seg.end = refined_offset

        logger.info(
            "Boundary refinement: %d segments processed (threshold=%.6f)",
            len(segments), speech_threshold,
        )
        return segments

    @staticmethod
    def _find_energy_transition(
        audio: np.ndarray,
        search_start: int,
        search_end: int,
        frame_size: int,
        hop: int,
        threshold: float,
        sample_rate: int,
        direction: str,
    ) -> Optional[float]:
        """在搜索范围内找到语音能量跃迁点。

        Args:
            direction: "onset" 找能量上升点（静音→语音），
                       "offset" 找能量下降点（语音→静音）

        Returns:
            跃迁点时间（秒），未找到则返回 None
        """
        if search_end <= search_start + frame_size:
            return None

        frames = []  # (sample_index, rms)
        for i in range(search_start, search_end - frame_size + 1, hop):
            frame = audio[i : i + frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            frames.append((i, rms))

        if not frames:
            return None

        if direction == "onset":
            # 从前向后：找第一个 RMS > threshold 的帧
            for sample_idx, rms in frames:
                if rms > threshold:
                    return sample_idx / sample_rate
        else:
            # 从后向前：找最后一个 RMS > threshold 的帧
            for sample_idx, rms in reversed(frames):
                if rms > threshold:
                    # 返回该帧结束时间（帧起始 + 帧长）
                    return (sample_idx + frame_size) / sample_rate

        return None

    @classmethod
    def get_segment_rms(
        cls,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int,
    ) -> float:
        """计算音频片段内的 RMS 能量（用于间隙静音验证）。

        Args:
            audio: 音频数据
            start_time: 起始时间（秒）
            end_time: 结束时间（秒）
            sample_rate: 采样率

        Returns:
            RMS 值（0.0 表示纯静音）
        """
        start_sample = max(0, int(start_time * sample_rate))
        end_sample = min(len(audio), int(end_time * sample_rate))
        if end_sample <= start_sample:
            return 0.0
        segment = audio[start_sample:end_sample]
        return float(np.sqrt(np.mean(segment ** 2)))

    @classmethod
    def estimate_silence_rms(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        percentile: float = 20.0,
    ) -> float:
        """估算音频的静音 RMS 阈值。

        在整个音频中采样，取底部 percentile% 的 RMS 中位数。

        Args:
            audio: 音频数据
            sample_rate: 采样率
            percentile: 用于静音估计的百分位

        Returns:
            静音 RMS 阈值
        """
        total_samples = len(audio)
        frame_size = int(0.01 * sample_rate)
        if frame_size < 1 or total_samples < frame_size:
            return 0.001

        rms_samples = []
        for i in range(0, total_samples - frame_size, int(0.1 * sample_rate)):
            frame = audio[i : i + frame_size]
            rms_samples.append(float(np.sqrt(np.mean(frame ** 2))))

        if not rms_samples:
            return 0.001

        rms_samples.sort()
        idx = max(0, int(len(rms_samples) * percentile / 100.0))
        raw_estimate = rms_samples[idx] if rms_samples else 0.001
        # 绝对最小阈值：16-bit 音频归一化到 [-1,1] 后，
        # -60dB 约等于 0.001，-80dB 约等于 0.0001
        # 取 -72dB (≈0.00025) 作为绝对下限，防止极干净音频阈值过低
        absolute_min = 0.00025
        return max(raw_estimate, absolute_min)

    @classmethod
    def estimate_noise_floor_per_chunk(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        chunk_duration: Optional[float] = None,
    ) -> dict:
        """每个处理单元独立采样环境底噪。

        在音频块中寻找最安静但非纯静音的区域采样底噪，
        使用鲁棒中位数估计，适应不同录音环境。

        策略：
        1. 优先在块首/块尾采样（通常接近静音但可能有底噪）
        2. 如果首/尾是纯静音（RMS=0），扫描全块找最安静的 1 秒
        3. 回退到全局百分位估计

        嘈杂环境（底噪高）→ 自动提高语音检测阈值
        安静环境（底噪低）→ 自动降低语音检测阈值

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率
            chunk_duration: 块时长（秒），为 None 时使用实际音频时长

        Returns:
            dict: {
                "noise_rms": float,           # 底噪 RMS 估计值
                "speech_threshold": float,    # 建议的语音判定阈值
                "is_noisy_environment": bool, # 是否嘈杂环境
            }
        """
        total_duration = len(audio) / sample_rate if sample_rate > 0 else 0
        if chunk_duration is None:
            chunk_duration = total_duration

        frame_size = int(0.01 * sample_rate)  # 10ms
        # 绝对最小底噪值：-72dB ≈ 0.00025（16-bit 归一化后）
        min_noise = 0.00025

        if frame_size < 1:
            return {
                "noise_rms": 0.001,
                "speech_threshold": 0.003,
                "is_noisy_environment": False,
            }

        def robust_noise_rms(segment: np.ndarray) -> Optional[float]:
            """鲁棒底噪估计。

            返回 None 表示该段为纯静音，无法用于估计。
            """
            total_rms = float(np.sqrt(np.mean(segment ** 2)))
            # 纯静音检测：总体 RMS 极低 → 跳过
            if total_rms < min_noise:
                return None
            if len(segment) < frame_size:
                return total_rms

            rms_vals = []
            for i in range(0, len(segment) - frame_size + 1, frame_size):
                frame = segment[i : i + frame_size]
                rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))
            if not rms_vals:
                return None

            rms_vals.sort()
            # 取底部 20%-50% 区间的中位数（鲁棒估计，排除偶发杂音）
            lo = max(0, int(len(rms_vals) * 0.2))
            hi = min(len(rms_vals), int(len(rms_vals) * 0.5) + 1)
            robust = rms_vals[lo:hi]
            result = float(np.median(robust)) if robust else rms_vals[len(rms_vals)//2]
            return result if result > min_noise else None

        noise_rms = None

        # 方式 1：优先在块首/块尾采样
        head_samples = int(min(0.5, total_duration * 0.05) * sample_rate)
        tail_start = int(max(0, total_duration - 0.5) * sample_rate)

        if head_samples > 0:
            head_result = robust_noise_rms(audio[:head_samples])
            if head_result is not None:
                noise_rms = head_result

        if tail_start < len(audio):
            tail_result = robust_noise_rms(audio[tail_start:])
            if tail_result is not None:
                noise_rms = (
                    min(noise_rms, tail_result)
                    if noise_rms is not None
                    else tail_result
                )

        # 方式 2：首尾都是纯静音 → 扫描全块找最安静的 1 秒区间
        if noise_rms is None:
            logger.debug(
                "Head/tail are pure silence, scanning for quietest non-silent region"
            )
            window_size = int(1.0 * sample_rate)  # 1 秒窗口
            step = window_size // 2
            min_window_rms = float("inf")
            best_window = None

            for start in range(0, len(audio) - window_size + 1, step):
                window = audio[start : start + window_size]
                w_rms = float(np.sqrt(np.mean(window ** 2)))
                if min_noise < w_rms < min_window_rms:
                    min_window_rms = w_rms
                    best_window = window

            if best_window is not None:
                result = robust_noise_rms(best_window)
                if result is not None:
                    noise_rms = result

        # 方式 3：全块扫描也没找到 → 回退到全局百分位估计
        if noise_rms is None:
            noise_rms = cls.estimate_silence_rms(audio, sample_rate, percentile=10.0)
            logger.debug(
                "Fallback to global silence estimate: %.6f", noise_rms
            )

        noise_rms = max(noise_rms, min_noise)

        # 判定环境类型：RMS > 0.02 视为嘈杂环境
        is_noisy = noise_rms > 0.02

        # 嘈杂环境用较高的语音/静音比值（减少误检）
        speech_ratio = 4.0 if is_noisy else 2.5

        logger.info(
            "Noise floor: %.6f RMS (%s environment, speech_ratio=%.1f)",
            noise_rms,
            "noisy" if is_noisy else "quiet",
            speech_ratio,
        )

        return {
            "noise_rms": noise_rms,
            "speech_threshold": max(noise_rms * speech_ratio, 0.001),
            "is_noisy_environment": is_noisy,
        }

    @classmethod
    def convert_format(
        cls, input_path: Path, output_path: Path, target_format: str = "wav"
    ) -> Path:
        """转换音频文件格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_format: 目标格式 (wav / mp3 / flac ...)

        Returns:
            输出文件路径
        """
        pydub = cls._get_pydub()
        audio = pydub.AudioSegment.from_file(str(input_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(output_path), format=target_format)
        return output_path

    # ---- 视频音频提取 ----

    # 常见视频格式扩展名
    VIDEO_EXTENSIONS = {
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv",
        ".wmv", ".m4v", ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
        ".rmvb", ".vob", ".divx",
    }

    # 常见视频容器格式的魔数字节签名
    VIDEO_MAGIC = {
        b"\x00\x00\x00\x18ftyp": ".mp4",       # ISO Base Media (MP4/M4V/MOV/3GP)
        b"\x00\x00\x00\x20ftyp": ".mp4",       # ISO BMFF variant header size
        b"\x1aE\xdf\xa3":       ".mkv",         # Matroska / WebM EBML
        b"RIFF":                None,            # AVI = RIFF container — check next
        b"\x00\x00\x01\xba":    ".mpg",          # MPEG-PS
        b"\x00\x00\x01\xb3":    ".mpg",          # MPEG
        b"\x47":                ".ts",           # MPEG-TS
        b"FLV":                 ".flv",          # Flash Video
    }

    @classmethod
    def _has_video_magic(cls, path: Path) -> bool:
        """检测文件头部是否为已知的视频容器格式（魔数检查）。

        避免依赖、仅读取前 12 字节，用于判断文件是否为视频容器
        （无论扩展名是否匹配）。
        """
        try:
            with open(path, "rb") as f:
                header = f.read(12)
        except OSError:
            return False

        if len(header) < 8:
            return False

        for magic, ext in cls.VIDEO_MAGIC.items():
            if not header.startswith(magic):
                continue
            if ext is not None:
                return True
            # RIFF 可能是 AVI — 检查 FOURCC（偏移 8 的 "AVI "）
            if magic == b"RIFF" and len(header) >= 12:
                return header[8:12] == b"AVI "
        return False

    @classmethod
    def is_video_file(cls, path: Path) -> bool:
        """判断是否为视频文件（先查扩展名，再查文件头魔数）。

        两层检查自动处理扩展名与实际格式不匹配的情况
        （例如 .wav 命名的 MP4 视频）。
        """
        if path.suffix.lower() in cls.VIDEO_EXTENSIONS:
            return True
        return cls._has_video_magic(path)

    @classmethod
    def extract_audio_from_video(
        cls,
        video_path: Path,
        output_dir: Path,
        output_name: str = "extracted_audio",
    ) -> Path:
        """从视频文件中提取音轨，转为标准 WAV 格式。

        使用 ffmpeg 直接提取，避免 pydub 加载整个文件到内存。
        输出: 16kHz / mono / 16-bit PCM WAV。

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            output_name: 输出文件名（不含扩展名）

        Returns:
            提取后的 WAV 文件路径
        """
        import subprocess

        output_path = output_dir / f"{output_name}.wav"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 使用 ffmpeg 提取音频流，重采样到 16kHz mono
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn",                    # 丢弃视频流
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", str(cls.DEFAULT_SAMPLE_RATE),  # 16kHz
            "-ac", str(cls.DEFAULT_CHANNELS),     # mono
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"音频提取超时（>10分钟）: {video_path.name}"
            )
        except subprocess.CalledProcessError as e:
            # ffmpeg 失败时，尝试 pydub 降级方案
            import logging
            logging.getLogger(__name__).warning(
                "ffmpeg extraction failed, trying pydub fallback: %s", e
            )
            pydub = cls._get_pydub()
            audio = pydub.AudioSegment.from_file(str(video_path))
            audio = audio.set_frame_rate(cls.DEFAULT_SAMPLE_RATE)
            audio = audio.set_channels(cls.DEFAULT_CHANNELS)
            audio = audio.set_sample_width(cls.DEFAULT_SAMPLE_WIDTH)
            audio.export(str(output_path), format="wav")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"音频提取失败，输出文件为空: {video_path.name}"
            )

        return output_path
