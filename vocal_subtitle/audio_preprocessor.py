"""音频预处理管线 — 前置降噪模块 (文档 5.12.1)

在 VAD 处理前对音频进行降噪，减少突发噪音和稳态底噪对下游
检测模块（ffmpeg silence、Silero VAD、RMS 能量扫描）的干扰。

降噪引擎（按复杂度递增）:
- spectral_gate: 零模型依赖，纯信号处理，开箱即用（默认）
- rnnoise: 轻量 RNN 降噪，适合实时处理
- deepfilternet: 最高降噪质量，适合嘈杂环境

突发噪音抑制:
- 检测 <200ms 的突发高能事件（关门、键盘、物品坠落）
- 用邻域插值替代，防止被 ffmpeg 误判为语音段

执行时机: 宏观切块（方案〇）之前，"第 -1 级"。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------


@dataclass
class DenoiseConfig:
    """前置降噪配置

    Attributes:
        enabled: 是否启用降噪（默认关闭，纯净录音不需要）
        engine: 降噪引擎 ("spectral_gate" | "rnnoise" | "deepfilternet")
        spectral_noise_reduction_db: 谱减法降噪量 (dB)
        spectral_noise_estimation_frames: 噪声估计帧数
        burst_noise_protection: 是否启用突发噪音保护
        burst_noise_threshold_db: 突发噪音判定阈值 (dB, 超过局部 RMS)
        burst_noise_max_duration_ms: 突发噪音最大持续时长
        output_sample_rate: 输出采样率（保持与输入一致）
    """

    enabled: bool = False
    engine: str = "spectral_gate"             # "spectral_gate" | "rnnoise" | "deepfilternet"

    # 谱减法参数
    spectral_noise_reduction_db: float = 12.0
    spectral_noise_estimation_frames: int = 10

    # 突发噪音保护
    burst_noise_protection: bool = True
    burst_noise_threshold_db: float = 15.0    # 超过局部 RMS 此值视为突发噪音
    burst_noise_max_duration_ms: int = 200

    # 输出
    output_sample_rate: int = 16000


# ------------------------------------------------------------------
# 预处理器
# ------------------------------------------------------------------


class AudioPreprocessor:
    """音频预处理管线（前置降噪）

    使用示例:
        preprocessor = AudioPreprocessor(DenoiseConfig(enabled=True))
        cleaned_audio, report = preprocessor.process(audio, sample_rate)
    """

    def __init__(self, config: Optional[DenoiseConfig] = None):
        self.config = config or DenoiseConfig()

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, Dict]:
        """对输入音频执行降噪

        Args:
            audio: 输入音频 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            (denoised_audio, quality_report)
        """
        cfg = self.config
        report: Dict = {
            "denoise_applied": False,
            "engine": cfg.engine,
            "burst_events_detected": 0,
            "input_rms": float(np.sqrt(np.mean(audio ** 2))),
        }

        if not cfg.enabled:
            report["reason"] = "disabled"
            return audio, report

        if len(audio) == 0:
            report["reason"] = "empty_audio"
            return audio, report

        # Stage 1: 突发噪音检测与抑制
        if cfg.burst_noise_protection:
            audio, burst_count = self._suppress_burst_noise(audio, sample_rate)
            report["burst_events_detected"] = burst_count
            if burst_count > 0:
                logger.info("Suppressed %d burst noise events", burst_count)

        # Stage 2: 稳态降噪
        if cfg.engine == "spectral_gate":
            audio = self._apply_spectral_gate(audio, sample_rate)
        elif cfg.engine == "rnnoise":
            audio = self._apply_rnnoise(audio, sample_rate)
        elif cfg.engine == "deepfilternet":
            audio = self._apply_deepfilternet(audio, sample_rate)
        else:
            logger.warning("Unknown denoise engine: %s, skipping", cfg.engine)
            report["reason"] = f"unknown_engine:{cfg.engine}"
            return audio, report

        report["denoise_applied"] = True
        report["output_rms"] = float(np.sqrt(np.mean(audio ** 2)))
        report["rms_reduction_db"] = round(
            20 * np.log10(max(report["output_rms"], 1e-8)
                          / max(report["input_rms"], 1e-8)), 1,
        )

        logger.info(
            "Denoise complete: engine=%s, rms_reduction=%.1fdB, burst=%d",
            cfg.engine, report["rms_reduction_db"], report["burst_events_detected"],
        )
        return audio, report

    # ------------------------------------------------------------------
    # 突发噪音抑制
    # ------------------------------------------------------------------

    def _suppress_burst_noise(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        """突发噪音检测与抑制

        检测能量在极短时间内（< max_duration）急剧上升后回落的事件，
        识别为突发噪音（关门、键盘等），用局部邻域插值替代。

        Returns:
            (processed_audio, burst_count)
        """
        cfg = self.config
        frame_ms = 5  # 5ms 帧，更精细的检测
        frame_size = int(frame_ms / 1000 * sample_rate)
        if frame_size < 1:
            return audio, 0

        hop = max(1, frame_size // 2)
        total_frames = (len(audio) - frame_size) // hop + 1
        if total_frames < 3:
            return audio, 0

        # 计算逐帧 RMS
        rms_vals = np.zeros(total_frames, dtype=np.float64)
        for i in range(total_frames):
            start = i * hop
            frame = audio[start:start + frame_size]
            rms_vals[i] = float(np.sqrt(np.mean(frame ** 2)))

        # 局部中位数 RMS（鲁棒估计，窗口 500ms）
        median_window = int(0.5 / (frame_ms / 1000))
        if median_window < 3:
            median_window = max(3, total_frames // 4)

        # 计算局部背景 RMS（滚动中位数）
        local_bg_rms = np.zeros(total_frames, dtype=np.float64)
        for i in range(total_frames):
            lo = max(0, i - median_window // 2)
            hi = min(total_frames, i + median_window // 2 + 1)
            window = rms_vals[lo:hi]
            # 排除最高 30% 的值（避免噪音本身污染背景估计）
            threshold = np.percentile(window, 70)
            bg_samples = window[window <= threshold]
            if len(bg_samples) > 0:
                local_bg_rms[i] = float(np.median(bg_samples))
            else:
                local_bg_rms[i] = float(np.median(window))

        # 突发阈值 = 背景 RMS × 10^(threshold_db/20)
        burst_factor = 10 ** (cfg.burst_noise_threshold_db / 20)
        max_burst_frames = int(
            cfg.burst_noise_max_duration_ms / (frame_ms)
        )

        # 检测并标记突发噪音帧
        burst_mask = np.zeros(total_frames, dtype=bool)
        in_burst = False
        burst_start_frame = 0
        burst_count = 0

        for i in range(total_frames):
            is_burst = rms_vals[i] > local_bg_rms[i] * burst_factor

            if is_burst and not in_burst:
                burst_start_frame = i
                in_burst = True
            elif not is_burst and in_burst:
                burst_duration_frames = i - burst_start_frame
                if burst_duration_frames <= max_burst_frames:
                    # 确认为突发噪音 → 标记
                    burst_mask[burst_start_frame:i] = True
                    burst_count += 1
                in_burst = False

        # 处理末尾未闭合的 burst
        if in_burst:
            burst_duration_frames = total_frames - burst_start_frame
            if burst_duration_frames <= max_burst_frames:
                burst_mask[burst_start_frame:] = True
                burst_count += 1

        if burst_count == 0:
            return audio, 0

        # 对标记区域做插值替代
        audio_out = audio.copy()
        i = 0
        while i < total_frames:
            if not burst_mask[i]:
                i += 1
                continue

            # 找到连续 burst 区域的起止
            burst_start = i
            while i < total_frames and burst_mask[i]:
                i += 1
            burst_end = i

            # 转换为采样点
            sample_start = burst_start * hop
            sample_end = min(len(audio), burst_end * hop + frame_size)

            self._interpolate_burst(audio_out, sample_start, sample_end)

        return audio_out, burst_count

    @staticmethod
    def _interpolate_burst(
        audio: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        """用邻域生成匹配能量的舒适噪声替代突发噪音段

        使用噪音段前后邻域的 RMS 作为目标能量，生成低能量随机噪声。
        避免突兀的完全静音（完全静音反而引人注意）。
        """
        if start >= end or start < 0 or end > len(audio):
            return

        # 取噪音前后的邻域（各 100ms）
        margin = min(1600, (end - start) // 2)  # ~100ms @ 16kHz
        pre_samples = audio[max(0, start - margin):start]
        post_samples = audio[end:min(len(audio), end + margin)]

        # 计算邻域 RMS
        pre_rms = float(np.sqrt(np.mean(pre_samples ** 2))) if len(pre_samples) > 0 else 0.0
        post_rms = float(np.sqrt(np.mean(post_samples ** 2))) if len(post_samples) > 0 else 0.0

        if pre_rms == 0 and post_rms == 0:
            # 两边都是静音 → 直接置零
            audio[start:end] = 0.0
            return

        target_rms = (pre_rms + post_rms) / 2

        # 生成匹配邻域能量的舒适噪声
        burst_len = end - start
        replacement = np.random.randn(burst_len).astype(np.float32) * target_rms * 0.3
        audio[start:end] = replacement

    # ------------------------------------------------------------------
    # 稳态降噪引擎
    # ------------------------------------------------------------------

    def _apply_spectral_gate(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """谱减法降噪（零模型依赖）

        基于短时傅里叶变换 (STFT) 的经典谱减法。
        估计噪声频谱 → 从信号频谱中减去 → 逆变换。
        """
        cfg = self.config

        # STFT 参数
        n_fft = 1024
        hop_length = n_fft // 4  # 75% 重叠

        # 计算 STFT
        try:
            # 使用简单的 numpy STFT
            stft_matrix = self._compute_stft(audio, n_fft, hop_length)
        except Exception as e:
            logger.warning("STFT computation failed: %s, skipping spectral gate", e)
            return audio

        if stft_matrix.size == 0:
            return audio

        # 幅度谱和相位谱
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # 估计噪声：取前 N 帧（假设音频开头为静音/底噪）
        noise_frames = min(cfg.spectral_noise_estimation_frames, magnitude.shape[1])
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # 谱减法
        reduction_factor = 10 ** (cfg.spectral_noise_reduction_db / 20)
        # 过减法：减去 (reduction_factor × noise)，但不下零
        magnitude_reduced = np.maximum(
            magnitude - reduction_factor * noise_profile,
            0.0,  # 频谱floor
        )

        # 逆 STFT
        try:
            reconstructed = self._compute_istft(
                magnitude_reduced * np.exp(1j * phase),
                n_fft, hop_length,
                target_length=len(audio),
            )
        except Exception as e:
            logger.warning("ISTFT failed: %s, returning original", e)
            return audio

        # 确保输出与输入等长且归一化
        if len(reconstructed) > len(audio):
            reconstructed = reconstructed[:len(audio)]
        elif len(reconstructed) < len(audio):
            reconstructed = np.pad(reconstructed, (0, len(audio) - len(reconstructed)))

        # 避免过度放大
        peak = np.max(np.abs(reconstructed))
        if peak > 1.0:
            reconstructed = reconstructed / peak

        return reconstructed.astype(np.float32)

    @staticmethod
    def _compute_stft(
        audio: np.ndarray,
        n_fft: int,
        hop_length: int,
    ) -> np.ndarray:
        """计算短时傅里叶变换（纯 numpy 实现，零依赖）"""
        window = np.hanning(n_fft)

        num_frames = (len(audio) - n_fft) // hop_length + 1
        if num_frames < 1:
            return np.array([[]])

        stft = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex128)

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft[:, i] = spectrum

        return stft

    @staticmethod
    def _compute_istft(
        stft_matrix: np.ndarray,
        n_fft: int,
        hop_length: int,
        target_length: int,
    ) -> np.ndarray:
        """逆短时傅里叶变换（Overlap-Add）"""
        window = np.hanning(n_fft)
        num_frames = stft_matrix.shape[1]

        output = np.zeros(target_length + n_fft, dtype=np.float64)
        window_sum = np.zeros(target_length + n_fft, dtype=np.float64)

        for i in range(num_frames):
            spectrum = stft_matrix[:, i]
            frame = np.fft.irfft(spectrum, n=n_fft)
            frame = frame * window

            start = i * hop_length
            output[start:start + n_fft] += frame
            window_sum[start:start + n_fft] += window ** 2

        # 归一化（避免窗口重叠导致的幅度变化）
        mask = window_sum > 1e-8
        output[mask] /= window_sum[mask]

        return output[:target_length]

    def _apply_rnnoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """RNNoise 降噪（需要 rnnoise 库）

        当前为降级实现：RNNoise 不可用时回退到谱减法。
        """
        try:
            import rnnoise
        except ImportError:
            logger.warning(
                "rnnoise not installed, falling back to spectral_gate. "
                "Install with: pip install rnnoise"
            )
            return self._apply_spectral_gate(audio, sample_rate)

        # RNNoise 需要 16kHz、16bit PCM、10ms 帧（160 samples）
        if sample_rate != 16000:
            logger.warning("RNNoise requires 16kHz, got %dHz. Falling back.", sample_rate)
            return self._apply_spectral_gate(audio, sample_rate)

        denoiser = rnnoise.RNNoise()
        frame_size = 480  # 30ms @ 16kHz (rnnoise 实际使用 10ms 内部帧)

        output = np.zeros_like(audio)
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            # 转为 16-bit PCM
            pcm_frame = (frame * 32767).astype(np.int16)
            denoised_pcm = denoiser.process_frame(pcm_frame.tobytes())
            denoised = np.frombuffer(denoised_pcm, dtype=np.int16).astype(np.float32) / 32767.0
            output[i:i + len(denoised)] = denoised[:len(denoised)]

        return output

    def _apply_deepfilternet(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """DeepFilterNet 降噪（需要 deepfilternet 库）

        当前为降级实现：DeepFilterNet 不可用时回退到谱减法。
        """
        try:
            from deepfilternet import DeepFilterNet
        except ImportError:
            logger.warning(
                "deepfilternet not installed, falling back to spectral_gate. "
                "Install with: pip install deepfilternet"
            )
            return self._apply_spectral_gate(audio, sample_rate)

        try:
            model = DeepFilterNet()
            # DeepFilterNet 接受 (1, T) 或 (T,) 的 float32 张量
            import torch

            audio_tensor = torch.from_numpy(audio.copy()).float().unsqueeze(0)
            with torch.no_grad():
                enhanced = model.enhance(audio_tensor)
            return enhanced.squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            logger.warning("DeepFilterNet failed: %s, falling back to spectral_gate", e)
            return self._apply_spectral_gate(audio, sample_rate)
