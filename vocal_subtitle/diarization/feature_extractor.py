"""音色特征提取器

从语音片段中提取多维声学特征向量，用于说话人聚类。

特征维度 (~100 维):
    MFCC mean/std           13 × 2 = 26  (音色核心表征)
    Delta MFCC mean/std     13 × 2 = 26  (音色动态变化)
    Delta-Delta MFCC mean   13 × 1 = 13  (音色加速度)
    Spectral centroid        1 × 2 = 2   (声音亮度)
    Spectral bandwidth       1 × 2 = 2   (频谱宽度)
    Spectral contrast        7 × 1 = 7   (频谱峰谷对比度)
    Pitch (F0)               1 × 2 = 2   (基频/音高)
    Formants F1, F2          2 × 2 = 4   (共振峰，声道特征)
    Zero-crossing rate       1 × 2 = 2   (信号过零率)
    RMS energy               1 × 2 = 2   (音量)
    Spectral rolloff         1 × 1 = 1   (频谱滚降点)
                            -------------
                            共 ~87 维 (PCA 前)

依赖: librosa (ISC license), numpy, scipy
所有导入均为延迟加载，仅在 diarization.enabled=True 时触发。
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# 最小有效段长（秒），短于此值的片段用零填充
_MIN_SEGMENT_DURATION = 0.25


class FeatureExtractor:
    """声学特征提取器（无状态）

    使用示例:
        extractor = FeatureExtractor()
        features = extractor.extract_features(audio_segment, sample_rate=16000)
        feature_matrix = extractor.extract_features_batch(segments, sample_rate=16000)
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def extract_features(
        self, audio: np.ndarray, sample_rate: int | None = None
    ) -> np.ndarray:
        """提取单个音频片段的特征向量

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率（默认用初始化时的值）

        Returns:
            1D float64 特征向量
        """
        sr = sample_rate or self.sample_rate
        audio = self._preprocess(audio, sr)

        features = []
        features.extend(self._mfcc_features(audio, sr))
        features.extend(self._spectral_features(audio, sr))
        features.extend(self._pitch_features(audio, sr))
        features.extend(self._formant_features(audio, sr))
        features.extend(self._temporal_features(audio, sr))
        features.extend(self._energy_features(audio))

        result = np.array(features, dtype=np.float64)
        # 替换 NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def extract_features_batch(
        self, audio_segments: List[np.ndarray], sample_rate: int | None = None
    ) -> np.ndarray:
        """批量提取特征矩阵

        Args:
            audio_segments: 音频数据列表
            sample_rate: 采样率

        Returns:
            float64 矩阵 (n_segments × n_features)
        """
        sr = sample_rate or self.sample_rate
        feature_list = []
        for seg in audio_segments:
            feats = self.extract_features(seg, sr)
            feature_list.append(feats)

        if not feature_list:
            return np.empty((0, 0), dtype=np.float64)

        return np.vstack(feature_list)

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def _preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """预处理音频片段，确保可提取特征"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 过短的片段用零填充
        min_samples = int(_MIN_SEGMENT_DURATION * sr)
        if len(audio) < min_samples:
            padded = np.zeros(min_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            return padded

        return audio

    # ------------------------------------------------------------------
    # MFCC 特征 (65 维)
    # ------------------------------------------------------------------

    def _mfcc_features(self, audio: np.ndarray, sr: int) -> List[float]:
        """提取 MFCC + Delta + Delta-Delta 统计量"""
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, using zero MFCC features")
            return [0.0] * 65

        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # MFCC mean/std (26d)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Delta MFCC mean/std (26d)
            delta = librosa.feature.delta(mfcc)
            delta_mean = np.mean(delta, axis=1)
            delta_std = np.std(delta, axis=1)

            # Delta-Delta MFCC mean (13d)
            delta2 = librosa.feature.delta(mfcc, order=2)
            delta2_mean = np.mean(delta2, axis=1)

            feats = []
            feats.extend(mfcc_mean.tolist())
            feats.extend(mfcc_std.tolist())
            feats.extend(delta_mean.tolist())
            feats.extend(delta_std.tolist())
            feats.extend(delta2_mean.tolist())
            return feats
        except Exception as e:
            logger.debug("MFCC extraction failed: %s", e)
            return [0.0] * 65

    # ------------------------------------------------------------------
    # 频谱特征 (12 维)
    # ------------------------------------------------------------------

    def _spectral_features(self, audio: np.ndarray, sr: int) -> List[float]:
        """提取频谱质心、带宽、对比度、滚降点"""
        try:
            import librosa
        except ImportError:
            return [0.0] * 12

        try:
            S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

            # 频谱质心 (2d)
            centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            centroid_feats = [np.mean(centroid), np.std(centroid)]

            # 频谱带宽 (2d)
            bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
            bandwidth_feats = [np.mean(bandwidth), np.std(bandwidth)]

            # 频谱对比度 (7d)
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
            contrast_feats = np.mean(contrast, axis=1).tolist()

            # 频谱滚降点 (1d)
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
            rolloff_feats = [np.mean(rolloff)]

            feats = []
            feats.extend(centroid_feats)
            feats.extend(bandwidth_feats)
            feats.extend(contrast_feats)
            feats.extend(rolloff_feats)
            return feats
        except Exception as e:
            logger.debug("Spectral feature extraction failed: %s", e)
            return [0.0] * 12

    # ------------------------------------------------------------------
    # 基频 / 音高特征 (2 维)
    # ------------------------------------------------------------------

    def _pitch_features(self, audio: np.ndarray, sr: int) -> List[float]:
        """提取基频 F0 统计量"""
        try:
            import librosa
        except ImportError:
            return [0.0, 0.0]

        try:
            # 使用 pYIN 算法（更准确）或 yin 作为备选
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),   # ~65 Hz
                fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
                sr=sr,
            )
            # 仅使用有声音帧
            f0_voiced = f0[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else f0
            f0_clean = f0_voiced[~np.isnan(f0_voiced)]
            if len(f0_clean) == 0:
                return [0.0, 0.0]
            return [float(np.mean(f0_clean)), float(np.std(f0_clean))]
        except Exception as e:
            logger.debug("Pitch extraction failed: %s", e)
            return [0.0, 0.0]

    # ------------------------------------------------------------------
    # 共振峰特征 (4 维)
    # ------------------------------------------------------------------

    def _formant_features(self, audio: np.ndarray, sr: int) -> List[float]:
        """提取前两个共振峰 F1, F2 的均值和标准差

        共振峰表征声道形状，是区分不同说话人的关键特征。
        使用 LPC (线性预测编码) 估计共振峰频率。
        """
        try:
            from scipy.signal import lfilter
        except ImportError:
            return [0.0, 0.0, 0.0, 0.0]

        try:
            # 对短片段提取共振峰
            frame_length = int(0.03 * sr)  # 30ms 帧
            hop_length = int(0.015 * sr)   # 15ms 步长
            n_frames = (len(audio) - frame_length) // hop_length + 1

            if n_frames < 3:
                return [0.0, 0.0, 0.0, 0.0]

            f1_vals, f2_vals = [], []
            lpc_order = int(sr / 1000) + 2  # 经验公式

            for i in range(n_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]

                if len(frame) < lpc_order + 1:
                    continue

                # 预加重
                frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

                try:
                    # LPC 分析
                    from numpy.fft import rfft

                    corr = np.correlate(frame, frame, mode="full")
                    corr = corr[len(corr)//2:]
                    R = corr[:lpc_order + 1]

                    # Levinson-Durbin 递归
                    a = np.zeros(lpc_order + 1)
                    a[0] = 1.0
                    e = R[0] if R[0] > 1e-10 else 1e-10

                    for k in range(1, lpc_order + 1):
                        lam = -np.dot(a[:k][::-1], R[1:k+1]) / e if e > 1e-10 else 0.0
                        a[1:k+1] = a[1:k+1] + lam * a[:k][::-1]
                        e = e * (1.0 - lam * lam)
                        if e < 1e-10:
                            e = 1e-10

                    # 求 LPC 多项式的根 → 共振峰
                    roots = np.roots(a)
                    # 仅保留单位圆内的共轭根
                    roots = roots[np.abs(roots) < 1.0]
                    angles = np.angle(roots)
                    freqs = angles * (sr / (2 * np.pi))
                    # 仅保留正频率且带宽合理的根 (带宽由极点幅度决定)
                    bandwidths = -0.5 * (sr / np.pi) * np.log(np.abs(roots))
                    valid = (freqs > 50) & (freqs < 4000) & (bandwidths < 500) & (bandwidths > 0)
                    valid_freqs = np.sort(freqs[valid])

                    if len(valid_freqs) >= 2:
                        f1_vals.append(valid_freqs[0])
                        f2_vals.append(valid_freqs[1])
                    elif len(valid_freqs) == 1:
                        f1_vals.append(valid_freqs[0])
                except Exception:
                    continue

            if not f1_vals:
                return [0.0, 0.0, 0.0, 0.0]

            f1 = np.array(f1_vals)
            f2 = np.array(f2_vals) if f2_vals else np.zeros_like(f1)

            return [
                float(np.mean(f1)), float(np.std(f1)),
                float(np.mean(f2)), float(np.std(f2)),
            ]
        except Exception as e:
            logger.debug("Formant extraction failed: %s", e)
            return [0.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # 时域特征 (2 维)
    # ------------------------------------------------------------------

    def _temporal_features(self, audio: np.ndarray, sr: int) -> List[float]:
        """提取过零率统计量"""
        try:
            # 自定义计算避免 librosa 依赖
            zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            # 分帧计算标准差
            frame_len = int(0.025 * sr)
            hop_len = int(0.010 * sr)
            n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
            frame_zcrs = []
            for i in range(min(n_frames, 100)):  # 最多采样 100 帧
                start = i * hop_len
                frame = audio[start:start + frame_len]
                if len(frame) < 2:
                    continue
                fzcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                frame_zcrs.append(fzcr)
            zcr_std = float(np.std(frame_zcrs)) if frame_zcrs else 0.0
            return [zcr, zcr_std]
        except Exception as e:
            logger.debug("ZCR extraction failed: %s", e)
            return [0.0, 0.0]

    # ------------------------------------------------------------------
    # 能量特征 (2 维)
    # ------------------------------------------------------------------

    def _energy_features(self, audio: np.ndarray) -> List[float]:
        """提取 RMS 能量统计量"""
        try:
            rms = np.sqrt(np.mean(audio ** 2))

            # 分帧能量标准差
            frame_len = 1024
            hop_len = 512
            n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)
            frame_energies = []
            for i in range(min(n_frames, 100)):
                start = i * hop_len
                frame = audio[start:start + frame_len]
                if len(frame) < 2:
                    continue
                frame_energies.append(np.sqrt(np.mean(frame ** 2)))
            rms_std = float(np.std(frame_energies)) if frame_energies else 0.0
            return [rms, rms_std]
        except Exception as e:
            logger.debug("Energy extraction failed: %s", e)
            return [0.0, 0.0]
