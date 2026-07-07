"""Silero VAD 语音检测引擎

基于 Silero VAD 神经网络模型的语音活动检测。
模型大小: ~1.5MB | 精度: 高 | 协议: MIT

要求: pip install torch torchaudio
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

from .base import SpeechSegment, VADEngine

logger = logging.getLogger(__name__)


class SileroVAD(VADEngine):
    """Silero VAD 引擎

    轻量级神经网络 VAD，默认推荐。
    协议: MIT (代码 + 模型权重)

    使用示例:
        engine = SileroVAD()
        engine.load_model()
        segments = engine.detect(Path("audio.wav"), threshold=0.5)
    """

    def __init__(self):
        self._model = None
        self._utils = None
        self._get_speech_timestamps = None

    @property
    def name(self) -> str:
        return "silero"

    def load_model(self) -> None:
        """加载 Silero VAD 模型"""
        if self._model is not None:
            return

        logger.info("Loading Silero VAD model")

        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch and torchaudio are required for Silero VAD. "
                "Install with: pip install torch torchaudio"
            )

        model, utils = self._load_model_inner()

        self._model = model
        self._utils = utils
        (
            self._get_speech_timestamps,
            _,
            self._read_audio,
            _,
            _,
        ) = utils

    def _load_model_inner(self):
        """实际模型加载逻辑（无代理环境）

        加载策略：
        1. 优先从本地缓存加载（importlib，零网络访问，秒级完成）
        2. 缓存不存在时回退到 torch.hub.load（需要 GitHub 访问）
        """
        import os
        import importlib.util

        model, utils = None, None
        errors = []

        cache_dir = os.path.expanduser(
            "~/.cache/torch/hub/snakers4_silero-vad_master"
        )

        # 方式 1（优先）: 本地缓存直接加载，完全离线，避免 torch.hub.load
        # 即使 force_reload=False 也会尝试连接 GitHub 验证仓库，在无网/
        # 需代理环境下会长时间挂起（TCP 超时可达数十分钟）
        if os.path.isdir(cache_dir):
            try:
                logger.info("Loading Silero VAD from local cache: %s", cache_dir)
                hubconf_path = os.path.join(cache_dir, "hubconf.py")
                spec = importlib.util.spec_from_file_location(
                    "silero_hubconf", hubconf_path
                )
                hubconf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hubconf)
                model, utils = hubconf.silero_vad()
            except Exception as e:
                errors.append(f"local_cache: {e}")
                logger.warning(
                    "Failed to load Silero VAD from local cache: %s", e
                )
        else:
            logger.info("Local cache not found at %s", cache_dir)

        # 方式 2（回退）: 本地缓存不存在时，通过 torch.hub.load 下载
        if model is None:
            import torch
            try:
                logger.info("Loading Silero VAD from torch.hub (GitHub)...")
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                )
            except Exception as e:
                errors.append(f"github: {e}")
                logger.warning(
                    "Failed to load Silero VAD from GitHub (network issue?): %s", e
                )

        if model is None:
            raise RuntimeError(
                "Failed to load Silero VAD model. "
                "Please ensure either GitHub is accessible or the model "
                "is cached locally. Errors: %s" % "; ".join(errors)
            )

        return model, utils

    def detect(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """检测音频文件中的语音区间

        Args:
            audio_path: 音频文件路径
            threshold: 语音概率阈值
            min_speech_duration_ms: 最小语音时长 (ms)
            min_silence_duration_ms: 最小静音时长 (ms)

        Returns:
            SpeechSegment 列表
        """
        if self._model is None:
            self.load_model()

        logger.info("Detecting speech in: %s (threshold=%.2f)", audio_path, threshold)

        try:
            wav = self._read_audio(str(audio_path), sampling_rate=16000)
            speech_timestamps = self._get_speech_timestamps(
                wav,
                self._model,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                return_seconds=True,
            )
        except Exception as e:
            logger.error("Silero VAD detection failed: %s", e)
            raise

        segments = [
            SpeechSegment(
                start=ts["start"],
                end=ts["end"],
                confidence=getattr(ts, "confidence", 1.0),
            )
            for ts in speech_timestamps
        ]

        logger.info("Found %d speech segments", len(segments))
        return segments

    def detect_on_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """在 numpy 数组上检测语音区间

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率
            threshold: 语音概率阈值
            min_speech_duration_ms: 最小语音时长 (ms)
            min_silence_duration_ms: 最小静音时长 (ms)

        Returns:
            SpeechSegment 列表
        """
        if self._model is None:
            self.load_model()

        # 确保是 float32 且是单声道
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        import torch

        if sample_rate != 16000:
            # 需要重采样
            try:
                import torchaudio
                import torchaudio.functional as F

                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                audio_tensor = F.resample(
                    audio_tensor, orig_freq=sample_rate, new_freq=16000
                )
                audio = audio_tensor.squeeze(0).numpy()
                sample_rate = 16000
            except ImportError:
                logger.warning(
                    "torchaudio not available, skipping resample. "
                    "VAD may be inaccurate with non-16kHz audio."
                )

        wav = torch.from_numpy(audio)

        speech_timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        segments = [
            SpeechSegment(
                start=ts["start"],
                end=ts["end"],
                confidence=getattr(ts, "confidence", 1.0),
            )
            for ts in speech_timestamps
        ]

        return segments
