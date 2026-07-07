"""faster-whisper 语音识别引擎

基于 CTranslate2 加速的 Whisper 识别引擎。
代码协议: MIT | 模型权重协议: MIT

要求: pip install faster-whisper
"""

import logging
import os
from typing import List, Optional

import numpy as np

from .base import ASREngine, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)


class FasterWhisperEngine(ASREngine):
    """faster-whisper 引擎

    基于 CTranslate2 优化的 Whisper 推理引擎，支持 GPU/CPU。

    关键特性:
    - CTranslate2 加速推理，速度比原始 Whisper 快 4 倍
    - 支持词级时间戳
    - 支持 8-bit 量化 (compute_type="int8")
    - 支持 CUDA / CPU 双模式

    使用示例:
        engine = FasterWhisperEngine(model="large-v3", device="cuda")
        engine.load_model()
        results = engine.transcribe(audio, language="zh")
    """

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        word_timestamps: bool = True,
        condition_on_previous_text: bool = False,
        vad_filter: bool = False,
    ):
        """
        Args:
            model: 模型大小 (large-v3 / medium / small / tiny)
            device: 推理设备 (cuda / cpu)
            compute_type: 计算精度 (float16 / int8_float16 / int8)
            beam_size: beam search 宽度
            word_timestamps: 是否输出词级时间戳
            condition_on_previous_text: 是否使用上文条件（建议 False 防幻觉）
            vad_filter: 是否启用内置 VAD（通常关闭，使用外部 VAD）
        """
        self._model = None
        self._model_size = model
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._word_timestamps = word_timestamps
        self._condition_on_previous_text = condition_on_previous_text
        self._vad_filter = vad_filter

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def model_name(self) -> str:
        return self._model_size

    def load_model(self) -> None:
        """加载 faster-whisper 模型"""
        if self._model is not None:
            return

        # CPU 不支持 float16，自动降级为 int8
        compute_type = self._compute_type
        if self._device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.warning(
                "CPU does not support float16 compute, auto-downgraded to int8"
            )

        logger.info(
            "Loading faster-whisper model: %s (device=%s, compute=%s)",
            self._model_size,
            self._device,
            compute_type,
        )

        try:
            from faster_whisper import WhisperModel

            # 优先使用本地缓存，避免不必要的网络请求
            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
                logger.info("Loaded from local cache")
            except Exception as local_err:
                # 本地不存在 → 清理 SOCKS 代理后从 HuggingFace Hub 下载
                logger.info(
                    "Not in local cache, downloading from HuggingFace: %s",
                    local_err,
                )
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=compute_type,
                )
        except ImportError:
            raise ImportError(
                "faster-whisper is required. "
                "Install with: pip install faster-whisper"
            )
        except Exception as e:
            logger.error("Failed to load faster-whisper model: %s", e)
            raise

    def detect_language(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[str]:
        """检测音频语言（利用 Whisper 编码器做全局语言检测）

        对前 30 秒音频运行一次轻量识别，从中提取检测到的语言代码。
        这比在每个短 VAD 段上分别做自动检测可靠得多 —
        Whisper 的语言检测需要足够的音频上下文（~30s）才能准确工作。

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            检测到的语言代码 (zh/en/ja/...) 或 None（检测失败）
        """
        if self._model is None:
            self.load_model()

        # 取前 30 秒做语言检测（Whisper 需要足够长的上下文）
        max_samples = 30 * sample_rate
        sample = (
            audio[:max_samples].astype(np.float32)
            if len(audio) > max_samples
            else audio.astype(np.float32)
        )
        # 确保是 1D 数组（单声道）
        if sample.ndim > 2:
            sample = sample.squeeze()
        if sample.ndim == 2:
            sample = sample.mean(axis=0)  # 立体声 → 单声道

        try:
            _, info = self._model.transcribe(
                sample,
                language=None,           # 触发自动检测
                beam_size=1,             # 最小 beam，只关心语言检测
                word_timestamps=False,   # 不需要词级时间戳
                condition_on_previous_text=False,
                vad_filter=False,
            )
            detected = info.language
            logger.info(
                "Language detected: %s (probability=%.2f)",
                detected, info.language_probability,
            )
            return detected
        except Exception as e:
            logger.warning("Language detection failed: %s", e)
            return None

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        """识别音频

        Args:
            audio: 音频数据 (float32, [-1, 1])
            sample_rate: 采样率
            language: 语言代码 (zh/en/ja/...)

        Returns:
            转录结果列表
        """
        if self._model is None:
            self.load_model()

        # 确保 audio 是 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        logger.info(
            "Transcribing: duration=%.1fs, language=%s",
            len(audio) / sample_rate,
            language or "auto",
        )

        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self._beam_size,
                word_timestamps=self._word_timestamps,
                condition_on_previous_text=self._condition_on_previous_text,
                vad_filter=self._vad_filter,
                **kwargs,
            )

            results = []
            for seg in segments:
                words = []
                if seg.words:
                    for w in seg.words:
                        words.append(
                            WordTimestamp(
                                word=w.word,
                                start=w.start,
                                end=w.end,
                                confidence=w.probability,
                            )
                        )

                results.append(
                    TranscriptionSegment(
                        text=seg.text.strip(),
                        start=seg.start,
                        end=seg.end,
                        words=words,
                        avg_logprob=seg.avg_logprob,
                    )
                )

            logger.info(
                "Transcription complete: %d segments, language=%s",
                len(results),
                info.language,
            )
            return results

        except Exception as e:
            logger.error("faster-whisper transcription failed: %s", e)
            raise
