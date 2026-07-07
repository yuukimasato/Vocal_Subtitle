"""Fun-ASR-Nano 语音识别引擎

基于 FunASR 的中文优化 ASR 引擎。
代码协议: Apache 2.0 | 模型权重协议: Apache 2.0

Fun-ASR-Nano 是阿里达摩院推出的轻量级中文语音识别模型，
对中文场景有更好的优化。

要求: pip install funasr
"""

import logging
from typing import List, Optional

import numpy as np

from .base import ASREngine, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)


class FunASREngine(ASREngine):
    """Fun-ASR-Nano 引擎

    中文语音识别优化，轻量级模型。
    协议: Apache 2.0

    使用示例:
        engine = FunASREngine(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        engine.load_model()
        results = engine.transcribe(audio, language="zh")
    """

    # 推荐的中文模型
    DEFAULT_MODEL = (
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )

    def __init__(
        self,
        model: str = "",
        device: str = "cuda",
        ncpu: int = 4,
    ):
        """
        Args:
            model: FunASR 模型 ID，默认使用中文大模型
            device: 推理设备 (cuda / cpu)
            ncpu: CPU 线程数
        """
        self._model = None
        self._model_id = model or self.DEFAULT_MODEL
        self._device = device
        self._ncpu = ncpu

    @property
    def name(self) -> str:
        return "funasr"

    @property
    def model_name(self) -> str:
        return self._model_id.split("/")[-1] if "/" in self._model_id else self._model_id

    def detect_language(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[str]:
        """FunASR 仅支持中文识别，始终返回 "zh"。

        与 Faster-Whisper 不同，FunASR-Nano 是中文专属模型，
        无法识别英文、日文等其他语言。对于非中文音频，
        使用此引擎将导致输出乱码中文字幕。

        Args:
            audio: 音频数据 (未使用)
            sample_rate: 采样率 (未使用)

        Returns:
            始终返回 "zh"
        """
        logger.warning(
            "FunASR is a Chinese-only ASR engine. "
            "If your audio is NOT in Chinese, the output subtitles "
            "will be garbage. Consider switching to faster-whisper "
            "(--asr-engine faster-whisper) for multi-language support."
        )
        return "zh"

    def load_model(self) -> None:
        """加载 FunASR 模型"""
        if self._model is not None:
            return

        logger.info("Loading FunASR model: %s", self._model_id)

        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self._model_id,
                device=self._device,
                ncpu=self._ncpu,
                disable_pbar=True,
                disable_log=False,
            )
        except ImportError:
            raise ImportError(
                "funasr is required. Install with: pip install funasr"
            )
        except Exception as e:
            logger.error("Failed to load FunASR model: %s", e)
            raise

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
            language: 语言代码 (FunASR 主要支持 zh)

        Returns:
            转录结果列表
        """
        if self._model is None:
            self.load_model()

        logger.info(
            "Transcribing with FunASR: duration=%.1fs",
            len(audio) / sample_rate,
        )

        try:
            # FunASR 需要 int16 格式的音频
            if audio.dtype == np.float32:
                audio_int16 = (audio * 32767).astype(np.int16)
            else:
                audio_int16 = audio.astype(np.int16)

            # 调用 FunASR
            result = self._model.generate(
                input=audio_int16,
                batch_size_s=300,
                **kwargs,
            )

            # 解析结果
            segments = []
            if result and len(result) > 0:
                res = result[0]

                # FunASR 返回的文本
                text = res.get("text", "")
                timestamp_list = res.get("timestamp", [])

                if timestamp_list:
                    # 有词级时间戳
                    for ts_item in timestamp_list:
                        if isinstance(ts_item, list) and len(ts_item) >= 3:
                            word_text = str(ts_item[0]) if ts_item[0] else ""
                            word_start = float(ts_item[1]) / 1000.0  # ms → s
                            word_end = float(ts_item[2]) / 1000.0
                            segments.append(
                                TranscriptionSegment(
                                    text=word_text,
                                    start=word_start,
                                    end=word_end,
                                    words=[
                                        WordTimestamp(
                                            word=word_text,
                                            start=word_start,
                                            end=word_end,
                                        )
                                    ],
                                )
                            )
                else:
                    # 无词级时间戳，创建单个段
                    segments.append(
                        TranscriptionSegment(
                            text=text.strip(),
                            start=0.0,
                            end=len(audio) / sample_rate,
                        )
                    )

            logger.info("FunASR complete: %d segments", len(segments))
            return segments

        except Exception as e:
            logger.error("FunASR transcription failed: %s", e)
            raise
