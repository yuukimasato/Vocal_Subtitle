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
from .funasr_manager import DEFAULT_FUNASR_MODEL, find_local_model, normalize_model_id

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
    DEFAULT_MODEL = DEFAULT_FUNASR_MODEL

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
        self._model_id = normalize_model_id(model)
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

            local_model = find_local_model(self._model_id)
            self._model = AutoModel(
                model=str(local_model) if local_model else self._model_id,
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
            # FunASR/PyTorch 的部分前端会对输入调用 mean()；传入 int16
            # 会触发 torch.mean(Short) 异常。ASREngine 的公共契约本身就是
            # float32、[-1, 1]，因此在这里保持浮点格式并做边界清理。
            audio_array = np.asarray(audio)
            if np.issubdtype(audio_array.dtype, np.integer):
                scale = float(max(abs(np.iinfo(audio_array.dtype).min), np.iinfo(audio_array.dtype).max))
                audio_float32 = audio_array.astype(np.float32) / scale
            else:
                audio_float32 = audio_array.astype(np.float32, copy=False)
            audio_float32 = np.nan_to_num(
                audio_float32, nan=0.0, posinf=1.0, neginf=-1.0
            )
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)

            # 调用 FunASR
            result = self._model.generate(
                input=audio_float32,
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

                parsed_timestamp_count = 0
                pair_timestamp_bounds = []
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
                            parsed_timestamp_count += 1
                        elif isinstance(ts_item, (list, tuple)) and len(ts_item) >= 2:
                            # 部分 FunASR 版本只返回 [start_ms, end_ms]，
                            # 文本仍在结果级 text 字段中。
                            parsed_timestamp_count += 1
                            pair_timestamp_bounds.append(
                                (float(ts_item[0]) / 1000.0, float(ts_item[1]) / 1000.0)
                            )
                if not segments and pair_timestamp_bounds and text.strip():
                    segments.append(
                        TranscriptionSegment(
                            text=text.strip(),
                            start=pair_timestamp_bounds[0][0],
                            end=pair_timestamp_bounds[-1][1],
                        )
                    )
                elif not parsed_timestamp_count:
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
