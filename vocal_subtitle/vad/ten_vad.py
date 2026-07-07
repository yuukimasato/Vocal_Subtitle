"""TEN VAD 语音检测引擎

基于 TEN (agora-extension-vad) 的实时语音活动检测。
模型大小: ~306KB | 精度: 更高 | 协议: Apache 2.0

TEN VAD 来自 TEN Framework，是一个高性能轻量级 VAD 引擎。
适用于低延迟、实时场景。

注意: 需要安装 ten 或 agora 相关 Python SDK。
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

from .base import SpeechSegment, VADEngine

logger = logging.getLogger(__name__)


class TENVAD(VADEngine):
    """TEN VAD 引擎

    轻量级高性能 VAD，适用于低延迟场景。
    协议: Apache 2.0

    注意: TEN VAD 实现需要对应的 Python SDK。
    当前实现包含框架结构，具体 SDK 调用需要根据
    TEN 官方 API 文档调整。
    """

    def __init__(self):
        self._model = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "ten"

    def load_model(self) -> None:
        """加载 TEN VAD 模型"""
        if self._is_loaded:
            return

        logger.info("Loading TEN VAD model")

        try:
            # TEN VAD SDK 加载逻辑
            # 具体 API 调用方式取决于 TEN 的 Python SDK 版本
            # 参考: https://github.com/TEN-framework
            #
            # 示例加载代码（需要根据实际 SDK 调整）:
            # import ten_vad
            # self._model = ten_vad.VAD(threshold=0.5)
            #
            # 目前作为占位实现，如果未安装 SDK 则降级
            self._is_loaded = True
            logger.warning(
                "TEN VAD SDK not fully integrated. "
                "Please install the TEN framework Python SDK. "
                "Falling back to basic frame-based detection."
            )
        except Exception as e:
            logger.error("Failed to load TEN VAD: %s", e)
            raise

    def detect(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """检测语音区间"""
        from ..utils.audio_utils import AudioUtils

        audio, sr = AudioUtils.load_audio(audio_path)
        return self.detect_on_array(
            audio, sr, threshold, min_speech_duration_ms, min_silence_duration_ms
        )

    def detect_on_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
    ) -> List[SpeechSegment]:
        """在 numpy 数组上检测语音区间

        当前实现使用基于能量的简单帧级检测作为降级方案。
        """
        if not self._is_loaded:
            self.load_model()

        # 帧参数
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        min_speech_frames = max(1, min_speech_duration_ms // frame_duration_ms)
        min_silence_frames = max(1, min_silence_duration_ms // frame_duration_ms)

        # 计算每帧能量
        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return []

        segments = []
        in_speech = False
        speech_start_frame = 0
        silence_count = 0

        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame**2))

            is_speech = rms > (threshold * 0.1)  # 能量阈值

            if is_speech and not in_speech:
                # 开始语音
                in_speech = True
                speech_start_frame = i
                silence_count = 0
            elif not is_speech and in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # 语音段结束
                    speech_end_frame = i - silence_count
                    if speech_end_frame - speech_start_frame >= min_speech_frames:
                        segments.append(
                            SpeechSegment(
                                start=speech_start_frame * frame_duration_ms / 1000.0,
                                end=speech_end_frame * frame_duration_ms / 1000.0,
                                confidence=rms / (threshold * 0.1),
                            )
                        )
                    in_speech = False
                    silence_count = 0
            elif is_speech and in_speech:
                silence_count = 0

        # 处理末尾语音段
        if in_speech:
            end_frame = num_frames - silence_count
            if end_frame - speech_start_frame >= min_speech_frames:
                segments.append(
                    SpeechSegment(
                        start=speech_start_frame * frame_duration_ms / 1000.0,
                        end=end_frame * frame_duration_ms / 1000.0,
                        confidence=1.0,
                    )
                )

        logger.info("TEN VAD fallback found %d segments", len(segments))
        return segments
