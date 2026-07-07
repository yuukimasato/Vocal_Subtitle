"""WebRTC VAD 语音检测引擎

基于 WebRTC 原生 VAD 模块的轻量级检测。
模型大小: N/A (基于信号处理) | 精度: 中 | 协议: BSD-3

适用于极端资源受限环境的降级方案。
需要: pip install webrtcvad
"""

import logging
from pathlib import Path
from typing import List

import numpy as np

from .base import SpeechSegment, VADEngine

logger = logging.getLogger(__name__)


class WebRTCVAD(VADEngine):
    """WebRTC VAD 引擎

    基于 WebRTC 信号处理的 VAD，无需神经网络模型。
    协议: BSD-3

    适用于资源极端受限场景（如树莓派、无 GPU）。

    使用示例:
        engine = WebRTCVAD()
        segments = engine.detect(Path("audio.wav"), aggressiveness=2)
    """

    # WebRTC VAD 支持的采样率
    VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}

    def __init__(self):
        self._vad = None
        self._aggressiveness: int = 2

    @property
    def name(self) -> str:
        return "webrtc"

    def load_model(self, aggressiveness: int = 2) -> None:
        """初始化 WebRTC VAD

        Args:
            aggressiveness: 检测激进程度 (0-3)，越大越激进
        """
        if self._vad is not None:
            return

        logger.info("Initializing WebRTC VAD (aggressiveness=%d)", aggressiveness)

        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(aggressiveness)
            self._aggressiveness = aggressiveness
        except ImportError:
            raise ImportError(
                "webrtcvad is required. Install it with: pip install webrtcvad"
            )
        except Exception as e:
            logger.error("Failed to initialize WebRTC VAD: %s", e)
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

        WebRTC VAD 要求帧长为 10/20/30ms 的整数倍，
        且仅支持 8000/16000/32000/48000 Hz 采样率。
        """
        if self._vad is None:
            self.load_model()

        # 确保采样率兼容
        if sample_rate not in self.VALID_SAMPLE_RATES:
            logger.warning(
                "Sample rate %d not supported by WebRTC VAD, "
                "resampling to 16000",
                sample_rate,
            )
            sample_rate = 16000

        # 转换为 int16 PCM
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)

        # 使用 30ms 帧长
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        # 分帧
        frames = []
        for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
            frame = audio_int16[i : i + frame_size].tobytes()
            frames.append(frame)

        if not frames:
            return []

        # 逐帧检测
        vad_results = []
        for frame in frames:
            try:
                is_speech = self._vad.is_speech(frame, sample_rate)
                vad_results.append(is_speech)
            except Exception:
                vad_results.append(False)

        # 平滑与合并
        min_speech_frames = max(1, min_speech_duration_ms // frame_duration_ms)
        min_silence_frames = max(1, min_silence_duration_ms // frame_duration_ms)

        segments = []
        in_speech = False
        speech_start_frame = 0
        silence_count = 0

        for i, is_speech in enumerate(vad_results):
            if is_speech and not in_speech:
                in_speech = True
                speech_start_frame = i
                silence_count = 0
            elif not is_speech and in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    speech_end_frame = i - silence_count
                    if (
                        speech_end_frame - speech_start_frame
                        >= min_speech_frames
                    ):
                        segments.append(
                            SpeechSegment(
                                start=speech_start_frame * frame_duration_ms / 1000.0,
                                end=speech_end_frame * frame_duration_ms / 1000.0,
                                confidence=1.0,
                            )
                        )
                    in_speech = False
                    silence_count = 0
            elif is_speech and in_speech:
                silence_count = 0

        # 处理末尾段
        if in_speech:
            end_frame = len(vad_results) - silence_count
            if end_frame - speech_start_frame >= min_speech_frames:
                segments.append(
                    SpeechSegment(
                        start=speech_start_frame * frame_duration_ms / 1000.0,
                        end=end_frame * frame_duration_ms / 1000.0,
                        confidence=1.0,
                    )
                )

        logger.info("WebRTC VAD found %d segments", len(segments))
        return segments
