"""Stage 2: 语音活动检测 (VAD) 模块

提供多种 VAD 引擎抽象接口和实现。

引擎列表:
- SileroVAD: 神经网络 VAD, ~1.5MB, MIT, 默认首选
- TENVAD: TEN 框架 VAD, ~306KB, Apache 2.0, 低延迟
- WebRTCVAD: 信号处理 VAD, BSD-3, 资源受限降级
- FFmpegSilenceVAD: ffmpeg silencedetect, 采样级精度, 纯能量
- BoundaryFusion: 三方法边界融合 (Silero + ffmpeg + RMS)
"""

from .base import SpeechSegment, VADEngine
from .silero_vad import SileroVAD
from .ten_vad import TENVAD
from .webrtc_vad import WebRTCVAD
from .ffmpeg_vad import FFmpegSilenceVAD, unified_ffmpeg_pass
from .boundary_fusion import BoundaryFusion, FusionConfig

__all__ = [
    "VADEngine",
    "SpeechSegment",
    "SileroVAD",
    "TENVAD",
    "WebRTCVAD",
    "FFmpegSilenceVAD",
    "unified_ffmpeg_pass",
    "BoundaryFusion",
    "FusionConfig",
]
