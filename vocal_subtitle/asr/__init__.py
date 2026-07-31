"""Stage 4: ASR 语音识别模块

提供多种 ASR 引擎抽象接口和实现。

引擎列表:
- FasterWhisperEngine: CTranslate2 加速 Whisper, MIT, 默认首选
- WhisperCppEngine: whisper.cpp, MIT, CPU/Apple Silicon
- FunASREngine: Fun-ASR-Nano, Apache 2.0, 中文优化

边界冗余 (Stage 4.6):
- BoundaryConfidenceEstimator: 边界置信度评估
- SlidingWindowReASR: 滑动窗口冗余重识别
- BoundaryArbitrator: LLM 语义仲裁
"""

from .base import ASREngine, TranscriptionSegment, WordTimestamp
from .faster_whisper_engine import FasterWhisperEngine
from .funasr_engine import FunASREngine
from .whisper_cpp_engine import WhisperCppEngine
from .router import ASRRouteDecision, ASRRouter
from .quality_gate import ASRQualityResult, evaluate_asr_quality
from .global_path import GlobalASRPath, GlobalASRService
from .review_path import ASRReviewPath, ASRReviewService
from .segmented_path import SegmentedASRPath, SegmentedASRService

__all__ = [
    "ASREngine",
    "TranscriptionSegment",
    "WordTimestamp",
    "FasterWhisperEngine",
    "WhisperCppEngine",
    "FunASREngine",
    "ASRRouteDecision",
    "ASRRouter",
    "ASRQualityResult",
    "evaluate_asr_quality",
    "ASRReviewPath",
    "ASRReviewService",
    "GlobalASRPath",
    "GlobalASRService",
    "SegmentedASRPath",
    "SegmentedASRService",
]
