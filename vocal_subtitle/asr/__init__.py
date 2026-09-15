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
from .boundary_arbitration import (
    ArbitrationConfig,
    ArbitrationResult,
    BoundaryArbitrator,
)
from .boundary_confidence import (
    BoundaryConfidence,
    BoundaryConfidenceEstimator,
    BoundaryRedundancyConfig,
)
from .boundary_reasr import (
    BoundaryReASRResult,
    SlidingWindow,
    SlidingWindowConfig,
    SlidingWindowReASR,
)
from .contracts import ASRFailureRequest, ASRReviewRequest, ASRRuntimePorts
from .engine_pairing import EnginePairDecision, EnginePairRouter
from .evidence import (
    DECISION_POLICY_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    RISK_POLICY_VERSION,
    CandidateEvidence,
    DecisionEvidenceBundle,
    EvidenceBundle,
    EvidenceDecision,
    EvidenceWord,
)
from .evidence_cache import EvidenceCacheKeyContext, EvidenceCachePort
from .evidence_decision import EvidenceDecisionEngine, decisions_to_subtitle_events
from .evidence_review import EvidenceReviewRuntimePorts, EvidenceReviewService
from .faster_whisper_engine import FasterWhisperEngine
from .funasr_engine import FunASREngine
from .global_path import GlobalASRPath, GlobalASRService
from .global_transcriber import (
    GlobalTranscriber,
    GlobalTranscriberConfig,
    GlobalTranscriptionResult,
)
from .hallucination import (
    HallucinationFilterPolicy,
    HallucinationFilterResult,
    filter_transcription_segments,
)
from .local_recovery import (
    LocalRecoveryConfig,
    LocalRecoveryEngine,
    LocalRecoveryRequest,
    LocalRecoveryResult,
)
from .optional_adapters import (
    LazyAudioClassifierSED,
    LazyQwenASR,
    LazyQwenForcedAligner,
)
from .quality_gate import ASRQualityResult, evaluate_asr_quality
from .qwen_engine import QwenASREngine
from .review_engines import (
    CallbackForcedAligner,
    CallbackSED,
    CallbackSemanticReview,
    ContextReASRPort,
    ForcedAlignerPort,
    QwenASRPort,
    ReviewUnavailable,
    SEDPort,
    SemanticReviewPort,
    WindowedASRContextReASR,
    WindowedASREngine,
    WindowedASRQwen,
    WindowTranscriptionPort,
)
from .review_path import ASRReviewPath, ASRReviewService
from .review_scheduler import ReviewScheduler, ReviewSchedulerConfig, ReviewWindow
from .review_telemetry import resource_snapshot, timed_call
from .risk_scoring import EvidenceRiskScorer, RiskAssessment, RiskScoringConfig
from .router import ASRRouteDecision, ASRRouter
from .secondary_evidence import (
    SecondaryEvidenceCollector,
    secondary_evidence_to_bundles,
)
from .segmented_path import SegmentedASRPath, SegmentedASRService
from .text_normalizer import TextNormalizer
from .whisper_cpp_engine import WhisperCppEngine
from .whisperx_engine import WhisperXEngine, WhisperXUnavailableError
from .window_execution import CancellationToken, WindowExecutionCoordinator

__all__ = [
    "ASREngine",
    "TranscriptionSegment",
    "WordTimestamp",
    "FasterWhisperEngine",
    "WhisperCppEngine",
    "FunASREngine",
    "QwenASREngine",
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
    "CandidateEvidence",
    "DecisionEvidenceBundle",
    "DECISION_POLICY_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "RISK_POLICY_VERSION",
    "EvidenceBundle",
    "EvidenceDecision",
    "EvidenceWord",
    "EvidenceDecisionEngine",
    "decisions_to_subtitle_events",
    "EvidenceReviewRuntimePorts",
    "EvidenceReviewService",
    "EvidenceCacheKeyContext",
    "EvidenceCachePort",
    "EvidenceRiskScorer",
    "RiskAssessment",
    "RiskScoringConfig",
    "ReviewScheduler",
    "ReviewSchedulerConfig",
    "ReviewWindow",
    "WindowedASRContextReASR",
    "WindowedASREngine",
    "WindowTranscriptionPort",
    "WindowedASRQwen",
    "LazyAudioClassifierSED",
    "LazyQwenASR",
    "LazyQwenForcedAligner",
    "SecondaryEvidenceCollector",
    "secondary_evidence_to_bundles",
    "EnginePairDecision",
    "EnginePairRouter",
    "ContextReASRPort",
    "CallbackForcedAligner",
    "CallbackSED",
    "CallbackSemanticReview",
    "QwenASRPort",
    "ForcedAlignerPort",
    "SEDPort",
    "SemanticReviewPort",
    "ReviewUnavailable",
    "resource_snapshot",
    "timed_call",
    "BoundaryConfidence",
    "BoundaryConfidenceEstimator",
    "BoundaryRedundancyConfig",
    "BoundaryReASRResult",
    "SlidingWindow",
    "SlidingWindowConfig",
    "SlidingWindowReASR",
    "ArbitrationConfig",
    "ArbitrationResult",
    "BoundaryArbitrator",
    "HallucinationFilterPolicy",
    "HallucinationFilterResult",
    "filter_transcription_segments",
    "WhisperXEngine",
    "WhisperXUnavailableError",
    "GlobalTranscriber",
    "GlobalTranscriberConfig",
    "GlobalTranscriptionResult",
    "LocalRecoveryConfig",
    "LocalRecoveryEngine",
    "LocalRecoveryRequest",
    "LocalRecoveryResult",
    "ASRFailureRequest",
    "ASRReviewRequest",
    "ASRRuntimePorts",
    "TextNormalizer",
    "CancellationToken",
    "WindowExecutionCoordinator",
]
