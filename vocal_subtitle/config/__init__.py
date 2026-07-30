"""Configuration dataclasses — pure data models with defaults.

This module holds every PipelineConfig sub-dataclass. No filesystem I/O,
environment reads, YAML loading or model construction happens here.
"""

from .models import (  # noqa: F401
    AcousticValidationConfig,
    ASRAutoRoutingConfig,
    ASRConfig,
    BoundaryRedundancyConfig,
    BoundaryRefinementConfig,
    CacheConfig,
    ContextReASRConfig,
    DegradationConfig,
    DiarizationConfig,
    FeedbackConfig,
    FFmpegVADConfig,
    FusionConfig,
    GapHandlingConfig,
    GlobalASRConfig,
    LLMOptimizeConfig,
    LoggingConfig,
    MacroChunkConfig,
    MergeDecisionConfig,
    MergingConfig,
    NoiseReductionConfig,
    PipelineConfig,
    SeparationConfig,
    SpeakerEmbeddingConfig,
    SpeakerRoleConfig,
    StreamingConfig,
    SubtitleBuildConfig,
    VADConfig,
)

# Re-export loader symbols so `from vocal_subtitle.config import ConfigLoader` works.
from ..config_loader import (  # noqa: F401, E0401
    ConfigLoader,
    validate_config_consistency,
)
