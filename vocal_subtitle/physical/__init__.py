"""Physical timeline and global-coordinate infrastructure."""

from .context import build_context_windows
from .coordinate import CoordinateMapper, CoordinateRange, MacroChunkCoordinate
from .ir_cache import (
    decode_ir_value,
    encode_ir_value,
    fingerprint_ir,
    load_ir_value,
    make_ir_cache_key,
    persist_ir_value,
)
from .ir import (
    GlobalSpeakerTimeline,
    GlobalTranscript,
    GlobalTranscriptSegment,
    GlobalWord,
    adapt_diarization_result,
    adapt_transcription_segments,
)
from .evidence_adapter import (
    EvidenceAdaptResult,
    adapt_ffmpeg_result,
    adapt_speech_segments,
    build_timeline_from_context,
)
from .shadow import ShadowBuildResult, build_shadow_artifacts
from .allocator import (
    AllocationResult,
    PhysicalSpan,
    WordAllocation,
    allocate_words,
    repair_late_words,
)
from .coverage import (
    PhysicalCoverageRange,
    PhysicalCoverageReport,
    audit_physical_coverage,
)
from .events import GlobalSubtitleEvent, build_events
from .subtitle_bins import (
    PhysicalSubtitleBin,
    assign_word_to_bin,
    build_physical_subtitle_bins,
)
from .timeline import (
    ContextWindow,
    PhysicalClip,
    PhysicalTimeline,
    SpeechEvidenceSpan,
)

__all__ = [
    "ContextWindow",
    "CoordinateMapper",
    "CoordinateRange",
    "MacroChunkCoordinate",
    "PhysicalClip",
    "PhysicalTimeline",
    "SpeechEvidenceSpan",
    "ShadowBuildResult",
    "EvidenceAdaptResult",
    "GlobalSpeakerTimeline",
    "GlobalTranscript",
    "GlobalTranscriptSegment",
    "GlobalWord",
    "build_context_windows",
    "build_timeline_from_context",
    "build_shadow_artifacts",
    "adapt_diarization_result",
    "adapt_ffmpeg_result",
    "adapt_speech_segments",
    "adapt_transcription_segments",
    "decode_ir_value",
    "encode_ir_value",
    "fingerprint_ir",
    "load_ir_value",
    "make_ir_cache_key",
    "persist_ir_value",
    "AllocationResult",
    "PhysicalSpan",
    "WordAllocation",
    "allocate_words",
    "repair_late_words",
    "PhysicalCoverageRange",
    "PhysicalCoverageReport",
    "audit_physical_coverage",
    "GlobalSubtitleEvent",
    "build_events",
    "PhysicalSubtitleBin",
    "assign_word_to_bin",
    "build_physical_subtitle_bins",
]
