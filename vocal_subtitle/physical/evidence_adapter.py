"""Adapters from existing VAD/fusion outputs to physical evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from ..vad.base import SpeechSegment
from .timeline import PhysicalTimeline, SpeechEvidenceSpan

if TYPE_CHECKING:
    from ..pipeline_context import PipelineContext


@dataclass
class EvidenceAdaptResult:
    evidence_spans: List[SpeechEvidenceSpan] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    skipped_count: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)

    def record_skip(self, message: str, source: Optional[str] = None) -> None:
        self.skipped_count += 1
        self.diagnostics.setdefault("skipped", []).append(message)
        if source:
            self.diagnostics.setdefault("skipped_by_source", {}).setdefault(source, 0)
            self.diagnostics["skipped_by_source"][source] += 1

    def record_success(self, span: SpeechEvidenceSpan) -> None:
        self.evidence_spans.append(span)
        self.source_counts[span.source] = self.source_counts.get(span.source, 0) + 1


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_range(item: Any) -> Tuple[float, float]:
    if isinstance(item, Mapping):
        return _finite(item.get("start"), "start"), _finite(item.get("end"), "end")
    if isinstance(item, (tuple, list)):
        if len(item) < 2:
            raise ValueError("range tuple must contain start and end")
        return _finite(item[0], "start"), _finite(item[1], "end")
    return _finite(getattr(item, "start", None), "start"), _finite(getattr(item, "end", None), "end")


def _read_confidence(item: Any) -> Optional[float]:
    value = item.get("confidence") if isinstance(item, Mapping) else getattr(item, "confidence", None)
    if value is None:
        return None
    return _finite(value, "confidence")


def _add_items(
    items: Sequence[Any],
    timeline: PhysicalTimeline,
    result: EvidenceAdaptResult,
    source: str,
    *,
    time_offset: float,
    physical_clip_id: Optional[str],
) -> EvidenceAdaptResult:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    offset = _finite(time_offset, "time_offset")
    for index, item in enumerate(items or []):
        try:
            local_start, local_end = _read_range(item)
            if local_start < 0 or local_end <= local_start:
                raise ValueError("range must satisfy 0 <= start < end")
            start = local_start + offset
            end = local_end + offset
            if start < 0.0 or end > timeline.duration:
                raise ValueError("global evidence range exceeds timeline duration")
            metadata = {
                "boundary_type": "detected_evidence",
                "source": source,
                "time_offset": offset,
            }
            if isinstance(item, Mapping) and isinstance(item.get("metadata"), Mapping):
                metadata.update(dict(item["metadata"]))
            evidence = timeline.add_evidence(
                start,
                end,
                source,
                confidence=_read_confidence(item),
                physical_clip_id=physical_clip_id,
                metadata=metadata,
                evidence_id=f"evidence:{source}:{index:06d}:{len(timeline.speech_evidence_spans):06d}",
            )
            result.record_success(evidence)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            result.record_skip(f"{source}[{index}]: {exc}", source)
    result.evidence_spans.sort(key=lambda item: (item.start, item.end, item.id))
    return result


def adapt_speech_segments(
    segments: Sequence[SpeechSegment],
    timeline: PhysicalTimeline,
    source: str,
    *,
    time_offset: float = 0.0,
    physical_clip_id: Optional[str] = None,
) -> EvidenceAdaptResult:
    return _add_items(segments, timeline, EvidenceAdaptResult(), source, time_offset=time_offset, physical_clip_id=physical_clip_id)


def adapt_ffmpeg_result(
    result: Mapping[str, Any],
    timeline: PhysicalTimeline,
    *,
    time_offset: float = 0.0,
    physical_clip_id: Optional[str] = None,
) -> EvidenceAdaptResult:
    if not isinstance(result, Mapping):
        raise ValueError("ffmpeg result must be a mapping")
    output = EvidenceAdaptResult()
    fields = (
        ("coarse_speech", "ffmpeg_coarse"),
        ("skeleton", "ffmpeg_skeleton"),
        ("rms", "rms"),
        ("rms_segments", "rms"),
        ("fused_segments", "boundary_fusion"),
    )
    seen = set()
    for field_name, source in fields:
        if field_name not in result or field_name in seen:
            continue
        seen.add(field_name)
        value = result.get(field_name) or []
        current = _add_items(value, timeline, output, source, time_offset=time_offset, physical_clip_id=physical_clip_id)
        output = current
    if "raw_silence_intervals" in result:
        output.diagnostics["ffmpeg_raw_silence_intervals"] = len(result.get("raw_silence_intervals") or [])
    if isinstance(result.get("diagnostics"), Mapping):
        output.diagnostics["input_diagnostics"] = dict(result["diagnostics"])
    return output


def _merge_results(target: EvidenceAdaptResult, source_result: EvidenceAdaptResult) -> None:
    target.evidence_spans.extend(source_result.evidence_spans)
    target.skipped_count += source_result.skipped_count
    for key, value in source_result.source_counts.items():
        target.source_counts[key] = target.source_counts.get(key, 0) + value
    for key, value in source_result.diagnostics.items():
        if isinstance(value, list):
            target.diagnostics.setdefault(key, []).extend(value)
        elif isinstance(value, dict):
            target.diagnostics.setdefault(key, {}).update(value)
        else:
            target.diagnostics[key] = value


def build_timeline_from_context(
    context: PipelineContext,
    duration: float,
    *,
    time_offset: float = 0.0,
    physical_clip_id: Optional[str] = None,
) -> EvidenceAdaptResult:
    """Collect existing detector outputs without changing their algorithms."""
    if context is None or not hasattr(context, "ffmpeg_unified_result"):
        raise ValueError("context must provide PipelineContext detector fields")
    timeline = getattr(context, "_physical_timeline", None)
    if not isinstance(timeline, PhysicalTimeline):
        timeline = PhysicalTimeline.from_duration(duration)
    output = EvidenceAdaptResult()
    unified = context.ffmpeg_unified_result
    if unified:
        _merge_results(output, adapt_ffmpeg_result(unified, timeline, time_offset=time_offset, physical_clip_id=physical_clip_id))
    for field_name, source in (("silero_segments", "silero"), ("fused_segments", "boundary_fusion")):
        _merge_results(output, adapt_speech_segments(getattr(context, field_name, []) or [], timeline, source, time_offset=time_offset, physical_clip_id=physical_clip_id))
    if not unified:
        _merge_results(output, adapt_speech_segments(context.ffmpeg_segments or [], timeline, "ffmpeg_coarse", time_offset=time_offset, physical_clip_id=physical_clip_id))
    if getattr(context, "diagnostics", None):
        output.diagnostics["context_diagnostics"] = list(context.diagnostics)
    return output
