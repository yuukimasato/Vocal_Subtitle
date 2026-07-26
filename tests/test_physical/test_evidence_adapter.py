from pathlib import Path

import numpy as np

from vocal_subtitle.pipeline_context import PipelineContext
from vocal_subtitle.physical.evidence_adapter import (
    adapt_ffmpeg_result,
    adapt_speech_segments,
    build_timeline_from_context,
)
from vocal_subtitle.physical.timeline import PhysicalTimeline
from vocal_subtitle.vad.base import SpeechSegment


def test_detector_sources_keep_provenance_and_overlap():
    timeline = PhysicalTimeline.from_duration(10.0)
    silero = adapt_speech_segments(
        [SpeechSegment(1.0, 2.0, confidence=0.8)], timeline, "silero"
    )
    ffmpeg = adapt_ffmpeg_result(
        {"coarse_speech": [SpeechSegment(1.5, 2.5, confidence=0.9)], "skeleton": [(1.2, 2.7)]},
        timeline,
    )

    assert [item.source for item in silero.evidence_spans] == ["silero"]
    assert {item.source for item in ffmpeg.evidence_spans} == {"ffmpeg_coarse", "ffmpeg_skeleton"}
    assert len(timeline.speech_evidence_spans) == 3
    assert all(item.metadata["boundary_type"] == "detected_evidence" for item in timeline.speech_evidence_spans)


def test_offset_and_invalid_duration_are_explicitly_handled():
    timeline = PhysicalTimeline.from_duration(10.0)
    result = adapt_speech_segments(
        [SpeechSegment(1.0, 2.0), SpeechSegment(9.0, 11.0)],
        timeline,
        "ffmpeg_coarse",
        time_offset=2.0,
    )

    assert [(item.start, item.end) for item in result.evidence_spans] == [(3.0, 4.0)]
    assert result.skipped_count == 1
    assert result.source_counts == {"ffmpeg_coarse": 1}


def test_unified_ffmpeg_result_is_preferred_over_legacy_coarse_context_field():
    context = PipelineContext(audio_path=Path("audio.wav"), audio=np.zeros(16000))
    context.ffmpeg_segments = [SpeechSegment(5.0, 6.0)]
    context.ffmpeg_unified_result = {"coarse_speech": [SpeechSegment(1.0, 2.0)]}

    result = build_timeline_from_context(context, 10.0)
    assert [item.source for item in result.evidence_spans] == ["ffmpeg_coarse"]
    assert result.evidence_spans[0].start == 1.0

