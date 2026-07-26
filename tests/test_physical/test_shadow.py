from pathlib import Path

import numpy as np

from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
from vocal_subtitle.pipeline_context import PipelineContext
from vocal_subtitle.physical.shadow import build_shadow_artifacts
from vocal_subtitle.vad.base import SpeechSegment


def test_shadow_builds_physical_and_global_artifacts_without_running_models():
    context = PipelineContext(audio_path=Path("audio.wav"), audio=np.zeros(160000))
    context.silero_segments = [SpeechSegment(1.0, 2.0, confidence=0.8)]
    context.asr_segments = [TranscriptionSegment(
        text="hello", start=1.0, end=2.0,
        words=[WordTimestamp("hello", 1.0, 2.0)],
    )]

    result = build_shadow_artifacts([context], 10.0)

    assert result.status == "ok"
    assert result.statistics["evidence_count"] == 1
    assert result.global_transcript.words[0].raw_start == 1.0
    assert result.to_dict()["physical_timeline"]["schema_version"] == "physical-timeline-v1"

