from types import SimpleNamespace

import numpy as np

from vocal_subtitle.asr.base import ASREngine, TranscriptionSegment, WordTimestamp
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.mapping.subtitle_builder import SubtitleBuilder
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.pipeline import Pipeline, PipelineStats
from vocal_subtitle.physical.allocator import (
    WordAllocation,
    allocate_words,
    repair_late_words,
)
from vocal_subtitle.physical.coverage import audit_physical_coverage
from vocal_subtitle.physical.ir import (
    GlobalSpeakerTimeline,
    GlobalTranscript,
    GlobalTranscriptSegment,
    GlobalWord,
)
from vocal_subtitle.physical.subtitle_bins import (
    PhysicalSubtitleBin,
    build_physical_subtitle_bins,
)
from vocal_subtitle.physical.timeline import PhysicalTimeline
from vocal_subtitle.webui import api


def _allocation(word_id: str, start: float, end: float) -> WordAllocation:
    word = GlobalWord(
        id=word_id,
        text=word_id,
        raw_start=start,
        raw_end=end,
        confidence=0.9,
        source_window_id="test",
        segment_id="segment",
    )
    return WordAllocation(word=word, physical_spans=(), accepted=True)


def test_coverage_audit_groups_uncovered_physical_bins():
    bins = [
        PhysicalSubtitleBin("bin-1", 0.0, 1.0, "skeleton", physical_clip_id="clip-a"),
        PhysicalSubtitleBin("bin-2", 1.05, 2.0, "skeleton", physical_clip_id="clip-a"),
        PhysicalSubtitleBin("bin-3", 3.0, 4.0, "skeleton", physical_clip_id="clip-a"),
    ]

    report = audit_physical_coverage(
        bins,
        [_allocation("covered", 0.2, 0.5)],
        merge_gap=0.1,
    )

    assert report.covered_physical_bin_count == 1
    assert report.uncovered_physical_bin_count == 2
    assert report.recovery_ranges[0].bin_ids == ("bin-2",)
    assert report.recovery_ranges[1].bin_ids == ("bin-3",)
    assert report.tail_gap_seconds == 3.5


def test_allocator_rejects_asr_word_inside_macro_clip_but_outside_speech_bin():
    timeline = PhysicalTimeline.from_duration(4.0)
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-000001")
    bins = build_physical_subtitle_bins(timeline)
    word = GlobalWord(
        id="hallucinated",
        text="静音误识别",
        raw_start=2.0,
        raw_end=2.4,
        confidence=0.9,
        source_window_id="test",
        segment_id="segment",
    )
    transcript = GlobalTranscript(
        audio_duration=4.0,
        words=[word],
        segments=[
            GlobalTranscriptSegment(
                id="segment",
                text=word.text,
                raw_start=word.raw_start,
                raw_end=word.raw_end,
                word_ids=[word.id],
            )
        ],
        backend="test",
        status="ok",
    )

    result = allocate_words(transcript, timeline, subtitle_bins=bins)

    assert result.accepted == []
    assert [item.word.id for item in result.rejected] == ["hallucinated"]
    assert result.rejected[0].warnings == ("outside_speech_bin",)
    assert result.diagnostics["speech_bin_rejected_count"] == 1


def test_late_word_is_repaired_to_previous_word_and_bin_end():
    previous = GlobalWord(
        id="previous",
        text="二十分",
        raw_start=0.5,
        raw_end=0.7,
        confidence=0.9,
        source_window_id="window",
        segment_id="segment",
    )
    late = GlobalWord(
        id="late",
        text="钟",
        raw_start=1.03,
        raw_end=1.7,
        confidence=0.9,
        source_window_id="window",
        segment_id="segment",
    )
    transcript = GlobalTranscript(
        audio_duration=2.0,
        words=[previous, late],
        segments=[
            GlobalTranscriptSegment(
                id="segment",
                text="二十分 钟",
                raw_start=0.5,
                raw_end=1.7,
                word_ids=["previous", "late"],
            )
        ],
        backend="test",
        status="ok",
    )
    bins = [
        PhysicalSubtitleBin(
            "bin-1",
            0.0,
            1.0,
            "skeleton",
            physical_clip_id="clip-a",
        )
    ]

    repaired = repair_late_words(transcript, bins)

    assert [(word.raw_start, word.raw_end) for word in repaired.words] == [
        (0.5, 0.7),
        (0.7, 1.0),
    ]
    assert repaired.words[1].metadata["timestamp_repaired"] is True
    assert repaired.diagnostics["late_word_repairs"][0]["word_id"] == "late"


class _RecoveryEngine(ASREngine):
    def __init__(self, recover: bool):
        self.recover = recover
        self.calls = 0

    @property
    def name(self):
        return "recovery-test"

    @property
    def model_name(self):
        return "recovery-test"

    def load_model(self):
        return None

    def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return [
                TranscriptionSegment(
                    text="head",
                    start=0.2,
                    end=0.4,
                    words=[WordTimestamp("head", 0.2, 0.4, confidence=0.9)],
                )
            ]
        if not self.recover:
            return []
        # The recovery window starts at 0.7s with the default 0.5s context.
        return [
            TranscriptionSegment(
                text="tail",
                start=0.7,
                end=0.9,
                words=[WordTimestamp("tail", 0.7, 0.9, confidence=0.9)],
            )
        ]


def _run_recovery_case(monkeypatch, recover: bool):
    config = PipelineConfig()
    config.cache.enabled = False
    config.asr.global_asr.enabled = True
    config.asr.global_asr.left_context = 0.5
    config.asr.global_asr.right_context = 0.5
    pipeline = Pipeline(config)
    pipeline._progress = SimpleNamespace()
    engine = _RecoveryEngine(recover)
    pipeline._get_global_asr_engine = lambda: engine

    timeline = PhysicalTimeline.from_duration(2.0)
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-000001")
    timeline.add_evidence(1.2, 2.0, "ffmpeg_skeleton", physical_clip_id="clip-000001")
    shadow = SimpleNamespace(
        physical_timeline=timeline,
        global_speaker_timeline=GlobalSpeakerTimeline(
            duration=2.0,
            turns=[],
            exclusive_turns=[],
            backend="test",
            status="unknown",
        ),
    )
    stats = PipelineStats(input_path="audio.wav", duration_seconds=2.0)
    events, diagnostics, transcript = pipeline._run_global_transcription_path(
        np.zeros(32000, dtype=np.float32),
        16000,
        shadow,
        stats,
    )
    return engine, events, diagnostics, transcript


def test_global_path_recovers_uncovered_tail(monkeypatch):
    engine, events, diagnostics, transcript = _run_recovery_case(monkeypatch, True)

    assert engine.calls == 2
    assert [event.text for event in events] == ["head", "tail"]
    assert diagnostics["recovery"]["status"] == "recovered"
    assert diagnostics["physical_coverage"]["complete"] is True
    assert transcript.status == "ok"


def test_global_path_marks_failed_tail_recovery_degraded(monkeypatch):
    _, events, diagnostics, transcript = _run_recovery_case(monkeypatch, False)

    assert [event.text for event in events] == ["head"]
    assert diagnostics["recovery"]["status"] == "incomplete"
    assert diagnostics["physical_coverage"]["complete"] is False
    assert transcript.status == "degraded"


def test_builder_quantization_respects_physical_bounds_and_previous_end():
    events = [
        SubtitleEvent(
            index=1,
            start=0.0004,
            end=1.0004,
            text="first",
            physical_start=0.0004,
            physical_end=1.0004,
        ),
        SubtitleEvent(
            index=2,
            start=1.0000,
            end=2.0004,
            text="second",
            physical_start=1.0000,
            physical_end=2.0004,
        ),
    ]

    subtitles = SubtitleBuilder()._to_ssa(events, fmt="srt")

    assert [(item.start, item.end) for item in subtitles] == [(1, 1000), (1000, 2000)]


def test_webui_payload_round_trip_preserves_physical_metadata():
    payload = {
        "index": 1,
        "start": 1.0,
        "end": 2.0,
        "text": "tail",
        "physical_start": 0.9,
        "physical_end": 2.1,
        "physical_region_id": "region-a",
        "physical_bin_id": "subtitle-bin-000002",
        "physical_bin_start": 0.9,
        "physical_bin_end": 2.1,
        "time_source": "physical_bin",
        "physical_spans": [
            {"physical_clip_id": "clip-a", "start": 0.9, "end": 2.1}
        ],
        "source_word_ids": ["word-2"],
    }

    event = api._subtitle_event_from_payload(payload)
    round_trip = api._subtitle_event_to_payload(event)

    assert round_trip["physical_start"] == 0.9
    assert round_trip["physical_end"] == 2.1
    assert round_trip["physical_region_id"] == "region-a"
    assert round_trip["physical_bin_id"] == "subtitle-bin-000002"
    assert round_trip["source_word_ids"] == ["word-2"]
