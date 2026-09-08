"""Main-pipeline routing tests for the bounded global ASR path."""

from types import SimpleNamespace

import numpy as np
import pytest

from vocal_subtitle.asr.base import (
    ASREngine,
    ASRInvalidResultError,
    TranscriptionSegment,
    WordTimestamp,
)
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.pipeline_context import NoiseProfile, PipelineContext
from vocal_subtitle.physical.ir import GlobalSpeakerTimeline
from vocal_subtitle.physical.timeline import PhysicalTimeline
from vocal_subtitle.utils.audio_utils import AudioUtils
from vocal_subtitle.vad.base import SpeechSegment


def _event(text="global"):
    return SubtitleEvent(
        index=1,
        start=0.1,
        end=0.4,
        text=text,
        source_word_ids=["word-000000"],
        physical_start=0.1,
        physical_end=0.4,
    )


def _transcript():
    return SimpleNamespace(
        status="ok",
        words=[object()],
        segments=[object()],
    )


def _prepare_run(monkeypatch, tmp_path, routing="auto"):
    config = PipelineConfig()
    config.asr.engine = "faster-whisper"
    config.asr.global_asr.enabled = True
    config.asr.global_asr.routing = routing
    config.cache.enabled = False
    config.cache.full_pipeline_cache = False
    config.macro_chunking.enabled = False
    config.acoustic_validation.skeleton_mode = True
    config.acoustic_validation.enabled = False
    config.diarization.enabled = False
    config.merge_decision.llm_tier = "rule_only"
    config.llm_optimize.enabled = False

    pipeline = Pipeline(config)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        AudioUtils,
        "load_audio",
        staticmethod(lambda _: (np.zeros(16000, dtype=np.float32), 16000)),
    )
    pipeline._run_early_detection = lambda *args: (
        PipelineContext(
            audio_path=input_path,
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
        ),
        [],
        None,
        NoiseProfile(0.0, 0.0, False),
    )
    pipeline._build_physical_shadow = lambda *args: SimpleNamespace(
        status="ok",
        diagnostics={},
        statistics={},
    )
    pipeline._post_process_events = lambda events, *args, **kwargs: events
    pipeline._finalize_events = lambda events, stats, duration: events
    pipeline._get_subtitle_builder = lambda: object()
    pipeline._export_subtitles_multi_format = (
        lambda builder, events, output_path, output_format, session_dir, label:
        {output_format: str(output_path)}
    )
    return pipeline, input_path


def test_auto_global_success_keeps_segmented_as_primary_path(monkeypatch, tmp_path):
    """Global ASR is observed as evidence after segmented production."""
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path)
    calls = {"global": 0, "segmented": 0}

    def run_global(**kwargs):
        calls["global"] += 1
        return [_event()], {"physical_coverage": {"complete": True}}, _transcript()

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        pipeline._global_review_timeline = PhysicalTimeline.from_duration(1.0)
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path = run_global
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(input_path, output_path=tmp_path / "out.srt", skip_separation=True)

    assert calls == {"global": 1, "segmented": 1}
    assert result["stats"].asr_path == "segmented"
    assert result["stats"].global_attempted is True
    assert result["events"][0].text == "segmented"


def test_auto_global_failure_keeps_segmented_output_and_records_evidence_failure(
    monkeypatch, tmp_path
):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path)
    calls = {"global": 0, "segmented": 0}

    def run_global(**kwargs):
        calls["global"] += 1
        raise RuntimeError("model crashed")

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        pipeline._global_review_timeline = PhysicalTimeline.from_duration(1.0)
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path = run_global
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(input_path, output_path=tmp_path / "out.srt", skip_separation=True)

    assert calls == {"global": 1, "segmented": 1}
    assert result["stats"].asr_path == "segmented"
    assert result["stats"].fallback_category == ""
    assert result["stats"].global_diagnostics["evidence"]["failure_category"] == "execution_failed"
    assert [event.text for event in result["events"]] == ["segmented"]


def test_dependency_unavailable_falls_back_to_segmented_path(monkeypatch, tmp_path):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path, routing="auto")
    segmented_calls = []
    pipeline._run_global_transcription_path = lambda **kwargs: (
        (_ for _ in ()).throw(ImportError("WhisperX is not installed"))
    )
    def run_skeleton(**kwargs):
        segmented_calls.append(True)
        pipeline._global_review_timeline = PhysicalTimeline.from_duration(1.0)
        return [_event("segmented")], 1, None

    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(
        input_path, output_path=tmp_path / "out.srt", skip_separation=True
    )

    assert segmented_calls == [True]
    assert result["stats"].fallback_category == ""
    assert result["stats"].global_diagnostics["evidence"]["failure_category"] == "dependency_unavailable"
    assert [event.text for event in result["events"]] == ["segmented"]


def test_early_detection_shadow_preserves_detector_sources(monkeypatch, tmp_path):
    config = PipelineConfig()
    config.vad.ffmpeg_enabled = True
    config.fusion.enabled = False
    pipeline = Pipeline(config)
    audio = np.zeros(16000, dtype=np.float32)
    audio_path = tmp_path / "vocals.wav"
    audio_path.write_bytes(b"placeholder")
    silero = [SpeechSegment(0.1, 0.5, 0.9)]
    ffmpeg = {
        "coarse_speech": [(0.0, 0.6)],
        "skeleton": [(0.1, 0.5)],
        "raw_silence_intervals": [],
    }
    pipeline._run_vad = lambda *_: silero
    pipeline._run_ffmpeg_vad = lambda path, ctx, prefix: ffmpeg

    context, selected, ffmpeg_result, noise = pipeline._run_early_detection(
        audio, 16000, audio_path
    )
    shadow = pipeline._build_physical_shadow(
        audio, 16000, audio_path, context, selected, ffmpeg_result, noise
    )
    sources = {item.source for item in shadow.physical_timeline.speech_evidence_spans}

    assert selected == silero
    assert sources == {"silero", "ffmpeg_coarse", "ffmpeg_skeleton"}
    assert shadow.physical_timeline.validate() == []


def test_global_path_passes_resolved_language_to_engine():
    class RecordingEngine(ASREngine):
        def __init__(self):
            self.languages = []

        @property
        def name(self):
            return "recording"

        @property
        def model_name(self):
            return "recording"

        def load_model(self):
            return None

        def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
            self.languages.append(language)
            return [
                TranscriptionSegment(
                    text="hello",
                    start=0.2,
                    end=0.4,
                    words=[WordTimestamp("hello", 0.2, 0.4, confidence=0.9)],
                )
            ]

    config = PipelineConfig()
    config.asr.language = "en"
    pipeline = Pipeline(config)
    engine = RecordingEngine()
    pipeline._get_global_asr_engine = lambda: engine
    timeline = PhysicalTimeline.from_duration(1.0)
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-000001")
    shadow = SimpleNamespace(
        physical_timeline=timeline,
        global_speaker_timeline=GlobalSpeakerTimeline(
            duration=1.0,
            turns=[],
            exclusive_turns=[],
            backend="test",
            status="unknown",
        ),
    )
    stats = SimpleNamespace(duration_seconds=1.0)

    events, _, _ = pipeline._run_global_transcription_path(
        np.zeros(16000, dtype=np.float32), 16000, shadow, stats
    )

    assert engine.languages == ["en"]
    assert [event.text for event in events] == ["hello"]


def test_global_path_repairs_segment_bounds_to_cover_native_word_timestamps():
    class BoundaryDriftEngine(ASREngine):
        @property
        def name(self):
            return "boundary-drift"

        @property
        def model_name(self):
            return "boundary-drift"

        def load_model(self):
            return None

        def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
            return [
                TranscriptionSegment(
                    text="hello",
                    start=0.2,
                    end=0.4,
                    words=[WordTimestamp("hello", 0.2, 0.6, confidence=0.9)],
                )
            ]

    config = PipelineConfig()
    config.asr.language = "en"
    pipeline = Pipeline(config)
    pipeline._get_global_asr_engine = lambda: BoundaryDriftEngine()
    timeline = PhysicalTimeline.from_duration(1.0)
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-000001")
    shadow = SimpleNamespace(
        physical_timeline=timeline,
        global_speaker_timeline=GlobalSpeakerTimeline(
            duration=1.0,
            turns=[],
            exclusive_turns=[],
            backend="test",
            status="unknown",
        ),
    )
    stats = SimpleNamespace(duration_seconds=1.0)

    _events, _diagnostics, transcript = pipeline._run_global_transcription_path(
        np.zeros(16000, dtype=np.float32), 16000, shadow, stats
    )

    assert transcript.validate() == []
    assert transcript.segments[0].raw_end == 0.6


def test_global_path_repairs_short_ffmpeg_tail_from_longer_vad_evidence():
    timeline = PhysicalTimeline.from_duration(5.0)
    timeline.add_evidence(
        0.0, 3.0, "ffmpeg_skeleton", physical_clip_id="clip-000001"
    )
    timeline.add_evidence(0.0, 4.2, "silero", physical_clip_id="clip-000001")

    repair = Pipeline._repair_tail_evidence(timeline, 5.0)

    assert repair["status"] == "extended"
    assert repair["alternative_end"] == 4.2
    assert max(
        item.end
        for item in timeline.speech_evidence_spans
        if item.source == "ffmpeg_skeleton"
    ) == 4.2


def test_segmented_asr_surfaces_all_segment_failures():
    class FailingEngine(ASREngine):
        @property
        def name(self):
            return "failing"

        @property
        def model_name(self):
            return "failing"

        def load_model(self):
            return None

        def transcribe(self, *args, **kwargs):
            raise RuntimeError("backend failed")

    pipeline = Pipeline(PipelineConfig())
    pipeline.config.cache.enabled = False
    pipeline._progress = SimpleNamespace(update_stage=lambda *args, **kwargs: None)
    pipeline._get_asr_engine = lambda: FailingEngine()
    segments = [SpeechSegment(0.0, 0.5, 0.9), SpeechSegment(1.0, 1.5, 0.9)]

    with pytest.raises(ASRInvalidResultError, match="no usable subtitles"):
        pipeline._run_asr(np.zeros(24000, dtype=np.float32), 16000, segments)


def test_segmented_asr_allows_no_speech_input():
    class NoopEngine(ASREngine):
        @property
        def name(self):
            return "noop"

        @property
        def model_name(self):
            return "noop"

        def load_model(self):
            return None

        def transcribe(self, *args, **kwargs):
            return []

    pipeline = Pipeline(PipelineConfig())
    pipeline.config.cache.enabled = False
    pipeline._progress = SimpleNamespace(update_stage=lambda *args, **kwargs: None)
    pipeline._get_asr_engine = lambda: NoopEngine()

    assert pipeline._run_asr(np.zeros(16000, dtype=np.float32), 16000, []) == []


def test_skeleton_segment_skips_empty_asr_result_and_continues(monkeypatch, tmp_path):
    pipeline = Pipeline(PipelineConfig())
    pipeline._progress = SimpleNamespace(
        start_stage=lambda *args, **kwargs: None,
        finish_stage=lambda *args, **kwargs: None,
    )
    pipeline.config.acoustic_validation.skeleton_min_speech = 0.1
    pipeline._process_chunk_pipeline = lambda **kwargs: (
        (_ for _ in ()).throw(ASRInvalidResultError("short burst"))
        if kwargs["chunk_label"] == "Seg 1/2"
        else ([_event("later")], 1, None)
    )

    monkeypatch.setattr(
        "vocal_subtitle.vad.ffmpeg_vad.unified_ffmpeg_pass",
        lambda *args, **kwargs: {"skeleton": [(0.0, 0.5), (1.0, 2.0)]},
    )
    monkeypatch.setattr(AudioUtils, "save_audio", lambda *args, **kwargs: None)

    events, segment_count, _ = pipeline._process_skeleton_segmented(
        np.zeros(2 * 16000, dtype=np.float32),
        16000,
        tmp_path / "input.wav",
    )

    assert segment_count == 1
    assert [event.text for event in events] == ["later"]
    assert events[0].start == pytest.approx(1.1)


def test_skeleton_segment_retries_failed_group_by_physical_members(monkeypatch, tmp_path):
    pipeline = Pipeline(PipelineConfig())
    pipeline._progress = SimpleNamespace(
        start_stage=lambda *args, **kwargs: None,
        finish_stage=lambda *args, **kwargs: None,
    )
    pipeline.config.acoustic_validation.skeleton_min_speech = 0.1
    calls = []

    def process_chunk(**kwargs):
        calls.append((kwargs["chunk_label"], len(kwargs["audio"])))
        if "fallback" not in kwargs["chunk_label"]:
            raise ASRInvalidResultError("context window rejected")
        return ([_event(kwargs["chunk_label"])], 1, None)

    pipeline._process_chunk_pipeline = process_chunk

    monkeypatch.setattr(
        "vocal_subtitle.vad.ffmpeg_vad.unified_ffmpeg_pass",
        lambda *args, **kwargs: {
            "skeleton": [(0.0, 0.5), (0.7, 1.2)],
        },
    )
    monkeypatch.setattr(AudioUtils, "save_audio", lambda *args, **kwargs: None)

    events, segment_count, _ = pipeline._process_skeleton_segmented(
        np.zeros(2 * 16000, dtype=np.float32),
        16000,
        tmp_path / "input.wav",
    )

    assert [label for label, _ in calls] == [
        "Seg 1/1",
        "Seg 1/1 fallback 1/2",
        "Seg 1/1 fallback 2/2",
    ]
    assert segment_count == 2
    assert len(events) == 2
    assert events[0].start == pytest.approx(0.1)
    assert events[1].start == pytest.approx(0.1 + 0.7)
    assert all(event.time_offset_trace for event in events)


def test_skeleton_segment_raises_when_all_asr_results_are_empty(monkeypatch, tmp_path):
    pipeline = Pipeline(PipelineConfig())
    pipeline._progress = SimpleNamespace(
        start_stage=lambda *args, **kwargs: None,
        finish_stage=lambda *args, **kwargs: None,
    )
    pipeline.config.acoustic_validation.skeleton_min_speech = 0.1
    pipeline._process_chunk_pipeline = lambda **kwargs: (
        (_ for _ in ()).throw(ASRInvalidResultError("no subtitles"))
    )

    monkeypatch.setattr(
        "vocal_subtitle.vad.ffmpeg_vad.unified_ffmpeg_pass",
        lambda *args, **kwargs: {"skeleton": [(0.0, 0.5)]},
    )
    monkeypatch.setattr(AudioUtils, "save_audio", lambda *args, **kwargs: None)

    with pytest.raises(ASRInvalidResultError, match="any of 1 skeleton speech segments"):
        pipeline._process_skeleton_segmented(
            np.zeros(16000, dtype=np.float32),
            16000,
            tmp_path / "input.wav",
        )


def test_finalize_events_uses_pipeline_subtitle_config():
    pipeline = Pipeline(PipelineConfig())
    pipeline.config.subtitle.max_duration = 2.0
    stats = SimpleNamespace(subtitle_count=0, quality_diagnostics={})
    event = SubtitleEvent(
        index=1,
        start=0.0,
        end=10.0,
        text="这是一条需要根据实例配置拆分的较长中文字幕句子以便验证最终化逻辑",
        physical_start=0.0,
        physical_end=10.0,
    )

    result = pipeline._finalize_events([event], stats, audio_duration=10.0)

    diagnostics = stats.quality_diagnostics["finalization"]
    assert diagnostics["split_long_event_count"] > 0
    assert diagnostics["input_event_count"] > 1
    assert stats.subtitle_count == len(result)
