"""Main-pipeline routing tests for the bounded global ASR path."""

from types import SimpleNamespace

import numpy as np
import pytest

from vocal_subtitle.asr.base import ASREngine, TranscriptionSegment, WordTimestamp
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


def test_auto_global_success_skips_skeleton_segmented_path(monkeypatch, tmp_path):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path)
    calls = {"global": 0, "segmented": 0}

    def run_global(**kwargs):
        calls["global"] += 1
        return [_event()], {"physical_coverage": {"complete": True}}, _transcript()

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path = run_global
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(input_path, output_path=tmp_path / "out.srt", skip_separation=True)

    assert calls == {"global": 1, "segmented": 0}
    assert result["stats"].asr_path == "global"
    assert result["stats"].global_attempted is True
    assert result["events"][0].text == "global"


def test_auto_global_failure_falls_back_once_and_discards_global_events(
    monkeypatch, tmp_path
):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path)
    calls = {"global": 0, "segmented": 0}

    def run_global(**kwargs):
        calls["global"] += 1
        raise RuntimeError("model crashed")

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path = run_global
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(input_path, output_path=tmp_path / "out.srt", skip_separation=True)

    assert calls == {"global": 1, "segmented": 1}
    assert result["stats"].asr_path == "legacy_degraded"
    assert result["stats"].fallback_category == "execution_failed"
    assert [event.text for event in result["events"]] == ["segmented"]


def test_explicit_global_failure_does_not_fallback(monkeypatch, tmp_path):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path, routing="global")
    segmented_calls = []
    pipeline._run_global_transcription_path = lambda **kwargs: (
        (_ for _ in ()).throw(ImportError("WhisperX is not installed"))
    )
    pipeline._process_skeleton_segmented = lambda **kwargs: segmented_calls.append(True)

    with pytest.raises(ImportError, match="WhisperX"):
        pipeline.run(input_path, output_path=tmp_path / "out.srt", skip_separation=True)

    assert segmented_calls == []


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
