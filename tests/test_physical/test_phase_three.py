from types import SimpleNamespace

import numpy as np

from vocal_subtitle.acoustic_validator import AcousticValidator
from vocal_subtitle.asr.base import ASREngine, TranscriptionSegment, WordTimestamp
from vocal_subtitle.asr.global_transcriber import (
    GlobalTranscriber,
    GlobalTranscriberConfig,
)
from vocal_subtitle.asr.whisperx_engine import normalize_whisperx_transcript
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.diarization.base import SpeakerTurn
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.physical.allocator import allocate_words
from vocal_subtitle.physical.events import build_events
from vocal_subtitle.physical.ir import (
    GlobalSpeakerTimeline,
    GlobalTranscript,
    GlobalWord,
)
from vocal_subtitle.physical.timeline import ContextWindow, PhysicalTimeline
from vocal_subtitle.physical.subtitle_bins import build_physical_subtitle_bins
from vocal_subtitle.pipeline import Pipeline, PipelineStats
from vocal_subtitle.utils.progress import ProgressManager


def _word(word_id, text, start, end, speaker_id=None):
    return GlobalWord(
        id=word_id,
        text=text,
        raw_start=start,
        raw_end=end,
        confidence=0.9,
        source_window_id="window",
        segment_id="segment",
        speaker_id=speaker_id,
    )


def _transcript(words):
    from vocal_subtitle.physical.ir import GlobalTranscriptSegment

    return GlobalTranscript(
        audio_duration=3.0,
        words=words,
        segments=[
            GlobalTranscriptSegment(
                id="segment",
                text=" ".join(word.text for word in words),
                raw_start=words[0].raw_start,
                raw_end=words[-1].raw_end,
                word_ids=[word.id for word in words],
            )
        ],
        backend="test",
        status="ok",
    )


def test_whisperx_normalization_is_lazy_and_maps_speaker_labels():
    transcript = normalize_whisperx_transcript(
        [
            {
                "text": "hello world",
                "start": 0.2,
                "end": 1.0,
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "hello", "start": 0.2, "end": 0.5, "score": 0.8},
                    {"word": "world", "start": 0.6, "end": 1.0},
                ],
            }
        ],
        source_window_id="ctx-a",
        segment_id_prefix="seg-a",
        time_offset=10.0,
        audio_duration=20.0,
    )

    assert [(word.raw_start, word.raw_end) for word in transcript.words] == [
        (10.2, 10.5),
        (10.6, 11.0),
    ]
    assert transcript.words[0].speaker_id == 1
    assert transcript.segments[0].metadata["speaker_id"] == 1


def test_allocator_preserves_cross_clip_word_as_whole_with_multiple_spans():
    timeline = PhysicalTimeline(3.0)
    timeline.add_clip(0.0, 1.0, clip_id="clip-a")
    timeline.add_clip(1.0, 2.0, clip_id="clip-b")
    timeline.add_evidence(0.0, 1.0, "vad", physical_clip_id="clip-a")
    timeline.add_evidence(1.0, 2.0, "vad", physical_clip_id="clip-b")
    transcript = _transcript([_word("w1", "今天天气", 0.8, 1.2, speaker_id=0)])

    result = allocate_words(transcript, timeline)
    assert len(result.accepted) == 1
    assert result.accepted[0].word.text == "今天天气"
    assert [span.clip_id for span in result.accepted[0].physical_spans] == [
        "clip-a",
        "clip-b",
    ]
    assert "cross_physical_boundary" in result.accepted[0].warnings

    event = build_events(result)[0]
    assert event.text == "今天天气"
    assert event.source_word_ids == ["w1"]
    assert len(event.physical_spans) == 2


def test_global_events_split_when_speech_evidence_has_a_gap():
    timeline = PhysicalTimeline(3.0)
    timeline.add_clip(0.0, 3.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 0.7, "ffmpeg_skeleton", physical_clip_id="clip-a")
    timeline.add_evidence(1.2, 2.0, "ffmpeg_skeleton", physical_clip_id="clip-a")
    transcript = _transcript(
        [
            _word("w1", "第一句", 0.2, 0.5),
            _word("w2", "第二句", 1.3, 1.6),
        ]
    )

    result = allocate_words(transcript, timeline)
    events = build_events(result)

    assert [event.text for event in events] == ["第一句", "第二句"]
    assert events[1].hard_split_before is True
    assert events[0].physical_spans[0].start == 0.2
    assert events[0].physical_spans[0].end == 0.5


def test_global_event_hard_boundary_survives_subtitle_export():
    timeline = PhysicalTimeline(3.0)
    timeline.add_clip(0.0, 3.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 0.7, "ffmpeg_skeleton", physical_clip_id="clip-a")
    timeline.add_evidence(1.2, 2.0, "ffmpeg_skeleton", physical_clip_id="clip-a")
    transcript = _transcript(
        [
            _word("w1", "第一句", 0.2, 0.5),
            _word("w2", "第二句", 1.3, 1.6),
        ]
    )
    events = build_events(allocate_words(transcript, timeline))

    from vocal_subtitle.mapping.subtitle_builder import SubtitleBuilder

    output = SubtitleBuilder().build_to_string(
        [event.to_subtitle_event() for event in events], fmt="srt"
    )

    assert output.count("\n\n") == 2


def test_physical_subtitle_bins_prefer_precise_skeleton_evidence():
    timeline = PhysicalTimeline(3.0)
    timeline.add_clip(0.0, 3.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 3.0, "boundary_fusion", physical_clip_id="clip-a")
    timeline.add_evidence(0.1, 0.8, "ffmpeg_skeleton", physical_clip_id="clip-a")
    timeline.add_evidence(1.2, 2.0, "ffmpeg_skeleton", physical_clip_id="clip-a")

    bins = build_physical_subtitle_bins(timeline)

    assert [(item.start, item.end) for item in bins] == [(0.1, 0.8), (1.2, 2.0)]


def test_physical_bin_event_preserves_bin_envelope_without_faking_word_time():
    timeline = PhysicalTimeline(3.0)
    timeline.add_clip(0.0, 3.0, clip_id="clip-a")
    timeline.add_evidence(0.1, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-a")
    transcript = _transcript([_word("w1", "今天天气", 0.3, 0.8, speaker_id=0)])

    from vocal_subtitle.physical.subtitle_bins import build_physical_subtitle_bins

    allocation = allocate_words(transcript, timeline)
    event = build_events(
        allocation,
        subtitle_bins=build_physical_subtitle_bins(timeline),
    )[0]

    assert event.text == "今天天气"
    assert (event.start, event.end) == (0.3, 0.8)
    assert (event.physical_bin_start, event.physical_bin_end) == (0.1, 1.0)
    subtitle_event = event.to_subtitle_event()
    assert (subtitle_event.physical_start, subtitle_event.physical_end) == (0.3, 0.8)
    assert event.time_source == "timing_degraded"


def test_micro_bin_fragment_merges_at_contiguous_whole_word_boundary():
    timeline = PhysicalTimeline(2.0)
    timeline.add_clip(0.0, 2.0, clip_id="clip-a")
    timeline.add_evidence(0.1, 0.45, "ffmpeg_skeleton", physical_clip_id="clip-a")
    timeline.add_evidence(0.5, 0.8, "ffmpeg_skeleton", physical_clip_id="clip-a")
    transcript = _transcript(
        [
            _word("w1", "婉儿", 0.2, 0.4, speaker_id=0),
            _word("w2", "别", 0.4, 0.6, speaker_id=0),
        ]
    )

    events = build_events(
        allocate_words(transcript, timeline),
        subtitle_bins=build_physical_subtitle_bins(timeline),
    )

    assert len(events) == 1
    assert events[0].text == "婉儿别"
    assert events[0].source_word_ids == ["w1", "w2"]


def test_long_physical_bin_splits_at_deep_energy_valley():
    sample_rate = 1000
    audio = np.ones(4 * sample_rate, dtype=np.float32)
    audio[2000:2100] = 0.0
    timeline = PhysicalTimeline(4.0)
    timeline.add_clip(0.0, 4.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 4.0, "ffmpeg_skeleton", physical_clip_id="clip-a")

    bins = build_physical_subtitle_bins(
        timeline,
        audio=audio,
        sample_rate=sample_rate,
        min_internal_silence=0.04,
    )

    assert [(round(item.start, 2), round(item.end, 2)) for item in bins] == [
        (0.0, 2.0),
        (2.1, 4.0),
    ]
    assert all(item.source.endswith("+energy_valley") for item in bins)


def test_global_shadow_pipeline_skips_legacy_asr(monkeypatch, tmp_path):
    pipeline = Pipeline(PipelineConfig())
    pipeline._progress = ProgressManager(use_tqdm=False)
    pipeline._run_vad = lambda audio, sample_rate: [
        SimpleNamespace(start=0.0, end=1.0, confidence=1.0)
    ]
    pipeline._run_ffmpeg_vad = lambda vocals_path, ctx, prefix="": {
        "coarse_speech": [(0.0, 1.0)],
        "skeleton": [(0.0, 1.0)],
    }
    pipeline._run_merging = lambda segments, audio, sample_rate, total_duration: segments

    def fail_if_called(*args, **kwargs):
        raise AssertionError("global shadow construction must not call legacy ASR")

    monkeypatch.setattr(pipeline, "_run_asr", fail_if_called)
    events, count, context = pipeline._process_chunk_pipeline(
        np.zeros(16000, dtype=np.float32),
        16000,
        tmp_path / "audio.wav",
        run_asr=False,
    )

    assert events == []
    assert count == 1
    # The PipelineContext stores ASR fragments, not raw segments
    assert getattr(context, "asr_fragments", []) == [] or getattr(context, "asr_segments", []) == []


def test_allocator_rejects_outside_words_and_falls_back_to_exclusive_turn():
    timeline = PhysicalTimeline(3.0)
    timeline.add_clip(0.5, 1.5, clip_id="clip-a")
    transcript = _transcript(
        [
            _word("inside", "hello", 0.6, 0.9),
            _word("outside", "noise", 2.0, 2.2),
        ]
    )
    speakers = GlobalSpeakerTimeline(
        duration=3.0,
        turns=[SpeakerTurn(0.5, 1.0, 4)],
        exclusive_turns=[SpeakerTurn(0.5, 1.0, 4)],
        backend="test",
        status="ok",
    )

    result = allocate_words(transcript, timeline, speakers)
    assert [item.word.id for item in result.accepted] == ["inside"]
    assert result.accepted[0].speaker_id == 4
    assert result.accepted[0].speaker_source == "exclusive_turn"
    assert [item.word.id for item in result.rejected] == ["outside"]


class _FakeEngine(ASREngine):
    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "fake"

    @property
    def model_name(self):
        return "fake"

    def load_model(self):
        return None

    def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
        self.calls += 1
        return [
            TranscriptionSegment(
                text="hello",
                start=0.4,
                end=0.7,
                words=[WordTimestamp("hello", 0.4, 0.7, confidence=0.8)],
            )
        ]


class _AligningFakeEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.alignment_calls = 0

    def align(self, audio, *, sample_rate=16000, segments=None, language=None):
        self.alignment_calls += 1
        return segments or []


def test_global_transcriber_deduplicates_overlapping_window_words():
    engine = _FakeEngine()
    transcriber = GlobalTranscriber(engine, GlobalTranscriberConfig())
    windows = [
        ContextWindow("w1", 0.0, 1.0, "clip-a", 0.0, 0.0),
        ContextWindow("w2", 0.0, 1.0, "clip-b", 0.0, 0.0),
    ]
    result = transcriber.transcribe(
        np.zeros(16000, dtype=np.float32), 16000, windows=windows
    )

    assert engine.calls == 2
    assert len(result.transcript.words) == 1
    assert result.diagnostics["deduplicated_word_count"] == 1


def test_global_transcriber_uses_optional_alignment_hook_and_keeps_ir_valid():
    engine = _AligningFakeEngine()
    transcriber = GlobalTranscriber(
        engine,
        GlobalTranscriberConfig(alignment=True),
    )
    result = transcriber.transcribe(
        np.zeros(16000, dtype=np.float32),
        16000,
        windows=[ContextWindow("w1", 0.0, 1.0, "clip-a", 0.0, 0.0)],
    )

    assert engine.alignment_calls == 1
    assert result.diagnostics["alignment_status"] == "applied"
    assert result.transcript.validate() == []


def test_pipeline_global_entry_builds_subtitle_events_from_shadow_ir():
    config = PipelineConfig()
    config.asr.global_asr.enabled = True
    config.asr.global_asr.backend = "faster-whisper"
    pipeline = Pipeline(config)
    pipeline._get_global_asr_engine = lambda: _FakeEngine()
    timeline = PhysicalTimeline.from_duration(1.0)
    timeline.add_evidence(
        0.0,
        1.0,
        "ffmpeg_skeleton",
        physical_clip_id="clip-000001",
    )
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
    stats = PipelineStats(input_path="audio.wav", duration_seconds=1.0)

    events, _, transcript = pipeline._run_global_transcription_path(
        np.zeros(16000, dtype=np.float32),
        16000,
        shadow,
        stats,
    )

    assert transcript.status == "ok"
    assert len(events) == 1
    assert events[0].source_word_ids
    assert events[0].physical_spans
    assert events[0].time_source == "boundary_decision"
    assert len(events[0].revision_trace) == 2
    assert all(
        item["decision"]["accepted"] for item in events[0].revision_trace
    )


def test_micro_gap_merge_does_not_cross_physical_clip_and_preserves_provenance():
    left = SubtitleEvent(
        index=1,
        start=0.0,
        end=0.4,
        text="hello",
        speaker_id=0,
        physical_start=0.0,
        physical_end=0.4,
        physical_spans=[{"physical_clip_id": "a", "start": 0.0, "end": 0.4}],
        source_word_ids=["w1"],
    )
    right = SubtitleEvent(
        index=2,
        start=0.43,
        end=0.8,
        text="world",
        speaker_id=0,
        physical_start=0.43,
        physical_end=0.8,
        physical_spans=[{"physical_clip_id": "b", "start": 0.43, "end": 0.8}],
        source_word_ids=["w2"],
    )

    merged, count = AcousticValidator._merge_micro_gaps([left, right], max_gap=0.05)
    assert count == 0
    assert len(merged) == 2

    same_clip = SubtitleEvent(
        index=2,
        start=0.43,
        end=0.8,
        text="world",
        speaker_id=0,
        physical_start=0.43,
        physical_end=0.8,
        physical_spans=[{"physical_clip_id": "a", "start": 0.43, "end": 0.8}],
        source_word_ids=["w2"],
    )
    merged, count = AcousticValidator._merge_micro_gaps([left, same_clip], max_gap=0.05)
    assert count == 1
    assert merged[0].source_word_ids == ["w1", "w2"]
