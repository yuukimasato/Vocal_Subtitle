"""全局 speaker timeline 到 ASR/字幕边界的回归测试。"""

from vocal_subtitle.asr.base import WordTimestamp
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.diarization.base import DiarizationResult, SpeakerTurn
from vocal_subtitle.mapping.time_mapper import SubtitleEvent, TimeMapper
from vocal_subtitle.merging.llm_merge_engine import LLMMergeEngine, MergeDecisionConfig
from vocal_subtitle.pipeline import Pipeline, PipelineStats
from vocal_subtitle.vad.base import SpeechSegment


def test_pipeline_projects_complex_alternation_without_local_renumbering():
    pipeline = Pipeline(PipelineConfig())
    pipeline._global_diarization = DiarizationResult(
        turns=[], exclusive_turns=[], speaker_count=3, backend="test",
    )
    pipeline._global_turns = [
        SpeakerTurn(0.0, 1.0, 0),
        SpeakerTurn(1.0, 2.0, 1),
        SpeakerTurn(2.0, 3.0, 0),
        SpeakerTurn(3.0, 4.0, 0),
        SpeakerTurn(4.0, 5.0, 1),
        SpeakerTurn(5.0, 6.0, 2),
        SpeakerTurn(6.0, 7.0, 1),
    ]

    segments, speaker_ids = pipeline._project_global_speakers(
        [SpeechSegment(0.0, 7.0)], time_offset=0.0, duration=7.0,
    )

    projected_times = [
        (round(segment.start, 3), round(segment.end, 3))
        for segment in segments
    ]
    assert projected_times == [
        (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
        (4.0, 5.0), (5.0, 6.0), (6.0, 7.0),
    ]
    assert speaker_ids == [0, 1, 0, 0, 1, 2, 1]


def test_single_speaker_projection_envelope_does_not_split_on_pause_turns():
    pipeline = Pipeline(PipelineConfig())
    pipeline._global_diarization = DiarizationResult(
        turns=[], exclusive_turns=[], speaker_count=1, backend="test",
    )
    pipeline._global_turns = [SpeakerTurn(0.0, 10.0, 0)]

    segments, speaker_ids = pipeline._project_global_speakers(
        [SpeechSegment(2.0, 2.8), SpeechSegment(5.0, 5.4)],
        time_offset=0.0,
        duration=8.0,
    )

    assert [(segment.start, segment.end) for segment in segments] == [
        (2.0, 2.8), (5.0, 5.4),
    ]
    assert speaker_ids == [0, 0]


def test_merge_engine_rejects_different_numeric_speakers():
    engine = LLMMergeEngine(MergeDecisionConfig(llm_tier="rule_only"))
    fragments = [
        {
            "id": 1, "start": 0.0, "end": 1.0, "text": "A",
            "speaker": "Speaker A", "speaker_id": 0,
            "gap_to_next_sec": 0.05,
        },
        {
            "id": 2, "start": 1.05, "end": 2.0, "text": "B",
            "speaker": "Speaker B", "speaker_id": 1,
            "gap_to_next_sec": None,
        },
    ]

    merged = engine.merge(fragments)

    assert len(merged) == 2
    assert [fragment["speaker_id"] for fragment in merged] == [0, 1]


def test_dedup_does_not_assign_unknown_event_to_known_speaker():
    events = [
        SubtitleEvent(1, 0.0, 2.0, "same text", speaker_id=None),
        SubtitleEvent(2, 0.2, 1.8, "same text", speaker_id=0),
    ]

    result = TimeMapper._deduplicate_overlapping(events)

    assert len(result) == 2
    assert {event.speaker_id for event in result} == {None, 0}


def test_final_boundary_check_splits_word_timestamps_by_speaker():
    pipeline = Pipeline(PipelineConfig())
    pipeline._global_turns = [
        SpeakerTurn(0.0, 1.5, 0),
        SpeakerTurn(1.5, 3.0, 1),
    ]
    stats = PipelineStats(input_path=None, duration_seconds=3.0)
    event = SubtitleEvent(
        1,
        0.0,
        3.0,
        "甲乙",
        words=[
            WordTimestamp("甲", 0.0, 1.0),
            WordTimestamp("乙", 1.0, 2.5),
        ],
        speaker_id=0,
        original_text="原始甲乙",
        physical_start=0.0,
        physical_end=3.0,
        physical_spans=[{"physical_clip_id": "clip-a", "start": 0.0, "end": 3.0}],
        logical_sentence_id=7,
        source_word_ids=["word-a", "word-b"],
        alignment_warning="test-warning",
        hard_split_before=True,
        physical_region_id="region-a",
    )

    result = pipeline._enforce_speaker_boundaries([event], stats)

    assert [(item.start, item.end, item.speaker_id, item.text) for item in result] == [
        (0.0, 1.5, 0, "甲"),
        (1.5, 3.0, 1, "乙"),
    ]
    assert [item.speaker_label for item in result] == ["Speaker A", "Speaker B"]
    assert [item.source_word_ids for item in result] == [["word-a"], ["word-b"]]
    assert all(item.physical_region_id == "region-a" for item in result)
    assert all(item.physical_spans for item in result)
    assert all(item.alignment_warning == "test-warning" for item in result)
    assert result[1].words[0].start == 0.0
    assert stats.mixed_event_count == 1


def test_final_boundary_check_marks_wordless_cross_speaker_event_unknown():
    pipeline = Pipeline(PipelineConfig())
    pipeline._global_turns = [
        SpeakerTurn(0.0, 1.0, 0),
        SpeakerTurn(1.0, 2.0, 1),
    ]
    stats = PipelineStats(input_path=None, duration_seconds=2.0)
    result = pipeline._enforce_speaker_boundaries(
        [SubtitleEvent(1, 0.0, 2.0, "无法按词分配")], stats,
    )

    assert len(result) == 1
    assert result[0].start == 0.0
    assert result[0].end == 1.0
    assert result[0].speaker_id is None
    assert stats.mixed_event_count == 1


def test_stats_expose_diarization_diagnostics():
    stats = PipelineStats(input_path=None, duration_seconds=1.0)
    stats.diarization_backend = "test"
    stats.diarization_status = "fallback"
    stats.fallback_reason = "backend unavailable"

    payload = stats.to_dict()

    assert payload["diarization_backend"] == "test"
    assert payload["diarization_status"] == "fallback"
    assert payload["fallback_reason"] == "backend unavailable"
    assert "atomic_span_count" in payload


def test_pyannote_does_not_reuse_legacy_diarization_cache():
    fallback = DiarizationResult(
        turns=[SpeakerTurn(0.0, 1.0, 0)],
        exclusive_turns=[SpeakerTurn(0.0, 1.0, 0)],
        speaker_count=1,
        backend="legacy-global-fallback",
        status="fallback",
    )
    pyannote = DiarizationResult(
        turns=[SpeakerTurn(0.0, 1.0, 0)],
        exclusive_turns=[SpeakerTurn(0.0, 1.0, 0)],
        speaker_count=1,
        backend="pyannote-community-1",
        status="ok",
    )

    assert not Pipeline._is_usable_diarization_cache(fallback, "pyannote")
    assert not Pipeline._is_usable_diarization_cache(fallback, "auto")
    assert Pipeline._is_usable_diarization_cache(pyannote, "pyannote")
    assert Pipeline._is_usable_diarization_cache(fallback, "legacy")


def test_time_mapper_does_not_extend_known_event_into_unknown_event():
    mapper = TimeMapper(seamless_threshold=1.0, natural_pause_max=1.0)
    events = [
        SubtitleEvent(1, 0.0, 1.0, "已知", speaker_id=0),
        SubtitleEvent(2, 1.2, 2.0, "未知", speaker_id=None),
    ]

    mapper._merge_gaps(events)

    assert events[0].end == 1.0
