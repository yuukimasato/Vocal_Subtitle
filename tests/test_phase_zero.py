"""阶段零：物理优先时间轴、speaker 上限和 ASR 元数据回归。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
from vocal_subtitle.asr.faster_whisper_engine import FasterWhisperEngine
from vocal_subtitle.config import ConfigLoader, PipelineConfig
from vocal_subtitle.diarization.base import DiarizationResult, SpeakerTurn
from vocal_subtitle.diarization.canonicalizer import canonicalize_diarization_result
from vocal_subtitle.mapping.time_mapper import TimeMapper
from vocal_subtitle.pipeline import Pipeline, PipelineStats
from vocal_subtitle.utils.cache_manager import CacheManager
from vocal_subtitle.vad.base import SpeechSegment


def test_single_event_uses_physical_boundaries_when_asr_is_early():
    events = TimeMapper().map(
        [[TranscriptionSegment(
            "测试", 0.1, 0.3, [WordTimestamp("测试", 0.12, 0.2)],
        )]],
        [SpeechSegment(4.0, 5.0)],
    )

    assert (events[0].start, events[0].end) == (4.0, 5.0)


def test_multiple_events_keep_physical_outer_envelope():
    events = TimeMapper().map(
        [[
            TranscriptionSegment("甲", 0.1, 0.4),
            TranscriptionSegment("乙", 0.5, 0.7),
        ]],
        [SpeechSegment(4.0, 5.0)],
    )

    assert events[0].start == 4.0
    assert events[-1].end == 5.0
    assert all(4.0 <= event.start < event.end <= 5.0 for event in events)


def test_canonicalizer_caps_speakers_without_inventing_missing_ids():
    result = DiarizationResult(
        turns=[
            SpeakerTurn(0.0, 1.0, 8),
            SpeakerTurn(1.0, 2.0, 3),
            SpeakerTurn(2.0, 3.0, 5),
        ],
        exclusive_turns=[
            SpeakerTurn(0.0, 1.0, 8),
            SpeakerTurn(1.0, 2.0, 3),
            SpeakerTurn(2.0, 3.0, 5),
        ],
        speaker_count=3,
        backend="test",
    )

    normalized = canonicalize_diarization_result(result, max_speakers=2)

    assert normalized.speaker_count == 2
    assert len({turn.speaker_id for turn in normalized.turns}) == 2
    assert normalized.diagnostics["raw_diarization_speaker_count"] == 3
    assert normalized.diagnostics["canonicalization_status"] == "degraded"


def test_expected_speakers_is_loaded_and_partitions_cache():
    config = ConfigLoader._parse_config({
        "pipeline": {"diarization": {"expected_speakers": 2}}
    })
    assert config.diarization.expected_speakers == 2

    first = CacheManager.make_key(Path("audio.wav"), expected_speakers=2)
    second = CacheManager.make_key(Path("audio.wav"), expected_speakers=3)
    assert first != second


def test_faster_whisper_preserves_quality_metadata():
    engine = FasterWhisperEngine(device="cpu")
    engine._model = SimpleNamespace(transcribe=lambda *args, **kwargs: (
        iter([SimpleNamespace(
            text=" test ", start=0.0, end=1.0, words=[SimpleNamespace(
                word="test", start=0.1, end=0.8, probability=0.9,
            )], avg_logprob=-0.2, no_speech_prob=0.1, compression_ratio=1.2,
        )]),
        SimpleNamespace(language="en", language_probability=0.99),
    ))

    results = engine.transcribe(np.zeros(16000, dtype=np.float32), language="en")

    assert results[0].no_speech_prob == 0.1
    assert results[0].compression_ratio == 1.2
    assert results[0].words[0].speaker_id is None


def test_stats_expose_canonicalization_fields():
    stats = PipelineStats(input_path=Path("audio.wav"), duration_seconds=1.0)
    stats.raw_diarization_speaker_count = 3
    stats.canonical_speaker_count = 2
    stats.speaker_merge_map = {8: 0, 3: 1, 5: 0}
    stats.canonicalization_status = "degraded"

    payload = stats.to_dict()

    assert payload["raw_diarization_speaker_count"] == 3
    assert payload["canonical_speaker_count"] == 2
    assert payload["speaker_merge_map"] == {8: 0, 3: 1, 5: 0}


def test_ffmpeg_vad_uses_acoustic_validation_settings(monkeypatch, tmp_path):
    calls = []

    def fake_pass(path, *, noise_db, min_silence_duration):
        calls.append((path, noise_db, min_silence_duration))
        return {"coarse_speech": [], "skeleton": []}

    import vocal_subtitle.vad.ffmpeg_vad as ffmpeg_vad

    monkeypatch.setattr(ffmpeg_vad, "unified_ffmpeg_pass", fake_pass)
    config = PipelineConfig()
    config.acoustic_validation.skeleton_noise_db = -51.0
    config.acoustic_validation.skeleton_min_silence = 0.23
    pipeline = Pipeline(config)
    context = SimpleNamespace(
        ffmpeg_unified_result=None,
        acoustic_skeleton=[],
        add_diagnostic=lambda message: None,
    )

    pipeline._run_ffmpeg_vad(tmp_path / "vocals.wav", context)

    assert calls == [(tmp_path / "vocals.wav", -51.0, 0.23)]


def test_final_pipeline_clamp_reapplies_physical_envelope():
    from vocal_subtitle.mapping.time_mapper import SubtitleEvent

    event = SubtitleEvent(
        index=1,
        start=0.0,
        end=3.0,
        text="越界",
        physical_start=1.0,
        physical_end=2.0,
    )

    result = Pipeline._clamp_to_physical_envelopes([event])

    assert [(item.start, item.end) for item in result] == [(1.0, 2.0)]


def test_physical_envelope_is_shifted_with_chunk_time():
    from vocal_subtitle.mapping.time_mapper import SubtitleEvent

    event = SubtitleEvent(
        index=1,
        start=0.0,
        end=1.0,
        text="块内",
        physical_start=0.0,
        physical_end=1.0,
    )
    chunk_offset = 10.0
    event.start += chunk_offset
    event.end += chunk_offset
    event.physical_start += chunk_offset
    event.physical_end += chunk_offset

    result = Pipeline._clamp_to_physical_envelopes([event])

    assert [(item.start, item.end) for item in result] == [(10.0, 11.0)]
