"""Speaker fusion tests without loading optional model weights."""

import numpy as np

from vocal_subtitle.asr.base import WordTimestamp
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.diarization.base import DiarizationResult, SpeakerTurn
from vocal_subtitle.diarization import speaker_fusion
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


class _FakeEmbedding:
    model_loaded = True
    name = "fake-ecapa"

    def extract_embedding(self, audio, sample_rate):
        mean = float(np.mean(audio)) if len(audio) else 0.0
        return np.array([1.0, 0.0]) if mean >= 0 else np.array([0.0, 1.0])


def _config(*, global_model="none", local_refinement="embedding", expected=None):
    config = PipelineConfig()
    config.diarization.enabled = True
    config.diarization.fusion_mode = "embedding"
    config.diarization.global_model = global_model
    config.diarization.local_refinement = local_refinement
    config.diarization.expected_speakers = expected
    config.diarization.min_change_confidence = 0.2
    return config


def test_local_refinement_splits_one_subtitle_at_word_boundary():
    audio = np.ones(6 * 16000, dtype=np.float32)
    audio[3 * 16000 :] = -1.0
    event = SubtitleEvent(
        1,
        2.0,
        4.8,
        "甲。乙",
        words=[
            WordTimestamp("甲。", 2.1, 2.7),
            WordTimestamp("乙", 3.2, 3.8),
        ],
        source_word_ids=["w1", "w2"],
    )

    result = speaker_fusion.run_speaker_fusion(
        [event], audio, 16000, _config(expected=2),
        embedding_engine=_FakeEmbedding(),
    )

    assert result.local_split_count == 1
    assert len(result.events) == 2
    assert [item.text for item in result.events] == ["甲。", "乙"]
    assert [item.start for item in result.events] == [2.1, 3.2]
    assert [item.source_word_ids for item in result.events] == [["w1"], ["w2"]]


def test_local_refinement_respects_minimum_part_duration():
    audio = np.ones(4 * 16000, dtype=np.float32)
    audio[2 * 16000 :] = -1.0
    event = SubtitleEvent(
        1,
        0.0,
        3.0,
        "甲乙",
        words=[WordTimestamp("甲", 0.2, 0.3), WordTimestamp("乙", 2.2, 2.8)],
    )
    config = _config(expected=2)
    config.diarization.min_local_segment_seconds = 0.25

    result = speaker_fusion.run_speaker_fusion(
        [event], audio, 16000, config, embedding_engine=_FakeEmbedding(),
    )

    assert result.local_split_count == 0
    assert len(result.events) == 1


def test_global_turns_split_event_and_map_to_embedding_identity(monkeypatch):
    audio = np.ones(4 * 16000, dtype=np.float32)
    event = SubtitleEvent(
        1,
        0.0,
        2.0,
        "甲乙",
        words=[WordTimestamp("甲", 0.2, 0.8), WordTimestamp("乙", 1.2, 1.8)],
    )
    global_result = DiarizationResult(
        turns=[SpeakerTurn(0.0, 1.0, 0), SpeakerTurn(1.0, 2.0, 1)],
        exclusive_turns=[SpeakerTurn(0.0, 1.0, 0), SpeakerTurn(1.0, 2.0, 1)],
        speaker_count=2,
        backend="pyannote-community-1",
        status="ok",
    )
    monkeypatch.setattr(
        speaker_fusion,
        "_run_global_pass",
        lambda audio, sample_rate, config: (
            global_result,
            "pyannote/speaker-diarization-community-1",
            "ok",
        ),
    )
    config = _config(global_model="community-1", local_refinement="off")
    config.diarization.fusion_mode = "dual"

    result = speaker_fusion.run_speaker_fusion(
        [event], audio, 16000, config, embedding_engine=_FakeEmbedding(),
    )

    assert result.backend == "fused"
    assert len(result.events) == 2
    assert [item.speaker_id for item in result.events] == [0, 1]
    # The synthetic embedding has no identity change, so the global turns
    # remain authoritative when the two-speaker mapping is underdetermined.
    assert all(item.speaker_source == "global" for item in result.events)


def test_global_pass_uses_shared_default_model_cache(monkeypatch):
    captured = {}

    class _FakeGlobalEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def diarize(self, **kwargs):
            return DiarizationResult(status="ok")

    monkeypatch.setattr(speaker_fusion, "is_model_cached", lambda *args: True)
    monkeypatch.setattr(
        "vocal_subtitle.diarization.pyannote_engine.PyannoteDiarizationEngine",
        _FakeGlobalEngine,
    )

    config = _config(global_model="auto", local_refinement="off")
    config.diarization.fusion_mode = "dual"
    config.speaker_embedding.cache_dir = ""

    result, model_ref, status = speaker_fusion._run_global_pass(
        np.zeros(16000, dtype=np.float32), 16000, config,
    )

    assert result is not None
    assert model_ref == "pyannote/speaker-diarization-community-1"
    assert status == "ok"
    assert captured["cache_dir"] == str(speaker_fusion.DEFAULT_CACHE_DIR)


def test_missing_both_lines_keeps_unknown_without_alternation():
    config = _config(global_model="none", local_refinement="off")
    config.speaker_embedding.enabled = False
    events = [SubtitleEvent(1, 0.0, 1.0, "甲"), SubtitleEvent(2, 1.2, 2.0, "乙")]

    result = speaker_fusion.run_speaker_fusion(
        events, np.zeros(32000, dtype=np.float32), 16000, config,
    )

    assert result.status == "degraded"
    assert result.speaker_count == 0
    assert [item.speaker_id for item in result.events] == [None, None]
    assert result.diagnostics["unknown_count"] == 2


def test_known_multi_speaker_count_does_not_split_identical_embeddings():
    audio = np.ones(8 * 16000, dtype=np.float32)
    events = [
        SubtitleEvent(1, 0.0, 1.0, "甲"),
        SubtitleEvent(2, 2.0, 3.0, "乙"),
    ]

    result = speaker_fusion.run_speaker_fusion(
        events, audio, 16000, _config(expected=2),
        embedding_engine=_FakeEmbedding(),
    )

    assert result.backend == "unknown"
    assert result.diagnostics["embedding_status"] == "failed"
    assert [item.speaker_id for item in result.events] == [None, None]
