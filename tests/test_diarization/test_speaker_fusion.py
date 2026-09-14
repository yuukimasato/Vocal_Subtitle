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


def test_silent_embedding_windows_are_skipped():
    class _CountingEmbedding(_FakeEmbedding):
        calls = 0

        def extract_embedding(self, audio, sample_rate):
            self.calls += 1
            return super().extract_embedding(audio, sample_rate)

    engine = _CountingEmbedding()
    config = _config(expected=1)
    events = [SubtitleEvent(1, 0.0, 1.0, "静音")]

    result = speaker_fusion.run_speaker_fusion(
        events, np.zeros(4 * 16000, dtype=np.float32), 16000, config,
        embedding_engine=engine,
    )

    assert result.diagnostics["embedding_status"] == "failed"
    assert engine.calls == 0


def test_fallback_cluster_assigns_speakers_when_embedding_unavailable():
    """嵌入引擎缺失（未装 speechbrain）时回退 MFCC+音高聚类，不再全部 unknown。"""
    audio = (
        0.3 * np.sin(2 * np.pi * 220.0 * np.arange(3 * 16000) / 16000)
    ).astype(np.float32)
    events = [
        SubtitleEvent(i + 1, i * 1.0, i * 1.0 + 0.9, f"第{i + 1}句")
        for i in range(3)
    ]
    config = _config()
    config.diarization.distance_threshold = 0.5
    config.diarization.min_speakers = 1
    config.diarization.max_speakers = 10

    result = speaker_fusion.run_speaker_fusion(
        events, audio, 16000, config,
        embedding_engine=None, language="zh",
    )

    assert result.backend == "agglomerative"
    assert result.status == "ok"
    assert all(event.speaker_id == 0 for event in result.events)
    assert all(event.speaker_label == "说话人A" for event in result.events)


def test_fallback_label_defaults_to_english_without_language():
    audio = (
        0.3 * np.sin(2 * np.pi * 220.0 * np.arange(2 * 16000) / 16000)
    ).astype(np.float32)
    events = [SubtitleEvent(1, 0.0, 0.9, "hello")]

    result = speaker_fusion.run_speaker_fusion(
        events, audio, 16000, _config(), embedding_engine=None,
    )

    assert result.backend == "agglomerative"
    assert result.events[0].speaker_label == "Speaker A"


def test_fallback_skipped_on_silent_audio():
    """全静音音频没有任何声学证据，回退不应凭空分配说话人。"""
    events = [SubtitleEvent(1, 0.0, 1.0, "甲"), SubtitleEvent(2, 1.2, 2.0, "乙")]
    config = _config()
    config.speaker_embedding.enabled = False

    result = speaker_fusion.run_speaker_fusion(
        events, np.zeros(32000, dtype=np.float32), 16000, config,
    )

    assert result.backend == "unknown"
    assert [item.speaker_id for item in result.events] == [None, None]


def test_speaker_evidence_conflict_falls_back_to_global(monkeypatch):
    """两线归属冲突时择优回退到全局线，不再置空（2026-09-13 契约）。

    嵌入线是 3s 粗窗口聚类，短事件/跨 turn 事件频繁误标，是冲突的主要
    来源；全局 diarization 是专门的"谁在何时说话"模型，证据更强。
    """
    audio = np.ones(4 * 16000, dtype=np.float32)
    events = [
        SubtitleEvent(1, 0.2, 1.8, "甲"),
        SubtitleEvent(2, 2.2, 3.8, "乙"),
    ]
    global_result = DiarizationResult(
        turns=[SpeakerTurn(0.0, 2.0, 0), SpeakerTurn(2.0, 4.0, 1)],
        exclusive_turns=[SpeakerTurn(0.0, 2.0, 0), SpeakerTurn(2.0, 4.0, 1)],
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
    evidence = speaker_fusion.EmbeddingEvidence(
        labels=[1, 0],
        spans=[(0.0, 2.0), (2.0, 4.0)],
        centroids={0: np.array([0.0, 1.0]), 1: np.array([1.0, 0.0])},
        model="fake-ecapa",
        silhouette=0.5,
        status="ok",
    )
    monkeypatch.setattr(
        speaker_fusion,
        "_extract_embedding_evidence",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setattr(
        speaker_fusion,
        "_map_global_to_embedding",
        lambda result, events, labels: {0: 0, 1: 1},
    )

    config = _config(global_model="community-1", local_refinement="off")
    config.diarization.fusion_mode = "dual"

    result = speaker_fusion.run_speaker_fusion(
        events, audio, 16000, config, embedding_engine=_FakeEmbedding(),
    )

    # 事件0: 全局说 0、嵌入说 1 → 冲突 → 回退全局线（旧行为是置空）
    # 事件1: 全局说 1、映射后 1、嵌入说 1 → fused
    assert [item.speaker_id for item in result.events] == [0, 1]
    assert [item.speaker_source for item in result.events] == [
        "global_conflict_fallback",
        "fused",
    ]
    assert result.conflict_count == 1
    assert result.diagnostics["unknown_count"] == 0
    assert all(item.speaker_label for item in result.events)


# ---- auto 单说话人复核 (_verify_single_speaker_auto) ----

def _turns_result(*speaker_ids):
    return DiarizationResult(
        turns=[
            SpeakerTurn(start=float(i) * 3.0, end=float(i) * 3.0 + 2.0, speaker_id=sid)
            for i, sid in enumerate(speaker_ids)
        ],
        status="ok",
    )


class _FakeVerifyEngine:
    result = None

    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, audio, sample_rate, min_speakers=None, max_speakers=10):
        return type(self).result


def test_single_speaker_verify_adopts_multi_speaker_retry(monkeypatch):
    monkeypatch.setattr(speaker_fusion, "is_model_cached", lambda name, cache: True)
    monkeypatch.setattr(
        speaker_fusion, "resolve_global_model_ref",
        lambda name: f"pyannote/speaker-diarization-{name}",
    )
    _FakeVerifyEngine.result = _turns_result(0, 1, 0, 1)
    monkeypatch.setattr(
        "vocal_subtitle.diarization.pyannote_engine.PyannoteDiarizationEngine",
        _FakeVerifyEngine,
    )

    config = PipelineConfig()
    config.diarization.global_model = "auto"
    first = _turns_result(0, 0, 0)

    result, model_ref = speaker_fusion._verify_single_speaker_auto(
        np.zeros(16000, dtype=np.float32), 16000,
        config.diarization, "pyannote/speaker-diarization-community-1",
        first, token=None, cache_dir="cache/speaker_models",
    )

    assert {turn.speaker_id for turn in result.turns} == {0, 1}
    assert model_ref == "pyannote/speaker-diarization-diarization-3.1"


def test_single_speaker_verify_keeps_first_when_retry_also_single(monkeypatch):
    monkeypatch.setattr(speaker_fusion, "is_model_cached", lambda name, cache: True)
    monkeypatch.setattr(
        speaker_fusion, "resolve_global_model_ref",
        lambda name: f"pyannote/speaker-diarization-{name}",
    )
    _FakeVerifyEngine.result = _turns_result(0, 0)
    monkeypatch.setattr(
        "vocal_subtitle.diarization.pyannote_engine.PyannoteDiarizationEngine",
        _FakeVerifyEngine,
    )

    config = PipelineConfig()
    config.diarization.global_model = "auto"
    first = _turns_result(0, 0, 0)

    result, model_ref = speaker_fusion._verify_single_speaker_auto(
        np.zeros(16000, dtype=np.float32), 16000,
        config.diarization, "pyannote/speaker-diarization-community-1",
        first, token=None, cache_dir="cache/speaker_models",
    )

    assert result is first
    assert model_ref == "pyannote/speaker-diarization-community-1"


def test_single_speaker_verify_skipped_for_explicit_model(monkeypatch):
    def _fail_engine(*args, **kwargs):
        raise AssertionError("verify engine must not load for explicit model")

    monkeypatch.setattr(
        "vocal_subtitle.diarization.pyannote_engine.PyannoteDiarizationEngine",
        _fail_engine,
    )

    config = PipelineConfig()
    config.diarization.global_model = "diarization-3.1"
    first = _turns_result(0, 0, 0)

    result, model_ref = speaker_fusion._verify_single_speaker_auto(
        np.zeros(16000, dtype=np.float32), 16000,
        config.diarization, "pyannote/speaker-diarization-3.1",
        first, token=None, cache_dir="cache/speaker_models",
    )

    assert result is first
    assert model_ref == "pyannote/speaker-diarization-3.1"
