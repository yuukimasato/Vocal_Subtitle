from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vocal_subtitle.asr.base import TranscriptionSegment
from vocal_subtitle.config import PipelineConfig, validate_config_consistency
from vocal_subtitle.pipeline import Pipeline, PipelineStats
from vocal_subtitle.vad.base import SpeechSegment


class _Progress:
    def update_stage(self, *_args, **_kwargs):
        pass


class _Cache:
    def __init__(self):
        self.values = {}

    @staticmethod
    def make_key(path, **params):
        return repr((str(path), sorted(params.items())))

    def get(self, stage, key):
        return self.values.get((stage, key))

    def set(self, stage, key, value):
        self.values[(stage, key)] = value


class _ASR:
    name = "fake"
    model_name = "fake-model"

    def __init__(self):
        self.calls = 0

    def load_model(self):
        pass

    def transcribe(self, *_args, **_kwargs):
        self.calls += 1
        return [TranscriptionSegment("感谢观看", 0.0, 1.0)]


def test_phase_one_config_defaults_and_validation():
    config = PipelineConfig()

    assert config.asr.no_speech_threshold == 0.6
    assert config.asr.log_prob_threshold == -1.0
    assert config.asr.compression_ratio_threshold == 2.4
    assert config.asr.hallucination_filter_version == "v1"

    config.asr.no_speech_threshold = -1
    warnings = validate_config_consistency(config)
    assert any("no_speech_threshold" in warning for warning in warnings)


def test_phase_one_cache_key_contains_filter_policy():
    from vocal_subtitle.utils.cache_manager import CacheManager

    base = dict(
        hallucination_filter_version="v1",
        hallucination_filter_enabled=True,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        filter_training_phrases=True,
        filter_adjacent_duplicates=True,
    )
    first = CacheManager.make_key(Path("segment.wav"), **base)
    changed = dict(base, hallucination_filter_version="v2")

    assert first != CacheManager.make_key(Path("segment.wav"), **changed)


def test_pipeline_records_filter_diagnostics():
    pipeline = Pipeline(PipelineConfig())
    pipeline._filter_asr_results(
        [TranscriptionSegment("感谢观看", 0.0, 1.0)]
    )
    stats = PipelineStats(input_path=Path("audio.wav"), duration_seconds=1.0)

    pipeline._apply_hallucination_stats(stats)

    assert stats.hallucination_filter_version == "v1"
    assert stats.hallucination_dropped_count == 1
    assert stats.hallucination_drop_reasons == {"training_phrase": 1}
    assert stats.to_dict()["hallucination_dropped_count"] == 1


def test_cached_and_fresh_asr_results_share_filter(monkeypatch):
    config = PipelineConfig()
    config.cache.enabled = True
    config.asr.language = "en"
    pipeline = Pipeline(config)
    engine = _ASR()
    cache = _Cache()
    pipeline._asr_engine = engine
    pipeline._cache = cache
    pipeline._progress = _Progress()
    pipeline._file_hash = "audio-hash"
    monkeypatch.setattr(pipeline, "_get_cache", lambda: cache)

    first = pipeline._run_asr(
        np.zeros(16000, dtype=np.float32),
        16000,
        [SpeechSegment(0.0, 1.0)],
    )
    second = pipeline._run_asr(
        np.zeros(16000, dtype=np.float32),
        16000,
        [SpeechSegment(0.0, 1.0)],
    )

    assert first == [[]]
    assert second == [[]]
    assert engine.calls == 1
    assert pipeline._hallucination_dropped_count == 2
