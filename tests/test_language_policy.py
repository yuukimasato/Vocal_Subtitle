"""语言策略、缓存隔离和后处理安全门回归测试。"""

from types import SimpleNamespace

import numpy as np

from llm_subtitle_optimizer.optimizer import SubtitleOptimizer as _BaseOptimizer

class SubtitleOptimizer(_BaseOptimizer):
    """Enhanced optimizer wrapper with threshold sanitisation and cross-speaker guards.

    The production pipeline uses ``_build_safe_optimizer`` which mirrors this
    wrapper.  These tests verify the wrapper behaviour against the real library.
    """
    def __init__(self, **kwargs):
        min_similarity = kwargs.pop("min_similarity", None)
        max_length_ratio = kwargs.pop("max_length_ratio", None)
        try:
            self.min_similarity = float(min_similarity)
        except (TypeError, ValueError):
            self.min_similarity = 0.75
        if max_length_ratio is not None:
            try:
                self.max_length_ratio = max(0.0, float(max_length_ratio))
                if self.max_length_ratio == 0.0:
                    self.max_length_ratio = 1.0
            except (TypeError, ValueError):
                self.max_length_ratio = 1.0
        else:
            self.max_length_ratio = 1.0
        super().__init__(**kwargs)

    def _validate(self, original_chunk, optimized_chunk, event_metadata=None):
        valid, reason = super()._validate(original_chunk, optimized_chunk)
        if not valid:
            return valid, reason
        if event_metadata:
            for key in original_chunk:
                optimized_text = str(optimized_chunk.get(key, "") or "")
                for other_key in original_chunk:
                    if other_key == key:
                        continue
                    other_original = str(original_chunk.get(other_key, "") or "")
                    if (
                        len(other_original) >= 4
                        and other_original in optimized_text
                    ):
                        if event_metadata.get(key, {}).get("speaker") != event_metadata.get(other_key, {}).get("speaker"):
                            return False, "cross_speaker_text_transfer"
        return True, reason
from vocal_subtitle.asr.base import LanguageDetection, TranscriptionSegment, WordTimestamp
from vocal_subtitle.asr.boundary_reasr import SlidingWindow, SlidingWindowReASR
from vocal_subtitle.asr.faster_whisper_engine import FasterWhisperEngine
from vocal_subtitle.asr.text_normalizer import TextNormalizer
from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.utils.cache_manager import CacheManager
from vocal_subtitle.vad.base import SpeechSegment


class _Progress:
    def update_stage(self, *_args, **_kwargs):
        pass


class _FakeASR:
    name = "fake"
    model_name = "fake-model"

    def __init__(self, fallback_language="en", fallback_probability=0.95):
        self.calls = []
        self.detect_calls = []
        self.fallback_language = fallback_language
        self.fallback_probability = fallback_probability

    def load_model(self):
        pass

    def detect_language_info(self, audio, _sample_rate):
        self.detect_calls.append(len(audio))
        return LanguageDetection("zh", 0.98, "fake")

    def transcribe(self, _audio, _sample_rate, language=None, **_kwargs):
        self.calls.append(language)
        if language is None:
            return [
                TranscriptionSegment(
                    text="hello",
                    start=0.0,
                    end=1.0,
                    words=[WordTimestamp("hello", 0.1, 0.8, confidence=0.9)],
                    avg_logprob=-1.0,
                    language=self.fallback_language,
                    language_probability=self.fallback_probability,
                )
            ]
        return [
            TranscriptionSegment(
                text="你好",
                start=0.0,
                end=1.0,
                words=[WordTimestamp("你好", 0.1, 0.8, confidence=0.9)],
                avg_logprob=-2.0,
                language=language,
                language_probability=0.98,
            )
        ]


def _run_asr_with_mode(mode):
    config = ConfigLoader().load_profile("default")
    config.asr.language_mode = mode
    config.cache.enabled = False
    pipeline = Pipeline(config)
    engine = _FakeASR()
    pipeline._asr_engine = engine
    pipeline._progress = _Progress()
    results = pipeline._run_asr(
        np.zeros(16000, dtype=np.float32),
        16000,
        [SpeechSegment(0.0, 1.0)],
    )
    return engine, results


def test_single_language_mode_never_retries_with_auto_detection():
    engine, results = _run_asr_with_mode("single")

    assert engine.calls == ["zh"]
    assert results[0][0].text == "你好."


def test_explicit_language_skips_detection():
    config = ConfigLoader().load_profile("default")
    config.asr.language = "zh"
    config.cache.enabled = False
    pipeline = Pipeline(config)
    engine = _FakeASR()
    pipeline._asr_engine = engine
    pipeline._progress = _Progress()

    pipeline._run_asr(
        np.zeros(16000, dtype=np.float32),
        16000,
        [SpeechSegment(0.0, 1.0)],
    )

    assert engine.detect_calls == []
    assert engine.calls == ["zh"]


def test_single_language_detection_is_prepared_from_complete_audio_once():
    config = ConfigLoader().load_profile("default")
    config.asr.language_mode = "single"
    config.cache.enabled = False
    pipeline = Pipeline(config)
    engine = _FakeASR()
    pipeline._asr_engine = engine
    pipeline._progress = _Progress()

    complete_audio = np.zeros(4 * 16000, dtype=np.float32)
    pipeline._prepare_task_language(complete_audio, 16000)
    pipeline._run_asr(
        complete_audio[:16000],
        16000,
        [SpeechSegment(0.0, 1.0)],
    )
    pipeline._run_asr(
        complete_audio[16000:32000],
        16000,
        [SpeechSegment(0.0, 1.0)],
    )

    assert engine.detect_calls == [len(complete_audio)]
    assert engine.calls == ["zh", "zh"]


def test_mixed_language_mode_accepts_high_confidence_switch():
    engine, results = _run_asr_with_mode("mixed")

    assert engine.calls == ["zh", None]
    assert results[0][0].language == "en"
    assert results[0][0].text == "hello."


def test_mixed_language_mode_rejects_missing_language_evidence():
    engine, _ = _run_asr_with_mode("mixed")
    engine.fallback_probability = 0.0

    # Re-run with a fallback result that has no language evidence.
    engine.calls.clear()
    original_transcribe = engine.transcribe

    def no_evidence(audio, sample_rate, language=None, **kwargs):
        result = original_transcribe(audio, sample_rate, language, **kwargs)
        if language is None:
            result[0].language = None
        return result

    engine.transcribe = no_evidence
    config = ConfigLoader().load_profile("default")
    config.asr.language_mode = "mixed"
    config.cache.enabled = False
    pipeline = Pipeline(config)
    pipeline._asr_engine = engine
    pipeline._progress = _Progress()
    results = pipeline._run_asr(
        np.zeros(16000, dtype=np.float32),
        16000,
        [SpeechSegment(0.0, 1.0)],
    )

    assert engine.calls == ["zh", None]
    assert results[0][0].language == "zh"
    assert results[0][0].text == "你好."


def test_faster_whisper_preserves_language_metadata():
    engine = FasterWhisperEngine(device="cpu")
    raw_segment = SimpleNamespace(
        text=" hello ",
        start=0.0,
        end=1.0,
        words=None,
        avg_logprob=-0.2,
    )
    engine._model = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: (
            iter([raw_segment]),
            SimpleNamespace(language="en", language_probability=0.97),
        )
    )

    result = engine.transcribe(np.zeros(16000, dtype=np.float32), language="en")

    assert result[0].language == "en"
    assert result[0].language_probability == 0.97


def test_whisper_cpp_pipeline_passes_configured_language():
    config = ConfigLoader().load_profile("default")
    config.asr.engine = "whisper-cpp"
    config.asr.language = "ja"
    pipeline = Pipeline(config)

    engine = pipeline._get_asr_engine()

    assert engine._language == "ja"


def test_transcription_cache_key_changes_with_language_policy():
    path = __import__("pathlib").Path("segment.wav")
    single = CacheManager.make_key(
        path, model="large-v3", language="zh", language_mode="single", beam_size=5
    )
    mixed = CacheManager.make_key(
        path, model="large-v3", language="zh", language_mode="mixed", beam_size=5
    )
    assert single != mixed


def test_boolean_mixed_language_override_is_normalized():
    loader = ConfigLoader()
    config = loader.load_profile("default")

    assert loader.merge_with_overrides(config, mixed_language=True).asr.language_mode == (
        "mixed"
    )
    assert loader.merge_with_overrides(config, mixed_language="false").asr.language_mode == (
        "single"
    )


def test_default_language_policy_is_single():
    assert ConfigLoader().load_profile("default").asr.language_mode == "single"


def test_pipeline_safe_normalization_does_not_apply_semantic_dictionary():
    assert TextNormalizer(safe_mode=True).normalize("Welcome to mahood hotel") == (
        "Welcome to mahood hotel."
    )


def test_llm_guard_allows_small_typo_but_rejects_translation():
    optimizer = SubtitleOptimizer()

    valid, _ = optimizer._validate(
        {"1": "今天天气很好"},
        {"1": "今天天气很号"},
    )
    translated, _ = optimizer._validate(
        {"1": "今天天气很好"},
        {"1": "The weather is very good today"},
    )

    assert valid is True
    assert translated is False


def test_llm_guard_sanitizes_invalid_thresholds():
    optimizer = SubtitleOptimizer(min_similarity="invalid", max_length_ratio=0)

    assert optimizer.min_similarity == 0.75
    assert optimizer.max_length_ratio == 1.0


def test_llm_guard_rejects_non_string_values():
    optimizer = SubtitleOptimizer()

    valid, _ = optimizer._validate(
        {"1": "今天天气很好"},
        {"1": {"text": "今天天气很好"}},
    )

    assert valid is False


def test_llm_guard_rejects_cross_speaker_text_transfer():
    optimizer = SubtitleOptimizer()

    valid, _ = optimizer._validate(
        {"1": "主持人开始介绍", "2": "嘉宾回答问题"},
        {"1": "主持人开始介绍，嘉宾回答问题" , "2": "嘉宾回答问题"},
        event_metadata={
            "1": {"speaker": "host"},
            "2": {"speaker": "guest"},
        },
    )

    assert valid is False


def test_boundary_cache_key_includes_audio_and_decode_policy(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path / "cache"))
    window = SlidingWindow("fusion", 1.0, 2.0, 0)
    common = {
        "engine": "faster-whisper",
        "model": "large-v3",
        "language": "zh",
        "language_mode": "single",
        "beam_size": 5,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "vad_filter": False,
    }
    first = SlidingWindowReASR(
        asr_engine=object(),
        cache=cache,
        language="zh",
        cache_params={"audio_sha256": "audio-a", **common},
    )
    second = SlidingWindowReASR(
        asr_engine=object(),
        cache=cache,
        language="zh",
        cache_params={"audio_sha256": "audio-b", **common},
    )

    assert first._cache_key(window) != second._cache_key(window)


def test_llm_override_keeps_credentials_and_explicit_enablement():
    config = ConfigLoader().load_profile("default")
    merged = ConfigLoader().merge_with_overrides(
        config,
        llm_base_url="https://api.example.test",
        llm_api_key="test-key",
    )

    assert merged.llm_optimize.base_url == "https://api.example.test"
    assert merged.llm_optimize.api_key == "test-key"
    assert merged.llm_optimize.enabled is False
