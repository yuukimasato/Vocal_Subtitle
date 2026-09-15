from types import SimpleNamespace

import pytest

from vocal_subtitle.asr.contracts import EvidenceReviewRequest
from vocal_subtitle.asr.evidence import (
    CandidateEvidence,
    EvidenceBundle,
    EvidenceDecision,
    EvidenceWord,
    candidate_from_subtitle_event,
    candidates_from_global_transcript,
    candidates_from_segments,
    evidence_word_from_asr,
)
from vocal_subtitle.asr.evidence_review import (
    EvidenceReviewRuntimePorts,
    EvidenceReviewService,
)


def candidate(identifier="seg-1", text="hello", start=1.0, end=2.0, **kwargs):
    return CandidateEvidence(
        id=identifier,
        source=kwargs.pop("source", "segmented"),
        text=text,
        start=start,
        end=end,
        **kwargs,
    )


def test_evidence_contract_preserves_missing_confidence_and_round_trips():
    word = EvidenceWord(
        id="w1",
        text="hello",
        start=None,
        end=None,
        confidence=None,
        time_source="segment_boundary",
    )
    item = candidate(words=(word,), confidence=None)

    restored = CandidateEvidence.from_dict(item.to_dict())

    assert restored.confidence is None
    assert restored.words[0].confidence is None
    assert restored.words[0].start is None


def test_evidence_bundle_and_decision_round_trip():
    item = candidate(confidence=None)
    bundle = EvidenceBundle(segmented=(item,))
    assert EvidenceBundle.from_dict(bundle.to_dict()).segmented[0] == item

    decision = EvidenceDecision(
        candidate_ids=(item.id,),
        decision="keep",
        final_text=item.text,
        final_words=(),
        start=item.start,
        end=item.end,
        time_source="segment_boundary",
        confidence=None,
        risk_score=0.1,
        risk_level="low",
    )
    assert EvidenceDecision.from_dict(decision.to_dict()) == decision


def test_evidence_contract_rejects_unknown_time_source():
    with pytest.raises(ValueError, match="time_source"):
        EvidenceWord(id="w1", text="hello", time_source="global")


def test_global_boundary_fallback_is_not_reported_as_native_word_time():
    word = SimpleNamespace(
        raw_start=1.0,
        raw_end=1.5,
        text="hello",
        confidence=None,
        metadata={"time_source": "segment_boundary"},
    )
    event = SimpleNamespace(
        index=1,
        start=1.0,
        end=1.5,
        text="hello",
        words=[word],
        source_word_ids=["w1"],
    )

    evidence = candidate_from_subtitle_event(event, source="global")

    assert evidence.words[0].time_source == "segment_boundary"
    assert evidence.words[0].confidence is None


def test_invalid_asr_word_time_is_degraded_to_missing_evidence_time():
    word = SimpleNamespace(
        word="hello",
        start=0.4,
        end=0.0,
        confidence=0.7,
    )

    evidence = evidence_word_from_asr(word, "w-invalid")

    assert evidence.start is None
    assert evidence.end is None
    assert evidence.diagnostics["timing_degraded"] is True
    assert evidence.diagnostics["invalid_timing"]["reason"] == (
        "missing_or_non_monotonic_word_time"
    )


def test_invalid_segment_word_time_does_not_abort_candidate_adaptation():
    segments = [
        SimpleNamespace(
            start=0.0,
            end=0.8,
            text="hello",
            words=[SimpleNamespace(word="hello", start=0.6, end=0.0)],
        )
    ]

    result = candidates_from_segments(segments, source="segmented")

    assert len(result) == 1
    assert result[0].text == "hello"
    assert result[0].has_word_times is False
    assert result[0].diagnostics["invalid_word_timing_count"] == 1


def test_invalid_segment_word_time_keeps_authoritative_review_on_segment_boundary():
    config = SimpleNamespace(
        enabled=True,
        shadow_mode=False,
        context_reasr_enabled=False,
        qwen_enabled=False,
        forced_aligner_enabled=False,
        sed_enabled=False,
        semantic_review_enabled=False,
        unresolved_keeps_candidate=True,
        require_multi_source_drop=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    event = SimpleNamespace(
        index=1,
        start=1.0,
        end=1.8,
        text="hello",
        words=[SimpleNamespace(word="hello", start=0.6, end=0.0)],
    )

    result = EvidenceReviewService().run(
        EvidenceReviewRequest(events=[event]),
        EvidenceReviewRuntimePorts(config=config),
    )

    assert len(result.decisions) == 1
    assert result.decisions[0].final_text == "hello"
    assert result.diagnostics["status"] == "ok"
    assert result.diagnostics["invalid_word_timing_count"] == 1


def test_auto_qwen_pair_does_not_load_disabled_optional_engine(monkeypatch):
    import numpy as np

    import vocal_subtitle.application.asr_path as asr_path_module
    from vocal_subtitle.application.offline_production import OfflineProductionResult
    from vocal_subtitle.config import ConfigLoader
    from vocal_subtitle.mapping.time_mapper import SubtitleEvent
    from vocal_subtitle.pipeline import Pipeline

    config = ConfigLoader().load_profile("default")
    config.evidence_review.qwen_enabled = False
    config.asr.engine_pair.secondary = "auto"
    pipeline = Pipeline(config)
    pipeline._resolved_language = "en"

    class CaptureCoordinator:
        def run(self, request, ports):
            assert ports.qwen is None
            assert ports.secondary is None
            assert ports.secondary_name == "qwen"
            return OfflineProductionResult(
                events=list(request.events),
                diagnostics={
                    "production_path": "authoritative",
                    "review_status": "ok",
                    "decision_count": 0,
                },
            )

    monkeypatch.setattr(
        asr_path_module,
        "OfflineProductionCoordinator",
        lambda: CaptureCoordinator(),
    )

    events, diagnostics = pipeline._apply_evidence_review(
        [SubtitleEvent(index=1, start=0.0, end=0.5, text="hello")],
        audio=np.zeros(8000, dtype="float32"),
        stats=type("Stats", (), {"duration_seconds": 0.5})(),
    )

    assert len(events) == 1
    assert diagnostics["production_path"] == "authoritative"


def test_global_transcript_adapter_preserves_missing_confidence_and_time_source():
    transcript = SimpleNamespace(
        backend="global-test",
        status="degraded",
        words=[
            SimpleNamespace(
                id="global-word-1",
                text="hello",
                raw_start=1.0,
                raw_end=1.4,
                confidence=None,
                source_window_id="window-1",
                metadata={"time_source": "segment_boundary"},
            )
        ],
        segments=[
            SimpleNamespace(
                id="global-segment-1",
                text="hello",
                raw_start=1.0,
                raw_end=1.4,
                word_ids=["global-word-1"],
                avg_logprob=None,
                language="en",
                metadata={},
            )
        ],
    )

    result = candidates_from_global_transcript(transcript)

    assert result[0].confidence is None
    assert result[0].words[0].confidence is None
    assert result[0].words[0].time_source == "segment_boundary"


def test_legacy_global_event_does_not_fabricate_missing_confidence():
    from vocal_subtitle.physical.events import GlobalSubtitleEvent
    from vocal_subtitle.physical.ir import GlobalWord

    event = GlobalSubtitleEvent(
        index=1,
        start=1.0,
        end=1.4,
        text="hello",
        words=[
            GlobalWord(
                id="global-word-1",
                text="hello",
                raw_start=1.0,
                raw_end=1.4,
                confidence=None,
                source_window_id="window-1",
                segment_id="segment-1",
                metadata={"time_source": "segment_boundary"},
            )
        ],
        time_source="timing_degraded",
    )

    evidence = candidate_from_subtitle_event(event.to_subtitle_event(), source="global")

    assert evidence.words[0].confidence is None
    assert evidence.words[0].time_source == "segment_boundary"


def test_evidence_review_service_runs_without_pipeline():
    config = SimpleNamespace(
        enabled=True,
        shadow_mode=True,
        context_reasr_enabled=True,
        unresolved_keeps_candidate=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    event = SimpleNamespace(index=1, start=1.0, end=1.3, text="hello", words=[])

    result = EvidenceReviewService().run(
        EvidenceReviewRequest(events=[event]),
        EvidenceReviewRuntimePorts(config=config),
    )

    assert result.events == [event]
    assert len(result.decisions) == 1
    assert result.diagnostics["status"] == "ok"
    assert result.diagnostics["decisions"][0]["candidate_ids"]


def test_global_evidence_can_replace_only_a_high_risk_segmented_candidate():
    from vocal_subtitle.physical.timeline import PhysicalTimeline

    config = SimpleNamespace(
        enabled=True,
        shadow_mode=False,
        context_reasr_enabled=False,
        qwen_enabled=False,
        forced_aligner_enabled=False,
        sed_enabled=False,
        semantic_review_enabled=False,
        unresolved_keeps_candidate=True,
        require_multi_source_drop=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    global_word = EvidenceWord(
        "global-word",
        "the right phrase",
        1.02,
        1.18,
        0.95,
    )
    global_candidate = candidate(
        "global-1",
        text="the right phrase",
        start=1.0,
        end=1.2,
        source="global",
        window_id="global",
        confidence=0.95,
        words=(global_word,),
    )
    timeline = PhysicalTimeline.from_duration(2.0)
    timeline.add_evidence(0.9, 1.3, "ffmpeg_skeleton")

    result = EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[
                SimpleNamespace(
                    index=1, start=1.0, end=1.2, text="the wrong phrase", words=[]
                )
            ],
            global_evidence=(global_candidate,),
            physical_timeline=timeline,
        ),
        EvidenceReviewRuntimePorts(config=config),
    )

    assert result.decisions[0].decision == "replace"
    assert result.decisions[0].final_text == "the right phrase"
    assert result.diagnostics["global_evidence"]["accepted_alternative_count"] == 1


def test_global_evidence_without_word_times_remains_signal_only():
    config = SimpleNamespace(
        enabled=True,
        shadow_mode=False,
        context_reasr_enabled=False,
        qwen_enabled=False,
        forced_aligner_enabled=False,
        sed_enabled=False,
        semantic_review_enabled=False,
        unresolved_keeps_candidate=True,
        require_multi_source_drop=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    result = EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[
                SimpleNamespace(index=1, start=1.0, end=1.2, text="wrong", words=[])
            ],
            global_evidence=(
                candidate(
                    "global-untimed",
                    text="correct",
                    start=1.0,
                    end=1.2,
                    source="global",
                    window_id="global",
                ),
            ),
        ),
        EvidenceReviewRuntimePorts(config=config),
    )

    assert result.decisions[0].decision == "unresolved"
    assert result.decisions[0].final_text == "wrong"
    assert result.diagnostics["global_evidence"]["accepted_alternative_count"] == 0
    assert (
        "missing_word_timestamps"
        in result.diagnostics["global_evidence"]["rejected"][0]["reasons"]
    )


def test_empty_asr_segments_are_ignored_at_evidence_boundary():
    segments = [
        SimpleNamespace(start=0.0, end=0.2, text="", words=[]),
        SimpleNamespace(start=0.2, end=0.5, text="hello", words=[]),
    ]

    result = candidates_from_segments(segments, source="context_reasr")

    assert [item.text for item in result] == ["hello"]


def test_empty_segmented_events_do_not_abort_review():
    config = SimpleNamespace(
        enabled=True,
        shadow_mode=False,
        context_reasr_enabled=False,
        unresolved_keeps_candidate=True,
        require_multi_source_drop=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    events = [
        SimpleNamespace(index=1, start=1.0, end=1.3, text="", words=[]),
        SimpleNamespace(index=2, start=1.3, end=1.6, text="hello", words=[]),
    ]

    result = EvidenceReviewService().run(
        EvidenceReviewRequest(events=events),
        EvidenceReviewRuntimePorts(config=config),
    )

    assert len(result.decisions) == 1
    assert result.decisions[0].final_text == "hello"
    assert result.diagnostics["invalid_segment_count"] == 1


def test_optional_review_ports_are_injected_and_failures_are_reported():
    class FakeQwen:
        name = "qwen-test"

        def __init__(self):
            self.calls = 0

        def review(self, audio, sample_rate, window, *, language=None):
            self.calls += 1
            return [
                candidate(
                    "qwen-1",
                    text="hello",
                    start=window.start + 0.1,
                    end=window.start + 0.4,
                    source="qwen",
                    confidence=0.9,
                    words=(
                        EvidenceWord(
                            "qwen-word-1",
                            "hello",
                            window.start + 0.1,
                            window.start + 0.4,
                            0.9,
                        ),
                    ),
                )
            ]

    config = SimpleNamespace(
        enabled=True,
        shadow_mode=True,
        context_reasr_enabled=False,
        qwen_enabled=True,
        forced_aligner_enabled=True,
        sed_enabled=True,
        semantic_review_enabled=True,
        unresolved_keeps_candidate=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    qwen = FakeQwen()
    result = EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[
                SimpleNamespace(
                    index=1,
                    start=1.0,
                    end=1.1,
                    text="hallucinated phrase",
                    words=[],
                )
            ],
            audio=__import__("numpy").zeros(32000, dtype="float32"),
            input_hash="input-qwen",
        ),
        EvidenceReviewRuntimePorts(config=config, qwen=qwen),
    )

    assert qwen.calls == 1
    assert result.diagnostics["optional_engines"]["qwen"]["status"] == "ok"
    assert result.diagnostics["optional_engines"]["qwen"]["phase"] == "qwen"
    assert (
        result.diagnostics["optional_engines"]["forced_aligner"]["status"]
        == "unavailable"
    )
    assert result.diagnostics["optional_engines"]["sed"]["status"] == "unavailable"
    assert (
        result.diagnostics["optional_engines"]["semantic_review"]["status"]
        == "unavailable"
    )


def test_risk_only_skips_secondary_asr_after_context_agreement():
    class FakeContext:
        name = "context-test"

        def review(self, audio, sample_rate, window, *, language=None):
            return [
                candidate(
                    "context-agree",
                    text="hello",
                    start=1.1,
                    end=1.4,
                    source="context_reasr",
                    confidence=0.9,
                    words=(
                        EvidenceWord(
                            "context-word",
                            "hello",
                            1.1,
                            1.4,
                            0.9,
                        ),
                    ),
                )
            ]

    class FakeQwen:
        name = "qwen-test"

        def __init__(self):
            self.calls = 0

        def review(self, audio, sample_rate, window, *, language=None):
            self.calls += 1
            return []

    config = SimpleNamespace(
        enabled=True,
        shadow_mode=True,
        context_reasr_enabled=True,
        qwen_enabled=True,
        forced_aligner_enabled=False,
        sed_enabled=False,
        semantic_review_enabled=False,
        unresolved_keeps_candidate=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    qwen = FakeQwen()
    result = EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[
                SimpleNamespace(index=1, start=1.0, end=1.3, text="hello", words=[])
            ],
            audio=__import__("numpy").zeros(32000, dtype="float32"),
        ),
        EvidenceReviewRuntimePorts(
            config=config, context_reasr=FakeContext(), qwen=qwen
        ),
    )

    assert qwen.calls == 0
    assert (
        "context_reasr_agreement"
        in result.diagnostics["residual_risk"][0]["evidence_codes"]
    )
    assert (
        result.diagnostics["optional_engines"]["qwen"]["reason"] == "residual_risk_gate"
    )


def test_full_quality_schedules_low_risk_candidate_for_qwen():
    class FakeQwen:
        name = "qwen-test"

        def __init__(self):
            self.calls = 0

        def review(self, audio, sample_rate, window, *, language=None):
            self.calls += 1
            return []

    config = SimpleNamespace(
        enabled=True,
        shadow_mode=True,
        context_reasr_enabled=False,
        qwen_enabled=True,
        forced_aligner_enabled=False,
        sed_enabled=False,
        semantic_review_enabled=False,
        unresolved_keeps_candidate=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    qwen = FakeQwen()
    result = EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[
                SimpleNamespace(
                    index=1,
                    start=1.0,
                    end=2.0,
                    text="stable phrase",
                    words=[
                        SimpleNamespace(
                            word="stable",
                            start=0.1,
                            end=0.5,
                            confidence=0.95,
                        )
                    ],
                )
            ],
            audio=__import__("numpy").zeros(32000, dtype="float32"),
            review_policy="full_quality",
        ),
        EvidenceReviewRuntimePorts(config=config, qwen=qwen),
    )

    assert qwen.calls == 1
    assert result.diagnostics["review_policy"] == "full_quality"
    assert result.diagnostics["optional_engines"]["qwen"]["policy"] == "full_quality"


def test_secondary_evidence_collects_auxiliary_results_without_changing_decisions():
    from vocal_subtitle.asr.evidence import EvidenceWord
    from vocal_subtitle.asr.evidence_review import (
        EvidenceReviewRuntimePorts,
        EvidenceReviewService,
    )

    class FakeAligner:
        name = "aligner-test"

        def __init__(self):
            self.calls = 0

        def align(self, audio, sample_rate, text, window, *, language=None):
            self.calls += 1
            return [
                EvidenceWord(
                    "aligned-1",
                    text,
                    window.start + 0.1,
                    window.start + 0.2,
                    None,
                    "qwen_forced_alignment",
                )
            ]

    class FakeSED:
        name = "sed-test"

        def __init__(self):
            self.calls = 0

        def detect(self, audio, sample_rate, window):
            self.calls += 1
            return {"label": "breathing", "score": 0.8}

    class FakeSemantic:
        name = "semantic-test"

        def __init__(self):
            self.calls = 0

        def review(self, text, context=""):
            self.calls += 1
            return {"risk": "non_speech", "context": context}

    config = SimpleNamespace(
        enabled=True,
        shadow_mode=True,
        context_reasr_enabled=False,
        qwen_enabled=False,
        forced_aligner_enabled=True,
        sed_enabled=True,
        semantic_review_enabled=True,
        unresolved_keeps_candidate=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    aligner, sed, semantic = FakeAligner(), FakeSED(), FakeSemantic()
    result = EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[
                SimpleNamespace(index=1, start=1.0, end=1.3, text="hello", words=[])
            ],
            audio=__import__("numpy").zeros(32000, dtype="float32"),
        ),
        EvidenceReviewRuntimePorts(
            config=config,
            forced_aligner=aligner,
            sed=sed,
            semantic_review=semantic,
        ),
    )

    secondary = result.diagnostics["secondary_evidence"]
    assert aligner.calls == 1
    assert sed.calls == 1
    assert semantic.calls == 1
    assert secondary["forced_aligner"]["windows"][0]["word_count"] == 1
    assert secondary["sed"]["windows"][0]["evidence"]["label"] == "breathing"
    assert (
        secondary["semantic_review"]["windows"][0]["evidence"]["risk"] == "non_speech"
    )
    assert {item["source"] for item in secondary["bundles"]} == {
        "forced_aligner",
        "sed",
        "semantic_review",
    }
    assert secondary["bundles"][0]["candidate_ids"]
    assert result.decisions[0].decision in {"keep", "unresolved"}
    assert "semantic_non_speech" in result.decisions[0].evidence_codes


def test_callback_adapters_keep_optional_engine_contracts_narrow():
    from vocal_subtitle.asr.review_engines import (
        CallbackForcedAligner,
        CallbackSED,
        CallbackSemanticReview,
    )

    window = SimpleNamespace(start=1.0, end=2.0)
    aligner = CallbackForcedAligner(
        lambda audio, sample_rate, text, item, *, language: [
            (text, item.start, item.end, language)
        ]
    )
    sed = CallbackSED(
        lambda audio, sample_rate, item: {"label": "breathing", "window": item.start}
    )
    semantic = CallbackSemanticReview(
        lambda text, context: {"class": "review", "text": text, "context": context}
    )

    assert aligner.align(None, 16000, "hello", window, language="en")[0][0] == "hello"
    assert sed.detect(None, 16000, window)["label"] == "breathing"
    assert semantic.review("hello", "before")["context"] == "before"


def test_lazy_qwen_adapter_projects_window_coordinates_without_optional_sdk():
    import numpy as np

    from vocal_subtitle.asr.optional_adapters import LazyQwenASR

    class FakeQwen:
        def transcribe(self, *, audio, language):
            assert len(audio[0]) == 8000
            assert audio[1] == 16000
            assert language == "English"
            return [
                {
                    "text": "hello",
                    "start": 0.1,
                    "end": 0.4,
                    "words": [{"word": "hello", "start": 0.1, "end": 0.4}],
                }
            ]

    window = SimpleNamespace(id="qwen-window", start=0.5, end=1.0)
    adapter = LazyQwenASR(
        "fake-model",
        model_loader=lambda path, device: FakeQwen(),
    )

    candidates = adapter.review(
        np.zeros(24000, dtype=np.float32),
        16000,
        window,
        language="en",
    )

    assert adapter.availability()["status"] == "ready"
    assert candidates[0].source == "qwen"
    assert candidates[0].start == pytest.approx(0.6)
    assert candidates[0].words[0].start == pytest.approx(0.6)
    assert candidates[0].words[0].time_source == "native_word_timestamp"


def test_lazy_forced_aligner_emits_secondary_absolute_word_times():
    import numpy as np

    from vocal_subtitle.asr.optional_adapters import LazyQwenForcedAligner

    class FakeAligner:
        def align(self, *, audio, text, language):
            assert len(audio[0]) == 8000
            assert audio[1] == 16000
            return {
                "words": [{"text": text, "start": 0.2, "end": 0.45, "confidence": None}]
            }

    window = SimpleNamespace(id="align-window", start=1.0, end=1.5)
    adapter = LazyQwenForcedAligner(
        "fake-aligner",
        model_loader=lambda path, device: FakeAligner(),
    )

    words = adapter.align(
        np.zeros(32000, dtype=np.float32),
        16000,
        "hello",
        window,
        language="en",
    )

    assert words[0].start == pytest.approx(1.2)
    assert words[0].end == pytest.approx(1.45)
    assert words[0].time_source == "qwen_forced_alignment"
    assert words[0].confidence is None


def test_lazy_sed_adapter_returns_structured_window_evidence():
    import numpy as np

    from vocal_subtitle.asr.optional_adapters import LazyAudioClassifierSED

    class FakeClassifier:
        def __call__(self, payload):
            assert payload["sampling_rate"] == 16000
            assert len(payload["raw"]) == 8000
            return [{"label": "breathing", "score": 0.91}]

    window = SimpleNamespace(id="sed-window", start=0.5, end=1.0)
    adapter = LazyAudioClassifierSED(
        "fake-sed",
        model_loader=lambda path, device: FakeClassifier(),
    )

    result = adapter.detect(
        np.zeros(24000, dtype=np.float32),
        16000,
        window,
    )

    assert adapter.availability()["status"] == "ready"
    assert result["status"] == "ok"
    assert result["labels"] == [{"label": "breathing", "score": 0.91}]
    assert result["start"] == pytest.approx(0.5)


def test_optional_adapter_reports_missing_runtime_or_model_without_loading():
    from vocal_subtitle.asr.optional_adapters import LazyAudioClassifierSED, LazyQwenASR

    qwen = LazyQwenASR("/missing/qwen-model")
    sed = LazyAudioClassifierSED("/missing/sed-model")

    assert qwen.availability()["reason"] == "model_path_missing"
    assert sed.availability()["reason"] == "model_path_missing"


def test_windowed_context_reasr_adapter_projects_window_times():
    from types import SimpleNamespace

    import numpy as np

    from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
    from vocal_subtitle.asr.review_engines import (
        WindowedASRContextReASR,
        WindowedASRQwen,
    )

    class FakeEngine:
        name = "fake"
        model_name = "fake-model"

        def load_model(self):
            return None

        def transcribe(self, audio, sample_rate, language=None):
            assert len(audio) == 8000
            assert sample_rate == 16000
            assert language == "en"
            return [
                TranscriptionSegment(
                    text="hello",
                    start=0.1,
                    end=0.4,
                    words=[WordTimestamp("hello", 0.1, 0.4, confidence=0.9)],
                )
            ]

    window = SimpleNamespace(id="review-1", start=0.5, end=1.0)
    candidates = WindowedASRContextReASR(lambda: FakeEngine()).review(
        np.zeros(24000, dtype=np.float32),
        16000,
        window,
        language="en",
    )

    assert candidates[0].start == pytest.approx(0.6)
    assert candidates[0].end == pytest.approx(0.9)
    assert candidates[0].window_id == "review-1"

    qwen_candidates = WindowedASRQwen(lambda: FakeEngine()).review(
        __import__("numpy").zeros(24000, dtype="float32"),
        16000,
        window,
        language="en",
    )
    assert qwen_candidates[0].source == "qwen"
