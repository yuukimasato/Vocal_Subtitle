"""Independent contract tests for the extracted ASR and merge services."""

from types import SimpleNamespace

from vocal_subtitle.asr.contracts import (
    ASRFailureRequest,
    ASRRuntimePorts,
    GlobalASRRequest,
    GlobalASRResult,
)
from vocal_subtitle.asr.global_path import GlobalASRService
from vocal_subtitle.asr.review_path import ASRReviewRequest, ASRReviewService
from vocal_subtitle.merging.llm_decider import LLMMergeDecider
from vocal_subtitle.merging.local_decider import LocalMergeDecider


def test_global_asr_service_uses_explicit_runner_port():
    request = GlobalASRRequest(audio=[], sample_rate=16000, shadow=None, stats=None)
    ports = ASRRuntimePorts(
        config=None,
        get_engine=lambda: None,
        get_engine_for=lambda *args, **kwargs: None,
        get_language=lambda: None,
        set_language=lambda value: None,
        quality_gate_kwargs=lambda: {},
        global_runner=lambda item: (["event"], {"source": "fake"}, SimpleNamespace(status="ok", words=["word"])),
    )
    result = GlobalASRService().run(request, ports)
    assert isinstance(result, GlobalASRResult)
    assert result.events == ["event"]
    assert result.diagnostics == {"source": "fake"}
    assert result.evidence == []


def test_global_asr_service_prefers_transcript_ir_for_evidence():
    transcript = SimpleNamespace(
        backend="global-test",
        status="degraded",
        words=[SimpleNamespace(
            id="w1",
            text="hello",
            raw_start=1.0,
            raw_end=1.4,
            confidence=None,
            source_window_id="window-1",
            metadata={"time_source": "segment_boundary"},
        )],
        segments=[SimpleNamespace(
            id="s1",
            text="hello",
            raw_start=1.0,
            raw_end=1.4,
            word_ids=["w1"],
            avg_logprob=None,
            language="en",
            metadata={},
        )],
    )
    request = GlobalASRRequest(audio=[], sample_rate=16000, shadow=None, stats=None)
    ports = ASRRuntimePorts(
        config=None,
        get_engine=lambda: None,
        get_engine_for=lambda *args, **kwargs: None,
        get_language=lambda: None,
        set_language=lambda value: None,
        quality_gate_kwargs=lambda: {},
        global_runner=lambda item: (["legacy-event"], {}, transcript),
    )

    result = GlobalASRService().run(request, ports)

    assert result.evidence[0].confidence is None
    assert result.evidence[0].words[0].time_source == "segment_boundary"


def test_asr_review_service_is_independent_of_pipeline():
    transcript = SimpleNamespace(status="ok", words=["word"])
    ASRReviewService.validate(
        ASRReviewRequest(events=["event"], transcript=transcript, diagnostics={})
    )
    assert ASRReviewService.classify_failure(
        ASRFailureRequest(RuntimeError("quality gate failed"))
    ) == "quality_gate_failed"


def test_local_merge_decider_handles_deterministic_rule_without_model():
    config = SimpleNamespace(local_nlp_gap_range=(0.0, 1.0))
    groups, unresolved = LocalMergeDecider(config).decide(
        [
            {"id": 0, "text": "incomplete,", "gap_to_next_sec": 0.2},
            {"id": 1, "text": "continuation", "gap_to_next_sec": 999},
        ]
    )
    assert groups[0]["ids"] == [0, 1]
    assert unresolved == []


def test_llm_decider_falls_back_without_cloud_endpoint():
    config = SimpleNamespace(
        llm_base_url=None,
        llm_fallback_to_rules=True,
        llm_model="fake",
        llm_temperature=0.0,
        llm_timeout=1,
        llm_api_key=None,
    )
    groups = LLMMergeDecider(config).decide(
        [
            {"id": 0, "text": "hello", "gap_to_next_sec": 0.1},
            {"id": 1, "text": "world", "gap_to_next_sec": 999},
        ]
    )
    assert groups[0]["ids"] == [0, 1]
