"""global 候选角色路由:signal 默认只参与风险评分,alternative 需显式准入。

- 默认(或显式 ``global_alternative_enabled=False``):global 证据仅用于风险
  评分,不进入 EvidenceDecision 的替代候选集合;
- ``global_alternative_enabled=True``:通过词时间、窗口归属、物理校验的
  global 候选才能成为 ``global_alternative`` 参与替换;
- 缺词时间或窗口归属非法的 global 候选被拒绝并记录原因。
"""

from types import SimpleNamespace

from vocal_subtitle.asr.contracts import EvidenceReviewRequest
from vocal_subtitle.asr.evidence import CandidateEvidence, EvidenceWord
from vocal_subtitle.asr.evidence_review import (
    EvidenceReviewRuntimePorts,
    EvidenceReviewService,
)
from vocal_subtitle.config.models import EvidenceReviewConfig

from vocal_subtitle.physical.timeline import PhysicalTimeline


def _config(**overrides):
    base = dict(
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
    base.update(overrides)
    return SimpleNamespace(**base)


def _segmented_event(text="the wrong phrase"):
    return SimpleNamespace(index=1, start=1.0, end=1.2, text=text, words=[])


def _global_candidate(**overrides):
    base = dict(
        identifier="global-1",
        text="the right phrase",
        start=1.0,
        end=1.2,
        window_id="global",
        words=(
            EvidenceWord("global-word", "the right phrase", 1.02, 1.18, 0.95),
        ),
    )
    base.update(overrides)
    return CandidateEvidence(
        id=base["identifier"],
        source="global",
        text=base["text"],
        start=base["start"],
        end=base["end"],
        window_id=base["window_id"],
        words=base["words"],
        confidence=0.95,
    )


def _timeline():
    timeline = PhysicalTimeline.from_duration(2.0)
    timeline.add_evidence(0.9, 1.3, "ffmpeg_skeleton")
    return timeline


def _run(config, global_evidence):
    return EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=[_segmented_event()],
            global_evidence=global_evidence,
            physical_timeline=_timeline(),
        ),
        EvidenceReviewRuntimePorts(config=config),
    )


def test_review_config_defaults_to_signal_only_admission():
    config = EvidenceReviewConfig()

    assert config.global_alternative_enabled is False


def test_signal_only_config_keeps_global_out_of_replacement_set():
    result = _run(
        _config(global_alternative_enabled=False),
        (_global_candidate(text="完全不同的内容"),),
    )

    global_diag = result.diagnostics["global_evidence"]
    assert global_diag["accepted_alternative_count"] == 0
    assert global_diag["alternative_admission"] == "disabled"
    assert global_diag["rejected"] == []
    # 保留 segmented 主候选文本,不被 global 文本替换。
    assert result.decisions[0].final_text == "the wrong phrase"
    # global 证据仍然作为 signal 参与风险评分。
    assert result.diagnostics["global_evidence_count"] == 1
    risk_codes = {
        code
        for item in result.diagnostics["risk"]
        for code in item["evidence_codes"]
    }
    assert "global_text_conflict" in risk_codes


def test_enabled_alternative_with_valid_range_is_admitted_for_replacement():
    result = _run(
        _config(global_alternative_enabled=True),
        (_global_candidate(),),
    )

    global_diag = result.diagnostics["global_evidence"]
    assert global_diag["alternative_admission"] == "enabled"
    assert global_diag["accepted_alternative_count"] == 1
    assert result.decisions[0].decision == "replace"
    assert result.decisions[0].final_text == "the right phrase"
    assert result.decisions[0].selected_candidate_id == "global-1"


def test_enabled_admission_still_rejects_missing_word_times_and_window():
    result = _run(
        _config(global_alternative_enabled=True),
        (
            _global_candidate(
                identifier="global-untimed",
                words=(),
            ),
            _global_candidate(
                identifier="global-nowindow",
                window_id="",
            ),
        ),
    )

    global_diag = result.diagnostics["global_evidence"]
    assert global_diag["accepted_alternative_count"] == 0
    reasons = {
        item["candidate_id"]: item["reasons"]
        for item in global_diag["rejected"]
    }
    assert "missing_word_timestamps" in reasons["global-untimed"]
    assert "missing_window_id" in reasons["global-nowindow"]
    assert result.decisions[0].final_text == "the wrong phrase"
