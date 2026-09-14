"""风险门控 Context Re-ASR(高精度方案 Task 6)。

- 低风险窗口不触发 re-ASR(耗时与调用次数不得无条件增加);
- high/critical 或明确文本冲突窗口才调用,一次运行每个窗口最多一次;
- ``context_reasr_min_level`` 提升到 high 时跳过中风险窗口;
- re-ASR 失败时保留主候选并记录失败原因;
- 诊断输出 reviewed_window_count / reasr_replaced_count / reasr_failed_count /
  reasr_time_ms / reasr_call_ratio。

风险档位由事件属性自然构成(评分器对事件候选打分):
- 低风险:有词时间 + 词置信 0.9 → 0 分;
- 中风险:无词时间 + 无置信 → 0.30;
- 高风险:中风险基础 + 高文本密度(cps≥15)→ ≥0.50;
- 冲突窗口:干净候选 + 全局证据文本强烈冲突 → medium + global_text_conflict。
"""

from types import SimpleNamespace

import numpy as np

from vocal_subtitle.asr.contracts import EvidenceReviewRequest
from vocal_subtitle.asr.evidence import CandidateEvidence
from vocal_subtitle.asr.evidence_review import (
    EvidenceReviewRuntimePorts,
    EvidenceReviewService,
)


def _timed_word(start, end, confidence=0.9):
    return SimpleNamespace(word="hello", start=start, end=end, confidence=confidence)


def _event(index=1, text="hello", start=1.0, end=2.0, words=()):
    return SimpleNamespace(
        index=index, start=start, end=end, text=text, words=list(words),
    )


def _config(**overrides):
    base = dict(
        enabled=True,
        shadow_mode=False,
        context_reasr_enabled=True,
        global_alternative_enabled=False,
        qwen_enabled=False,
        forced_aligner_enabled=False,
        sed_enabled=False,
        semantic_review_enabled=False,
        unresolved_keeps_candidate=True,
        require_multi_source_drop=True,
        left_context=0.5,
        right_context=0.5,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
        max_workers=1,
        window_timeout_seconds=10.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class CountingReASR:
    """记录调用次数的 fake Context Re-ASR 引擎。"""

    name = "counting-reasr"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def review(self, audio, sample_rate, window, language=None):
        self.calls.append((window.start, window.end, tuple(window.candidate_ids)))
        if self.fail:
            raise RuntimeError("reasr engine crashed")
        return [
            CandidateEvidence(
                id=f"reasr-{len(self.calls)}",
                source="context_reasr",
                text="hello",
                start=window.start + 0.1,
                end=window.end - 0.1,
                window_id=window.id,
            )
        ]


def _run(config, events, engine, global_evidence=()):
    return EvidenceReviewService().run(
        EvidenceReviewRequest(
            events=events,
            audio=np.zeros(32000, dtype=np.float32),
            sample_rate=16000,
            global_evidence=global_evidence,
        ),
        EvidenceReviewRuntimePorts(config=config, context_reasr=engine),
    )


def test_low_risk_windows_do_not_invoke_reasr():
    engine = CountingReASR()
    events = [_event(words=[_timed_word(0.1, 0.9)])]

    result = _run(_config(), events, engine)

    assert engine.calls == []
    assert result.diagnostics["review"]["reviewed_window_count"] == 0


def test_high_risk_window_invokes_reasr_exactly_once():
    engine = CountingReASR()
    # 无词时间(0.30) + cps=18(≥15 → +0.20) = 0.50 high。
    events = [_event(text="0123456789abcdefgh")]

    result = _run(_config(), events, engine)

    assert len(engine.calls) == 1
    review_diag = result.diagnostics["review"]
    assert review_diag["reviewed_window_count"] == 1
    assert review_diag["reasr_failed_count"] == 0
    assert "reasr_time_ms" in review_diag
    assert "reasr_call_ratio" in review_diag
    assert "reasr_replaced_count" in review_diag


def test_medium_conflict_window_still_gets_reviewed():
    engine = CountingReASR()
    # 干净候选(0 分) + 全局证据文本强烈冲突(+0.25) → medium + conflict code。
    events = [_event(words=[_timed_word(0.1, 0.9)])]
    global_evidence = (
        CandidateEvidence(
            id="global-1",
            source="global",
            text="完全不同的全局识别内容",
            start=1.0,
            end=2.0,
            window_id="global",
        ),
    )

    result = _run(_config(), events, engine, global_evidence=global_evidence)

    assert len(engine.calls) == 1
    risk_codes = {
        code
        for item in result.diagnostics["risk"]
        for code in item["evidence_codes"]
    }
    assert "global_text_conflict" in risk_codes


def test_min_level_high_skips_medium_windows():
    engine = CountingReASR()
    medium = _event(index=1, text="hello")  # 0.30 medium
    high = _event(index=2, text="0123456789abcdefgh", start=5.0, end=6.0)  # high

    result = _run(
        _config(context_reasr_min_level="high"), [medium, high], engine,
    )

    # 只复核了 high 候选(index=2)所在的窗口,medium 窗口被跳过。
    reviewed_ids = {
        candidate_id for _, _, ids in engine.calls for candidate_id in ids
    }
    assert reviewed_ids == {"segmented:event:000002"}
    assert result.diagnostics["review"]["reviewed_window_count"] == 1


def test_reasr_failure_keeps_primary_candidate_and_records_reason():
    engine = CountingReASR(fail=True)
    events = [_event(text="0123456789abcdefgh")]

    result = _run(_config(), events, engine)

    assert len(engine.calls) == 1
    review_diag = result.diagnostics["review"]
    assert review_diag["reasr_failed_count"] == 1
    assert review_diag["status"] == "degraded"
    assert "reasr engine crashed" in review_diag["windows"][0]["error"]
    # 主候选保留:final text 仍是 segmented 文本。
    assert result.decisions[0].final_text == "0123456789abcdefgh"
