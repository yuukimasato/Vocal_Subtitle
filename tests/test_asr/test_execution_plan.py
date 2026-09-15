"""ASRExecutionPlan 状态机(2026-09-15 重构计划 Task 4)。

五类路径转移:
- 显式引擎:无回退;
- auto 路由:主引擎为选中引擎,无回退;
- FunASR 质量门禁回退:auto + funasr + faster-whisper 回退引擎;
- 显式 global 失败:不允许静默降级;
- global_primary:门禁失败回退 segmented。
"""

from types import SimpleNamespace

from vocal_subtitle.asr.execution_plan import build_execution_plan


def _decision(requested="auto", selected="faster-whisper", fallback=None):
    return SimpleNamespace(
        requested_engine=requested,
        selected_engine=selected,
        fallback_engine=fallback,
    )


def test_explicit_engine_has_no_fallback():
    plan = build_execution_plan(
        config=SimpleNamespace(),
        decision=_decision("faster-whisper"),
        requested_path="segmented",
    )

    assert plan.primary_engine == "faster-whisper"
    assert plan.fallback_engine is None
    assert plan.fallback_trigger == "none"


def test_auto_route_selects_probed_engine_without_fallback():
    plan = build_execution_plan(
        config=SimpleNamespace(),
        decision=_decision("auto", "faster-whisper"),
        requested_path="segmented",
    )

    assert plan.requested_engine == "auto"
    assert plan.primary_engine == "faster-whisper"
    assert plan.fallback_engine is None


def test_funasr_fallback_is_explicit_transition():
    plan = build_execution_plan(
        config=SimpleNamespace(),
        decision=_decision("auto", "funasr", fallback="faster-whisper"),
        requested_path="segmented",
    )

    assert plan.primary_engine == "funasr"
    assert plan.fallback_engine == "faster-whisper"
    assert plan.fallback_trigger == "funasr_quality_gate"


def test_explicit_global_failure_does_not_degrade_silently():
    plan = build_execution_plan(
        config=SimpleNamespace(),
        decision=_decision("auto"),
        requested_path="global",
    )

    assert plan.hard_fail_on_primary_error is True


def test_global_primary_falls_back_to_segmented():
    plan = build_execution_plan(
        config=SimpleNamespace(),
        decision=_decision("auto"),
        requested_path="global_primary",
    )

    assert plan.requested_path == "global_primary"
    assert plan.fallback_engine == "segmented"
    assert plan.fallback_trigger == "gate_or_failure"
    assert plan.hard_fail_on_primary_error is False


def test_plan_serialization_round_trip():
    plan = build_execution_plan(
        config=SimpleNamespace(),
        decision=_decision("auto", "funasr", fallback="faster-whisper"),
        requested_path="segmented",
    )

    payload = plan.to_dict()
    assert payload["fallback_trigger"] == "funasr_quality_gate"
    assert isinstance(payload, dict)
