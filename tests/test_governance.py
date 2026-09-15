"""Tests for the vocal_subtitle.governance module.

Covers:
- engine_lifecycle: EngineLifecycle, EngineStatus, EngineRegistry, LifecycleManager
- experiment_registry: ExperimentStatus, ExperimentRecord, ExperimentRegistry
- release: ReleaseStatus, ReleaseManager, PreReleaseChecklist, RollbackStrategy
"""

import pytest

from vocal_subtitle.governance.engine_lifecycle import (
    ALLOWED_LIFECYCLE_TRANSITIONS,
    EngineLifecycle,
    EngineRegistry,
    EngineStatus,
    LifecycleManager,
)
from vocal_subtitle.governance.experiment_registry import (
    ALLOWED_EXPERIMENT_TRANSITIONS,
    ExperimentRegistry,
    ExperimentStatus,
)
from vocal_subtitle.governance.release import (
    AlertReport,
    ObservabilityMetrics,
    PreReleaseChecklist,
    ReleaseManager,
    ReleaseStatus,
)

# ---------------------------------------------------------------------------
# EngineLifecycle enum and transitions
# ---------------------------------------------------------------------------


def test_engine_lifecycle_enum_values():
    assert EngineLifecycle.UNAVAILABLE.value == "unavailable"
    assert EngineLifecycle.MODEL_MISSING.value == "model_missing"
    assert EngineLifecycle.READY_SHADOW.value == "ready_shadow"
    assert EngineLifecycle.READY_REVIEW.value == "ready_review"
    assert EngineLifecycle.READY_DEFAULT.value == "ready_default"
    assert str(EngineLifecycle.READY_SHADOW) == "ready_shadow"
    assert len(EngineLifecycle) == 5


def test_allowed_lifecycle_transitions_match_spec():
    assert ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.UNAVAILABLE] == frozenset(
        {EngineLifecycle.MODEL_MISSING}
    )
    assert ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.MODEL_MISSING] == frozenset(
        {EngineLifecycle.READY_SHADOW}
    )
    assert ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.READY_SHADOW] == frozenset(
        {EngineLifecycle.READY_REVIEW, EngineLifecycle.MODEL_MISSING}
    )
    assert ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.READY_REVIEW] == frozenset(
        {EngineLifecycle.READY_DEFAULT, EngineLifecycle.READY_SHADOW}
    )
    assert ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.READY_DEFAULT] == frozenset(
        {EngineLifecycle.READY_SHADOW}
    )
    assert set(ALLOWED_LIFECYCLE_TRANSITIONS) == set(EngineLifecycle)


def test_invalid_edges_absent():
    assert (
        EngineLifecycle.READY_DEFAULT
        not in ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.UNAVAILABLE]
    )
    assert (
        EngineLifecycle.READY_SHADOW
        not in ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.UNAVAILABLE]
    )
    assert (
        EngineLifecycle.READY_DEFAULT
        not in ALLOWED_LIFECYCLE_TRANSITIONS[EngineLifecycle.MODEL_MISSING]
    )


# ---------------------------------------------------------------------------
# EngineStatus
# ---------------------------------------------------------------------------


def test_engine_status_to_dict():
    status = EngineStatus(
        engine="whisper-test",
        model="tiny",
        status=EngineLifecycle.READY_SHADOW,
        device="cuda",
        model_path="/models/tiny.ct2",
        model_hash="sha256:abcd",
        language_support="zh, en",
    )
    d = status.to_dict()
    assert d["engine"] == "whisper-test"
    assert d["status"] == "ready_shadow"
    assert d["device"] == "cuda"
    assert d["model_hash"] == "sha256:abcd"


def test_engine_status_defaults():
    status = EngineStatus(engine="x")
    assert status.status is EngineLifecycle.UNAVAILABLE
    assert status.device == "cpu"
    assert status.model == ""


# ---------------------------------------------------------------------------
# EngineRegistry
# ---------------------------------------------------------------------------


def test_registry_prepopulated_entries():
    registry = EngineRegistry()
    names = {e.engine for e in registry.list_all()}
    assert names == {
        "uvr",
        "spleeter",
        "open-unmix",
        "silero",
        "webrtc",
        "faster-whisper",
        "funasr",
        "qwen-asr",
        "whisper.cpp",
        "global-asr-evidence",
        "context-reasr",
        "qwen-review",
        "forced-aligner",
        "sed",
        "semantic-review",
        "speechbrain-ecapa",
        "pyannote",
    }
    assert len(registry.list_all()) == 17


def test_registry_get_engine():
    registry = EngineRegistry()
    uvr = registry.get("uvr")
    assert uvr is not None
    assert uvr.engine == "uvr"
    assert uvr.status is EngineLifecycle.READY_DEFAULT

    qwen_review = registry.get("qwen-review")
    assert qwen_review.status is EngineLifecycle.MODEL_MISSING

    assert registry.get("does-not-exist") is None


def test_registry_list_by_status():
    registry = EngineRegistry()
    assert {
        e.engine for e in registry.list_by_status(EngineLifecycle.READY_DEFAULT)
    } == {
        "uvr",
        "silero",
        "faster-whisper",
        "global-asr-evidence",
        "speechbrain-ecapa",
    }
    assert {
        e.engine for e in registry.list_by_status(EngineLifecycle.READY_SHADOW)
    } == {
        "open-unmix",
        "webrtc",
        "funasr",
        "qwen-asr",
        "whisper.cpp",
        "pyannote",
    }
    assert {e.engine for e in registry.list_by_status(EngineLifecycle.UNAVAILABLE)} == {
        "spleeter",
        "context-reasr",
        "semantic-review",
    }
    assert registry.list_by_status(EngineLifecycle.READY_REVIEW) == []


def test_registry_list_by_category():
    registry = EngineRegistry()
    assert {e.engine for e in registry.list_by_category("separation")} == {
        "uvr",
        "spleeter",
        "open-unmix",
    }
    assert {e.engine for e in registry.list_by_category("vad")} == {"silero", "webrtc"}
    assert {e.engine for e in registry.list_by_category("asr")} == {
        "faster-whisper",
        "funasr",
        "qwen-asr",
        "whisper.cpp",
    }
    assert registry.list_by_category("unknown") == []


def test_registry_to_dict():
    registry = EngineRegistry()
    d = registry.to_dict()
    assert len(d) == 17
    assert d["faster-whisper"]["status"] == "ready_default"
    assert d["qwen-review"]["status"] == "model_missing"


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------


def test_promote_full_chain():
    mgr = LifecycleManager()
    engine = "spleeter"  # starts unavailable
    assert mgr.registry.get(engine).status is EngineLifecycle.UNAVAILABLE

    assert mgr.transition(engine, EngineLifecycle.MODEL_MISSING) is True
    assert mgr.registry.get(engine).status is EngineLifecycle.MODEL_MISSING

    assert mgr.promote_to_shadow(engine, model_path="/models/spleeter") is True
    assert mgr.registry.get(engine).status is EngineLifecycle.READY_SHADOW

    assert mgr.promote_to_review(engine) is True
    assert mgr.registry.get(engine).status is EngineLifecycle.READY_REVIEW

    assert mgr.promote_to_default(engine) is True
    entry = mgr.registry.get(engine)
    assert entry.status is EngineLifecycle.READY_DEFAULT
    assert entry.model_path == "/models/spleeter"


def test_rollback_to_shadow():
    mgr = LifecycleManager()
    engine = "uvr"  # default ready_default
    assert mgr.rollback_to_shadow(engine) is True
    assert mgr.registry.get(engine).status is EngineLifecycle.READY_SHADOW


def test_force_transition_bypasses_validation():
    mgr = LifecycleManager()
    assert mgr.transition("spleeter", EngineLifecycle.READY_DEFAULT, force=True) is True
    assert mgr.registry.get("spleeter").status is EngineLifecycle.READY_DEFAULT


def test_invalid_transition_raises_value_error():
    mgr = LifecycleManager()
    with pytest.raises(ValueError, match="Invalid transition"):
        mgr.transition("spleeter", EngineLifecycle.READY_DEFAULT)
    assert mgr.registry.get("spleeter").status is EngineLifecycle.UNAVAILABLE


def test_transition_unknown_engine_raises():
    mgr = LifecycleManager()
    with pytest.raises(ValueError, match="Unknown engine"):
        mgr.transition("nonexistent-engine", EngineLifecycle.READY_SHADOW)


# ---------------------------------------------------------------------------
# ExperimentRegistry
# ---------------------------------------------------------------------------


def test_experiment_status_enum_values():
    assert [s.value for s in ExperimentStatus] == [
        "proposed",
        "shadow",
        "review",
        "enabled",
        "rolled_back",
    ]


def test_allowed_experiment_transitions():
    assert ALLOWED_EXPERIMENT_TRANSITIONS[ExperimentStatus.PROPOSED] == frozenset(
        {ExperimentStatus.SHADOW}
    )
    assert ALLOWED_EXPERIMENT_TRANSITIONS[ExperimentStatus.SHADOW] == frozenset(
        {ExperimentStatus.REVIEW, ExperimentStatus.ROLLED_BACK}
    )
    assert ALLOWED_EXPERIMENT_TRANSITIONS[ExperimentStatus.REVIEW] == frozenset(
        {ExperimentStatus.ENABLED, ExperimentStatus.ROLLED_BACK}
    )
    assert ALLOWED_EXPERIMENT_TRANSITIONS[ExperimentStatus.ENABLED] == frozenset(
        {ExperimentStatus.ROLLED_BACK}
    )
    assert ALLOWED_EXPERIMENT_TRANSITIONS[ExperimentStatus.ROLLED_BACK] == frozenset(
        {ExperimentStatus.PROPOSED}
    )


def test_experiment_registry_defaults():
    registry = ExperimentRegistry()
    assert len(registry.list_all()) == 6
    assert registry.get("exp-20260802-qwen-review") is not None
    assert registry.get("exp-does-not-exist") is None


def test_experiment_status_chain():
    registry = ExperimentRegistry()
    exp_id = "exp-20260802-llm-optimize"
    assert registry.get(exp_id).status is ExperimentStatus.PROPOSED

    assert registry.approve_to_shadow(exp_id) is True
    assert registry.get(exp_id).status is ExperimentStatus.SHADOW

    assert registry.promote_to_review(exp_id) is True
    assert registry.get(exp_id).status is ExperimentStatus.REVIEW

    assert registry.enable(exp_id) is True
    assert registry.get(exp_id).status is ExperimentStatus.ENABLED

    assert registry.rollback(exp_id, reason="regression") is True
    assert registry.get(exp_id).status is ExperimentStatus.ROLLED_BACK

    assert registry.transition(exp_id, "proposed") is True
    assert registry.get(exp_id).status is ExperimentStatus.PROPOSED


def test_experiment_invalid_transition_rejected():
    registry = ExperimentRegistry()
    with pytest.raises(ValueError, match="Invalid transition"):
        registry.transition("exp-20260802-llm-optimize", "enabled")
    with pytest.raises(ValueError, match="Invalid status"):
        registry.transition("exp-20260802-qwen-review", "bogus")
    with pytest.raises(ValueError, match="Unknown experiment"):
        registry.transition("exp-does-not-exist", "shadow")


def test_experiment_list_by_status_and_category():
    registry = ExperimentRegistry()
    shadow_ids = {e.experiment_id for e in registry.list_by_status("shadow")}
    assert shadow_ids == {
        "exp-20260802-qwen-review",
        "exp-20260802-forced-aligner",
        "exp-20260802-sed-non-speech",
        "exp-20260802-vad-fusion",
    }
    assert registry.list_by_status("enabled") == []
    assert registry.list_by_status("bogus") == []

    engine_ids = {e.experiment_id for e in registry.list_by_category("engine")}
    assert engine_ids == {
        "exp-20260802-qwen-review",
        "exp-20260802-forced-aligner",
        "exp-20260802-sed-non-speech",
    }
    assert registry.list_by_category("bogus") == []


# ---------------------------------------------------------------------------
# ReleaseManager
# ---------------------------------------------------------------------------


def test_release_status_values():
    assert ReleaseStatus.DEVELOPMENT.value == "development"
    assert ReleaseStatus.PRODUCTION_USABLE.value == "production-usable"
    assert ReleaseStatus.QUALITY_IMPROVING.value == "quality-improving"


def test_classify_development_by_default():
    status, blockers = ReleaseManager.classify()
    assert status is ReleaseStatus.DEVELOPMENT
    assert blockers == ["d0_regression", "d1_replay", "smoke_test"]


def test_classify_production_usable():
    criteria = {key: True for key, _, _ in ReleaseManager.PRODUCTION_CRITERIA}
    status, blockers = ReleaseManager.classify(criteria)
    assert status is ReleaseStatus.PRODUCTION_USABLE
    assert blockers == []


def test_prerelease_checklist():
    checklist = PreReleaseChecklist(
        version="1.0.0",
        sections={"deps": [{"key": "lock", "passed": True}]},
    )
    assert checklist.all_passed() is True  # all checked items pass
    checklist.sections["deps"][0]["passed"] = False
    assert checklist.all_passed() is False  # one item fails
    assert PreReleaseChecklist(version="1.0.0").all_passed() is True  # empty = all pass


def test_observe_clean_metrics_no_alerts():
    report = ReleaseManager.observe(ObservabilityMetrics())
    assert isinstance(report, AlertReport)
    assert report.ok() is True
    assert report.alerts == []


def test_observe_crash_rate_alert():
    report = ReleaseManager.observe(ObservabilityMetrics(crash_rate=0.15))
    assert report.ok() is False
    assert report.alerts[0]["metric"] == "crash_rate"


def test_observe_multiple_alerts():
    report = ReleaseManager.observe(
        ObservabilityMetrics(
            crash_rate=0.12,
            degradation_rate=0.30,
            export_failure_rate=0.10,
            high_severity_count=2,
        )
    )
    assert len(report.alerts) >= 4


def test_rollback_config_strategy():
    plan = ReleaseManager.rollback_config("production")
    assert plan["strategy"] == "config"
    assert "vocal-subtitle feedback rollback" in plan["command"]


def test_disable_engine_overrides():
    qwen = ReleaseManager.disable_engine("qwen")
    assert qwen["strategy"] == "engine_disable"
    assert "qwen_enabled: false" in qwen["override_yaml"]


def test_known_limitations():
    limitations = ReleaseManager.list_known_limitations()
    assert len(limitations) == 8
    assert limitations[0]["limitation_id"] == "LIM-001"


def test_release_notes():
    notes = ReleaseManager.release_notes(
        version="0.3.0",
        status=ReleaseStatus.PRODUCTION_USABLE,
        changes={"新增": ["X"]},
        upgrades=["迁移"],
        known_issues=["问题一"],
    )
    assert "# Vocal Subtitle v0.3.0" in notes
    assert "production-usable" in notes


def test_check_pre_release(tmp_path):
    mgr = ReleaseManager(storage_dir=tmp_path)
    checklist = mgr.check_pre_release("0.2.0")
    assert checklist.version == "0.2.0"
    assert set(checklist.sections) == {
        "dependencies",
        "models",
        "tests",
        "d0_regression",
        "smoke",
        "export",
        "docs",
    }


def test_verify_item_persists(tmp_path):
    mgr = ReleaseManager(storage_dir=tmp_path)
    mgr.check_pre_release("0.2.0")
    assert (
        mgr.verify_item("0.2.0", "dependencies", "lock_files", True, note="locked")
        is True
    )
    mgr2 = ReleaseManager(storage_dir=tmp_path)
    assert mgr2.verify_item("0.2.0", "models", "silero_vad", True) is True


def test_current_state(tmp_path):
    mgr = ReleaseManager(storage_dir=tmp_path)
    state = mgr.current_state()
    assert state["current_version"] == "0.2.0"
    assert state["status"] == "development"
