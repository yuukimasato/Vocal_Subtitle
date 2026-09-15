"""工单 06 回归测试：参数 overrides 接线（feedback.apply_overrides_on_run，默认关）

覆盖：
- FeedbackConfig 默认关闭；8613 开关状态持久化读写
- 开关关闭时配置构建与接线前完全一致（同输入同配置零行为变化对照）
- 开关开启时合并 active_profile overrides，显式单次 overrides 优先
- 任务提交与后台运行两条配置构建路径都吃到合并结果
- 接线后的学习路径上震荡锁定与自动回滚护栏仍然生效
"""

import asyncio
import dataclasses
from pathlib import Path

import pytest

from vocal_subtitle.config import ConfigLoader, FeedbackConfig
from vocal_subtitle.feedback import UserProfileManager
from vocal_subtitle.feedback.health_scorer import should_auto_rollback
from vocal_subtitle.feedback.pipeline_stage import PipelineFeedbackMixin
from vocal_subtitle.feedback.user_profile import (
    APPLY_OVERRIDES_TOGGLE_FILE,
    load_apply_overrides_on_run,
    save_apply_overrides_on_run,
)
from vocal_subtitle.utils.file_hasher import compute_config_hash
from vocal_subtitle.webui import api, pipeline_tasks

RUN_OVERRIDES = {"language": "zh", "vad_threshold": 0.6}


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """隔离 ~/.vocal_subtitle：档案与开关状态都落在临时目录"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _install_learned_profile(overrides=None, feedback_count=2, history=None):
    """写入带 overrides 的 active_profile 并打开 8613"应用学习参数"开关"""
    mgr = UserProfileManager()
    profile = mgr.load("user_default")
    profile["overrides"] = overrides if overrides is not None else {}
    profile["feedback_count"] = feedback_count
    if history is not None:
        profile["history"] = history
    mgr.save(profile)
    save_apply_overrides_on_run(True)
    return mgr


# ---------------------------------------------------------------------------
# 开关默认值与持久化
# ---------------------------------------------------------------------------


def test_apply_overrides_on_run_defaults_false():
    """验收：feedback 配置新增 apply_overrides_on_run，默认 false"""
    assert FeedbackConfig().apply_overrides_on_run is False
    # 场景模板未配置 feedback 节时同样默认关
    config = ConfigLoader().load_profile("default")
    assert config.feedback.apply_overrides_on_run is False


def test_toggle_state_persists_in_profile_dir(isolated_home):
    """8613 开关状态持久化在用户配置目录，读写字节一致"""
    assert load_apply_overrides_on_run() is False  # 默认关
    assert save_apply_overrides_on_run(True) is True
    assert load_apply_overrides_on_run() is True
    assert save_apply_overrides_on_run(False) is False
    assert load_apply_overrides_on_run() is False
    toggle_file = (
        isolated_home / ".vocal_subtitle" / "profiles" / APPLY_OVERRIDES_TOGGLE_FILE
    )
    assert toggle_file.exists()


# ---------------------------------------------------------------------------
# 开关两态：配置构建对照
# ---------------------------------------------------------------------------


def test_switch_off_config_identical_to_prewiring(isolated_home):
    """验收（对照验证）：开关关闭时与接线前的配置构建产物完全一致"""
    loader = ConfigLoader()
    first = pipeline_tasks._build_run_config(loader, "default", dict(RUN_OVERRIDES))
    second = pipeline_tasks._build_run_config(loader, "default", dict(RUN_OVERRIDES))
    legacy = loader.merge_with_overrides(
        loader.load_profile("default"), **RUN_OVERRIDES
    )
    # 同输入同配置两次构建产物一致，且与接线前路径逐字段一致
    assert dataclasses.asdict(first) == dataclasses.asdict(second)
    assert dataclasses.asdict(first) == dataclasses.asdict(legacy)
    assert compute_config_hash(first) == compute_config_hash(legacy)


def test_switch_on_without_profile_overrides_is_noop(isolated_home):
    """开关开启但档案尚无 overrides 时不改变配置"""
    _install_learned_profile(overrides={})
    loader = ConfigLoader()
    wired = pipeline_tasks._build_run_config(loader, "default", dict(RUN_OVERRIDES))
    legacy = loader.merge_with_overrides(
        loader.load_profile("default"), **RUN_OVERRIDES
    )
    assert dataclasses.asdict(wired) == dataclasses.asdict(legacy)


def test_switch_on_merges_active_profile_overrides(isolated_home):
    """验收（合并生效）：开关开启时 active_profile 的 overrides 合入管线配置"""
    _install_learned_profile(overrides={"merging": {"padding": 0.22}})
    loader = ConfigLoader()
    wired = pipeline_tasks._build_run_config(loader, "default", {})
    baseline = loader.load_profile("default")
    assert wired.merging.padding == pytest.approx(0.22)
    assert baseline.merging.padding != pytest.approx(0.22)
    # 合并改变了有效配置 → 全管线缓存键随之隔离
    assert compute_config_hash(wired) != compute_config_hash(baseline)


def test_explicit_run_overrides_take_precedence_over_learned(isolated_home):
    """显式单次 overrides 后应用，优先于学习参数（用户当次意图优先）"""
    _install_learned_profile(overrides={"vad": {"threshold": 0.8}})
    loader = ConfigLoader()
    wired = pipeline_tasks._build_run_config(loader, "default", {"vad_threshold": 0.5})
    assert wired.vad.threshold == pytest.approx(0.5)
    # 无显式覆盖时学习参数生效
    learned_only = pipeline_tasks._build_run_config(loader, "default", {})
    assert learned_only.vad.threshold == pytest.approx(0.8)


def test_yaml_gate_enables_merge_without_toggle_file(isolated_home):
    """场景模板 YAML 显式开启 feedback.apply_overrides_on_run 也可作为开关"""
    _install_learned_profile(overrides={"merging": {"padding": 0.22}})
    config = ConfigLoader().load_profile("default")
    config.feedback.apply_overrides_on_run = True
    merged = pipeline_tasks._apply_active_profile_overrides(config)
    assert merged.merging.padding == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# 运行路径接线：后台线程与任务提交
# ---------------------------------------------------------------------------


class _HistorySpy:
    def __init__(self):
        self.updates = []

    def update(self, task_id, **fields):
        self.updates.append(fields)


class _SubmitHistorySpy:
    def __init__(self):
        self.created = None

    def find_by_hash(self, *args, **kwargs):
        return None

    def create(self, **kwargs):
        self.created = kwargs

    def update(self, *args, **kwargs):
        pass


class _WebSocketSpy:
    def create_progress_callback(self, task_id):
        return lambda event: None

    def broadcast_from_thread(self, task_id, message):
        return None

    def store_task_result(self, task_id, result):
        return None


def test_run_thread_consumes_wired_config(monkeypatch, tmp_path, isolated_home):
    """后台运行线程构建的管线配置吃到合并结果（两态对照）"""
    from vocal_subtitle.application.pipeline_result import PipelineStats

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.srt"
    input_path.write_bytes(b"RIFF")
    baseline = ConfigLoader().load_profile("default").merging.padding

    def _run_case(captured):
        history = _HistorySpy()

        class FakePipeline:
            def __init__(self, config):
                captured.append(config)

            def run(self, **kwargs):
                stats = PipelineStats(input_path=input_path, duration_seconds=1.0)
                stats.status = "completed"
                stats.run_id = "run-1"
                return {"subtitle_path": output_path, "stats": stats, "events": []}

        monkeypatch.setattr(api, "Pipeline", FakePipeline)
        monkeypatch.setattr(
            api, "_task_store", {"task-1": {"task_id": "task-1", "status": "pending"}}
        )
        monkeypatch.setattr(api, "_task_history", history)
        monkeypatch.setattr(pipeline_tasks, "ws_manager", _WebSocketSpy())
        monkeypatch.setattr(
            pipeline_tasks,
            "_persistence_manager",
            lambda: type("Persistence", (), {"persist_task": lambda *args: None})(),
        )
        pipeline_tasks.run_pipeline_in_thread(
            "task-1",
            input_path,
            output_path,
            "default",
            "srt",
            True,
            {},
            tmp_path,
        )
        assert history.updates[-1]["status"] == "completed"

    # 开关关闭：与基线一致（零行为变化）
    captured = []
    _run_case(captured)
    assert captured[0].merging.padding == pytest.approx(baseline)

    # 开关开启：合并 active_profile 的学习参数
    captured = []
    _install_learned_profile(overrides={"merging": {"padding": 0.22}})
    _run_case(captured)
    assert captured[0].merging.padding == pytest.approx(0.22)


def test_submit_builds_wired_config(monkeypatch, tmp_path, isolated_home):
    """任务提交阶段的配置构建（缓存键/历史记录来源）吃到合并结果"""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "_task_store", {})
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)

    def _noop_thread(*args):
        pass

    service = pipeline_tasks.PipelineTaskService()
    baseline = ConfigLoader().load_profile("default")

    # 开关关闭：历史记录中的配置与基线一致（零行为变化）
    spy_off = _SubmitHistorySpy()
    monkeypatch.setattr(api, "_task_history", spy_off)
    result = asyncio.run(
        service.submit(
            b"RIFF",
            "clip.wav",
            profile="default",
            output_format="srt",
            skip_separation=True,
            overrides_text="{}",
            thread_target=_noop_thread,
        )
    )
    assert result["task_id"]
    assert dataclasses.asdict(spy_off.created["config"]) == dataclasses.asdict(baseline)

    # 开关开启：合并学习参数
    _install_learned_profile(overrides={"merging": {"padding": 0.22}})
    spy_on = _SubmitHistorySpy()
    monkeypatch.setattr(api, "_task_history", spy_on)
    asyncio.run(
        service.submit(
            b"RIFF",
            "clip2.wav",
            profile="default",
            output_format="srt",
            skip_separation=True,
            overrides_text="{}",
            thread_target=_noop_thread,
        )
    )
    assert spy_on.created is not None
    assert spy_on.created["config"].merging.padding == pytest.approx(0.22)
    assert baseline.merging.padding != pytest.approx(0.22)


# ---------------------------------------------------------------------------
# 护栏回归：震荡锁定与自动回滚（接线后的学习路径）
# ---------------------------------------------------------------------------


OSCILLATION_HISTORY = [
    {
        "adjustments": {"merging.padding": [0.10, 0.14]},
        "timestamp": "2026-07-01T10:00:00",
        "diff_report_summary": "增大",
    },
    {
        "adjustments": {"merging.padding": [0.14, 0.09]},
        "timestamp": "2026-07-02T10:00:00",
        "diff_report_summary": "减小",
    },
    {
        "adjustments": {"merging.padding": [0.09, 0.13]},
        "timestamp": "2026-07-03T10:00:00",
        "diff_report_summary": "增大",
    },
    {
        "adjustments": {"merging.padding": [0.13, 0.08]},
        "timestamp": "2026-07-04T10:00:00",
        "diff_report_summary": "减小",
    },
]


class _WiredStage(PipelineFeedbackMixin):
    """最小管线桩：复用 mixin 的学习流程，D2 入库旁路"""

    def __init__(self, config):
        self.config = config
        self.ingested = None

    def _ingest_feedback_sample(self, **kwargs):
        self.ingested = kwargs


def _auto_events():
    """合成自动字幕事件（结束时间比参照字幕早 150ms → padding 增大归因）"""
    from vocal_subtitle.mapping.time_mapper import SubtitleEvent

    return [
        SubtitleEvent(
            index=1, start=0.0, end=2.0, text="今天天气不错", speaker_label=None
        ),
        SubtitleEvent(
            index=2, start=2.5, end=5.0, text="我们去看电影", speaker_label=None
        ),
    ]


def _write_reference_srt(tmp_path):
    path = tmp_path / "reference.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:02,150\n今天天气不错\n\n"
        "2\n00:00:02,500 --> 00:00:05,150\n我们去看电影\n",
        encoding="utf-8",
    )
    return path


def _wired_stage_config():
    """开关开启时的运行配置（与 run_pipeline_in_thread 同一构建函数）"""
    config = pipeline_tasks._build_run_config(ConfigLoader(), "default", {})
    # 关闭语义相似度避免测试加载句子向量模型
    config.feedback.alignment_semantic_enabled = False
    return config


def test_wired_learning_path_respects_oscillation_lock(isolated_home, tmp_path):
    """震荡检测标记 + locked_params 锁定：学习不推进已锁定参数"""
    baseline = ConfigLoader().load_profile("default").merging.padding
    _install_learned_profile(
        overrides={"merging": {"padding": baseline}},
        feedback_count=5,
        history=list(OSCILLATION_HISTORY),
    )
    mgr = UserProfileManager()
    locked = mgr.load("user_default")
    locked["locked_params"] = ["merging.padding"]
    mgr.save(locked)

    stage = _WiredStage(_wired_stage_config())
    report = stage._run_feedback_learning(
        _auto_events(),
        _write_reference_srt(tmp_path),
        str(tmp_path / "audio.wav"),
    )
    assert report is not None
    assert report["oscillations_detected"] >= 1
    reloaded = mgr.load("user_default")
    assert reloaded["overrides"]["merging"]["padding"] == pytest.approx(baseline)


def test_wired_learning_path_auto_rollback_restores_overrides(isolated_home, tmp_path):
    """学习推进参数后，健康度骤降触发自动回滚，档案恢复到学习前版本"""
    baseline = ConfigLoader().load_profile("default").merging.padding
    mgr = _install_learned_profile(
        overrides={"merging": {"padding": baseline}},
        feedback_count=2,
        history=[],
    )
    # 再保存一次生成 bak.1（内容与学习前一致），使学习后的版本可回滚
    mgr.save(mgr.load("user_default"))

    stage = _WiredStage(_wired_stage_config())
    report = stage._run_feedback_learning(
        _auto_events(),
        _write_reference_srt(tmp_path),
        str(tmp_path / "audio.wav"),
    )
    assert report is not None
    learned = mgr.load("user_default")
    assert learned["feedback_count"] == 3
    assert learned["overrides"]["merging"]["padding"] > baseline + 0.001

    # 自动回滚判定（pipeline_stage Step 4.5 同款调用）在接线后的配置上仍触发
    should, reason = should_auto_rollback(
        80.0,
        40.0,
        drop_threshold=stage.config.feedback.quality_drop_threshold,
    )
    assert should
    restored = mgr.rollback("user_default")
    assert restored["overrides"]["merging"]["padding"] == pytest.approx(baseline)
