"""工单 03：冷重跑异步学习任务（定案 D28/D21/D27）。

覆盖验收项四层：
- 任务创建：V3/V4 缺 task_id 的学习请求转内部学习任务，立即返回任务标识，
  任务存储与历史带 task_type="learn" + 场景标签；V1 场景与 dry_run 预览不走异步
- 完成回调：管线冷重跑完成后自动执行与同步路径同一套"对齐→diff→入库"，
  报告挂任务详情（learn_report），进度事件带"学习"标记
- 失败分支：管线失败原样透出；学习阶段失败任务转 failed 且错误可见
- 幂等：同一音频+同一参考字幕+同一场景重复提交返回原任务；
  失败任务允许重试；全管线缓存命中时同步秒级应答
"""

import hashlib
import json
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.feedback import sample_manager as sample_manager_module
from vocal_subtitle.utils.file_hasher import compute_config_hash
from vocal_subtitle.webui import api, feedback_services, pipeline_tasks
from vocal_subtitle.webui.app import create_app
from vocal_subtitle.webui.pipeline_tasks import (
    _build_run_config,
    _learn_dedupe_hash,
    run_learn_task_in_thread,
)


AUDIO_BYTES = b"fake audio bytes for cold-rerun learn"

REFERENCE_SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\n第一句\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\n第二句\n"
)

TASK_STATS = {
    "detected_language": "zh",
    "speaker_count": 2,
    "canonical_speaker_count": 2,
    "duration_seconds": 12.0,
}


def _serialized_event(index, start, end, text):
    """任务历史 result_json 中的事件形态（pipeline_tasks._serialize_events 的子集）"""
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "original_text": text,
        "speaker_id": None,
        "speaker_label": None,
    }


def _audio_hash():
    return hashlib.sha256(AUDIO_BYTES).hexdigest()


def _base_config_hash():
    """普通任务的全管线缓存键（与 submit_learn_task 的构建方式一致）"""
    return compute_config_hash(_build_run_config(ConfigLoader(), "default", {}))


def _session_dir(tmp_path):
    """上传音频的会话目录（SessionManager 字节哈希键的等价计算）"""
    directory = tmp_path / hashlib.sha256(AUDIO_BYTES).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.wav").write_bytes(AUDIO_BYTES)
    return directory


def _reference_hash(reference=REFERENCE_SRT):
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()


class FakeTaskHistory:
    """TaskHistoryManager 的最小替身（覆盖学习任务用到的接口）"""

    def __init__(self):
        self.records = {}

    def _sorted(self):
        return sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)

    def create(self, task_id, file_name, file_hash, file_size, profile, config, *,
               run_id="", task_type="", scenario="", config_hash=""):
        self.records[task_id] = {
            "id": task_id,
            "run_id": run_id,
            "input_file_name": file_name,
            "input_file_hash": file_hash,
            "input_file_size": file_size,
            "profile": profile,
            "config_json": "{}",
            "config_hash": config_hash,
            "status": "pending",
            "error_category": "",
            "progress_json": "{}",
            "result_json": None,
            "error": None,
            "total_duration_seconds": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "task_type": task_type,
            "scenario": scenario,
        }

    def update(self, task_id, **fields):
        self.records[task_id].update(fields)

    def get(self, task_id):
        return self.records.get(task_id)

    def find_by_hash(self, file_hash, config_hash, task_type=None):
        for rec in self._sorted():
            if (rec["input_file_hash"] == file_hash
                    and rec["config_hash"] == config_hash
                    and rec["status"] in ("completed", "degraded_completed")
                    and (task_type is None or rec.get("task_type") == task_type)):
                return rec
        return None

    def list(self, limit=50, offset=0, status=None):
        rows = [r for r in self._sorted() if status is None or r["status"] == status]
        return rows[offset:offset + limit]

    def count(self, status=None):
        return len([r for r in self.records.values()
                    if status is None or r["status"] == status])


class WsSpy:
    """WebSocketManager 替身：记录推送消息，捕获进度回调"""

    def __init__(self):
        self.messages = []

    def set_main_loop(self, loop):
        pass

    def create_progress_callback(self, task_id):
        def callback(event):
            self.messages.append({"task_id": task_id, "channel": "progress", **event})
        return callback

    def broadcast_from_thread(self, task_id, message):
        self.messages.append({"task_id": task_id, "channel": "broadcast", **message})


class ThreadRecorder:
    """替身线程目标：记录调用参数而不真正执行学习任务"""

    def __init__(self):
        self.calls = []
        self.event = threading.Event()

    def __call__(self, *args):
        self.calls.append(args)
        self.event.set()


class FakeSyncPipeline:
    """同步路径（inline-review / dry_run）的管线替身：两条与参考一致的事件"""

    def __init__(self, config):
        pass

    def run(self, **kwargs):
        events = [
            SimpleNamespace(index=1, start=1.0, end=3.0, text="第一句"),
            SimpleNamespace(index=2, start=4.0, end=6.0, text="第二句"),
        ]
        return {
            "events": events,
            "stats": SimpleNamespace(to_dict=lambda: dict(TASK_STATS)),
        }


def _make_fake_pipeline_run(store, history, captured, *, fail=False):
    """替身 run_pipeline_in_thread：模拟管线冷重跑的落库行为"""

    def fake_run(task_id, input_path, output_path, profile, output_format,
                 skip_separation, overrides, session_dir=None, progress_callback=None):
        captured["skip_separation"] = skip_separation
        captured["progress_callback"] = progress_callback
        if fail:
            store[task_id]["status"] = "failed"
            store[task_id]["error"] = "boom: 引擎不可用"
            history.update(task_id, status="failed", error="boom: 引擎不可用")
            return
        store[task_id]["status"] = "running"
        if progress_callback is not None:
            progress_callback({"type": "stage_start", "stage": "pipeline",
                               "total": 1, "description": "Pipeline 启动"})
        result = {
            "task_id": task_id,
            "run_id": "run-learn",
            "status": "completed",
            "input_path": str(input_path),
            "events": [
                _serialized_event(1, 1.0, 3.0, "第一句"),
                _serialized_event(2, 4.0, 6.0, "第二句"),
            ],
            "stats": dict(TASK_STATS),
            "artifacts": {"input": str(input_path)},
        }
        store[task_id]["status"] = "completed"
        store[task_id]["result"] = result
        history.update(task_id, status="completed", result_json=json.dumps(result, default=str))

    return fake_run


def _seed_completed_task(history, task_id, payload):
    """布置一条已完成的普通任务记录（全管线缓存/缓存命中场景）"""
    history.records[task_id] = {
        "id": task_id,
        "run_id": "run-orig",
        "input_file_name": "a.wav",
        "input_file_hash": _audio_hash(),
        "input_file_size": len(AUDIO_BYTES),
        "profile": "default",
        "config_json": "{}",
        "config_hash": payload.pop("_config_hash", _base_config_hash()),
        "status": "completed",
        "error_category": "",
        "progress_json": "{}",
        "result_json": json.dumps(payload, default=str),
        "error": None,
        "total_duration_seconds": 1.0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "task_type": "",
        "scenario": "",
    }


def _seed_learn_task(store, history, input_path):
    """手工布置一个待执行的内部学习任务（等价 submit_learn_task 的落库结果）"""
    task_id = "task-learn-1"
    store[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": None,
        "result": None,
        "error": None,
        "from_cache": False,
        "input_file_name": "a.wav",
        "session_dir": str(input_path.parent),
        "task_type": pipeline_tasks.LEARN_TASK_TYPE,
        "scenario": "existing-subtitle",
    }
    history.create(
        task_id=task_id,
        file_name="a.wav",
        file_hash=_audio_hash(),
        file_size=len(AUDIO_BYTES),
        profile="default",
        config=None,
        task_type=pipeline_tasks.LEARN_TASK_TYPE,
        scenario="existing-subtitle",
        config_hash=_learn_dedupe_hash(_base_config_hash(),
                                       _reference_hash(),
                                       "existing-subtitle"),
    )
    return task_id


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    """隔离的测试客户端：任务存储/历史/上传目录全部替换"""
    monkeypatch.setattr(api, "_task_store", {})
    monkeypatch.setattr(api, "_task_history", FakeTaskHistory())
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    return TestClient(create_app())


@pytest.fixture
def store(client):
    return api._task_store


@pytest.fixture
def history(client):
    return api._task_history


@pytest.fixture
def ws_spy(monkeypatch):
    spy = WsSpy()
    monkeypatch.setattr(pipeline_tasks, "ws_manager", spy)
    return spy


@pytest.fixture
def recorder(monkeypatch):
    rec = ThreadRecorder()
    monkeypatch.setattr(pipeline_tasks, "run_learn_task_in_thread", rec)
    return rec


@pytest.fixture
def stub_semantic_scorer(monkeypatch):
    """关闭对齐器的语义模型加载，保证测试离线且快速"""
    from vocal_subtitle.feedback import aligner as aligner_module

    monkeypatch.setattr(
        aligner_module.SemanticScorer,
        "is_available",
        property(lambda self: False),
    )


@pytest.fixture
def captured_samples(monkeypatch):
    """替换 FeedbackSampleManager，捕获 D2 入库元数据而不写真实样本库"""
    ingested = []

    class FakeSampleManager:
        def __init__(self, storage_dir=None):
            pass

        def ingest(self, **kwargs):
            ingested.append(kwargs)
            return SimpleNamespace(sample_id="sample-1")

    monkeypatch.setattr(sample_manager_module, "FeedbackSampleManager", FakeSampleManager)
    return ingested


@pytest.fixture
def guard_param_learner(monkeypatch):
    """事件与参考完全一致时不应触发参数学习器（避免测试写用户档案）"""

    def _boom(*args, **kwargs):
        raise AssertionError("identical events should not trigger param learning")

    monkeypatch.setattr(feedback_services, "UserProfileManager", _boom)
    monkeypatch.setattr(feedback_services, "ParamLearner", _boom)
    monkeypatch.setattr(feedback_services, "FewShotBuilder", _boom)


def _stub_pipeline_class(monkeypatch):
    monkeypatch.setattr(feedback_services, "_pipeline_class", lambda: FakeSyncPipeline)


# ---------------------------------------------------------------------------
# 任务创建
# ---------------------------------------------------------------------------

class TestLearnTaskCreation:
    def test_cold_rerun_creates_internal_learn_task(
        self, client, store, history, recorder,
    ):
        resp = _post_learn(client, "existing-subtitle")
        assert resp.status_code == 200
        data = resp.json()
        # 不阻塞请求：立即返回任务标识与学习标记
        assert data["task_type"] == "learn"
        assert data["status"] == "pending"
        assert data["scenario"] == "existing-subtitle"
        assert data["deduplicated"] is False
        task_id = data["task_id"]
        recorder.event.wait(timeout=5)

        # 任务存储带"学习"标记与场景标签
        assert store[task_id]["task_type"] == "learn"
        assert store[task_id]["scenario"] == "existing-subtitle"
        # 任务历史同样带标记（列表/详情可见性的数据源）
        record = history.get(task_id)
        assert record["task_type"] == "learn"
        assert record["scenario"] == "existing-subtitle"
        # 只提交一次，管线在后台线程执行
        assert len(recorder.calls) == 1

    def test_learn_task_visible_in_task_list_and_detail(
        self, client, recorder,
    ):
        task_id = _post_learn(client, "from-scratch-timing").json()["task_id"]

        tasks = client.get("/api/tasks").json()
        entry = next(t for t in tasks if t["task_id"] == task_id)
        assert entry["task_type"] == "learn"
        assert entry["scenario"] == "from-scratch-timing"

        status = client.get(f"/api/tasks/{task_id}").json()
        assert status["task_type"] == "learn"
        assert status["scenario"] == "from-scratch-timing"

        detail = client.get(f"/api/history/{task_id}").json()
        assert detail["task_type"] == "learn"
        assert detail["scenario"] == "from-scratch-timing"

        listing = client.get("/api/history").json()
        item = next(i for i in listing["items"] if i["id"] == task_id)
        assert item["task_type"] == "learn"
        assert item["scenario"] == "from-scratch-timing"

    def test_inline_review_without_task_id_stays_sync(
        self, client, monkeypatch, captured_samples, stub_semantic_scorer, recorder,
    ):
        """V1 场景（非冷重跑）不带 task_id 时保持旧客户端同步重跑行为"""
        _stub_pipeline_class(monkeypatch)
        resp = _post_learn(client, "inline-review")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["baseline_source"] == "pipeline_rerun"
        assert "task_type" not in data  # 同步应答，不是任务信封
        assert recorder.calls == []
        assert len(api._task_store) == 0

    def test_dry_run_preview_stays_sync(
        self, client, monkeypatch, captured_samples, stub_semantic_scorer, recorder,
    ):
        """dry_run 预览语义不变：即使 V3/V4 场景也不转异步任务"""
        _stub_pipeline_class(monkeypatch)
        files = {
            "reference": ("ref.srt", REFERENCE_SRT.encode("utf-8"), "application/octet-stream"),
            "audio": ("a.wav", AUDIO_BYTES, "audio/wav"),
        }
        resp = client.post(
            "/api/feedback/learn",
            data={"scenario": "existing-subtitle", "dry_run": "true"},
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "[dry-run]" in data["message"]
        assert recorder.calls == []
        assert len(api._task_store) == 0


def _post_learn(client, scenario, *, with_audio=True, reference=REFERENCE_SRT):
    files = {"reference": ("ref.srt", reference.encode("utf-8"), "application/octet-stream")}
    if with_audio:
        files["audio"] = ("a.wav", AUDIO_BYTES, "audio/wav")
    return client.post("/api/feedback/learn", data={"scenario": scenario}, files=files)


# ---------------------------------------------------------------------------
# 完成回调：对齐 → diff → 入库 → 报告挂任务详情
# ---------------------------------------------------------------------------

class TestLearnTaskCompletion:
    @pytest.fixture
    def completed(self, monkeypatch, tmp_path, store, history, ws_spy,
                  captured_samples, stub_semantic_scorer, guard_param_learner):
        input_path = _session_dir(tmp_path) / "input.wav"
        task_id = _seed_learn_task(store, history, input_path)
        reference_path = input_path.parent / "learn_reference_dummy.srt"
        reference_path.write_text(REFERENCE_SRT, encoding="utf-8")
        captured = {}
        monkeypatch.setattr(
            pipeline_tasks, "run_pipeline_in_thread",
            _make_fake_pipeline_run(store, history, captured),
        )
        run_learn_task_in_thread(
            task_id,
            input_path,
            reference_path,
            input_path.parent / "output.srt",
            "default",
            "existing-subtitle",
            session_dir=input_path.parent,
        )
        return {"task_id": task_id, "captured": captured}

    def test_report_mounted_on_task_detail(self, completed, store, history, ws_spy):
        task_id = completed["task_id"]
        # 任务存储：报告可读
        report = store[task_id]["learn_report"]
        assert report["status"] == "ok"
        assert report["baseline_source"] == "task_history"
        assert report["task_id"] == task_id
        # 任务详情：报告并入 result_json.learn_report
        result = json.loads(history.get(task_id)["result_json"])
        assert result["learn_report"]["status"] == "ok"
        assert result["learn_report"]["baseline_source"] == "task_history"
        # 任务保持完成态
        assert store[task_id]["status"] == "completed"
        assert history.get(task_id)["status"] == "completed"

    def test_d2_sample_ingested_with_scenario_and_stats(
        self, completed, captured_samples,
    ):
        assert len(captured_samples) == 1
        meta = captured_samples[0]
        assert meta["scene"] == "existing-subtitle"
        assert meta["language"] == "zh"
        assert meta["speaker_count"] == 2
        assert meta["audio_duration"] == 12.0

    def test_pipeline_phase_uses_full_run_and_marks_progress(self, completed, ws_spy):
        # D21：冷重跑完整跑一遍（不跳过分离），复用普通任务执行体
        assert completed["captured"]["skip_separation"] is False
        # 进度推送带"学习"标记与场景标签
        progress = [m for m in ws_spy.messages if m["channel"] == "progress"]
        assert progress and all(
            m.get("task_type") == "learn" and m.get("scenario") == "existing-subtitle"
            for m in progress
        )
        # 专属推送：learn_started / learn_complete
        types = {m["type"] for m in ws_spy.messages if m["channel"] == "broadcast"}
        assert "learn_started" in types
        assert "learn_complete" in types
        complete = next(m for m in ws_spy.messages if m.get("type") == "learn_complete")
        assert complete["task_id"] == completed["task_id"]
        assert complete["report"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 失败分支
# ---------------------------------------------------------------------------

class TestLearnTaskFailure:
    def test_pipeline_failure_is_visible_and_skips_learn(
        self, monkeypatch, tmp_path, store, history, ws_spy, captured_samples,
    ):
        input_path = _session_dir(tmp_path) / "input.wav"
        task_id = _seed_learn_task(store, history, input_path)
        captured = {}
        monkeypatch.setattr(
            pipeline_tasks, "run_pipeline_in_thread",
            _make_fake_pipeline_run(store, history, captured, fail=True),
        )
        run_learn_task_in_thread(
            task_id, input_path,
            input_path.parent / "learn_reference_dummy.srt",
            input_path.parent / "output.srt",
            "default", "existing-subtitle",
            session_dir=input_path.parent,
        )
        # 任务状态与错误可见
        assert store[task_id]["status"] == "failed"
        assert "boom" in store[task_id]["error"]
        assert history.get(task_id)["status"] == "failed"
        assert "boom" in history.get(task_id)["error"]
        # 学习阶段未执行：无报告、无入库、无完成推送
        assert "learn_report" not in store[task_id]
        assert captured_samples == []
        assert not any(m.get("type") == "learn_complete" for m in ws_spy.messages)

    def test_learn_phase_failure_marks_task_failed(
        self, monkeypatch, tmp_path, store, history, ws_spy,
        captured_samples, stub_semantic_scorer,
    ):
        input_path = _session_dir(tmp_path) / "input.wav"
        task_id = _seed_learn_task(store, history, input_path)
        broken_reference = input_path.parent / "learn_reference_broken.srt"
        broken_reference.write_text("", encoding="utf-8")  # 解析不到字幕事件
        captured = {}
        monkeypatch.setattr(
            pipeline_tasks, "run_pipeline_in_thread",
            _make_fake_pipeline_run(store, history, captured),
        )
        run_learn_task_in_thread(
            task_id, input_path, broken_reference,
            input_path.parent / "output.srt",
            "default", "existing-subtitle",
            session_dir=input_path.parent,
        )
        assert store[task_id]["status"] == "failed"
        assert "字幕事件" in store[task_id]["error"]
        assert history.get(task_id)["status"] == "failed"
        assert "字幕事件" in history.get(task_id)["error"]
        # 失败推送带学习标记与场景标签
        errors = [m for m in ws_spy.messages if m.get("type") == "error"]
        assert errors and errors[-1]["task_type"] == "learn"
        assert errors[-1]["scenario"] == "existing-subtitle"
        assert captured_samples == []


# ---------------------------------------------------------------------------
# 幂等 / 去重
# ---------------------------------------------------------------------------

class TestLearnTaskIdempotency:
    def test_duplicate_inflight_submission_returns_same_task(
        self, client, history, recorder,
    ):
        first = _post_learn(client, "existing-subtitle").json()
        second = _post_learn(client, "existing-subtitle").json()
        assert second["task_id"] == first["task_id"]
        assert second["deduplicated"] is True
        assert second["task_type"] == "learn"
        # 只提交一次后台任务，历史只有一条记录
        recorder.event.wait(timeout=5)
        assert len(recorder.calls) == 1
        assert len(history.records) == 1

    def test_completed_task_dedupe_returns_original(self, client, store, history, recorder):
        first = _post_learn(client, "existing-subtitle").json()
        recorder.event.wait(timeout=5)
        # 模拟任务已完成
        store[first["task_id"]]["status"] = "completed"
        history.records[first["task_id"]]["status"] = "completed"

        second = _post_learn(client, "existing-subtitle").json()
        assert second["task_id"] == first["task_id"]
        assert second["status"] == "completed"
        assert second["deduplicated"] is True
        assert len(recorder.calls) == 1

    def test_different_reference_creates_new_task(self, client, history, recorder):
        first = _post_learn(client, "existing-subtitle").json()
        other = REFERENCE_SRT.replace("第二句", "改过的第二句")
        second = _post_learn(client, "existing-subtitle", reference=other).json()
        assert second["task_id"] != first["task_id"]
        assert second["deduplicated"] is False
        assert len(history.records) == 2

    def test_failed_task_can_be_resubmitted(self, client, store, history, recorder):
        first = _post_learn(client, "existing-subtitle").json()
        recorder.event.wait(timeout=5)
        store[first["task_id"]]["status"] = "failed"
        history.records[first["task_id"]]["status"] = "failed"

        second = _post_learn(client, "existing-subtitle").json()
        assert second["task_id"] != first["task_id"]
        assert second["deduplicated"] is False
        assert len(recorder.calls) == 2


# ---------------------------------------------------------------------------
# 全管线缓存命中：不进队列，同步秒级应答
# ---------------------------------------------------------------------------

class TestPipelineCacheHit:
    def test_cache_hit_answers_synchronously(
        self, client, tmp_path, store, history, recorder,
        captured_samples, stub_semantic_scorer, guard_param_learner,
    ):
        session = _session_dir(tmp_path)
        _seed_completed_task(history, "task-orig", {
            "task_id": "task-orig",
            "run_id": "run-orig",
            "status": "completed",
            "input_path": str(session / "input.wav"),
            "events": [
                _serialized_event(1, 1.0, 3.0, "第一句"),
                _serialized_event(2, 4.0, 6.0, "第二句"),
            ],
            "stats": dict(TASK_STATS),
            "artifacts": {"input": str(session / "input.wav")},
        })

        resp = _post_learn(client, "existing-subtitle")
        assert resp.status_code == 200
        data = resp.json()
        # 秒级同步应答：task_id 引用已缓存任务，不创建内部任务
        assert data["status"] == "ok"
        assert data["baseline_source"] == "task_history"
        assert data["task_id"] == "task-orig"
        assert recorder.calls == []
        assert len(store) == 0
        # D2 元数据从任务 stats + 场景标签补全
        assert captured_samples[0]["scene"] == "existing-subtitle"
        assert captured_samples[0]["audio_duration"] == 12.0

    def test_cache_hit_with_cleaned_session_falls_back_to_task(
        self, client, tmp_path, store, history, recorder,
    ):
        _seed_completed_task(history, "task-orig", {
            "task_id": "task-orig",
            "status": "completed",
            "input_path": str(tmp_path / "cleaned-away" / "input.wav"),  # 已清理
            "events": [_serialized_event(1, 1.0, 3.0, "第一句")],
            "stats": dict(TASK_STATS),
        })

        resp = _post_learn(client, "existing-subtitle")
        assert resp.status_code == 200
        data = resp.json()
        # 会话产物已清理 → 回退为冷重跑内部任务
        assert data["task_type"] == "learn"
        assert data["status"] == "pending"
        assert data["task_id"] != "task-orig"
        recorder.event.wait(timeout=5)
        assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# 任务历史：task_type/scenario 列迁移与幂等键查询（真实 SQLite）
# ---------------------------------------------------------------------------

class TestTaskHistoryLearnColumns:
    def test_create_with_learn_marker_and_find_by_type(self, tmp_path):
        from vocal_subtitle.utils.task_history import TaskHistoryManager

        mgr = TaskHistoryManager(db_path=tmp_path / "history.db")
        config = _build_run_config(ConfigLoader(), "default", {})
        mgr.create(
            "t1", "a.wav", "h" * 64, 10, "default", config,
            task_type="learn", scenario="existing-subtitle", config_hash="c1",
        )
        record = mgr.get("t1")
        assert record["task_type"] == "learn"
        assert record["scenario"] == "existing-subtitle"
        # pending 不参与缓存命中
        assert mgr.find_by_hash("h" * 64, "c1") is None
        mgr.update("t1", status="completed")
        # 按任务类型过滤：学习任务幂等去重只命中 learn 记录
        assert mgr.find_by_hash("h" * 64, "c1", task_type="learn")["id"] == "t1"
        assert mgr.find_by_hash("h" * 64, "c1", task_type="") is None
        # 不传过滤参数时保持既有行为
        assert mgr.find_by_hash("h" * 64, "c1")["id"] == "t1"

    def test_create_without_new_params_keeps_legacy_behavior(self, tmp_path):
        from vocal_subtitle.utils.task_history import TaskHistoryManager

        mgr = TaskHistoryManager(db_path=tmp_path / "history.db")
        config = _build_run_config(ConfigLoader(), "default", {})
        mgr.create("t2", "b.wav", "h2" * 32, 10, "default", config)
        record = mgr.get("t2")
        # 旧调用形态：config_hash 仍按配置计算，标记列为空串
        assert record["config_hash"] == compute_config_hash(config)
        assert record["task_type"] == ""
        assert record["scenario"] == ""
