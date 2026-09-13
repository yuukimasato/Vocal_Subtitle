"""Regression tests for the WebUI background task state machine."""

from pathlib import Path

from vocal_subtitle.application.pipeline_result import PipelineStats
from vocal_subtitle.webui import api
from vocal_subtitle.webui import pipeline_tasks


class _HistorySpy:
    def __init__(self):
        self.status = "pending"
        self.transitions = []

    def set_preflight(self, task_id):
        self.update(task_id, status="preflight")

    def set_running(self, task_id, run_id=""):
        self.update(task_id, status="running", run_id=run_id)

    def update(self, task_id, **fields):
        if "status" in fields and fields["status"] != self.status:
            self.transitions.append(fields["status"])
            self.status = fields["status"]


class _WebSocketSpy:
    def create_progress_callback(self, task_id):
        return lambda event: None

    def broadcast_from_thread(self, task_id, message):
        return None

    def store_task_result(self, task_id, result):
        return None


def test_background_task_preserves_preflight_state_order(monkeypatch, tmp_path):
    task_id = "task-1"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.srt"
    input_path.write_bytes(b"RIFF")
    history = _HistorySpy()

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            history.set_preflight(task_id)
            history.set_running(task_id, run_id="run-1")
            stats = PipelineStats(input_path=input_path, duration_seconds=1.0)
            stats.status = "completed"
            stats.run_id = "run-1"
            return {"subtitle_path": output_path, "stats": stats, "events": []}

    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    monkeypatch.setattr(api, "_task_history", history)
    monkeypatch.setattr(pipeline_tasks, "ws_manager", _WebSocketSpy())
    monkeypatch.setattr(
        pipeline_tasks,
        "_persistence_manager",
        lambda: type("Persistence", (), {"persist_task": lambda *args: None})(),
    )
    monkeypatch.setattr(
        api,
        "_task_store",
        {task_id: {"task_id": task_id, "status": "pending"}},
    )

    pipeline_tasks.run_pipeline_in_thread(
        task_id,
        input_path,
        output_path,
        "default",
        "srt",
        True,
        {},
        tmp_path,
    )

    assert history.transitions == ["preflight", "running", "completed"]


def test_completed_run_survives_cleared_task_store(monkeypatch, tmp_path):
    """运行中条目被清空（如用户点了清空历史）时，成功结果不得被记为 failed。"""
    task_id = "cleared-task"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.srt"
    input_path.write_bytes(b"RIFF")
    history = _HistorySpy()
    store = {task_id: {"task_id": task_id, "status": "pending"}}

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self, **kwargs):
            store.pop(task_id)  # 模拟运行中途 task_store 条目被清除
            stats = PipelineStats(input_path=input_path, duration_seconds=1.0)
            stats.status = "completed"
            stats.run_id = "run-1"
            return {"subtitle_path": output_path, "stats": stats, "events": []}

    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    monkeypatch.setattr(api, "_task_history", history)
    monkeypatch.setattr(pipeline_tasks, "ws_manager", _WebSocketSpy())
    monkeypatch.setattr(
        pipeline_tasks,
        "_persistence_manager",
        lambda: type("Persistence", (), {"persist_task": lambda *args: None})(),
    )
    monkeypatch.setattr(api, "_task_store", store)

    pipeline_tasks.run_pipeline_in_thread(
        task_id,
        input_path,
        output_path,
        "default",
        "srt",
        True,
        {},
        tmp_path,
    )

    assert history.status == "completed"
    assert store[task_id]["status"] == "completed"
    assert store[task_id]["result"]["task_id"] == task_id
