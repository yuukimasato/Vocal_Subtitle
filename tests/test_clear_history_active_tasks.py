"""清空历史不得影响执行中/排队中的任务。

回归背景（2026-09-13）：用户在任务运行中点击"清空历史"，DELETE /api/history
无条件清空 task_store 并删除全部历史行；任务完成时写回 task_store 触发
KeyError，成功的运行在历史中被记成 failed。
"""

from fastapi.testclient import TestClient

from vocal_subtitle.webui import api, routes_history
from vocal_subtitle.webui.app import create_app


class FakeTaskHistory:
    def __init__(self):
        self.cleared = []

    def clear(self, older_than_days=None, exclude_ids=None):
        self.cleared.append(
            {"older_than_days": older_than_days, "exclude_ids": exclude_ids}
        )
        return 0


class FakeStorage:
    def __init__(self):
        self.upload_skip_names = None

    def clear_persistent_files(self):
        return 0

    def clear_uploads(self, skip_names=frozenset()):
        self.upload_skip_names = set(skip_names)
        return 0


def test_clear_history_keeps_active_tasks(monkeypatch):
    store = {
        "running-task": {
            "task_id": "running-task",
            "status": "running",
            "session_dir": "/cache/uploads/aaaa1111",
        },
        "pending-task": {
            "task_id": "pending-task",
            "status": "pending",
            "session_dir": "/cache/uploads/bbbb2222",
        },
        "done-task": {
            "task_id": "done-task",
            "status": "completed",
            "session_dir": "/cache/uploads/cccc3333",
        },
    }
    history = FakeTaskHistory()
    storage = FakeStorage()
    monkeypatch.setattr(api, "_task_store", store)
    monkeypatch.setattr(api, "_task_history", history)
    monkeypatch.setattr(routes_history, "_storage", storage)

    client = TestClient(create_app())
    resp = client.delete("/api/history")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # 执行中/排队中的任务全部保留，已结束的被移除
    assert "running-task" in store
    assert "pending-task" in store
    assert "done-task" not in store

    # 历史行清理排除执行中的任务
    assert history.cleared == [
        {"older_than_days": None, "exclude_ids": {"running-task", "pending-task"}}
    ]

    # 会话目录清理跳过执行中任务（含 pending），已结束的不跳过
    assert storage.upload_skip_names == {"aaaa1111", "bbbb2222"}


def test_clear_history_with_older_than_days_unchanged(monkeypatch):
    """按保留天数清理时行为不变，不传 exclude_ids。"""
    store = {
        "running-task": {"task_id": "running-task", "status": "running"},
        "old-task": {"task_id": "old-task", "status": "completed"},
    }
    history = FakeTaskHistory()
    monkeypatch.setattr(api, "_task_store", store)
    monkeypatch.setattr(api, "_task_history", history)
    monkeypatch.setattr(routes_history, "_storage", FakeStorage())

    client = TestClient(create_app())
    resp = client.delete("/api/history?older_than_days=30")

    assert resp.status_code == 200
    assert history.cleared == [{"older_than_days": 30, "exclude_ids": None}]
