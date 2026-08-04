import json

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.webui import api
from vocal_subtitle.webui.app import create_app


def _task():
    return {
        "task_id": "batch-task",
        "status": "completed",
        "result": {
            "subtitle_path": None,
            "llm_subtitle_path": None,
            "subtitle_count": 3,
            "stats": {"subtitle_count": 3},
            "events": [
                {"index": 1, "start": 0.0, "end": 1.0, "text": "用户修改后的最终文本", "speaker_id": 0, "speaker_label": "A"},
                {"index": 2, "start": 1.1, "end": 2.0, "text": "第二句", "speaker_id": None, "speaker_label": None},
                {"index": 3, "start": 2.1, "end": 3.0, "text": "第三句", "speaker_id": 1, "speaker_label": "B"},
            ],
        },
    }


@pytest.fixture
def client(monkeypatch):
    tasks = {"batch-task": _task()}
    monkeypatch.setattr(api, "_task_store", tasks)
    return TestClient(create_app())


def test_batch_speaker_updates_final_events_without_changing_text(client):
    response = client.put(
        "/api/subtitle/batch-task/batch",
        json={
            "action": "speaker",
            "indexes": [1, 3],
            "speaker_id": 1,
            "speaker_label": "B",
        },
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["speaker_id"] for event in events] == [1, None, 1]
    assert events[0]["text"] == "用户修改后的最终文本"
    assert api._task_store["batch-task"]["result"]["events"][0]["text"] == "用户修改后的最终文本"


def test_rewrite_uses_user_edited_text_for_final_and_llm_files(tmp_path):
    result = _task()["result"]
    result["subtitle_path"] = str(tmp_path / "final.srt")
    result["llm_subtitle_path"] = str(tmp_path / "llm.srt")
    result["events"][0]["text"] = "用户手动修改后的最终版"

    assert api._rewrite_subtitle_files(result) == []
    assert "用户手动修改后的最终版" in (tmp_path / "final.srt").read_text(encoding="utf-8")
    assert "用户手动修改后的最终版" in (tmp_path / "llm.srt").read_text(encoding="utf-8")


def test_batch_merge_returns_reindexed_final_events(client):
    response = client.put(
        "/api/subtitle/batch-task/batch",
        json={"action": "merge", "indexes": [2, 3], "separator": "space"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["subtitle_count"] == 2
    assert [event["index"] for event in payload["events"]] == [1, 2]
    assert payload["events"][1]["text"] == "第二句 第三句"
    assert api._task_store["batch-task"]["result"]["stats"]["subtitle_count"] == 2


def test_batch_merge_rejects_non_contiguous_selection(client):
    response = client.put(
        "/api/subtitle/batch-task/batch",
        json={"action": "merge", "indexes": [1, 3], "separator": "newline"},
    )

    assert response.status_code == 400
    assert "连续" in response.json()["detail"]


def test_batch_edit_loads_and_persists_history_only_task(monkeypatch):
    source = _task()
    updates = {}

    class HistoryStub:
        def get(self, task_id):
            return {"id": task_id, "result_json": json.dumps(source["result"])}

        def update(self, task_id, **fields):
            updates.update(fields)

    monkeypatch.setattr(api, "_task_store", {})
    monkeypatch.setattr(api, "_task_history", HistoryStub())
    client = TestClient(create_app())

    response = client.put(
        "/api/subtitle/batch-task/batch",
        json={"action": "speaker", "indexes": [2], "speaker_label": "新说话人"},
    )

    assert response.status_code == 200
    assert updates["status"] == "completed"
    persisted = json.loads(updates["result_json"])
    assert persisted["events"][1]["speaker_label"] == "新说话人"
