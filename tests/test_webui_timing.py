"""Waveform timing edits through the single-subtitle update API."""

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.webui import api
from vocal_subtitle.webui.app import create_app


def _task():
    return {
        "task_id": "timing-task",
        "status": "completed",
        "result": {
            "subtitle_path": None,
            "llm_subtitle_path": None,
            "subtitle_count": 2,
            "stats": {"subtitle_count": 2},
            "events": [
                {"index": 1, "start": 0.0, "end": 1.0, "text": "第一句", "speaker_id": None, "speaker_label": None},
                {"index": 2, "start": 1.2, "end": 2.0, "text": "第二句", "speaker_id": None, "speaker_label": None},
            ],
        },
    }


@pytest.fixture
def client(monkeypatch):
    tasks = {"timing-task": _task()}
    monkeypatch.setattr(api, "_task_store", tasks)
    return TestClient(create_app())


def test_timing_only_update_persists_and_rewrites_srt(client, tmp_path, monkeypatch):
    result = api._task_store["timing-task"]["result"]
    result["subtitle_path"] = str(tmp_path / "final.srt")

    response = client.put(
        "/api/subtitle/timing-task/2",
        json={"index": 2, "start": 1.35, "end": 2.25},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["start"] == 1.35
    assert payload["end"] == 2.25
    assert payload["text"] == "第二句"

    stored = api._task_store["timing-task"]["result"]["events"][1]
    assert stored["start"] == 1.35
    assert stored["end"] == 2.25
    assert stored["time_source"] == "manual_edit"

    srt_text = (tmp_path / "final.srt").read_text(encoding="utf-8")
    assert "00:00:01,350 --> 00:00:02,250" in srt_text


def test_partial_timing_update_keeps_other_edge(client):
    response = client.put(
        "/api/subtitle/timing-task/1",
        json={"index": 1, "end": 1.5},
    )

    assert response.status_code == 200
    stored = api._task_store["timing-task"]["result"]["events"][0]
    assert stored["start"] == 0.0
    assert stored["end"] == 1.5


def test_text_and_timing_update_in_one_request(client):
    response = client.put(
        "/api/subtitle/timing-task/1",
        json={"index": 1, "text": "改写文本", "start": 0.1, "end": 0.8},
    )

    assert response.status_code == 200
    stored = api._task_store["timing-task"]["result"]["events"][0]
    assert stored["text"] == "改写文本"
    assert stored["original_text"] is None
    assert (stored["start"], stored["end"]) == (0.1, 0.8)


def test_invalid_timing_returns_400(client):
    response = client.put(
        "/api/subtitle/timing-task/1",
        json={"index": 1, "start": 2.0, "end": 1.0},
    )

    assert response.status_code == 400


def test_empty_body_returns_400(client):
    response = client.put("/api/subtitle/timing-task/1", json={"index": 1})

    assert response.status_code == 400


def test_text_only_request_still_works(client):
    """旧客户端只发 text 的请求保持兼容。"""
    response = client.put(
        "/api/subtitle/timing-task/1",
        json={"index": 1, "text": "纯文本更新"},
    )

    assert response.status_code == 200
    stored = api._task_store["timing-task"]["result"]["events"][0]
    assert stored["text"] == "纯文本更新"
    assert stored["start"] == 0.0
    assert stored["end"] == 1.0
