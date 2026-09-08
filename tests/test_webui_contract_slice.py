"""Acceptance tests for the first WebUI product-slice contract."""

import json

from fastapi.testclient import TestClient

from vocal_subtitle.contracts.common import CONTRACT_VERSION
from vocal_subtitle.webui import api
from vocal_subtitle.webui.api_serializers import history_detail, history_item
from vocal_subtitle.webui.app import create_app


def test_history_serializers_backfill_run_id_from_result():
    task = {
        "id": "task-1",
        "input_file_name": "episode.wav",
        "status": "completed",
        "result_json": json.dumps({
            "contract_version": CONTRACT_VERSION,
            "task_id": "task-1",
            "run_id": "run-1",
            "stats": {"run_id": "run-1", "quality_status": "pass"},
            "events": [],
        }),
    }

    assert history_item(task)["run_id"] == "run-1"
    assert history_detail(task)["result_summary"]["run_id"] == "run-1"


def test_task_status_exposes_standard_contract_fields(monkeypatch):
    monkeypatch.setattr(api, "_task_store", {
        "task-1": {
            "task_id": "task-1",
            "status": "degraded_completed",
            "result": {
                "contract_version": CONTRACT_VERSION,
                "task_id": "task-1",
                "run_id": "run-1",
                "artifacts": {"subtitle": "/tmp/subtitle.srt"},
                "diagnostics": {"fallback": "review_unavailable"},
                "stats": {"run_id": "run-1"},
            },
        }
    })

    response = TestClient(create_app()).get("/api/tasks/task-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-1"
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["artifacts"]["subtitle"] == "/tmp/subtitle.srt"
    assert payload["diagnostics"]["fallback"] == "review_unavailable"


def test_input_audio_download_uses_session_input_file(tmp_path, monkeypatch):
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"RIFF-test-audio")
    monkeypatch.setattr(api, "_task_store", {
        "task-1": {
            "task_id": "task-1",
            "status": "completed",
            "session_dir": str(tmp_path),
            "input_file_name": "episode.wav",
            "result": {"run_id": "run-1"},
        }
    })

    response = TestClient(create_app()).get("/api/tasks/task-1/audio?type=input")

    assert response.status_code == 200
    assert response.content == b"RIFF-test-audio"
    assert "filename" in response.headers.get("content-disposition", "")


def test_waveform_workspace_assets_are_served():
    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="waveform-canvas"' in response.text
    assert "/js/waveform.js" in response.text
    waveform = client.get("/js/waveform.js")
    assert waveform.status_code == 200
    assert "/audio/stream?type=" in waveform.text
