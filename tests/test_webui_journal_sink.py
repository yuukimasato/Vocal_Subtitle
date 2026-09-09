"""journal sink 端点测试（方案 §3.3，V1.5 汇聚通道）。"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.webui import routes_journal
from vocal_subtitle.webui.app import create_app


@pytest.fixture
def sink_dir(tmp_path, monkeypatch):
    directory = tmp_path / "journal_sink"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes_journal, "_sink_dir", lambda: directory)
    return directory


@pytest.fixture
def client(sink_dir):
    return TestClient(create_app())


def _event(session_id="s-1", seq=0, **extra):
    event = {
        "schema": "edit-journal-v1",
        "type": "event",
        "session_id": session_id,
        "seq": seq,
        "ts": "2026-09-10T00:00:00Z",
        "actor": "human",
        "command": "updateCueTimes",
        "targets": ["cue-1"],
        "diff": [{"op": "modify", "id": "cue-1", "changes": [{"field": "start", "before": 1.0, "after": 0.9}]}],
        "context": {},
        "provenance": None,
    }
    event.update(extra)
    return event


class TestSinkProbe:
    def test_probe_returns_202(self, client):
        resp = client.get("/api/journal/sink")
        assert resp.status_code == 202
        data = resp.json()
        assert data["schema"] == "edit-journal-v1"
        assert data["accepted"] is True

    def test_probe_allows_null_origin(self, client):
        resp = client.get("/api/journal/sink", headers={"Origin": "null"})
        assert resp.headers.get("access-control-allow-origin") == "null"


class TestSinkPost:
    def test_post_ndjson_writes_file(self, client, sink_dir):
        lines = [
            json.dumps({"schema": "edit-journal-v1", "type": "header", "session_id": "s-1",
                        "file": {"subtitle": "x.srt"}, "run_id": "run-1"}),
            json.dumps(_event(seq=0)),
            json.dumps(_event(seq=1)),
        ]
        resp = client.post(
            "/api/journal/sink",
            content=("\n".join(lines) + "\n").encode(),
            headers={"Content-Type": "application/x-ndjson", "Origin": "null"},
        )
        assert resp.status_code == 202
        assert resp.json() == {"accepted": 3, "duplicates": 0, "rejected": 0}
        stored = (sink_dir / "s-1.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(stored) == 3
        assert json.loads(stored[1])["command"] == "updateCueTimes"
        # CORS：file:// 的 null origin 被回显（standalone 编辑器可用）
        assert resp.headers.get("access-control-allow-origin") == "null"

    def test_post_dedupes_by_session_seq(self, client, sink_dir):
        payload = (json.dumps(_event(seq=7)) + "\n").encode()
        first = client.post("/api/journal/sink", content=payload,
                            headers={"Content-Type": "application/x-ndjson"})
        second = client.post("/api/journal/sink", content=payload,
                             headers={"Content-Type": "application/x-ndjson"})
        assert first.json()["accepted"] == 1
        assert second.json()["duplicates"] == 1
        stored = (sink_dir / "s-1.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(stored) == 1

    def test_post_json_object_array(self, client, sink_dir):
        payload = json.dumps({"events": [_event(seq=0), _event(seq=1)]}).encode()
        resp = client.post("/api/journal/sink", content=payload,
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 202
        assert resp.json()["accepted"] == 2

    def test_invalid_payload_400(self, client):
        resp = client.post("/api/journal/sink", content=b"not json",
                           headers={"Content-Type": "application/x-ndjson"})
        assert resp.status_code == 400

    def test_wrong_schema_422(self, client):
        payload = (json.dumps({"schema": "other-v1", "type": "event"}) + "\n").encode()
        resp = client.post("/api/journal/sink", content=payload,
                           headers={"Content-Type": "application/x-ndjson"})
        assert resp.status_code == 422

    def test_event_missing_required_fields_rejected(self, client, sink_dir):
        payload = (json.dumps({"schema": "edit-journal-v1", "type": "event", "session_id": "s-2"}) + "\n").encode()
        resp = client.post("/api/journal/sink", content=payload,
                           headers={"Content-Type": "application/x-ndjson"})
        assert resp.status_code == 422
        assert resp.json()["rejected"] == 1
