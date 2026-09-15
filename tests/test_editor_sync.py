"""editor-sync 端点测试（editor-sync-v1：编辑台字幕同步通道）。"""

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.webui import routes_editor_sync
from vocal_subtitle.webui.app import create_app


@pytest.fixture
def sync_dir(tmp_path, monkeypatch):
    directory = tmp_path / "editor_sync"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes_editor_sync, "_sync_dir", lambda: directory)
    return directory


@pytest.fixture
def client(sync_dir):
    return TestClient(create_app())


def _payload(**extra):
    payload = {
        "name": "sample.srt",
        "media_name": "sample.wav",
        "format": "srt",
        "include_speakers": True,
        "cue_count": 2,
        "content": "1\n00:00:00,200 --> 00:00:02,536\n[说话人A]我的天\n",
        "updated_at": "2026-09-16T00:00:00Z",
    }
    payload.update(extra)
    return payload


class TestCapability:
    def test_capability(self, client):
        resp = client.get("/api/editor-sync/capability")
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["capability"] == "editor-sync-v1"

    def test_allows_null_origin(self, client):
        resp = client.get("/api/editor-sync/capability", headers={"Origin": "null"})
        assert resp.headers.get("access-control-allow-origin") == "null"


class TestSave:
    def test_save_and_list(self, client, sync_dir):
        resp = client.post("/api/editor-sync/subtitle", json=_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["id"] == "sample.srt"

        listing = client.get("/api/editor-sync/subtitles").json()
        assert listing["count"] == 1
        meta = listing["items"][0]
        assert meta["name"] == "sample.srt"
        assert meta["format"] == "srt"
        assert meta["cue_count"] == 2
        assert meta["include_speakers"] is True
        assert (sync_dir / "sample.srt").read_text(encoding="utf-8").startswith("1\n")

    def test_latest_wins_overwrite(self, client, sync_dir):
        client.post("/api/editor-sync/subtitle", json=_payload(cue_count=2))
        client.post(
            "/api/editor-sync/subtitle",
            json=_payload(cue_count=9, content="2\n最新稿\n"),
        )
        listing = client.get("/api/editor-sync/subtitles").json()
        assert listing["count"] == 1
        assert listing["items"][0]["cue_count"] == 9
        assert (sync_dir / "sample.srt").read_text(encoding="utf-8") == "2\n最新稿\n"

    def test_download_roundtrip_original_filename(self, client):
        client.post("/api/editor-sync/subtitle", json=_payload())
        resp = client.get("/api/editor-sync/subtitles/sample.srt/download")
        assert resp.status_code == 200
        assert "[说话人A]我的天" in resp.text
        assert "sample.srt" in resp.headers.get("content-disposition", "")

    def test_name_with_path_parts_sanitized(self, client, sync_dir):
        resp = client.post(
            "/api/editor-sync/subtitle", json=_payload(name="../../etc/passwd.srt")
        )
        assert resp.status_code == 200
        # 路径分隔符替换为下划线，前导 ../ 的点/下划线整体剥落 → 不含任何目录成分
        assert resp.json()["id"] == "etc_passwd.srt"
        assert (sync_dir / "etc_passwd.srt").is_file()
        # 下载路径同样净化，不落目录外
        assert (
            client.get(
                "/api/editor-sync/subtitles/..%2F..%2Fetc%2Fpasswd.srt/download"
            ).status_code
            == 404
        )

    def test_chinese_name_kept(self, client, sync_dir):
        resp = client.post(
            "/api/editor-sync/subtitle", json=_payload(name="第一集_说话人.srt")
        )
        assert resp.status_code == 200
        assert (sync_dir / "第一集_说话人.srt").is_file()

    def test_missing_name_422(self, client):
        resp = client.post("/api/editor-sync/subtitle", json=_payload(name="  "))
        assert resp.status_code == 422

    def test_bad_format_422(self, client):
        resp = client.post("/api/editor-sync/subtitle", json=_payload(format="docx"))
        assert resp.status_code == 422

    def test_missing_content_422(self, client):
        resp = client.post("/api/editor-sync/subtitle", json=_payload(content="   "))
        assert resp.status_code == 422

    def test_invalid_body_400(self, client):
        resp = client.post(
            "/api/editor-sync/subtitle",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_download_missing_404(self, client):
        assert (
            client.get("/api/editor-sync/subtitles/nope.srt/download").status_code
            == 404
        )
