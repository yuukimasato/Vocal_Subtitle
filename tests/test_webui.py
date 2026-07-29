"""WebUI API 端点测试"""

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.webui.app import create_app


@pytest.fixture
def client():
    """FastAPI TestClient"""
    app = create_app()
    return TestClient(app)


class TestProfilesAPI:
    """配置相关 API 测试"""

    def test_list_profiles(self, client):
        """GET /api/profiles 返回所有场景模板"""
        resp = client.get("/api/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 5

        names = [p["name"] for p in data]
        assert "default" in names
        assert "podcast" in names

        for p in data:
            assert "name" in p
            assert "description" in p
            assert "config_summary" in p
            assert "separation_engine" in p["config_summary"]

    def test_get_profile_config(self, client):
        """GET /api/profiles/{name} 返回完整配置"""
        resp = client.get("/api/profiles/default")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "default"
        assert "config" in data
        assert "separator" in data["config"]

    def test_get_nonexistent_profile(self, client):
        """GET /api/profiles/{name} 不存在的配置返回 404"""
        resp = client.get("/api/profiles/nonexistent")
        assert resp.status_code == 404

    def test_speaker_models_catalog(self, client):
        resp = client.get("/api/speaker-models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert {item["model_id"] for item in models} == {
            "speechbrain-ecapa",
            "pyannote-embedding",
            "community-1",
            "diarization-3.1",
        }


class TestDeviceAPI:
    """设备信息 API 测试"""

    def test_get_device_info(self, client):
        """GET /api/device 返回设备信息"""
        resp = client.get("/api/device")
        assert resp.status_code == 200
        data = resp.json()
        assert "device_type" in data
        assert "device_count" in data
        assert "recommended_model" in data


class TestSpeakerModelDownloadAPI:
    def test_download_passes_and_persists_token(self, client, monkeypatch):
        captured = {}

        def fake_store(token):
            captured["stored"] = token

        def fake_download(model_id, *, token=None):
            captured["download"] = (model_id, token)
            return {"model_id": model_id, "status": "ready", "cache_dir": "hidden"}

        monkeypatch.setattr(
            "vocal_subtitle.utils.hf_token_store.store_hf_token", fake_store,
        )
        monkeypatch.setattr(
            "vocal_subtitle.diarization.model_registry.download_model", fake_download,
        )

        response = client.post(
            "/api/speaker-models/community-1/download",
            data={"token": "hf_test_secret"},
        )

        assert response.status_code == 200
        assert captured == {
            "stored": "hf_test_secret",
            "download": ("community-1", "hf_test_secret"),
        }
        assert "hf_test_secret" not in response.text

    def test_model_status_is_local_and_reports_integrity(self, client, monkeypatch):
        monkeypatch.setattr(
            "vocal_subtitle.diarization.model_registry.model_status",
            lambda model_id: {
                "model_id": model_id,
                "cached": True,
                "status": "ready",
                "cache_integrity": "complete",
            },
        )

        response = client.get("/api/speaker-models/community-1/status")

        assert response.status_code == 200
        assert response.json()["cache_integrity"] == "complete"

    def test_download_classifies_auth_failure_without_leaking_token(
        self, client, monkeypatch
    ):
        class FakeGatedError(Exception):
            response = type("Response", (), {"status_code": 403})()

        def fake_download(model_id, *, token=None):
            raise FakeGatedError("token material must not be returned")

        monkeypatch.setattr(
            "vocal_subtitle.diarization.model_registry.download_model", fake_download
        )

        response = client.post(
            "/api/speaker-models/community-1/download",
            data={"token": "hf_test_secret"},
        )

        assert response.status_code == 502
        assert "Token 无效" in response.json()["detail"]
        assert "hf_test_secret" not in response.text


class TestTasksAPI:
    """任务管理 API 测试"""

    def test_get_nonexistent_task(self, client):
        """GET /api/tasks/{id} 不存在的任务返回 404"""
        resp = client.get("/api/tasks/nonexistent-id")
        assert resp.status_code == 404

    def test_list_tasks_empty(self, client):
        """GET /api/tasks 返回任务列表"""
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestSubtitleAPI:
    """字幕操作 API 测试"""

    def test_get_subtitles_task_not_found(self, client):
        """GET /api/subtitle/{id} 不存在的任务返回 404"""
        resp = client.get("/api/subtitle/nonexistent-id")
        assert resp.status_code == 404

    def test_export_task_not_found(self, client):
        """导出不存在的任务返回 404"""
        resp = client.get("/api/subtitle/nonexistent-id/export?format=srt")
        assert resp.status_code == 404


class TestStaticFiles:
    """静态文件服务测试"""

    def test_index_html_served(self, client):
        """GET / 返回 index.html"""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Vocal Subtitle" in resp.text
        assert 'data-action="check"' in resp.text
        assert "'speaker_embedding_hf_token'," not in resp.text
        assert "'speaker_embedding_hf_token': 'speaker_embedding_token'" not in resp.text

    def test_credential_fields_do_not_use_password_manager_semantics(self, client):
        html = client.get("/").text

        assert 'id="llm-api-key" name="llm-api-key" type="text" class="credential-mask"' in html
        assert 'data-key="speaker_embedding_hf_token" name="speaker-embedding-token" type="text" class="credential-mask"' in html
        assert 'autocomplete="off"' in html
        assert 'data-form-type="other"' in html
        assert 'type="password"' not in html

    def test_funasr_ui_exposes_local_first_prepare_flow(self, client):
        html = client.get("/").text

        assert "/api/asr/funasr/status" in html
        assert "/api/asr/funasr/prepare" in html
        assert "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" in html
        assert "syncASREngineOptions" in html
        assert "funasrPreparing" in html

    def test_funasr_prepare_endpoints_use_local_first_manager(self, client, monkeypatch):
        monkeypatch.setattr(
            "vocal_subtitle.webui.api.funasr_status",
            lambda model: {
                "engine": "funasr",
                "model": "normalized-model",
                "package_installed": True,
                "model_cached": True,
                "ready": True,
            },
        )
        status = client.get("/api/asr/funasr/status?model=large-v3")
        assert status.status_code == 200
        assert status.json()["ready"] is True

        monkeypatch.setattr(
            "vocal_subtitle.webui.api.ensure_funasr_ready",
            lambda model: {"engine": "funasr", "model": model, "ready": True},
        )
        prepared = client.post("/api/asr/funasr/prepare", json={"model": "large-v3"})
        assert prepared.status_code == 200
        assert prepared.json()["ready"] is True

    def test_api_docs_available(self, client):
        """GET /docs 返回 Swagger UI"""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        """GET /openapi.json 返回 OpenAPI schema"""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        paths = schema.get("paths", {})
        assert "/api/profiles" in paths
        assert "/api/device" in paths
        assert "/api/run" in paths
