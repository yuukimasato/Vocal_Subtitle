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
