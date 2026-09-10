"""Boundary tests for the explicit WebUI application services."""

from pathlib import Path

from vocal_subtitle.webui import api
from vocal_subtitle.webui import pipeline_tasks
from vocal_subtitle.webui.routes_history import _storage
from vocal_subtitle.webui.storage_services import WebUIStorageService


LEGACY_API_NAMES = (
    "get_subtitles",
    "export_subtitle",
    "run_pipeline",
    "get_task_status",
    "list_history",
    "feedback_learn",
    "feedback_preview",
    "get_funasr_status",
    "prepare_funasr",
)


def test_pipeline_task_service_honors_legacy_api_pipeline_override(monkeypatch):
    class FakePipeline:
        pass

    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    assert pipeline_tasks._pipeline_class() is FakePipeline


def test_storage_service_is_constructible_without_pipeline():
    service = WebUIStorageService(api.state)
    assert service.state is api.state
    assert isinstance(_storage, WebUIStorageService)


def test_pipeline_task_service_exports_compatibility_entrypoint():
    assert callable(pipeline_tasks.PipelineTaskService.run_in_thread)
    assert pipeline_tasks.Pipeline is api.Pipeline


def test_storage_size_handles_missing_directory(tmp_path):
    service = WebUIStorageService(api.state)
    assert service.directory_size_mb(Path(tmp_path) / "missing") == 0.0


def test_legacy_api_facade_exports_split_route_functions():
    assert all(callable(getattr(api, name)) for name in LEGACY_API_NAMES)
    assert api.SubtitleEventResponse is not None
    assert api._run_pipeline_in_thread is not None


def test_funasr_compat_adapters_do_not_recurse_through_api_exports(monkeypatch):
    monkeypatch.setattr(
        "vocal_subtitle.asr.funasr_manager.ensure_funasr_ready",
        lambda model: {"ready": True, "model": model},
    )
    monkeypatch.setattr(
        "vocal_subtitle.asr.funasr_manager.funasr_status",
        lambda model: {"ready": True, "model": model},
    )

    assert api.ensure_funasr_ready("tiny")["model"] == "tiny"
    assert api.funasr_status("tiny")["model"] == "tiny"
