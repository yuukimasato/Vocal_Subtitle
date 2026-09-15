"""Backend contract and compatibility adapter tests."""

import json
from dataclasses import dataclass
from pathlib import Path

from vocal_subtitle.application.contract_coordinator import BackendRunCoordinator
from vocal_subtitle.contracts import (
    CONTRACT_VERSION,
    ArtifactRegistryAdapter,
    ASREngineAdapter,
    EngineRegistryAdapter,
    ErrorInfo,
    PipelineRunAdapter,
    RunReport,
    RunRequest,
    RunResult,
    TaskHistoryAdapter,
    TaskRequest,
    TaskSnapshot,
    TaskState,
)
from vocal_subtitle.contracts.engine import EngineRequest
from vocal_subtitle.contracts.external import CLIAdapter, WebUIAdapter
from vocal_subtitle.governance.engine_lifecycle import EngineRegistry
from vocal_subtitle.utils.task_history import TaskHistoryManager


@dataclass
class _Config:
    mode: str = "offline"


def test_task_request_and_snapshot_round_trip():
    request = TaskRequest(
        task_id="task-1",
        input_path="input.wav",
        output_path="out.srt",
        overrides={"skip_separation": True},
    )
    assert TaskRequest.from_dict(request.to_dict()) == request

    snapshot = TaskSnapshot(
        task_id="task-1",
        state=TaskState.DEGRADED_COMPLETED,
        progress=2,
        error=ErrorInfo("resource_exhausted", message="fallback", recoverable=True),
    )
    restored = TaskSnapshot.from_dict(snapshot.to_dict())
    assert restored.state is TaskState.DEGRADED_COMPLETED
    assert restored.progress == 1.0
    assert restored.error.category == "resource_exhausted"


def test_task_history_adapter_preserves_legacy_payload(tmp_path):
    manager = TaskHistoryManager(db_path=tmp_path / "tasks.db")
    adapter = TaskHistoryAdapter(manager)
    request = TaskRequest(task_id="task-1", input_path=str(tmp_path / "audio.wav"))
    created = adapter.create(request, input_fingerprint="sha256:test", config=_Config())
    assert created.state is TaskState.PENDING

    adapter.transition(created.task_id, TaskState.PREFLIGHT)
    running = adapter.transition(created.task_id, TaskState.RUNNING, run_id="run-1")
    assert running.run_id == "run-1"
    done = adapter.transition(created.task_id, TaskState.COMPLETED)
    assert done.status == "completed"
    raw = manager.get("task-1")
    assert raw["input_file_hash"] == "sha256:test"


def test_pipeline_adapter_maps_legacy_result(tmp_path):
    class FakePipeline:
        def run(self, **kwargs):
            assert kwargs["input_path"] == tmp_path / "input.wav"
            assert kwargs["output_path"] == tmp_path / "out.srt"
            return {
                "subtitle_path": tmp_path / "out.srt",
                "events": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "stats": type(
                    "Stats",
                    (),
                    {"status": "completed", "run_id": "run-1", "task_id": "task-1"},
                )(),
                "from_cache": True,
            }

    request = RunRequest(
        task=TaskRequest(
            task_id="task-1",
            input_path=str(tmp_path / "input.wav"),
            output_path=str(tmp_path / "out.srt"),
        )
    )
    result = PipelineRunAdapter(FakePipeline()).run(request)
    assert result.task.state is TaskState.COMPLETED
    assert result.subtitle_path == str(tmp_path / "out.srt")
    assert result.from_cache is True
    assert result.to_dict()["contract_version"] == CONTRACT_VERSION


def test_engine_registry_adapter_maps_lifecycle():
    adapter = EngineRegistryAdapter(EngineRegistry())
    available = adapter.get("faster-whisper")
    missing = adapter.get("qwen-review")
    assert available.available is True
    assert available.identity.model == "large-v3"
    assert missing.available is False
    assert len(list(adapter.list_all())) > 5


def test_asr_engine_adapter_normalizes_success_and_failure():
    class FakeASR:
        name = "fake-asr"
        model_name = "fake-model"

        def load_model(self):
            return None

        def transcribe(self, audio, sample_rate=16000, **kwargs):
            assert sample_rate == 16000
            return ["segment"]

    adapter = ASREngineAdapter(FakeASR(), status="ready_default")
    assert adapter.prepare().ready is True
    result = adapter.execute(EngineRequest(payload=b"audio", language="zh"))
    assert result.success is True
    assert result.events == ("segment",)
    assert adapter.availability().available is True

    class Broken(FakeASR):
        def transcribe(self, *args, **kwargs):
            error = RuntimeError("backend down")
            error.category = "dependency_unavailable"
            raise error

    failed = ASREngineAdapter(Broken()).execute(EngineRequest(payload=b"audio"))
    assert failed.success is False
    assert failed.error.category == "dependency_missing"


def test_external_adapters_keep_frameworks_out_of_contracts():
    request = CLIAdapter.request(
        "input.wav", output_path="out.srt", skip_separation=True
    )
    assert request.overrides["skip_separation"] is True
    web_request = WebUIAdapter.request(
        {"input_path": "input.wav", "profile": "podcast"}
    )
    assert web_request.profile == "podcast"


def test_report_adapter_round_trip_and_persistence(tmp_path):
    class FakeBuilder:
        def __init__(self, run_id, task_id):
            self.run_id = run_id
            self.task_id = task_id

        def build(self, config_snapshot=None):
            return {
                "$schema": "run-report-v1",
                "run_id": self.run_id,
                "task_id": self.task_id,
                "config_snapshot": config_snapshot or {},
            }

        def persist(self, report, config=None):
            path = tmp_path / "persisted.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            return path

    from vocal_subtitle.contracts.adapters import RunReportAdapter

    adapter = RunReportAdapter(FakeBuilder)
    result = RunResult(
        task=TaskSnapshot("task-1", run_id="run-1", state=TaskState.COMPLETED)
    )
    report = adapter.build(result, config_snapshot={"profile": "default"})
    assert isinstance(report, RunReport)
    assert report.to_dict()["$schema"] == "run-report-v1"
    assert adapter.persist(report).exists()


def test_coordinator_preserves_result_when_reporting_fails():
    class Tasks:
        def __init__(self):
            self.snapshot = None

        def get(self, task_id):
            return self.snapshot

        def create(self, request, **kwargs):
            self.snapshot = TaskSnapshot(request.task_id or "task-1")
            return self.snapshot

        def transition(self, task_id, state, **kwargs):
            self.snapshot.state = state
            self.snapshot.status = state.value
            return self.snapshot

    class Runner:
        def run(self, request):
            return RunResult(
                task=TaskSnapshot(request.task.task_id, state=TaskState.COMPLETED),
                subtitle_path="out.srt",
                artifacts={"subtitle": "out.srt"},
            )

    class FailingReports:
        def build(self, result, **kwargs):
            raise OSError("report disk full")

        def persist(self, report, **kwargs):
            raise AssertionError("persist should not be called")

    tasks = Tasks()
    artifacts = ArtifactRegistryAdapter()
    coordinator = BackendRunCoordinator(
        tasks=tasks,
        runner=Runner(),
        reports=FailingReports(),
        artifacts=artifacts,
    )
    result = coordinator.run(
        RunRequest(task=TaskRequest(input_path="input.wav", task_id="task-1"))
    )
    assert result.status == "completed"
    assert result.diagnostics["report_error"]["category"] == "output_failed"
    assert artifacts.get("subtitle") == Path("out.srt")


def test_error_info_does_not_serialize_traceback():
    try:
        raise ValueError("bad input")
    except ValueError as exc:
        error = ErrorInfo.from_exception(exc, category="input_invalid")
    payload = error.to_dict()
    assert payload["category"] == "input_invalid"
    assert "traceback" not in payload
