"""reporting 模块单元测试

覆盖:
  - RunReportBuilder / _sanitize_for_yaml (run_report.py)
  - run_report_schema 数据类 (InputInfo, RunReport, etc.)
  - EngineAvailabilityChecker / EngineAvailabilitySnapshot (engine_availability.py)
  - DegradationLogger (degradation_log.py)
"""

import hashlib
import json
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from vocal_subtitle.reporting import (
    DegradationLogger,
    EngineAvailabilityChecker,
    EngineAvailabilitySnapshot,
    EngineStatusEntry,
    PipelinePathInfo,
    QualityInfo,
    RunReport,
    RunReportBuilder,
)
from vocal_subtitle.reporting.run_report import _sanitize_for_yaml
from vocal_subtitle.reporting.run_report_schema import (
    DegradationInfo,
    InputInfo,
    OutputInfo,
)


@dataclass
class _FakeConfig:
    model_size: str = "large-v3"
    device: str = "cpu"
    labels: tuple = ("asr", "vad")


ALL_ENGINE_KEYS = {
    "separation_uvr",
    "separation_spleeter",
    "separation_open_unmix",
    "vad_silero",
    "vad_webrtc",
    "asr_faster_whisper",
    "asr_funasr",
    "asr_qwen",
    "asr_whisper_cpp",
    "review_global_evidence",
    "review_context_reasr",
    "review_qwen",
    "review_forced_aligner",
    "review_sed",
    "review_semantic",
    "diarization_speechbrain",
    "diarization_pyannote",
}


def _write_audio_file(
    directory: Path, name: str = "input.wav", size: int = 1024
) -> Path:
    path = directory / name
    path.write_bytes(b"\x00" * size)
    return path


def _make_snapshot() -> EngineAvailabilitySnapshot:
    return EngineAvailabilitySnapshot(
        entries={
            "vad_silero": EngineStatusEntry(
                engine="silero",
                model="silero_vad",
                status="ready_default",
                device="cpu",
            ),
            "asr_faster_whisper": EngineStatusEntry(
                engine="faster-whisper",
                model="large-v3",
                status="ready_default",
                device="cpu",
            ),
        },
        host_info={"platform": "Linux", "python": "3.12"},
    )


# ---------------------------------------------------------------------------
# _sanitize_for_yaml
# ---------------------------------------------------------------------------


class TestSanitizeForYaml:
    def test_flat_tuple_becomes_list(self):
        result = _sanitize_for_yaml(("a", "b", "c"))
        assert result == ["a", "b", "c"]
        assert isinstance(result, list)

    def test_nested_tuple_inside_tuple(self):
        result = _sanitize_for_yaml((1, (2, (3, 4))))
        assert result == [1, [2, [3, 4]]]

    def test_tuple_inside_dict(self):
        result = _sanitize_for_yaml(
            {"route": ("quality_first", "cost_first"), "mode": "offline"}
        )
        assert result == {"route": ["quality_first", "cost_first"], "mode": "offline"}
        assert isinstance(result["route"], list)

    def test_tuple_inside_list(self):
        result = _sanitize_for_yaml([("a", 1), ("b", 2)])
        assert result == [["a", 1], ["b", 2]]

    def test_non_tuple_scalars_pass_through(self):
        for value in (1, 1.5, "text", True, None, b"bytes"):
            assert _sanitize_for_yaml(value) is value

    def test_empty_containers(self):
        assert _sanitize_for_yaml(()) == []
        assert _sanitize_for_yaml({}) == {}
        assert _sanitize_for_yaml([]) == []

    def test_no_python_tuple_tag_in_yaml_output(self):
        import yaml

        text = yaml.dump(_sanitize_for_yaml({"labels": ("a", "b")}), allow_unicode=True)
        assert "!!python/tuple" not in text


# ---------------------------------------------------------------------------
# RunReportBuilder
# ---------------------------------------------------------------------------


class TestRunReportBuilder:
    def test_full_build_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = _write_audio_file(root)
            builder = RunReportBuilder(
                "run-offline-001", "offline-20260803-001", reports_root=root
            )
            builder.set_input(audio, duration=12.34, sample_rate=16000, channels=1)
            builder.set_engine_snapshot(_make_snapshot())
            builder.set_engine_status(
                {
                    "faster-whisper": {
                        "selected": True,
                        "enabled": True,
                        "available": True,
                        "status": "completed",
                        "windows_processed": 3,
                        "windows_failed": 0,
                    },
                }
            )
            builder.set_pipeline_path(
                mode="offline", production_path="quality_first", route_version="v3"
            )
            builder.set_stage(
                "asr",
                status="completed",
                duration=10.0,
                engine="faster-whisper",
                model="large-v3",
                quality_score=0.95,
            )
            builder.set_stage("vad", status="completed", duration=5.0)
            builder.add_warning("asr", "low confidence segment")
            builder.add_error("asr", "engine retried", category="retry")
            builder.record_degradation(
                "asr",
                "faster-whisper",
                "whisper_cpp",
                reason="CUDA OOM",
                category="resource_exhausted",
            )
            builder.stage_timings["asr"] = 10.0
            builder.stage_timings["vad"] = 5.0

            report = builder.build(config_snapshot={"profile": "default"})
            data = report.to_dict()

            assert data["$schema"] == "run-report-v1"
            assert data["run_id"] == "run-offline-001"
            assert data["task_id"] == "offline-20260803-001"
            assert data["input"]["path"] == str(audio)
            assert data["input"]["format"] == "wav"
            assert data["input"]["duration_seconds"] == 12.34
            assert (
                data["engine_availability"]["vad_silero"]["status"] == "ready_default"
            )
            assert data["engine_status"]["asr_faster_whisper"]["windows_processed"] == 3
            assert data["pipeline_path"]["mode"] == "offline"
            assert data["pipeline_path"]["route_version"] == "v3"
            assert "asr" in data["stages"]
            assert "vad" in data["stages"]
            assert data["stages"]["asr"]["quality_score"] == 0.95
            assert data["warnings"] == [
                {"stage": "asr", "message": "low confidence segment"}
            ]
            assert data["errors"] == [
                {"stage": "asr", "message": "engine retried", "category": "retry"}
            ]
            assert data["timing"]["total_wall_seconds"] == 15.0

            report_path = builder.persist(report)
            assert report_path.exists()
            assert (builder.report_dir / "engine_availability.json").exists()
            assert (builder.report_dir / "stage_timings.json").exists()

            log_path = builder.report_dir / "degradation_log.jsonl"
            assert log_path.exists()
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            event = json.loads(lines[0])
            assert event["stage"] == "asr"

    def test_set_input_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = RunReportBuilder("run-1", "task-1", reports_root=Path(tmp))
            with pytest.raises(FileNotFoundError):
                builder.set_input(Path(tmp) / "missing.wav", duration=1.0)

    def test_report_dir_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = RunReportBuilder("run-1", "task-1", reports_root=Path(tmp))
            assert builder.report_dir == Path(tmp) / "run-1"

    def test_build_without_config_snapshot_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = RunReportBuilder("run-1", "task-1", reports_root=Path(tmp))
            report = builder.build()
            assert report.to_dict()["config_snapshot"] == {}

    def test_set_input_populates_all_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = _write_audio_file(root, size=2048)
            expected_hash = hashlib.sha256(b"\x00" * 2048).hexdigest()
            builder = RunReportBuilder("run-1", "task-1", reports_root=root)
            builder.set_input(audio, duration=3.5, sample_rate=44100, channels=2)
            info = builder.input_info
            assert info.file_hash == expected_hash
            assert info.file_size_bytes == 2048
            assert info.duration_seconds == 3.5
            assert info.sample_rate == 44100
            assert info.channels == 2

    def test_persist_with_config_writes_yaml_no_tuples(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builder = RunReportBuilder("run-1", "task-1", reports_root=root)
            builder.persist(builder.build(), config=_FakeConfig())
            snapshot_yaml = root / "run-1" / "config_snapshot.yaml"
            assert snapshot_yaml.exists()
            text = snapshot_yaml.read_text(encoding="utf-8")
            assert "!!python/tuple" not in text
            data = yaml.safe_load(text)
            assert data["device"] == "cpu"


# ---------------------------------------------------------------------------
# EngineAvailabilityChecker
# ---------------------------------------------------------------------------


class TestEngineAvailabilityChecker:
    def test_check_all_returns_all_engine_keys(self):
        snapshot = EngineAvailabilityChecker().check_all()
        assert set(snapshot.entries) == ALL_ENGINE_KEYS

    def test_check_all_entries_have_valid_status(self):
        snapshot = EngineAvailabilityChecker().check_all()
        valid = (
            "unavailable",
            "model_missing",
            "ready_shadow",
            "ready_review",
            "ready_default",
        )
        for entry in snapshot.entries.values():
            assert entry.status in valid

    def test_check_all_host_info_included(self):
        snapshot = EngineAvailabilityChecker().check_all()
        host = snapshot.host_info
        assert "platform" in host
        assert "cpu_count" in host
        assert isinstance(host["cpu_count"], int)

    def test_check_critical_returns_bool_map(self):
        checker = EngineAvailabilityChecker()
        snapshot = checker.check_all()
        result = checker.check_critical()
        assert set(result) == {"vad_available", "asr_available", "all_critical_ready"}
        expected_vad = any(
            snapshot.entries[key].status.startswith("ready")
            for key in ("vad_silero", "vad_webrtc")
        )
        expected_asr = any(
            snapshot.entries[key].status.startswith("ready")
            for key in (
                "asr_faster_whisper",
                "asr_funasr",
                "asr_qwen",
                "asr_whisper_cpp",
            )
        )
        assert result["vad_available"] is expected_vad
        assert result["asr_available"] is expected_asr
        assert result["all_critical_ready"] == (
            result["vad_available"] and result["asr_available"]
        )

    def test_deterministic_engine_statuses(self):
        entries = EngineAvailabilityChecker().check_all().entries
        assert entries["separation_spleeter"].status == "unavailable"
        assert entries["vad_silero"].status in {"unavailable", "ready_default"}
        assert entries["review_context_reasr"].status == "unavailable"
        assert entries["review_semantic"].status == "unavailable"

    def test_silero_requires_torchaudio_runtime(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setitem(sys.modules, "torchaudio", None)

        status = EngineAvailabilityChecker()._check_vad("silero")

        assert status[0] == "unavailable"
        assert "torchaudio" in status[1]

    def test_openunmix_requires_the_runtime_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openunmix", None)

        status = EngineAvailabilityChecker()._check_separation("open-unmix")

        assert status[0] == "unavailable"
        assert "openunmix" in status[1]


# ---------------------------------------------------------------------------
# DegradationLogger
# ---------------------------------------------------------------------------


class TestDegradationLogger:
    def test_record_then_flush_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            dlog = DegradationLogger(Path(tmp))
            dlog.record(
                "asr",
                "faster-whisper",
                "whisper_cpp",
                reason="CUDA OOM",
                category="resource_exhausted",
            )
            assert dlog.count == 1
            dlog.flush()
            assert dlog.path.exists()
            lines = dlog.path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            event = json.loads(lines[0])
            assert event["stage"] == "asr"
            assert event["from"] == "faster-whisper"
            assert event["to"] == "whisper_cpp"

    def test_multiple_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            dlog = DegradationLogger(Path(tmp))
            for i in range(3):
                dlog.record("asr", f"engine-{i}", "fallback")
            dlog.flush()
            lines = dlog.path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 3

    def test_empty_flush_creates_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dlog = DegradationLogger(Path(tmp))
            dlog.flush()
            assert not dlog.path.exists()

    def test_count_property_tracks_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            dlog = DegradationLogger(Path(tmp))
            dlog.record("asr", "a", "b")
            dlog.flush()
            dlog.record("vad", "c", "d")
            dlog.flush()
            assert dlog.count == 2


# ---------------------------------------------------------------------------
# RunReport Schema
# ---------------------------------------------------------------------------


class TestRunReportSchema:
    def test_input_info_to_dict(self):
        info = InputInfo(
            path="/tmp/in.wav",
            file_hash="abc",
            file_size_bytes=1024,
            duration_seconds=3.5,
            sample_rate=16000,
            channels=1,
            format="wav",
        )
        data = info.to_dict()
        assert data["path"] == "/tmp/in.wav"
        assert data["format"] == "wav"

    def test_engine_status_entry_valid_statuses(self):
        for status in (
            "unavailable",
            "model_missing",
            "ready_shadow",
            "ready_review",
            "ready_default",
        ):
            entry = EngineStatusEntry(engine="test", status=status)
            assert entry.status == status

    def test_pipeline_path_info_defaults(self):
        info = PipelinePathInfo()
        assert info.mode == "offline"
        assert info.production_path == "quality_first"

    def test_run_report_required_fields(self):
        report = RunReport()
        data = report.to_dict()
        assert data["$schema"] == "run-report-v1"
        assert data["run_id"] == ""
        assert data["errors"] == []
        assert data["warnings"] == []

    def test_quality_info_to_dict(self):
        quality = QualityInfo(status="fail", coverage_audit={"pct": 0.9})
        data = quality.to_dict()
        assert data["status"] == "fail"
        assert data["coverage_audit"] == {"pct": 0.9}

    def test_degradation_info_to_dict(self):
        info = DegradationInfo(overall_mode="fallback", fallback_reason="model missing")
        data = info.to_dict()
        assert data["overall_mode"] == "fallback"
        assert data["fallback_reason"] == "model missing"

    def test_output_info_to_dict(self):
        info = OutputInfo(subtitle_count=120, speaker_count=2, detected_language="zh")
        data = info.to_dict()
        assert data["subtitle_count"] == 120
        assert data["detected_language"] == "zh"
