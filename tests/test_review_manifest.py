"""review-manifest-v1 契约与落盘/端点测试（方案 §3.1）。"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.contracts.review_manifest import (
    REVIEW_MANIFEST_SCHEMA,
    build_review_manifest,
    manifest_filename,
    write_review_manifest,
)
from vocal_subtitle.utils.persistence_manager import PersistenceManager, PersistenceSettings
from vocal_subtitle.webui.app import create_app
from vocal_subtitle.webui.runtime_state import state


def _event(index, start, end, text, **extra):
    payload = {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "original_text": None,
        "speaker_id": None,
        "speaker_label": None,
    }
    payload.update(extra)
    return payload


class TestBuildReviewManifest:
    def test_schema_and_required_fields(self, tmp_path):
        subtitle = tmp_path / "final.srt"
        subtitle.write_text("1\n00:00:01,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
        manifest = build_review_manifest(
            task_id="t1",
            run_id="run-1",
            subtitle_path=subtitle,
            events=[_event(1, 1.0, 2.0, "你好")],
        )
        assert manifest["schema"] == REVIEW_MANIFEST_SCHEMA
        assert manifest["task_id"] == "t1"
        assert manifest["run_id"] == "run-1"
        assert manifest["subtitle"]["file"] == "final.srt"
        assert manifest["subtitle"]["format"] == "srt"
        assert manifest["subtitle"]["stage"] == "final"
        assert manifest["subtitle"]["hash"]  # 内容哈希已计算

    def test_cue_provenance_fields(self, tmp_path):
        subtitle = tmp_path / "final.srt"
        subtitle.write_text("x", encoding="utf-8")
        events = [
            _event(1, 12.1, 14.3, "大家好", speaker_id=0, speaker_label="主持人",
                   words=[{"word": "大", "start": 0.0, "end": 0.2, "confidence": 0.9},
                          {"word": "家", "start": 0.2, "end": 0.4, "confidence": 0.8}]),
            _event(2, 15.0, 16.0, "优化后", original_text="优化前"),
            _event(3, 17.0, 18.0, "无词级"),
        ]
        manifest = build_review_manifest(task_id="t", run_id="r", subtitle_path=subtitle, events=events)
        cue1, cue2, cue3 = manifest["cues"]
        # 词级相对时间 → 绝对时间；confidence 取均值
        assert cue1["words"][0] == {"w": "大", "t0": 12.1, "t1": 12.3, "confidence": 0.9}
        assert cue1["words"][1]["t0"] == 12.3
        assert cue1["confidence"] == pytest.approx(0.85)
        assert cue1["speaker_label"] == "主持人"
        # LLM 覆盖（original_text 保留）→ llm-optimized；否则 asr
        assert cue2["source_stage"] == "llm-optimized"
        assert cue1["source_stage"] == "asr"
        # 可选字段缺失则不出现
        assert "words" not in cue3
        assert "confidence" not in cue3
        assert "speaker_id" not in cue3

    def test_optional_top_level_fields(self, tmp_path):
        subtitle = tmp_path / "final.srt"
        subtitle.write_text("x", encoding="utf-8")
        manifest = build_review_manifest(
            task_id="t", run_id="r", subtitle_path=subtitle, events=[],
            input_name="lesson01.mp4", input_sha256="abc", duration=3600.0,
            profile="default", engines={"asr": "funasr", "vad": ""},
        )
        assert manifest["input"] == {"filename": "lesson01.mp4", "sha256": "abc", "duration": 3600.0}
        assert manifest["profile"] == "default"
        assert manifest["engines"] == {"asr": "funasr"}  # 空值引擎被剔除


class TestManifestWriter:
    def test_filename_and_write(self, tmp_path):
        subtitle = tmp_path / "lesson01.srt"
        subtitle.write_text("x", encoding="utf-8")
        assert manifest_filename(subtitle) == "lesson01.manifest.json"
        manifest = build_review_manifest(task_id="t", run_id="r", subtitle_path=subtitle, events=[])
        path = write_review_manifest(subtitle, manifest)
        assert path == tmp_path / "lesson01.manifest.json"
        assert json.loads(path.read_text(encoding="utf-8"))["schema"] == REVIEW_MANIFEST_SCHEMA


class TestPersistenceWiring:
    def test_persist_task_writes_manifest(self, tmp_path):
        manager = PersistenceManager(
            settings_path=tmp_path / "settings.json",
            files_dir=tmp_path / "files",
        )
        settings = PersistenceSettings(
            persist_final_srt=True, persist_final_ass=False,
            persist_asr_subtitle=False, persist_llm_subtitle=False,
            persist_vocals=False, persist_accompaniment=False,
        )
        task_result = {
            "run_id": "run-1",
            "input_path": str(tmp_path / "input" / "lesson01.mp4"),
            "subtitle_path": str(tmp_path / "input" / "output.srt"),
            "stats": {"duration_seconds": 60.0, "final_engine": "funasr"},
            "events": [_event(1, 1.0, 2.0, "你好", speaker_id=0, speaker_label="主持人")],
        }
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "output.srt").write_text("subtitle", encoding="utf-8")

        manager.persist_task("task-1", task_result, settings)

        persisted_manifest = tmp_path / "files" / "task-1" / "final.manifest.json"
        assert persisted_manifest.exists()
        data = json.loads(persisted_manifest.read_text(encoding="utf-8"))
        assert data["schema"] == REVIEW_MANIFEST_SCHEMA
        assert data["cues"][0]["speaker_label"] == "主持人"
        # 会话目录（uploads）内的主字幕旁也有一份
        session_manifest = tmp_path / "input" / "output.manifest.json"
        assert session_manifest.exists()

    def test_persist_task_without_final_subtitle_skips_manifest(self, tmp_path):
        manager = PersistenceManager(
            settings_path=tmp_path / "settings.json",
            files_dir=tmp_path / "files",
        )
        settings = PersistenceSettings(persist_final_srt=False, persist_final_ass=False)
        task_result = {"run_id": "run-1", "events": [_event(1, 1.0, 2.0, "你好")]}
        manager.persist_task("task-2", task_result, settings)
        assert not (tmp_path / "files" / "task-2" / "final.manifest.json").exists()


class TestManifestEndpoint:
    def test_get_task_manifest(self):
        task_id = "mtest-endpoint"
        state.task_store[task_id] = {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "run_id": "run-x",
                "input_path": "/tmp/lesson01.mp4",
                "subtitle_path": "/tmp/output.srt",
                "stats": {"duration_seconds": 3600.0, "final_engine": "funasr"},
                "events": [_event(1, 12.1, 14.3, "大家好", speaker_label="主持人")],
            },
        }
        try:
            client = TestClient(create_app())
            resp = client.get(f"/api/tasks/{task_id}/manifest")
            assert resp.status_code == 200
            data = resp.json()
            assert data["schema"] == REVIEW_MANIFEST_SCHEMA
            assert data["input"]["filename"] == "lesson01.mp4"
            assert data["cues"][0]["index"] == 1
        finally:
            state.task_store.pop(task_id, None)

    def test_get_manifest_unknown_task_404(self):
        client = TestClient(create_app())
        resp = client.get("/api/tasks/no-such-task/manifest")
        assert resp.status_code == 404
