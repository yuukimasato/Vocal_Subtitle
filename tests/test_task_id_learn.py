"""工单 01：task_id 引用学习 + 场景标签（定案 D25/D27）。

覆盖三层：
- API：/api/feedback/learn 的 task_id 分支（复用任务历史基线、不重跑管线）、
  task_id 与音频均缺 400、任务不存在 404、会话产物清理 410、旧客户端音频路径兼容
- 服务：基线解析与复用、D2 入库元数据从任务 stats 补全
- ingest：journal header 可选 scenario 字段与统计按场景分层（旧日志照常摄取）
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.feedback import sample_manager as sample_manager_module
from vocal_subtitle.feedback.journal_ingest import (
    journal_scenario,
    journal_statistics,
    load_journal_files,
)
from vocal_subtitle.webui import api
from vocal_subtitle.webui.app import create_app
from vocal_subtitle.webui.feedback_services import (
    FeedbackLearningService,
    TaskBaselineError,
)


REFERENCE_SRT = (
    "1\n00:00:01,000 --> 00:00:03,000\n第一句\n\n"
    "2\n00:00:04,000 --> 00:00:06,000\n第二句\n"
)

TASK_STATS = {
    "detected_language": "zh",
    "speaker_count": 2,
    "canonical_speaker_count": 2,
    "duration_seconds": 12.0,
}


def _serialized_event(index, start, end, text):
    """任务历史 result_json 中的事件形态（pipeline_tasks._serialize_events 的子集）"""
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "original_text": text,
        "speaker_id": None,
        "speaker_label": None,
    }


def _task_payload(session_dir):
    input_path = str(session_dir / "input.wav") if session_dir else ""
    return {
        "task_id": "task-1",
        "run_id": "run-1",
        "status": "completed",
        "events": [
            _serialized_event(1, 1.0, 3.0, "第一句"),
            _serialized_event(2, 4.0, 6.0, "第二句"),
        ],
        "stats": dict(TASK_STATS),
        "input_path": input_path,
        "artifacts": {"input": input_path} if input_path else {},
    }


def _history_record(payload):
    return {"id": "task-1", "status": "completed",
            "result_json": json.dumps(payload, default=str)}


class FakeTaskHistory:
    """TaskHistoryManager.get 的最小替身"""

    def __init__(self, records=None):
        self.records = records or {}

    def get(self, task_id):
        return self.records.get(task_id)


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def session_dir(tmp_path):
    """伪造任务会话目录（cache/uploads/{sha256[:16]}/ 的替身）"""
    directory = tmp_path / "abc123def4567890"
    directory.mkdir()
    (directory / "input.wav").write_bytes(b"fake audio")
    return directory


@pytest.fixture
def stub_semantic_scorer(monkeypatch):
    """关闭对齐器的语义模型加载，保证测试离线且快速"""
    from vocal_subtitle.feedback import aligner as aligner_module

    monkeypatch.setattr(
        aligner_module.SemanticScorer,
        "is_available",
        property(lambda self: False),
    )


@pytest.fixture
def captured_samples(monkeypatch):
    """替换 FeedbackSampleManager，捕获 D2 入库元数据而不写真实样本库"""
    ingested = []

    class FakeSampleManager:
        def __init__(self, storage_dir=None):
            pass

        def ingest(self, **kwargs):
            ingested.append(kwargs)
            return SimpleNamespace(sample_id="sample-1")

    monkeypatch.setattr(sample_manager_module, "FeedbackSampleManager", FakeSampleManager)
    return ingested


@pytest.fixture
def pipeline_calls(monkeypatch):
    """记录管线构造/运行调用；任何调用都说明发生了不应有的重跑"""
    calls = []

    def _factory():
        class ExplodingPipeline:
            def __init__(self, config):
                calls.append("construct")

            def run(self, **kwargs):
                calls.append("run")
                return {"events": [], "stats": {}}

        return ExplodingPipeline

    monkeypatch.setattr(
        "vocal_subtitle.webui.feedback_services._pipeline_class", _factory
    )
    return calls


# ---------------------------------------------------------------------------
# API 层：task_id 分支与错误分支
# ---------------------------------------------------------------------------

def _post_learn(client, data=None, with_reference=True):
    files = {}
    if with_reference:
        files["reference"] = ("ref.srt", REFERENCE_SRT.encode("utf-8"), "application/octet-stream")
    return client.post("/api/feedback/learn", data=data or {}, files=files)


class TestLearnApiTaskIdBranch:
    def test_task_id_reuses_history_baseline_without_rerun(
        self, client, monkeypatch, session_dir, pipeline_calls,
        captured_samples, stub_semantic_scorer,
    ):
        monkeypatch.setattr(
            api, "_task_history",
            FakeTaskHistory({"task-1": _history_record(_task_payload(session_dir))}),
        )
        resp = _post_learn(client, data={"task_id": "task-1", "scenario": "inline-review"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["baseline_source"] == "task_history"
        assert data["task_id"] == "task-1"
        assert pipeline_calls == []  # 不触发管线重跑
        # D2 元数据从任务 stats + 场景标签补全（不再是 unknown/0/空）
        assert len(captured_samples) == 1
        meta = captured_samples[0]
        assert meta["language"] == "zh"
        assert meta["speaker_count"] == 2
        assert meta["audio_duration"] == 12.0
        assert meta["scene"] == "inline-review"

    def test_task_id_wins_over_uploaded_audio(
        self, client, monkeypatch, session_dir, pipeline_calls, captured_samples,
        stub_semantic_scorer,
    ):
        monkeypatch.setattr(
            api, "_task_history",
            FakeTaskHistory({"task-1": _history_record(_task_payload(session_dir))}),
        )
        resp = client.post(
            "/api/feedback/learn",
            data={"task_id": "task-1"},
            files={
                "audio": ("a.wav", b"fake audio", "audio/wav"),
                "reference": ("ref.srt", REFERENCE_SRT.encode("utf-8"), "application/octet-stream"),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["baseline_source"] == "task_history"
        assert pipeline_calls == []  # 带任务基线时不重跑音频


class TestLearnApiErrors:
    def test_missing_task_id_and_audio_returns_400(self, client):
        resp = _post_learn(client, data={})
        assert resp.status_code == 400
        assert "task_id" in resp.json()["detail"]
        assert "音频" in resp.json()["detail"]

    def test_unknown_task_returns_404_with_audio_hint(self, client, monkeypatch):
        monkeypatch.setattr(api, "_task_history", FakeTaskHistory({}))
        resp = _post_learn(client, data={"task_id": "missing"})
        assert resp.status_code == 404
        assert "音频" in resp.json()["detail"]

    def test_cleaned_session_returns_410_with_audio_hint(self, client, monkeypatch, tmp_path):
        payload = _task_payload(None)
        payload["input_path"] = str(tmp_path / "cleaned-away" / "input.wav")
        monkeypatch.setattr(
            api, "_task_history",
            FakeTaskHistory({"task-1": _history_record(payload)}),
        )
        resp = _post_learn(client, data={"task_id": "task-1"})
        assert resp.status_code == 410
        assert "已清理" in resp.json()["detail"]
        assert "音频" in resp.json()["detail"]

    def test_invalid_scenario_returns_400(self, client, monkeypatch, session_dir):
        monkeypatch.setattr(
            api, "_task_history",
            FakeTaskHistory({"task-1": _history_record(_task_payload(session_dir))}),
        )
        resp = _post_learn(client, data={"task_id": "task-1", "scenario": "bogus"})
        assert resp.status_code == 400
        assert "场景" in resp.json()["detail"]


class TestLearnApiLegacyPath:
    def test_audio_only_legacy_path_still_reruns(
        self, client, monkeypatch, captured_samples, stub_semantic_scorer,
    ):
        """旧客户端兼容：不带 task_id、只传音频 → 照常走管线重跑"""
        from vocal_subtitle.webui import feedback_services

        class FakePipeline:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                events = [
                    SimpleNamespace(index=1, start=1.0, end=3.0, text="第一句"),
                    SimpleNamespace(index=2, start=4.0, end=6.0, text="第二句"),
                ]
                return {
                    "events": events,
                    "stats": SimpleNamespace(to_dict=lambda: dict(TASK_STATS)),
                }

        monkeypatch.setattr(feedback_services, "_pipeline_class", lambda: FakePipeline)
        resp = client.post(
            "/api/feedback/learn",
            data={},
            files={
                "audio": ("a.wav", b"fake audio", "audio/wav"),
                "reference": ("ref.srt", REFERENCE_SRT.encode("utf-8"), "application/octet-stream"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["baseline_source"] == "pipeline_rerun"
        # 重跑路径同样从管线 stats 补全 D2 元数据
        assert captured_samples[0]["language"] == "zh"
        assert captured_samples[0]["audio_duration"] == 12.0


# ---------------------------------------------------------------------------
# 服务层：基线解析/复用与元数据补全
# ---------------------------------------------------------------------------

class TestResolveTaskBaseline:
    def test_parses_history_events_and_stats(self, session_dir):
        service = FeedbackLearningService(
            task_history=FakeTaskHistory({"task-1": _history_record(_task_payload(session_dir))})
        )
        baseline = service.resolve_task_baseline("task-1")
        assert [e.text for e in baseline.events] == ["第一句", "第二句"]
        assert [e.index for e in baseline.events] == [1, 2]
        assert baseline.stats["detected_language"] == "zh"
        assert baseline.session_dir == session_dir

    def test_missing_task_raises_404_with_audio_hint(self):
        service = FeedbackLearningService(task_history=FakeTaskHistory({}))
        with pytest.raises(TaskBaselineError) as excinfo:
            service.resolve_task_baseline("nope")
        assert excinfo.value.status_code == 404
        assert "音频" in str(excinfo.value)

    def test_empty_events_raise_410(self, tmp_path):
        session_dir = tmp_path / "sess"
        session_dir.mkdir()
        payload = {"events": [], "stats": {}, "input_path": str(session_dir / "input.wav")}
        service = FeedbackLearningService(
            task_history=FakeTaskHistory({"task-1": _history_record(payload)})
        )
        with pytest.raises(TaskBaselineError) as excinfo:
            service.resolve_task_baseline("task-1")
        assert excinfo.value.status_code == 410

    def test_cleaned_session_dir_raises_410_with_audio_hint(self, tmp_path):
        payload = _task_payload(None)
        payload["input_path"] = str(tmp_path / "gone" / "input.wav")
        service = FeedbackLearningService(
            task_history=FakeTaskHistory({"task-1": _history_record(payload)})
        )
        with pytest.raises(TaskBaselineError) as excinfo:
            service.resolve_task_baseline("task-1")
        assert excinfo.value.status_code == 410
        assert "音频" in str(excinfo.value)


class TestServiceLearnMetadata:
    @pytest.fixture(autouse=False)
    def no_param_writes(self, monkeypatch):
        """dry_run=False 且归因非空时阻断参数写档（UserProfileManager/ParamLearner/FewShot）"""
        from vocal_subtitle.webui import feedback_services as fbs

        class _NoWriteProfile:
            def __init__(self, *a, **k):
                pass

            def load(self, name):
                return {"overrides": {}}

            def save(self, profile):
                pass

        class _NoLearner:
            def __init__(self, *a, **k):
                pass

            def learn_from_diff(self, **kw):
                return {}

        class _NoFewShot:
            def __init__(self, *a, **k):
                pass

            def load_cache(self, name):
                pass

            def build_merge_examples(self, actions):
                pass

            def build_format_examples(self, edits):
                pass

            def save_cache(self, name):
                pass

        monkeypatch.setattr(fbs, "UserProfileManager", _NoWriteProfile)
        monkeypatch.setattr(fbs, "ParamLearner", _NoLearner)
        monkeypatch.setattr(fbs, "FewShotBuilder", _NoFewShot)

    def test_learn_with_task_id_reuses_baseline_and_fills_metadata(
        self, session_dir, pipeline_calls, captured_samples, stub_semantic_scorer,
        no_param_writes, tmp_path,
    ):
        ref = tmp_path / "ref.srt"
        ref.write_text(REFERENCE_SRT, encoding="utf-8")
        service = FeedbackLearningService(
            task_history=FakeTaskHistory({"task-1": _history_record(_task_payload(session_dir))})
        )
        # 预览不产生入库副作用（定案 §5 遗留观察的复核结论）
        response = service.learn(
            None, ref, task_id="task-1", scenario="external-correction", dry_run=True,
        )
        assert response["status"] == "ok"
        assert captured_samples == []
        response = service.learn(
            None, ref, task_id="task-1", scenario="external-correction", dry_run=False,
        )
        assert response["status"] == "ok"
        assert response["baseline_source"] == "task_history"
        assert response["task_id"] == "task-1"
        assert pipeline_calls == []
        meta = captured_samples[0]
        assert meta["language"] == "zh"
        assert meta["speaker_count"] == 2
        assert meta["audio_duration"] == 12.0
        assert meta["scene"] == "external-correction"

    def test_speaker_count_falls_back_to_canonical(
        self, session_dir, captured_samples, stub_semantic_scorer, no_param_writes, tmp_path,
    ):
        payload = _task_payload(session_dir)
        payload["stats"] = {"detected_language": "en", "canonical_speaker_count": 3,
                            "duration_seconds": 30.0}
        ref = tmp_path / "ref.srt"
        ref.write_text(REFERENCE_SRT, encoding="utf-8")
        service = FeedbackLearningService(
            task_history=FakeTaskHistory({"task-1": _history_record(payload)})
        )
        service.learn(None, ref, task_id="task-1", scenario="existing-subtitle", dry_run=False)
        meta = captured_samples[0]
        assert meta["language"] == "en"
        assert meta["speaker_count"] == 3
        assert meta["audio_duration"] == 30.0

    def test_rerun_path_metadata_from_pipeline_stats(
        self, monkeypatch, captured_samples, stub_semantic_scorer, no_param_writes, tmp_path,
    ):
        from vocal_subtitle.webui import feedback_services

        class FakePipeline:
            def __init__(self, config):
                pass

            def run(self, **kwargs):
                return {
                    "events": [
                        SimpleNamespace(index=1, start=1.0, end=3.0, text="第一句"),
                        SimpleNamespace(index=2, start=4.0, end=6.0, text="第二句"),
                    ],
                    "stats": SimpleNamespace(to_dict=lambda: dict(TASK_STATS)),
                }

        monkeypatch.setattr(feedback_services, "_pipeline_class", lambda: FakePipeline)
        ref = tmp_path / "ref.srt"
        ref.write_text(REFERENCE_SRT, encoding="utf-8")
        service = FeedbackLearningService(task_history=FakeTaskHistory({}))
        response = service.learn(tmp_path / "a.wav", ref, dry_run=False)
        assert response["status"] == "ok"
        assert response["baseline_source"] == "pipeline_rerun"
        assert captured_samples[0]["language"] == "zh"


# ---------------------------------------------------------------------------
# ingest 层：journal header 可选 scenario 与统计分层
# ---------------------------------------------------------------------------

def _journal_event(session_id="s-1", seq=0, command="updateCueTimes"):
    return {
        "schema": "edit-journal-v1",
        "type": "event",
        "session_id": session_id,
        "seq": seq,
        "ts": "2026-09-10T00:00:00Z",
        "actor": "human",
        "command": command,
        "diff": [],
        "context": {},
    }


def _write_journal(path, header_extra, seqs, session_id="s-1"):
    header = {"schema": "edit-journal-v1", "type": "header", "session_id": session_id}
    header.update(header_extra)
    lines = [json.dumps(header)]
    lines += [json.dumps(_journal_event(session_id=session_id, seq=seq)) for seq in seqs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestJournalScenarioLayering:
    def test_header_scenario_is_read(self, tmp_path):
        path = _write_journal(
            tmp_path / "a.journal.jsonl", {"scenario": "inline-review"}, [0, 1]
        )
        files = load_journal_files([path])
        assert journal_scenario(files[0]) == "inline-review"

    def test_legacy_header_without_scenario_reads_empty(self, tmp_path):
        path = _write_journal(tmp_path / "old.journal.jsonl", {}, [0])
        files = load_journal_files([path])
        assert journal_scenario(files[0]) == ""

    def test_statistics_layered_by_scenario(self, tmp_path):
        """带场景的日志按场景分层；无 scenario 的旧日志照常计入总体"""
        modern = _write_journal(
            tmp_path / "modern.journal.jsonl", {"scenario": "inline-review"}, [0, 1],
            session_id="s-modern",
        )
        legacy = _write_journal(
            tmp_path / "legacy.journal.jsonl", {}, [0], session_id="s-legacy",
        )
        files = load_journal_files([modern, legacy])
        stats = journal_statistics(files)
        # 总体：两份日志全部事件
        assert stats.event_count == 3
        assert stats.session_count == 2
        # 分层：仅场景日志进入 by_scenario
        assert set(stats.by_scenario) == {"inline-review"}
        layered = stats.by_scenario["inline-review"]
        assert layered.event_count == 2
        assert layered.session_count == 1
        assert layered.commands["updateCueTimes"] == 2

    def test_statistics_covers_multiple_scenarios(self, tmp_path):
        first = _write_journal(
            tmp_path / "a.journal.jsonl", {"scenario": "inline-review"}, [0],
            session_id="s-a",
        )
        second = _write_journal(
            tmp_path / "b.journal.jsonl", {"scenario": "from-scratch-timing"}, [0, 5],
            session_id="s-b",
        )
        files = load_journal_files([first, second])
        stats = journal_statistics(files)
        assert set(stats.by_scenario) == {"inline-review", "from-scratch-timing"}
        assert stats.by_scenario["from-scratch-timing"].event_count == 2
        assert stats.event_count == 3

    def test_to_dict_includes_by_scenario(self, tmp_path):
        path = _write_journal(
            tmp_path / "a.journal.jsonl", {"scenario": "existing-subtitle"}, [0]
        )
        payload = journal_statistics(load_journal_files([path])).to_dict()
        assert payload["by_scenario"]["existing-subtitle"]["event_count"] == 1
        assert payload["event_count"] == 1
