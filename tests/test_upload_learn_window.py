"""工单 04：8613 上传型学习窗口（V2 external-correction，定案 D20/D23/D26）。

覆盖三层：
- API：按输入哈希查询任务的端点（/api/history/by-hash/{sha256}，
  含会话目录兜底：视频任务存的是提取音轨哈希，拖入原始视频按会话键反查）
- 全流程冒烟（TestClient）：建任务 → 哈希查询选中 → 上传 .srt 修正字幕 →
  learn(task_id, scenario=external-correction) → 报告字段齐全 → 会话清理后 410 提示
- 对齐行为（D20）：覆盖率 <50% 警告但放行；未匹配人工行标重构行、不参与时间轴学习
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.feedback import sample_manager as sample_manager_module
from vocal_subtitle.utils.session_manager import SessionManager
from vocal_subtitle.utils.task_history import TaskHistoryManager
from vocal_subtitle.webui import api
from vocal_subtitle.webui.app import create_app

TASK_STATS = {
    "detected_language": "zh",
    "speaker_count": 2,
    "canonical_speaker_count": 2,
    "duration_seconds": 30.0,
}

# 管线基线事件（任务历史 result_json 中已存事件的序列化形态）
AUTO_EVENTS = [
    (1.0, 3.0, "第一句"),
    (4.0, 6.0, "第二句"),
    (7.0, 9.0, "第三句"),
    (10.0, 12.0, "第四句"),
    (13.0, 15.0, "第五句"),
    (16.0, 18.0, "第六句"),
]


@dataclass
class _FakeConfig:
    """TaskHistoryManager.create 需要 dataclass 以计算 config_hash"""

    profile: str = "default"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _srt(events) -> str:
    blocks = []
    for idx, (start, end, text) in enumerate(events, 1):
        blocks.append(f"{idx}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
    return "\n".join(blocks)


def _ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _serialized_event(index, start, end, text):
    return {
        "index": index,
        "start": start,
        "end": end,
        "text": text,
        "original_text": text,
        "speaker_id": None,
        "speaker_label": None,
    }


def _register_task(manager, task_id, file_hash, session_dir, status="completed"):
    """写入一条任务历史记录（含已存事件基线与 stats）"""
    input_path = str(session_dir / "input.wav")
    payload = {
        "task_id": task_id,
        "run_id": "run-1",
        "status": status,
        "events": [
            _serialized_event(i, start, end, text)
            for i, (start, end, text) in enumerate(AUTO_EVENTS, 1)
        ],
        "stats": dict(TASK_STATS),
        "input_path": input_path,
        "artifacts": {"input": input_path},
    }
    manager.create(
        task_id=task_id,
        file_name="lesson01.wav",
        file_hash=file_hash,
        file_size=2048,
        profile="default",
        config=_FakeConfig(),
    )
    manager.update(task_id, status=status, result_json=json.dumps(payload, default=str))
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def history(tmp_path, monkeypatch):
    """注入临时任务历史库（替换 state.task_history 的解析目标 api._task_history）"""
    manager = TaskHistoryManager(db_path=tmp_path / "history.db")
    monkeypatch.setattr(api, "_task_history", manager)
    return manager


@pytest.fixture
def upload_root(tmp_path, monkeypatch):
    """注入临时上传目录（学习请求暂存与会话目录兜底查询都落在这里）"""
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", root)
    return root


@pytest.fixture
def session_dir(tmp_path):
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

    monkeypatch.setattr(
        sample_manager_module, "FeedbackSampleManager", FakeSampleManager
    )
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
# 哈希查询端点：/api/history/by-hash/{sha256}
# ---------------------------------------------------------------------------


class TestHistoryByHashEndpoint:
    def test_finds_completed_task_by_input_hash(self, client, history, session_dir):
        file_hash = _sha("lesson01")
        _register_task(history, "task-ok", file_hash, session_dir)
        resp = client.get(f"/api/history/by-hash/{file_hash}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["match_source"] == "input_file_hash"
        item = data["items"][0]
        assert item["id"] == "task-ok"
        assert item["input_file_hash"] == file_hash
        assert item["input_file_name"] == "lesson01.wav"
        assert item["status"] == "completed"

    def test_invalid_hash_returns_400(self, client, history):
        resp = client.get("/api/history/by-hash/not-a-hash")
        assert resp.status_code == 400
        assert "sha256" in resp.json()["detail"]

    def test_no_match_returns_empty_items(self, client, history):
        resp = client.get(f"/api/history/by-hash/{_sha('missing')}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["match_source"] == ""

    def test_failed_tasks_are_not_learnable_sources(self, client, history, session_dir):
        file_hash = _sha("lesson02")
        _register_task(history, "task-bad", file_hash, session_dir, status="failed")
        _register_task(history, "task-good", file_hash, session_dir)
        resp = client.get(f"/api/history/by-hash/{file_hash}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == "task-good"

    def test_session_dir_fallback_matches_original_video_bytes(
        self, client, history, session_dir, upload_root
    ):
        """视频任务存的是提取音轨哈希；拖入原始视频按会话目录键兜底反查（D26）"""
        original_hash = _sha("original-video-bytes")
        extracted_hash = _sha("extracted-audio-bytes")
        _register_task(history, "task-video", extracted_hash, session_dir)
        session_mgr = SessionManager(upload_root)
        video_session = session_mgr.session_dir(original_hash[:16])
        video_session.mkdir(parents=True)
        session_mgr.write_metadata(
            video_session,
            original_filename="lesson01.mkv",
            input_sha256=extracted_hash,
            profile="default",
            config_hash="cfg",
            task_id="task-video",
        )
        resp = client.get(f"/api/history/by-hash/{original_hash}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["match_source"] == "session_dir"
        assert data["items"][0]["id"] == "task-video"

    def test_session_dir_fallback_ignores_failed_task(
        self, client, history, session_dir, upload_root
    ):
        original_hash = _sha("another-video-bytes")
        _register_task(history, "task-failed", _sha("x"), session_dir, status="failed")
        session_mgr = SessionManager(upload_root)
        video_session = session_mgr.session_dir(original_hash[:16])
        video_session.mkdir(parents=True)
        session_mgr.write_metadata(
            video_session,
            original_filename="lesson02.mkv",
            input_sha256=_sha("x"),
            profile="default",
            config_hash="cfg",
            task_id="task-failed",
        )
        resp = client.get(f"/api/history/by-hash/{original_hash}")
        assert resp.status_code == 200
        assert resp.json()["items"] == []


class TestFindByFileHashUnit:
    def test_returns_recent_completed_first(self, history, session_dir):
        file_hash = _sha("dup")
        _register_task(history, "task-old", file_hash, session_dir)
        _register_task(history, "task-new", file_hash, session_dir)
        records = history.find_by_file_hash(file_hash)
        assert [r["id"] for r in records] == ["task-new", "task-old"]

    def test_empty_hash_returns_empty_list(self, history):
        assert history.find_by_file_hash("") == []


# ---------------------------------------------------------------------------
# TestClient 全流程冒烟：建任务 → 哈希查询 → 上传字幕 → 学习 → 报告 → 410
# ---------------------------------------------------------------------------


def _post_learn(client, events, **data):
    files = {
        "reference": (
            "corrected.srt",
            _srt(events).encode("utf-8"),
            "application/octet-stream",
        )
    }
    return client.post("/api/feedback/learn", data=data, files=files)


class TestUploadLearnFlow:
    def test_full_flow_hash_lookup_learn_and_report_then_cleaned_session_410(
        self,
        client,
        history,
        session_dir,
        pipeline_calls,
        captured_samples,
        stub_semantic_scorer,
        monkeypatch,
    ):
        # 确认学习（dry_run=false）且归因非空 → 阻断参数写档，专注验证 D2 入库元数据
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
        # 1. 建任务（模拟一次已完成的管线任务，事件基线已存服务端）
        file_hash = _sha("flow-source")
        _register_task(history, "task-flow", file_hash, session_dir)

        # 2. 拖入同一片源 → 内容哈希命中并自动选中
        resp = client.get(f"/api/history/by-hash/{file_hash}")
        assert resp.status_code == 200
        matched = resp.json()["items"][0]
        assert matched["id"] == "task-flow"

        # 3. 上传修正字幕（整体后移 0.15s 的时间修正）→ external-correction 引用学习
        corrected = [
            (start + 0.15, end + 0.15, text) for (start, end, text) in AUTO_EVENTS
        ]
        resp = _post_learn(
            client,
            corrected,
            task_id=matched["id"],
            scenario="external-correction",
            dry_run="true",
        )
        assert resp.status_code == 200
        report = resp.json()
        # 4. 报告字段齐全：覆盖率/配对数/时间偏移/参数调整/重构行
        assert report["status"] == "ok"
        assert report["baseline_source"] == "task_history"
        assert report["task_id"] == "task-flow"
        assert report["alignment_coverage"] == 1.0
        assert report["total_pairs"] == 6
        assert report["time_shifts_count"] == 6
        assert report["reconstructed_lines"] == 0
        assert report["inserted_lines"] == 0
        assert report["coverage_warning"] == ""
        assert "merging.padding" in report["param_adjustments"]
        adjustment = report["param_adjustments"]["merging.padding"]
        assert adjustment["direction"] == "increase"
        assert pipeline_calls == []  # 全程不重跑管线
        # 预览不产生入库副作用（定案 §5 遗留观察的复核结论）
        assert captured_samples == []
        # 确认学习：场景标签与任务 stats 元数据入库
        resp = _post_learn(
            client,
            corrected,
            task_id=matched["id"],
            scenario="external-correction",
            dry_run="false",
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert captured_samples[0]["scene"] == "external-correction"
        assert captured_samples[0]["language"] == "zh"
        assert captured_samples[0]["audio_duration"] == 30.0

        # 5. 来源任务会话产物被清理 → 报错并提示改走音频上传学习
        shutil.rmtree(session_dir)
        resp = _post_learn(
            client,
            corrected,
            task_id="task-flow",
            scenario="external-correction",
            dry_run="true",
        )
        assert resp.status_code == 410
        detail = resp.json()["detail"]
        assert "已清理" in detail
        assert "音频" in detail

    def test_unknown_task_hint_after_hash_gone(
        self, client, history, session_dir, pipeline_calls, stub_semantic_scorer
    ):
        """任务记录被清理（历史中无此任务）→ 404，同样提示音频路径"""
        resp = _post_learn(
            client,
            AUTO_EVENTS,
            task_id="ghost",
            scenario="external-correction",
            dry_run="true",
        )
        assert resp.status_code == 404
        assert "音频" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# D20 对齐行为：<50% 警告但放行；重构行不参与时间轴维度学习
# ---------------------------------------------------------------------------


class TestLowCoverageWarnButProceed:
    def test_below_50_percent_warns_and_still_learns(
        self,
        client,
        history,
        session_dir,
        pipeline_calls,
        captured_samples,
        stub_semantic_scorer,
    ):
        """2/12 配对（覆盖率 <50%）：警告但放行，6 行重构行不进时间轴学习"""
        file_hash = _sha("heavy-resplit")
        _register_task(history, "task-low", file_hash, session_dir)

        # 修正字幕：前两句与基线基本对齐（仅 20ms 微调），后 6 句为时段/文本都不同的重构行
        corrected = [
            (1.02, 3.02, "第一句"),
            (4.02, 6.02, "第二句"),
            (60.0, 62.0, "重写的新内容甲"),
            (63.0, 65.0, "重写的新内容乙"),
            (66.0, 68.0, "重写的新内容丙"),
            (69.0, 71.0, "重写的新内容丁"),
            (72.0, 74.0, "重写的新内容戊"),
            (75.0, 77.0, "重写的新内容己"),
        ]
        resp = _post_learn(
            client,
            corrected,
            task_id="task-low",
            scenario="external-correction",
            dry_run="false",
        )
        assert resp.status_code == 200
        report = resp.json()
        # 警告但放行：状态仍为 ok，学习流程完整走完
        assert report["status"] == "ok"
        assert report["baseline_source"] == "task_history"
        assert "50%" in report["coverage_warning"]
        assert "重构行" in report["coverage_warning"]
        # 未匹配的人工行标重构行；自动版多出行单独计数
        assert report["reconstructed_lines"] == 6
        assert report["inserted_lines"] == 4
        assert report["alignment_coverage"] < 0.5
        # 重构行不参与时间轴维度学习：时间偏移只来自 2 个 1:1 配对（20ms 微调
        # 低于 50ms 归因阈值 → 无参数调整），其余 6 行被排除
        assert report["time_shifts_count"] == 2
        assert report["param_adjustments"] == {}
        assert report["message"] == "无需调整参数"
        assert pipeline_calls == []
        # 样本照常入库（场景标签 external-correction）
        assert captured_samples[0]["scene"] == "external-correction"

    def test_aligner_default_behavior_still_raises(self, stub_semantic_scorer):
        """向后兼容：默认调用（不传 on_low_coverage）低于门控仍抛 AlignmentError"""
        from vocal_subtitle.feedback.aligner import AlignmentError, SubtitleAligner
        from vocal_subtitle.mapping.time_mapper import SubtitleEvent

        def _events(specs):
            return [
                SubtitleEvent(index=i, start=s, end=e, text=t)
                for i, (s, e, t) in enumerate(specs, 1)
            ]

        auto = _events([(0.0, 2.0, "完全不同的一句"), (3.0, 5.0, "完全不相关的另一句")])
        manual = _events([(20.0, 22.0, f"内容甲{i}") for i in range(10)])
        aligner = SubtitleAligner(semantic_enabled=False)
        with pytest.raises(AlignmentError):
            aligner.align(auto, manual)
        # warn 模式：同样的输入放行
        pairs = aligner.align(auto, manual, on_low_coverage="warn")
        assert any(p.match_type == "DELETE" for p in pairs)
