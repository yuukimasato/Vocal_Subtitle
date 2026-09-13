"""工单 10：8613 数据集工作区（导出套壳，定案 D35）。

覆盖：
- 服务层 summarize_export：预览与 export_dataset 完全同口径（accepted-only +
  场景过滤 + 追加式已导出剔除），保证"预览数字 = 导出条目数"
- HTTP 端点：GET /api/feedback/dataset/preview、POST /api/feedback/dataset/export
  （许可门禁 400 复用服务层语义 / 未知场景 400 / 空库 400 / 产物落盘）
- TestClient 全流程冒烟：预览 → 选许可 → 导出 → 路径与统计校验（浏览器冒烟
  的 TestClient 等价层）
- 委托验证：HTTP 端点与 CLI 共用同一服务层导出实现，无第二份拷贝
- 8613 静态页结构：数据集工作区/导航/脚本挂载存在
"""

import inspect
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import vocal_subtitle
import vocal_subtitle.feedback.dataset_export as dataset_export_module
from vocal_subtitle.feedback import sample_manager as sample_manager_module
from vocal_subtitle.feedback.dataset_export import (
    collect_accepted_samples,
    export_dataset,
    summarize_export,
)
from vocal_subtitle.feedback.sample_manager import FeedbackSampleManager
from vocal_subtitle.webui import api, routes_dataset
from vocal_subtitle.webui.app import create_app

SHA_A = "a" * 64
SHA_B = "b" * 64


def _ingest(manager, text, *, scene="inline-review", language="zh", speaker_count=2,
            duration=12.5, status="accepted", task_id="task-1", sha256=SHA_A,
            run_id="run-1"):
    """入库一条样本并置为指定审核状态（与 test_dataset_export 同一做法）"""
    sample = manager.ingest(
        auto_subtitle=f"1\n00:00:00,000 --> 00:00:02,000\n自动{text}\n",
        human_revision=f"1\n00:00:00,000 --> 00:00:02,000\n人工{text}\n",
        alignment={"method": "dtw", "coverage_ratio": 1.0, "confidence": 0.9},
        consent_level="anonymous",
        language=language,
        scene=scene,
        audio_duration=duration,
        speaker_count=speaker_count,
        edit_types={"text_correction": 1},
        task_id=task_id,
        audio_sha256=sha256,
        run_id=run_id,
    )
    assert sample is not None
    if status != "pending":  # 入库默认即 pending
        manager.review(sample.sample_id, status, reviewer="tester")
    return sample


def _strip_text(manager, sample, field="automatic_subtitle"):
    """把样本的字幕全文从落盘 JSON 中删掉（模拟旧版样本库只存哈希）"""
    path = Path(manager.storage_dir) / f"{sample.sample_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data[field]["text"]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures（真实样本库全是 pending，测试一律用隔离临时样本库）
# ---------------------------------------------------------------------------


@pytest.fixture
def library(tmp_path) -> FeedbackSampleManager:
    """指向临时目录的样本库（不触碰真实 cache/feedback_samples）"""
    return FeedbackSampleManager(tmp_path / "lib")


@pytest.fixture
def client(sample_library, data_root):
    """TestClient：样本库与默认数据集目录均已隔离到 tmp_path"""
    return TestClient(create_app())


@pytest.fixture
def sample_library(tmp_path, monkeypatch):
    """把路由内 FeedbackSampleManager 的解析整体替换为临时样本库"""
    monkeypatch.setattr(
        sample_manager_module, "FeedbackSampleManager",
        lambda: FeedbackSampleManager(tmp_path / "lib"),
    )
    return tmp_path / "lib"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """注入临时上传目录：默认数据集目录随之落到 tmp_path/datasets/<name>"""
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", root)
    return root


# ---------------------------------------------------------------------------
# 服务层 summarize_export：预览与导出同口径
# ---------------------------------------------------------------------------


class TestSummarizeExport:
    def test_preview_count_equals_export_count(self, library, tmp_path):
        """预览 exportable_count == 导出 exported_count（全场景与场景过滤两种口径）"""
        _ingest(library, "甲")
        _ingest(library, "乙", scene="from-scratch-timing", task_id="task-2", sha256=SHA_B)
        _ingest(library, "丙", status="pending")   # pending 不出门
        _ingest(library, "丁", status="rejected")  # rejected 不出门

        for scenarios, name in (([], "ds-all"), (["inline-review"], "ds-inline")):
            preview = summarize_export(library, scenarios)
            outcome = export_dataset(library, tmp_path / name, license="CC-BY-4.0", scenarios=scenarios)
            assert preview["exportable_count"] == outcome.exported_count
            assert preview["accepted_total"] == outcome.accepted_total

    def test_scenario_filter_and_language_distribution(self, library):
        _ingest(library, "甲", scene="inline-review", language="zh")
        _ingest(library, "乙", scene="from-scratch-timing", language="ja",
                task_id="task-2", sha256=SHA_B)
        preview = summarize_export(library)
        assert preview["accepted_total"] == 2
        assert preview["by_scenario"] == {"inline-review": 1, "from-scratch-timing": 1}
        assert preview["by_language"] == {"zh": 1, "ja": 1}
        assert preview["exportable_count"] == 2

        filtered = summarize_export(library, ["inline-review"])
        assert filtered["exportable_count"] == 1
        assert filtered["by_language"] == {"zh": 1}

    def test_excludes_already_exported_like_append_export(self, library, tmp_path):
        """追加式语义：已导出样本从预览中剔除，新样本计入（与导出增量一致）"""
        first = _ingest(library, "甲")
        out_dir = tmp_path / "ds"
        outcome = export_dataset(library, out_dir, license="CC-BY-4.0")
        assert outcome.exported_count == 1

        preview = summarize_export(library, out_dir=out_dir)
        assert preview["already_exported"] == 1
        assert preview["exportable_count"] == 0

        _ingest(library, "乙", task_id="task-2", sha256=SHA_B)
        preview = summarize_export(library, out_dir=out_dir)
        assert preview["exportable_count"] == 1
        # 导出验证：只写入新样本
        outcome = export_dataset(library, out_dir, license="CC-BY-4.0")
        assert outcome.exported_count == preview["exportable_count"] == 1
        assert first.sample_id not in outcome.exported_ids

    def test_skipped_missing_text_counted_not_exportable(self, library):
        sample = _ingest(library, "甲")
        _strip_text(library, sample)
        preview = summarize_export(library)
        assert preview["exportable_count"] == 0
        assert preview["skipped_missing_text"] == [sample.sample_id]

    def test_empty_library_zero_values_without_error(self, library):
        """无 accepted 样本时预览返回零值统计（导出才报错，预览友好展示）"""
        preview = summarize_export(library)
        assert preview["accepted_total"] == 0
        assert preview["exportable_count"] == 0
        assert preview["by_scenario"] == {}

    def test_no_out_dir_treated_as_first_export(self, library, tmp_path):
        _ingest(library, "甲")
        # 不传 out_dir：视为全新导出（目录里已有的导出不参与剔除）
        preview = summarize_export(library)
        assert preview["exportable_count"] == 1
        assert preview["existing_dataset_samples"] == 0


# ---------------------------------------------------------------------------
# HTTP 端点：预览 + 导出
# ---------------------------------------------------------------------------


class TestDatasetPreviewEndpoint:
    def test_preview_stats(self, client, sample_library):
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")
        _ingest(manager, "乙", scene="external-correction", language="en",
                task_id="task-2", sha256=SHA_B)
        _ingest(manager, "丙", status="pending")

        resp = client.get("/api/feedback/dataset/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema"] == "dataset-v1"
        assert data["accepted_total"] == 2
        assert data["exportable_count"] == 2
        assert data["by_scenario"] == {"inline-review": 1, "external-correction": 1}
        assert data["by_language"] == {"zh": 1, "en": 1}
        assert data["skipped_missing_text"] == []
        # 默认输出目录 = 用户数据目录 datasets/<name>
        assert data["out_dir"].endswith(str(Path("datasets") / "subtitle-feedback"))

    def test_preview_scenario_query_filter(self, client, sample_library):
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")
        _ingest(manager, "乙", scene="from-scratch-timing", task_id="task-2", sha256=SHA_B)
        resp = client.get("/api/feedback/dataset/preview", params={"scenarios": "inline-review"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenarios_filter"] == ["inline-review"]
        assert data["exportable_count"] == 1

    def test_preview_unknown_scenario_400(self, client):
        resp = client.get("/api/feedback/dataset/preview", params={"scenarios": "bogus"})
        assert resp.status_code == 400
        assert "未知场景标签" in resp.json()["detail"]

    def test_preview_empty_library_returns_zero_stats(self, client):
        resp = client.get("/api/feedback/dataset/preview")
        assert resp.status_code == 200
        assert resp.json()["exportable_count"] == 0


class TestDatasetExportEndpoint:
    def test_missing_license_400_reuses_service_gate(self, client, sample_library):
        """许可缺失 → 400：门禁语义来自服务层（LicenseRequiredError），HTTP 仅转译"""
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")
        resp = client.post("/api/feedback/dataset/export", json={"scenarios": []})
        assert resp.status_code == 400
        assert "许可" in resp.json()["detail"]

    def test_blank_license_400(self, client, sample_library):
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")
        resp = client.post("/api/feedback/dataset/export", json={"license": "   "})
        assert resp.status_code == 400
        assert "许可" in resp.json()["detail"]

    def test_unknown_scenario_400(self, client, sample_library):
        resp = client.post(
            "/api/feedback/dataset/export",
            json={"license": "CC-BY-4.0", "scenarios": ["bogus"]},
        )
        assert resp.status_code == 400
        assert "未知场景标签" in resp.json()["detail"]

    def test_empty_library_400(self, client):
        """真实样本库全是 pending 时的等价场景：无 accepted 样本 → 400 拒绝"""
        resp = client.post("/api/feedback/dataset/export", json={"license": "CC-BY-4.0"})
        assert resp.status_code == 400
        assert "accepted" in resp.json()["detail"]

    def test_export_writes_git_ready_dir_with_out_dir_override(
        self, client, sample_library, tmp_path
    ):
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")
        _ingest(manager, "乙", scene="external-correction", task_id="task-2", sha256=SHA_B)
        out_dir = tmp_path / "custom-ds"

        resp = client.post(
            "/api/feedback/dataset/export",
            json={"license": "CC-BY-4.0", "scenarios": ["inline-review"], "out_dir": str(out_dir)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["exported_count"] == 1  # 场景过滤生效
        assert data["accepted_total"] == 1
        assert data["license"] == "CC-BY-4.0"
        assert data["shards"] == ["data/shard-0001.jsonl"]

        # 产物与 CLI 导出同一服务层产出：README 卡片 + manifest + jsonl 分片
        assert (out_dir / "README.md").exists()
        assert (out_dir / "manifest.json").exists()
        assert "CC-BY-4.0" in (out_dir / "README.md").read_text(encoding="utf-8")
        lines = (out_dir / "data/shard-0001.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["schema"] == "dataset-v1"

    def test_export_default_dir_and_warnings(self, client, sample_library, data_root):
        """默认目录落用户数据目录 datasets/；缺字幕全文样本进入告警字段"""
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")
        broken = _ingest(manager, "乙", task_id="task-2", sha256=SHA_B)
        _strip_text(manager, broken)

        resp = client.post(
            "/api/feedback/dataset/export",
            json={"license": "CC0-1.0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        expected = data_root.parent / "datasets" / "subtitle-feedback"
        assert Path(data["out_dir"]) == expected
        assert expected.is_dir()
        assert data["exported_count"] == 1
        assert data["total_samples"] == 1
        assert data["skipped_missing_text"] == [broken.sample_id]

    def test_repeat_export_idempotent_and_preview_zero(self, client, sample_library):
        """重复导出幂等；同参数预览的"本次将导出"归零（预览 = 导出闭环）"""
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")

        first = client.post("/api/feedback/dataset/export", json={"license": "CC-BY-4.0"})
        assert first.status_code == 200
        assert first.json()["exported_count"] == 1

        preview = client.get("/api/feedback/dataset/preview")
        assert preview.json()["exportable_count"] == 0
        assert preview.json()["already_exported"] == 1

        second = client.post("/api/feedback/dataset/export", json={"license": "CC-BY-4.0"})
        assert second.status_code == 200
        assert second.json()["exported_count"] == 0
        assert second.json()["total_samples"] == 1


# ---------------------------------------------------------------------------
# TestClient 全流程冒烟：预览 → 选许可 → 导出 → 路径/统计校验
# （浏览器冒烟的等价层：与前端 ui-dataset-workspace.js 相同的调用序列）
# ---------------------------------------------------------------------------


class TestDatasetWorkspaceFlow:
    def test_preview_choose_license_export_verify_path_and_stats(
        self, client, sample_library
    ):
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲", scene="inline-review", language="zh")
        _ingest(manager, "乙", scene="inline-review", language="zh",
                task_id="task-2", sha256=SHA_B, run_id="run-2")
        _ingest(manager, "丙", scene="from-scratch-timing", language="ja",
                task_id="task-3", sha256=SHA_B, run_id="run-3")
        _ingest(manager, "丁", status="pending")  # 真实库全 pending 的等价物：不参与

        # 1. 预览（前端进入工作区自动加载，全部场景）
        resp = client.get("/api/feedback/dataset/preview")
        assert resp.status_code == 200
        preview = resp.json()
        assert preview["accepted_total"] == 3
        assert preview["exportable_count"] == 3
        assert preview["by_language"] == {"zh": 2, "ja": 1}
        assert preview["total_duration_seconds"] == 37.5

        # 2. 场景过滤后预览（前端勾选 inline-review 后点刷新）
        resp = client.get(
            "/api/feedback/dataset/preview", params={"scenarios": "inline-review"}
        )
        assert resp.status_code == 200
        filtered = resp.json()
        assert filtered["exportable_count"] == 2
        assert filtered["by_scenario"] == {"inline-review": 2}

        # 3. 选许可（前端下拉必选）→ 导出
        resp = client.post(
            "/api/feedback/dataset/export",
            json={"license": "CC-BY-4.0", "scenarios": ["inline-review"]},
        )
        assert resp.status_code == 200
        export = resp.json()
        # 预览数字 = 导出条目数（验收标准核心）
        assert export["exported_count"] == filtered["exportable_count"] == 2

        # 4. 路径与统计校验：默认目录下产物齐全
        out_dir = Path(export["out_dir"])
        assert out_dir.is_dir()
        assert (out_dir / "README.md").exists()
        assert (out_dir / "manifest.json").exists()
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["total_samples"] == export["total_samples"] == 2
        assert manifest["license"] == "CC-BY-4.0"
        assert manifest["scenarios_filter"] == ["inline-review"]
        shard_lines = (out_dir / "data/shard-0001.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(shard_lines) == export["exported_count"] == 2

        # 5. 导出后同参数预览归零、累计数一致（追加式语义前后呼应）
        resp = client.get(
            "/api/feedback/dataset/preview", params={"scenarios": "inline-review"}
        )
        after = resp.json()
        assert after["exportable_count"] == 0
        assert after["already_exported"] == 2
        assert after["existing_dataset_samples"] == 2


# ---------------------------------------------------------------------------
# 服务层复用：HTTP 与 CLI 共用同一导出实现，无第二份拷贝
# ---------------------------------------------------------------------------


class TestServiceLayerReuse:
    def test_export_endpoint_delegates_to_service_layer(
        self, client, sample_library, monkeypatch
    ):
        """HTTP 导出端点必须走服务层 export_dataset（打桩验证委托）"""
        calls = []
        real = dataset_export_module.export_dataset

        def spy(*args, **kwargs):
            calls.append({"license": kwargs.get("license"), "scenarios": kwargs.get("scenarios")})
            return real(*args, **kwargs)

        monkeypatch.setattr(dataset_export_module, "export_dataset", spy)
        manager = FeedbackSampleManager(sample_library)
        _ingest(manager, "甲")

        resp = client.post(
            "/api/feedback/dataset/export",
            json={"license": "CC-BY-4.0", "scenarios": ["inline-review"]},
        )
        assert resp.status_code == 200
        assert calls == [{"license": "CC-BY-4.0", "scenarios": ["inline-review"]}]

    def test_routes_module_holds_no_export_logic_copy(self):
        """路由模块只做 HTTP 适配：不含分片物化等写盘逻辑（统计读回 manifest 除外）"""
        source = inspect.getsource(routes_dataset)
        assert "shard-" not in source
        assert "def _next_shard_index" not in source
        assert "def _aggregate_stats" not in source
        assert "def render_readme" not in source

    def test_cli_and_http_use_same_service_symbols(self):
        """CLI 与 HTTP 的导出都指向同一个服务层函数对象"""
        from vocal_subtitle.cli_commands import feedback_commands

        cli_source = inspect.getsource(feedback_commands)
        assert "from ..feedback.dataset_export import" in cli_source
        assert "export_dataset(" in cli_source
        assert dataset_export_module.export_dataset is export_dataset


# ---------------------------------------------------------------------------
# 8613 静态页结构：数据集工作区挂载
# ---------------------------------------------------------------------------


STATIC_DIR = Path(vocal_subtitle.__file__).parent / "webui" / "static"


class TestStaticWorkspace:
    def test_index_has_dataset_workspace(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert 'data-workspace="dataset"' in html
        assert 'id="workspace-dataset"' in html
        assert 'id="dataset-license"' in html
        assert 'id="btn-dataset-export"' in html
        assert 'id="dataset-preview-stats"' in html
        assert "ui-dataset-workspace.js" in html

    def test_workspace_js_allows_dataset(self):
        source = (STATIC_DIR / "js" / "workspace.js").read_text(encoding="utf-8")
        assert "'dataset'" in source
        assert "DatasetWorkspace.refresh" in source

    def test_dataset_workspace_js_wiring(self):
        source = (STATIC_DIR / "js" / "ui-dataset-workspace.js").read_text(encoding="utf-8")
        assert "window.DatasetWorkspace" in source
        assert "datasetPreview" in source
        assert "datasetExport" in source

    def test_api_client_has_dataset_methods(self):
        source = (STATIC_DIR / "js" / "api-client.js").read_text(encoding="utf-8")
        assert "datasetPreview" in source
        assert "/api/feedback/dataset/preview" in source
        assert "/api/feedback/dataset/export" in source
