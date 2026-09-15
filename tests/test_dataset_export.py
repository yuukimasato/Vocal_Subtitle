"""dataset-v1 数据集导出测试（工单 09：过滤/格式快照/许可门禁/追加式导出/bundle-audio）。"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from vocal_subtitle.cli_commands.feedback_commands import feedback
from vocal_subtitle.feedback.dataset_export import (
    DATASET_SCHEMA,
    LicenseRequiredError,
    build_entry,
    collect_accepted_samples,
    export_dataset,
)
from vocal_subtitle.feedback.sample_manager import FeedbackSampleManager

SHA_A = "a" * 64
SHA_B = "b" * 64

# dataset-v1 条目字段（顺序以定案 §4 条目示例为准）
EXPECTED_ENTRY_KEYS = [
    "schema",
    "sample_id",
    "scenario",
    "language",
    "speaker_count",
    "audio_duration",
    "run_id",
    "task_id",
    "auto_subtitle",
    "human_revision",
    "edit_types",
    "audio_ref",
]


@pytest.fixture
def library(tmp_path) -> FeedbackSampleManager:
    """指向临时目录的样本库（不触碰真实 cache/feedback_samples）"""
    return FeedbackSampleManager(tmp_path)


def _ingest(
    manager,
    text,
    *,
    scene="inline-review",
    language="zh",
    speaker_count=2,
    duration=12.5,
    status="accepted",
    task_id="task-1",
    sha256=SHA_A,
    run_id="run-1",
    edit_types=None,
):
    """入库一条样本并置为指定审核状态"""
    sample = manager.ingest(
        auto_subtitle=f"1\n00:00:00,000 --> 00:00:02,000\n自动{text}\n",
        human_revision=f"1\n00:00:00,000 --> 00:00:02,000\n人工{text}\n",
        alignment={"method": "dtw", "coverage_ratio": 1.0, "confidence": 0.9},
        consent_level="anonymous",
        language=language,
        scene=scene,
        audio_duration=duration,
        speaker_count=speaker_count,
        edit_types=edit_types if edit_types is not None else {"text_correction": 1},
        task_id=task_id,
        audio_sha256=sha256,
        run_id=run_id,
    )
    assert sample is not None
    if (
        status != "pending"
    ):  # 入库默认即 pending；review 只接受 accepted/rejected/disputed
        manager.review(sample.sample_id, status, reviewer="tester")
    return sample


def _read_manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def _shard_files(out_dir: Path) -> list[str]:
    data_dir = out_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(path.name for path in data_dir.glob("shard-*.jsonl"))


class TestCollectAccepted:
    def test_only_accepted_exported(self, library):
        accepted = _ingest(library, "甲")
        _ingest(library, "乙", status="pending")
        _ingest(library, "丙", status="rejected")
        samples = collect_accepted_samples(library)
        assert [item["sample_id"] for item in samples] == [accepted.sample_id]

    def test_disputed_not_exported(self, library):
        _ingest(library, "甲", status="disputed")
        assert collect_accepted_samples(library) == []

    def test_scenario_filter(self, library):
        inline = _ingest(library, "甲", scene="inline-review")
        _ingest(
            library, "乙", scene="from-scratch-timing", task_id="task-2", sha256=SHA_B
        )
        filtered = collect_accepted_samples(library, ["inline-review"])
        assert [item["sample_id"] for item in filtered] == [inline.sample_id]
        assert len(collect_accepted_samples(library)) == 2

    def test_unknown_scene_excluded_by_filter(self, library):
        # 旧样本 scene 缺省落为 unknown：显式场景过滤时不应混入
        _ingest(library, "甲", scene="unknown")
        assert collect_accepted_samples(library, ["inline-review"]) == []


class TestEntryFormat:
    def test_entry_matches_spec(self, library):
        sample = _ingest(library, "甲")
        entry = build_entry(library.get(sample.sample_id))
        assert entry is not None
        assert list(entry.keys()) == EXPECTED_ENTRY_KEYS
        assert entry["schema"] == DATASET_SCHEMA == "dataset-v1"
        assert entry["sample_id"] == sample.sample_id
        assert entry["scenario"] == "inline-review"
        assert entry["language"] == "zh"
        assert entry["speaker_count"] == 2
        assert entry["audio_duration"] == 12.5
        assert entry["run_id"] == "run-1"
        assert entry["task_id"] == "task-1"
        assert "自动甲" in entry["auto_subtitle"]
        assert "人工甲" in entry["human_revision"]
        assert entry["edit_types"] == {"text_correction": 1}
        assert entry["audio_ref"] == {
            "task_id": "task-1",
            "sha256": SHA_A,
            "duration": 12.5,
        }

    def test_old_sample_without_text_builds_no_entry(self, library):
        """旧版样本库只落字幕哈希不存全文：无法物化文本对，条目应为 None"""
        sample = _ingest(library, "甲")
        path = Path(library.storage_dir) / f"{sample.sample_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["automatic_subtitle"]["text"]
        del data["human_revision"]["text"]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert build_entry(library.get(sample.sample_id)) is None

    def test_export_skips_missing_text_and_records_it(self, library, tmp_path):
        sample = _ingest(library, "甲")
        path = Path(library.storage_dir) / f"{sample.sample_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["automatic_subtitle"]["text"]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        outcome = export_dataset(library, tmp_path / "ds", license="CC-BY-4.0")
        assert outcome.exported_count == 0
        assert outcome.skipped_missing_text == [sample.sample_id]


class TestLicenseGate:
    def test_missing_license_rejected(self, library, tmp_path):
        with pytest.raises(LicenseRequiredError):
            export_dataset(library, tmp_path / "ds", license="")

    def test_blank_license_rejected(self, library, tmp_path):
        with pytest.raises(LicenseRequiredError):
            export_dataset(library, tmp_path / "ds", license="   ")

    def test_cli_non_interactive_without_license_rejected(self, tmp_path, monkeypatch):
        from vocal_subtitle.feedback import sample_manager as module

        monkeypatch.setattr(
            module,
            "FeedbackSampleManager",
            lambda: FeedbackSampleManager(tmp_path / "lib"),
        )
        runner = CliRunner()
        result = runner.invoke(
            feedback, ["export-dataset", "--out", str(tmp_path / "ds")]
        )
        assert result.exit_code == 1
        assert "许可" in result.output

    def test_license_recorded_in_readme(self, library, tmp_path):
        out_dir = tmp_path / "ds"
        _ingest(library, "甲")
        outcome = export_dataset(library, out_dir, license="CC-BY-4.0")
        readme = (out_dir / "README.md").read_text(encoding="utf-8")
        assert "CC-BY-4.0" in readme
        assert outcome.license == "CC-BY-4.0"


class TestAppendOnlyExport:
    def test_second_export_appends_new_shard(self, library, tmp_path):
        out_dir = tmp_path / "ds"
        _ingest(library, "甲")
        _ingest(library, "乙", task_id="task-2", sha256=SHA_B)
        first = export_dataset(library, out_dir, license="CC-BY-4.0")
        assert first.shards == ["data/shard-0001.jsonl"]
        first_bytes = (out_dir / "data/shard-0001.jsonl").read_bytes()

        _ingest(
            library,
            "丙",
            scene="external-correction",
            task_id="task-3",
            sha256=SHA_B,
            run_id="run-3",
        )
        second = export_dataset(library, out_dir, license="CC-BY-4.0")
        assert second.shards == ["data/shard-0002.jsonl"]
        assert (
            out_dir / "data/shard-0001.jsonl"
        ).read_bytes() == first_bytes  # 已存在分片未被改写
        assert _shard_files(out_dir) == ["shard-0001.jsonl", "shard-0002.jsonl"]

        manifest = _read_manifest(out_dir)
        assert manifest["total_samples"] == 3
        assert manifest["schema"] == "dataset-v1"
        assert [
            shard["file"] for shard in manifest["shards"]
        ] == first.shards + second.shards
        assert all(shard["sha256"] for shard in manifest["shards"])

    def test_repeat_export_is_idempotent(self, library, tmp_path):
        out_dir = tmp_path / "ds"
        _ingest(library, "甲")
        export_dataset(library, out_dir, license="CC-BY-4.0")
        manifest_before = _read_manifest(out_dir)
        readme_before = (out_dir / "README.md").read_bytes()

        outcome = export_dataset(library, out_dir, license="CC-BY-4.0")
        assert outcome.exported_count == 0
        assert outcome.shards == []
        assert _read_manifest(out_dir) == manifest_before
        assert (out_dir / "README.md").read_bytes() == readme_before

    def test_empty_library_rejected_on_first_export(self, library, tmp_path):
        with pytest.raises(ValueError, match="accepted"):
            export_dataset(library, tmp_path / "ds", license="CC-BY-4.0")

    def test_readme_card_contains_schema_stats_license(self, library, tmp_path):
        out_dir = tmp_path / "ds"
        _ingest(library, "甲", scene="inline-review", language="zh")
        _ingest(
            library,
            "乙",
            scene="from-scratch-timing",
            language="ja",
            task_id="task-2",
            sha256=SHA_B,
        )
        export_dataset(library, out_dir, license="CC0-1.0")
        readme = (out_dir / "README.md").read_text(encoding="utf-8")
        assert "dataset-v1" in readme
        assert "CC0-1.0" in readme
        assert "inline-review 1 条" in readme or "inline-review" in readme
        assert "from-scratch-timing" in readme
        assert "git tag" in readme  # 发行建议


class TestBundleAudio:
    def test_bundle_copies_session_audio_only(self, library, tmp_path):
        session_root = tmp_path / "uploads"
        session = session_root / SHA_A[:16]
        session.mkdir(parents=True)
        (session / "input.wav").write_bytes(b"RIFFfake")
        (session / "notes.txt").write_text("非音频文件", encoding="utf-8")
        _ingest(library, "甲")

        out_dir = tmp_path / "ds"
        outcome = export_dataset(
            library,
            out_dir,
            license="CC-BY-4.0",
            bundle_audio=True,
            session_root=session_root,
        )
        bundled = out_dir / "audio" / SHA_A[:16] / "input.wav"
        assert outcome.bundled_audio == [SHA_A[:16]]
        assert bundled.read_bytes() == b"RIFFfake"
        assert not (out_dir / "audio" / SHA_A[:16] / "notes.txt").exists()

    def test_bundle_missing_audio_warns_but_exports(self, library, tmp_path):
        session_root = tmp_path / "uploads"  # 会话目录不存在
        _ingest(library, "甲")
        out_dir = tmp_path / "ds"
        outcome = export_dataset(
            library,
            out_dir,
            license="CC-BY-4.0",
            bundle_audio=True,
            session_root=session_root,
        )
        assert outcome.bundled_audio == []
        assert outcome.missing_audio == [SHA_A[:16]]
        assert outcome.exported_count == 1  # 引用照写，导出不阻塞
        entry = json.loads(
            (out_dir / "data/shard-0001.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert entry["audio_ref"]["sha256"] == SHA_A

    def test_bundle_off_by_default(self, library, tmp_path):
        _ingest(library, "甲")
        out_dir = tmp_path / "ds"
        export_dataset(library, out_dir, license="CC-BY-4.0")
        assert not (out_dir / "audio").exists()

    def test_bundle_warning_written(self, library, tmp_path):
        _ingest(library, "甲")
        out_dir = tmp_path / "ds"
        export_dataset(library, out_dir, license="CC-BY-4.0", bundle_audio=True)
        readme = (out_dir / "README.md").read_text(encoding="utf-8")
        assert "不建议推送到 git" in readme


class TestCli:
    @pytest.fixture
    def cli_library(self, tmp_path, monkeypatch):
        """把 CLI 的样本库隔离到临时目录"""
        from vocal_subtitle.feedback import sample_manager as module

        monkeypatch.setattr(
            module,
            "FeedbackSampleManager",
            lambda: FeedbackSampleManager(tmp_path / "lib"),
        )
        return tmp_path / "lib"

    def test_cli_export_with_scenario_filter(self, cli_library, tmp_path):
        manager = FeedbackSampleManager(cli_library)
        _ingest(manager, "甲")
        _ingest(
            manager, "乙", scene="from-scratch-timing", task_id="task-2", sha256=SHA_B
        )
        out_dir = tmp_path / "ds"
        runner = CliRunner()
        result = runner.invoke(
            feedback,
            [
                "export-dataset",
                "--out",
                str(out_dir),
                "--scenarios",
                "inline-review",
                "--license",
                "CC-BY-4.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "shard-0001.jsonl" in result.output
        assert "CC-BY-4.0" in result.output
        assert (out_dir / "README.md").exists()
        assert (out_dir / "manifest.json").exists()
        lines = (
            (out_dir / "data/shard-0001.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["scenario"] == "inline-review"
        assert entry["schema"] == "dataset-v1"

    def test_cli_rejects_unknown_scenario(self, cli_library, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            feedback,
            [
                "export-dataset",
                "--out",
                str(tmp_path / "ds"),
                "--scenarios",
                "bogus",
                "--license",
                "CC-BY-4.0",
            ],
        )
        assert result.exit_code == 1
        assert "未知场景标签" in result.output

    def test_cli_reports_empty_library(self, cli_library, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            feedback,
            [
                "export-dataset",
                "--out",
                str(tmp_path / "ds"),
                "--license",
                "CC-BY-4.0",
            ],
        )
        assert result.exit_code == 1
        assert "accepted" in result.output

    def test_cli_bundle_audio_prints_git_warning(self, cli_library, tmp_path):
        manager = FeedbackSampleManager(cli_library)
        _ingest(manager, "甲")
        out_dir = tmp_path / "ds"
        runner = CliRunner()
        result = runner.invoke(
            feedback,
            [
                "export-dataset",
                "--out",
                str(out_dir),
                "--license",
                "CC-BY-4.0",
                "--bundle-audio",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "不建议" in result.output
        assert (
            out_dir / "audio" / SHA_A[:16] / "input.wav"
        ).exists() or "找不到实体文件" in result.output
