"""journal sink 保留策略测试（D30）：消费后归档/删除、TTL 兜底清理、幂等。

目录约定：cache/journal_sink/ 顶层是「待消费」文件，consumed/ 存已消费归档；
TTL 清理只动顶层超期 *.jsonl，consumed/ 内的归档不受二次清理。
"""

import json
import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from vocal_subtitle.cli import main
from vocal_subtitle.config import FeedbackConfig
from vocal_subtitle.config.loader import ConfigLoader
from vocal_subtitle.feedback.journal_retention import (
    CONSUMED_SUBDIR,
    RETENTION_ARCHIVE,
    RETENTION_DELETE,
    apply_retention,
    cleanup_expired,
    consumed_dir,
)


def _make_sink_file(sink_dir: Path, name: str = "s-1.jsonl", content: str | None = None) -> Path:
    path = sink_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        content or '{"schema": "edit-journal-v1", "type": "header", "session_id": "s-1"}\n',
        encoding="utf-8",
    )
    return path


def _age(path: Path, days: float) -> None:
    """把文件 mtime 往前拨 days 天"""
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


# ---------------------------------------------------------------------------
# 消费后保留策略（归档 / 删除）
# ---------------------------------------------------------------------------

class TestApplyRetention:
    def test_archive_moves_to_consumed(self, tmp_path):
        sink = tmp_path / "journal_sink"
        path = _make_sink_file(sink)
        result = apply_retention(path, policy=RETENTION_ARCHIVE, sink_dir=sink)
        assert result.action == "archived"
        assert not path.exists()
        archived = consumed_dir(sink) / "s-1.jsonl"
        assert archived.exists()
        assert "edit-journal-v1" in archived.read_text(encoding="utf-8")
        assert result.detail == str(archived)

    def test_delete_removes_file(self, tmp_path):
        sink = tmp_path / "journal_sink"
        path = _make_sink_file(sink)
        result = apply_retention(path, policy=RETENTION_DELETE, sink_dir=sink)
        assert result.action == "deleted"
        assert not path.exists()
        assert not consumed_dir(sink).exists()

    def test_missing_source_is_idempotent(self, tmp_path):
        """源文件已被归档/删除时重复执行不报错（幂等）"""
        sink = tmp_path / "journal_sink"
        result = apply_retention(sink / "s-1.jsonl", policy=RETENTION_ARCHIVE, sink_dir=sink)
        assert result.action == "skipped"
        assert result.reason == "missing"

    def test_repeated_archive_overwrites(self, tmp_path):
        """同 session 重新上送后再次消费：覆盖旧归档而非报错（幂等）"""
        sink = tmp_path / "journal_sink"
        first = _make_sink_file(sink)
        apply_retention(first, policy=RETENTION_ARCHIVE, sink_dir=sink)
        second = _make_sink_file(sink, content='{"schema": "edit-journal-v1", "type": "header", "session_id": "s-1", "v": 2}\n')
        result = apply_retention(second, policy=RETENTION_ARCHIVE, sink_dir=sink)
        assert result.action == "archived"
        archived = consumed_dir(sink) / "s-1.jsonl"
        assert '"v": 2' in archived.read_text(encoding="utf-8")

    def test_outside_sink_untouched(self, tmp_path):
        """sink 外的用户文件（本地 .journal.jsonl 导出）不参与保留策略"""
        outside = tmp_path / "exports" / "demo.journal.jsonl"
        outside.parent.mkdir(parents=True)
        outside.write_text("{}", encoding="utf-8")
        result = apply_retention(outside, policy=RETENTION_DELETE, sink_dir=tmp_path / "journal_sink")
        assert result.action == "skipped"
        assert result.reason == "outside-sink"
        assert outside.exists()

    def test_consumed_file_not_reprocessed(self, tmp_path):
        """consumed/ 内的归档不再被归档/删除逻辑处理"""
        sink = tmp_path / "journal_sink"
        archived = _make_sink_file(sink, name=f"{CONSUMED_SUBDIR}/s-1.jsonl")
        result = apply_retention(archived, policy=RETENTION_DELETE, sink_dir=sink)
        assert result.action == "skipped"
        assert archived.exists()

    def test_unknown_policy_raises(self, tmp_path):
        sink = tmp_path / "journal_sink"
        path = _make_sink_file(sink)
        with pytest.raises(ValueError):
            apply_retention(path, policy="shred", sink_dir=sink)
        assert path.exists()  # 非法策略不动文件


# ---------------------------------------------------------------------------
# TTL 兜底清理（只清从未被消费的超期文件）
# ---------------------------------------------------------------------------

class TestCleanupExpired:
    def test_expired_file_removed(self, tmp_path):
        sink = tmp_path / "journal_sink"
        expired = _make_sink_file(sink, name="old.jsonl")
        fresh = _make_sink_file(sink, name="new.jsonl")
        _age(expired, days=31)
        removed = cleanup_expired(sink, ttl_days=30)
        assert removed == [expired]
        assert not expired.exists()
        assert fresh.exists()

    def test_fresh_file_untouched(self, tmp_path):
        """未到期文件不受清理影响"""
        sink = tmp_path / "journal_sink"
        fresh = _make_sink_file(sink)
        _age(fresh, days=29)
        assert cleanup_expired(sink, ttl_days=30) == []
        assert fresh.exists()

    def test_consumed_archive_not_cleaned(self, tmp_path):
        """consumed/ 内的归档即使超期也不被 TTL 二次清理"""
        sink = tmp_path / "journal_sink"
        archived = _make_sink_file(sink, name=f"{CONSUMED_SUBDIR}/s-old.jsonl")
        _age(archived, days=400)
        assert cleanup_expired(sink, ttl_days=30) == []
        assert archived.exists()

    def test_non_jsonl_untouched(self, tmp_path):
        """非 .jsonl 文件不清理（防御性：sink 目录里可能混有其他产物）"""
        sink = tmp_path / "journal_sink"
        note = sink / "README.txt"
        sink.mkdir(parents=True, exist_ok=True)
        note.write_text("keep me", encoding="utf-8")
        _age(note, days=400)
        assert cleanup_expired(sink, ttl_days=30) == []
        assert note.exists()

    def test_idempotent_second_run(self, tmp_path):
        """重复运行幂等：第二次无可清理文件"""
        sink = tmp_path / "journal_sink"
        expired = _make_sink_file(sink)
        _age(expired, days=31)
        assert len(cleanup_expired(sink, ttl_days=30)) == 1
        assert cleanup_expired(sink, ttl_days=30) == []

    def test_ttl_zero_disables(self, tmp_path):
        """ttl_days <= 0 视为禁用（防配置错误清空 sink）"""
        sink = tmp_path / "journal_sink"
        expired = _make_sink_file(sink)
        _age(expired, days=400)
        assert cleanup_expired(sink, ttl_days=0) == []
        assert expired.exists()

    def test_missing_dir_returns_empty(self, tmp_path):
        assert cleanup_expired(tmp_path / "nope", ttl_days=30) == []

    def test_explicit_now_parameter(self, tmp_path):
        """now 可注入（清理例程测试/调度用）：now 推进一年后未到期文件超期"""
        from datetime import datetime, timedelta, timezone

        sink = tmp_path / "journal_sink"
        fresh = _make_sink_file(sink)
        assert cleanup_expired(sink, ttl_days=30) == []  # 当前时间未到期
        shifted = datetime.now(timezone.utc) + timedelta(days=365)
        assert cleanup_expired(sink, ttl_days=30, now=shifted) == [fresh]


# ---------------------------------------------------------------------------
# feedback 配置节（默认 30 天，不写死）
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = FeedbackConfig()
        assert cfg.journal_sink_retention == "archive"
        assert cfg.journal_sink_ttl_days == 30

    def test_yaml_overrides(self):
        """feedback 配置节可覆盖两个数值"""
        cfg = ConfigLoader._parse_config({
            "feedback": {"journal_sink_retention": "delete", "journal_sink_ttl_days": 7},
        })
        assert cfg.feedback.journal_sink_retention == "delete"
        assert cfg.feedback.journal_sink_ttl_days == 7


# ---------------------------------------------------------------------------
# CLI：ingest-journal 消费后清理 + cleanup-journal-sink 子命令
# ---------------------------------------------------------------------------

def _journal_event(session_id="s-cli", seq=0):
    return {
        "schema": "edit-journal-v1",
        "type": "event",
        "session_id": session_id,
        "seq": seq,
        "ts": "2026-09-10T00:00:00Z",
        "actor": "human",
        "command": "updateCueTimes",
        "diff": [{"op": "modify", "id": "cue-1",
                  "changes": [{"field": "start", "before": 1.0, "after": 0.99}]}],
        "context": {},
    }


def _original_srt(path: Path) -> None:
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHello world\n",
        encoding="utf-8",
    )


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """隔离 CLI 外部副作用：sink 目录与 D2 样本库都指向临时目录"""
    import vocal_subtitle.cli_commands.feedback_commands as fb_cmds
    from vocal_subtitle.feedback import sample_manager

    sink = tmp_path / "journal_sink"
    monkeypatch.setattr(fb_cmds, "_default_sink_dir", lambda: sink)
    monkeypatch.setattr(sample_manager.FeedbackSampleManager, "ingest",
                        lambda self, **kwargs: None)
    monkeypatch.setattr(sample_manager.FeedbackSampleManager, "count_by_status",
                        lambda self, status=None: 0)
    return sink


class TestCli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def _invoke_ingest(self, runner, sink, *extra):
        journal = _make_sink_file(sink, name="s-cli.jsonl")
        lines = [
            json.dumps({"schema": "edit-journal-v1", "type": "header", "session_id": "s-cli"}),
            json.dumps(_journal_event()),
        ]
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        original = sink / "original.srt"
        _original_srt(original)
        result = runner.invoke(
            main,
            ["feedback", "ingest-journal", str(journal), "--original", str(original), "--no-apply", *extra],
        )
        return result, journal

    def test_ingest_archives_consumed_sink_file(self, runner, cli_env):
        result, journal = self._invoke_ingest(runner, cli_env)
        assert result.exit_code == 0, result.output
        assert not journal.exists()
        archived = consumed_dir(cli_env) / "s-cli.jsonl"
        assert archived.exists()
        assert "已归档已消费日志" in result.output
        # 原始字幕与未消费文件不受影响
        assert (cli_env / "original.srt").exists()

    def test_ingest_delete_policy_via_config(self, runner, cli_env, monkeypatch):
        """策略可配置：journal_sink_retention=delete 时消费后直接删除"""
        import dataclasses

        import vocal_subtitle.cli_commands.feedback_commands as fb_cmds

        # 重新装饰以生成新 __init__（仅赋类属性会被继承的 dataclass __init__ 默认值覆盖）
        @dataclasses.dataclass
        class DeleteCfg(FeedbackConfig):
            journal_sink_retention: str = "delete"

        # 策略经由 _load_feedback_config（default 模板 feedback 节）读取 — 打到该接缝
        monkeypatch.setattr(fb_cmds, "_load_feedback_config", lambda: DeleteCfg())
        result, journal = self._invoke_ingest(runner, cli_env)
        assert result.exit_code == 0, result.output
        assert not journal.exists()
        assert not consumed_dir(cli_env).exists()
        assert "已删除已消费日志" in result.output

    def test_unconsumed_journal_not_retained(self, runner, cli_env, monkeypatch):
        """部分失败保护（B1 修复锁定）：重放被跳过的日志不算已消费，保留策略不动它"""
        import dataclasses

        import vocal_subtitle.cli_commands.feedback_commands as fb_cmds

        @dataclasses.dataclass
        class DeleteCfg(FeedbackConfig):
            journal_sink_retention: str = "delete"

        monkeypatch.setattr(fb_cmds, "_load_feedback_config", lambda: DeleteCfg())
        cli_env.mkdir(parents=True, exist_ok=True)
        # good.journal.jsonl 配对 good.srt（自动发现成功）；bad.journal.jsonl 无配对（重放跳过）
        good = cli_env / "good.journal.jsonl"
        good.write_text("\n".join([
            json.dumps({"schema": "edit-journal-v1", "type": "header", "session_id": "s-good"}),
            json.dumps(_journal_event()),
        ]) + "\n", encoding="utf-8")
        _original_srt(cli_env / "good.srt")
        bad = cli_env / "bad.journal.jsonl"
        bad.write_text("\n".join([
            json.dumps({"schema": "edit-journal-v1", "type": "header", "session_id": "s-bad"}),
            json.dumps(_journal_event()),
        ]) + "\n", encoding="utf-8")
        result = runner.invoke(
            main, ["feedback", "ingest-journal", str(good), str(bad), "--no-apply"],
        )
        assert result.exit_code == 0, result.output
        # 成功消费的按策略删除；未消费的原地保留并明确计数
        assert not good.exists()
        assert bad.exists()
        assert "已删除已消费日志" in result.output
        assert "未成功消费" in result.output

    def test_ingest_dry_run_keeps_sink_file(self, runner, cli_env):
        """dry-run 未真正消费，源文件原地保留"""
        result, journal = self._invoke_ingest(runner, cli_env, "--dry-run")
        assert result.exit_code == 0, result.output
        assert journal.exists()
        assert not consumed_dir(cli_env).exists()

    def test_cleanup_removes_expired_only(self, runner, tmp_path, monkeypatch):
        import vocal_subtitle.cli_commands.feedback_commands as fb_cmds

        sink = tmp_path / "journal_sink"
        monkeypatch.setattr(fb_cmds, "_default_sink_dir", lambda: sink)
        expired = _make_sink_file(sink, name="old.jsonl")
        fresh = _make_sink_file(sink, name="new.jsonl")
        _age(expired, days=31)

        result = runner.invoke(main, ["feedback", "cleanup-journal-sink", "--sink-dir", str(sink)])
        assert result.exit_code == 0, result.output
        assert not expired.exists()
        assert fresh.exists()
        assert "已清理 1 个超期未消费的 sink 文件" in result.output

        # 重复运行幂等
        again = runner.invoke(main, ["feedback", "cleanup-journal-sink", "--sink-dir", str(sink)])
        assert again.exit_code == 0
        assert "无超期未消费的 sink 文件" in again.output

    def test_cleanup_spares_consumed_archive(self, runner, tmp_path):
        sink = tmp_path / "journal_sink"
        archived = _make_sink_file(sink, name=f"{CONSUMED_SUBDIR}/s-old.jsonl")
        _age(archived, days=400)
        result = runner.invoke(main, ["feedback", "cleanup-journal-sink", "--sink-dir", str(sink)])
        assert result.exit_code == 0, result.output
        assert archived.exists()
        assert "不参与 TTL 清理" in result.output

    def test_cleanup_ttl_days_zero_disables(self, runner, tmp_path):
        sink = tmp_path / "journal_sink"
        expired = _make_sink_file(sink)
        _age(expired, days=400)
        result = runner.invoke(
            main, ["feedback", "cleanup-journal-sink", "--sink-dir", str(sink), "--ttl-days", "0"],
        )
        assert result.exit_code == 0, result.output
        assert expired.exists()
        assert "已禁用" in result.output

    def test_cleanup_ttl_days_override(self, runner, tmp_path):
        """--ttl-days 覆盖配置值（默认 30 天不写死）"""
        sink = tmp_path / "journal_sink"
        recent = _make_sink_file(sink)
        _age(recent, days=2)
        result = runner.invoke(
            main, ["feedback", "cleanup-journal-sink", "--sink-dir", str(sink), "--ttl-days", "1"],
        )
        assert result.exit_code == 0, result.output
        assert not recent.exists()
