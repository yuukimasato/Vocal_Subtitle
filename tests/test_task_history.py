"""TaskHistoryManager 状态机测试

对应 TASK_STATE_MACHINE.md (task-state-v1)。
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from vocal_subtitle.utils.task_history import (
    ALLOWED_TRANSITIONS,
    ErrorCategory,
    PreflightChecklist,
    TaskHistoryManager,
    TERMINAL_STATUSES,
    VALID_TASK_STATUSES,
)


@dataclass
class _FakeConfig:
    """用于测试的最小化 dataclass config，满足 asdict(config) 调用。"""
    mode: str = "offline"
    degradation_mode: str = "full"
    asr_engine: str = "auto"


class TestTaskStateMachine:
    """任务状态机验证"""

    def test_all_statuses_in_valid_set(self):
        """所有 7 个状态都在 VALID_TASK_STATUSES 中"""
        expected = {"pending", "preflight", "running", "completed",
                    "degraded_completed", "failed", "cancelled"}
        assert VALID_TASK_STATUSES == expected

    def test_terminal_statuses_are_subset_of_valid(self):
        """终态是有效状态的子集"""
        assert TERMINAL_STATUSES <= VALID_TASK_STATUSES
        assert len(TERMINAL_STATUSES) == 4
        assert "completed" in TERMINAL_STATUSES
        assert "degraded_completed" in TERMINAL_STATUSES
        assert "failed" in TERMINAL_STATUSES
        assert "cancelled" in TERMINAL_STATUSES

    def test_allowed_transitions_match_spec(self):
        """状态转换表与 TASK_STATE_MACHINE.md 一致"""
        # pending → preflight, cancelled
        assert "preflight" in ALLOWED_TRANSITIONS["pending"]
        assert "cancelled" in ALLOWED_TRANSITIONS["pending"]
        # preflight → running, failed, cancelled
        assert "running" in ALLOWED_TRANSITIONS["preflight"]
        assert "failed" in ALLOWED_TRANSITIONS["preflight"]
        assert "cancelled" in ALLOWED_TRANSITIONS["preflight"]
        # running → completed, degraded_completed, failed, cancelled
        assert "completed" in ALLOWED_TRANSITIONS["running"]
        assert "degraded_completed" in ALLOWED_TRANSITIONS["running"]
        assert "failed" in ALLOWED_TRANSITIONS["running"]
        assert "cancelled" in ALLOWED_TRANSITIONS["running"]
        # 终态无出边
        for terminal in TERMINAL_STATUSES:
            assert ALLOWED_TRANSITIONS[terminal] == frozenset()

    def test_pending_to_preflight_valid(self):
        """pending → preflight 是合法转换"""
        assert "preflight" in ALLOWED_TRANSITIONS["pending"]

    def test_running_to_degraded_completed_valid(self):
        """running → degraded_completed 是合法转换"""
        assert "degraded_completed" in ALLOWED_TRANSITIONS["running"]

    def test_completed_to_anything_invalid(self):
        """终态不能转换到任何其他状态"""
        assert len(ALLOWED_TRANSITIONS["completed"]) == 0


class TestErrorCategory:
    """错误分类测试"""

    def test_all_categories_have_messages(self):
        """所有 8 类错误都有用户消息模板"""
        categories = [
            ErrorCategory.INPUT_MISSING,
            ErrorCategory.FORMAT_UNSUPPORTED,
            ErrorCategory.MODEL_MISSING,
            ErrorCategory.DEPENDENCY_MISSING,
            ErrorCategory.RESOURCE_EXHAUSTED,
            ErrorCategory.ENGINE_TIMEOUT,
            ErrorCategory.RECOVERABLE_DEGRADATION,
            ErrorCategory.UNRECOVERABLE_FAILURE,
        ]
        for cat in categories:
            msg = ErrorCategory.user_message(cat, path="x", format="x",
                                              model="x", dep="x",
                                              engine="x", timeout="60",
                                              reason="x")
            assert msg
            assert len(msg) > 0

    def test_user_message_formatting(self):
        """用户消息模板正确格式化"""
        msg = ErrorCategory.user_message("input_missing", path="/tmp/test.wav")
        assert "/tmp/test.wav" in msg

        msg = ErrorCategory.user_message("engine_timeout", engine="faster-whisper", timeout="60")
        assert "faster-whisper" in msg
        assert "60" in msg

    def test_unknown_category_fallback(self):
        """未知错误分类返回通用消息"""
        msg = ErrorCategory.user_message("nonexistent", reason="test")
        assert msg  # 不会崩溃


class TestPreflightChecklist:
    """预检清单测试"""

    def test_all_seven_checks_present(self):
        """TASK_STATE_MACHINE.md 定义 7 项预检"""
        keys = {c["key"] for c in PreflightChecklist.checks}
        expected = {"input_exists", "format_supported", "separation_available",
                    "vad_available", "asr_available", "disk_space", "output_writable"}
        assert keys == expected

    def test_critical_checks_marked(self):
        """关键检查项正确标记"""
        critical_keys = {c["key"] for c in PreflightChecklist.checks if c["critical"]}
        # VAD、ASR、磁盘、输出、输入、格式是关键项
        assert "vad_available" in critical_keys
        assert "asr_available" in critical_keys
        assert "disk_space" in critical_keys

    def test_separation_is_non_critical(self):
        """分离引擎是可降级的"""
        sep_check = [c for c in PreflightChecklist.checks if c["key"] == "separation_available"][0]
        assert sep_check["critical"] is False


class TestTaskHistoryManager:
    """TaskHistoryManager CRUD 与状态机集成测试"""

    @pytest.fixture
    def db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_tasks.db"

    @pytest.fixture
    def mgr(self, db_path):
        return TaskHistoryManager(db_path=db_path)

    def test_create_task(self, mgr):
        """创建任务并验证初始状态"""
        mgr.create("task-001", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        task = mgr.get("task-001")
        assert task is not None
        assert task["id"] == "task-001"
        assert task["input_file_name"] == "test.wav"
        # 创建时状态为 pending
        assert task["status"] in ("pending", "preflight")

    def test_update_status_transition(self, mgr):
        """正常状态转换 pending → preflight → running → completed"""
        mgr.create("task-002", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-002", status="preflight")
        assert mgr.get("task-002")["status"] == "preflight"

        mgr.update("task-002", status="running")
        assert mgr.get("task-002")["status"] == "running"

        mgr.update("task-002", status="completed")
        assert mgr.get("task-002")["status"] == "completed"

    def test_update_degraded_completion(self, mgr):
        """降级完成状态转换"""
        mgr.create("task-003", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-003", status="preflight")
        mgr.update("task-003", status="running")
        mgr.update("task-003", status="degraded_completed")
        assert mgr.get("task-003")["status"] == "degraded_completed"

    def test_cancel_from_pending(self, mgr):
        """排队中取消"""
        mgr.create("task-004", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-004", status="cancelled")
        assert mgr.get("task-004")["status"] == "cancelled"

    def test_cancel_from_running(self, mgr):
        """运行中取消"""
        mgr.create("task-005", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-005", status="preflight")
        mgr.update("task-005", status="running")
        mgr.update("task-005", status="cancelled")
        assert mgr.get("task-005")["status"] == "cancelled"

    def test_store_result_json(self, mgr):
        """保存结果 JSON"""
        mgr.create("task-006", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        result = {"subtitle_count": 42, "speaker_count": 2}
        mgr.update("task-006", status="completed", result_json=json.dumps(result))
        task = mgr.get("task-006")
        # 验证状态更新
        assert task["status"] == "completed"

    def test_find_by_hash(self, mgr):
        """通过文件哈希 + 配置哈希查找缓存"""
        from vocal_subtitle.utils.file_hasher import compute_config_hash

        config = _FakeConfig()
        config_hash = compute_config_hash(config)

        mgr.create("task-cached", "test.wav", "sha256:abc123", 2048, "default", config)
        mgr.update("task-cached", status="preflight")
        mgr.update("task-cached", status="running")
        mgr.update("task-cached", status="completed",
                   result_json=json.dumps({"subtitle_path": "/tmp/out.srt"}))

        found = mgr.find_by_hash("sha256:abc123", config_hash)
        assert found is not None
        assert found["id"] == "task-cached"

    def test_find_by_hash_miss(self, mgr):
        """缓存未命中返回 None"""
        found = mgr.find_by_hash("sha256:nonexistent", "")
        assert found is None

    def test_list_tasks(self, mgr):
        """列出任务并按时间排序"""
        mgr.create("task-a", "a.wav", "sha256:a", 100, "default", _FakeConfig())
        mgr.create("task-b", "b.wav", "sha256:b", 200, "default", _FakeConfig())
        tasks = mgr.list(limit=10)
        assert len(tasks) >= 2

    def test_fixup_stale_running_tasks(self, mgr):
        """归属进程已死亡的残留 running 任务修复为 failed"""
        import os
        import subprocess
        import sys

        mgr.create("task-stale", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-stale", status="preflight")
        mgr.update("task-stale", status="running")
        # 将归属进程改写为一个已退出的 PID，模拟进程中断后的残留任务
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        with mgr._lock:
            conn = mgr._get_conn()
            try:
                conn.execute(
                    "UPDATE task_history SET owner_pid = ? WHERE id = 'task-stale'",
                    (p.pid,),
                )
                conn.commit()
            finally:
                conn.close()
        # 修复后应为 failed
        mgr.fixup_stale_running_tasks()
        task = mgr.get("task-stale")
        assert task["status"] == "failed"

    def test_fixup_spares_running_task_of_live_process(self, mgr):
        """存活进程（如正在执行的 CLI）拥有的 running 任务不被误标为 failed"""
        mgr.create("task-live", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-live", status="preflight")
        mgr.update("task-live", status="running")
        with mgr._lock:
            conn = mgr._get_conn()
            try:
                conn.execute(
                    "UPDATE task_history SET owner_pid = ? WHERE id = 'task-live'",
                    (os.getpid(),),
                )
                conn.commit()
            finally:
                conn.close()
        assert mgr.fixup_stale_running_tasks() == 0
        task = mgr.get("task-live")
        assert task["status"] == "running"

    def test_fixup_marks_legacy_rows_without_owner(self, mgr):
        """无归属进程（owner_pid=0，旧版本遗留）的 running 任务仍被修复"""
        mgr.create("task-legacy", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-legacy", status="running")
        with mgr._lock:
            conn = mgr._get_conn()
            try:
                conn.execute(
                    "UPDATE task_history SET owner_pid = 0 WHERE id = 'task-legacy'"
                )
                conn.commit()
            finally:
                conn.close()
        assert mgr.fixup_stale_running_tasks() == 1
        assert mgr.get("task-legacy")["status"] == "failed"

    def test_completed_task_clears_stale_error(self, mgr):
        """completed 状态不应残留 error（如重启 fixup 误标后任务仍正常完成）"""
        mgr.create("task-cleared", "test.wav", "sha256:abc", 1024, "default", _FakeConfig())
        mgr.update("task-cleared", status="running")
        # 模拟重启 fixup 误标
        mgr.update("task-cleared", status="failed",
                   error="Server restarted during task execution",
                   error_category="unrecoverable_failure")
        # 任务实际完成（外部进程继续执行到完成）
        mgr.update("task-cleared", status="completed",
                   result_json=json.dumps({"subtitle_count": 3}))
        task = mgr.get("task-cleared")
        assert task["status"] == "completed"
        assert not task["error"]
        assert not task["error_category"]
