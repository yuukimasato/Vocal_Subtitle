"""任务历史管理器

基于 SQLite 的持久化任务历史记录，支持：
- 完整任务状态机 (TASK_STATE_MACHINE.md v1)
  pending → preflight → running → completed | degraded_completed | failed | cancelled
- 错误分类 (TASK_STATE_MACHINE.md §错误分类)
- 基于文件哈希 + 配置哈希的缓存查找
- 分页查询和历史清理
- run_id 关联
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 数据库放置在项目缓存目录下
DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "cache"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "task_history.db"

# ------------------------------------------------------------------
# 任务状态枚举 (TASK_STATE_MACHINE.md)
# ------------------------------------------------------------------

VALID_TASK_STATUSES = frozenset({
    "pending",
    "preflight",
    "running",
    "completed",
    "degraded_completed",
    "failed",
    "cancelled",
})

TERMINAL_STATUSES = frozenset({
    "completed",
    "degraded_completed",
    "failed",
    "cancelled",
})

# 允许的状态转换 (TASK_STATE_MACHINE.md §转换规则)
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending":             frozenset({"preflight", "cancelled"}),
    "preflight":           frozenset({"running", "failed", "cancelled"}),
    "running":             frozenset({"completed", "degraded_completed", "failed", "cancelled"}),
    "completed":           frozenset(),
    "degraded_completed":  frozenset(),
    "failed":              frozenset(),
    "cancelled":           frozenset(),
}

# ------------------------------------------------------------------
# 错误分类 (TASK_STATE_MACHINE.md §错误分类)
# ------------------------------------------------------------------


class ErrorCategory:
    """任务失败的错误分类枚举"""

    INPUT_MISSING = "input_missing"
    FORMAT_UNSUPPORTED = "format_unsupported"
    MODEL_MISSING = "model_missing"
    DEPENDENCY_MISSING = "dependency_missing"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    ENGINE_TIMEOUT = "engine_timeout"
    RECOVERABLE_DEGRADATION = "recoverable_degradation"
    UNRECOVERABLE_FAILURE = "unrecoverable_failure"

    _USER_MESSAGES = {
        INPUT_MISSING: "找不到输入文件：{path}",
        FORMAT_UNSUPPORTED: "不支持的音频格式：{format}，支持 MP3/WAV/M4A/FLAC",
        MODEL_MISSING: "缺少模型：{model}，运行 `vocal-subtitle download-models`",
        DEPENDENCY_MISSING: "缺少系统依赖：{dep}，运行 `bash install.sh`",
        RESOURCE_EXHAUSTED: "资源不足，尝试使用更小的模型或 --cpu",
        ENGINE_TIMEOUT: "{engine} 执行超时（{timeout}s），已降级",
        RECOVERABLE_DEGRADATION: "部分可选功能不可用，继续生成基础字幕",
        UNRECOVERABLE_FAILURE: "处理失败：{reason}",
    }

    @classmethod
    def user_message(cls, category: str, **kwargs) -> str:
        template = cls._USER_MESSAGES.get(category, "处理失败：{reason}")
        return template.format(**kwargs) if kwargs else template


# ------------------------------------------------------------------
# 预检清单 (TASK_STATE_MACHINE.md §预检清单)
# ------------------------------------------------------------------


class PreflightChecklist:
    """预检阶段检查项"""

    checks: list[dict] = [
        {"key": "input_exists", "label": "输入文件存在且可读", "critical": True},
        {"key": "format_supported", "label": "音频格式支持", "critical": True},
        {"key": "separation_available", "label": "分离引擎可用", "critical": False},
        {"key": "vad_available", "label": "VAD 引擎可用", "critical": True},
        {"key": "asr_available", "label": "至少一个 ASR 引擎可用", "critical": True},
        {"key": "disk_space", "label": "磁盘空间充足", "critical": True},
        {"key": "output_writable", "label": "输出目录可写", "critical": True},
    ]


class TaskHistoryManager:
    """持久化任务历史记录管理器

    使用 SQLite 存储任务的完整生命周期信息。
    线程安全（同一实例内串行化写操作）。

    使用示例:
        history = TaskHistoryManager()
        history.create("abc123", "audio.wav", "sha256...", 1024000, "default", config)
        history.update("abc123", status="completed", result_json=r"...")
        tasks = history.list(limit=20)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Args:
            db_path: SQLite 数据库路径，默认 cache/task_history.db
        """
        self._db_path = Path(db_path or DEFAULT_DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（每次新建，线程安全）"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """初始化数据库表结构（含 schema 迁移）"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_history (
                        id              TEXT PRIMARY KEY,
                        run_id          TEXT NOT NULL DEFAULT '',
                        input_file_name TEXT NOT NULL,
                        input_file_hash TEXT NOT NULL DEFAULT '',
                        input_file_size INTEGER NOT NULL DEFAULT 0,
                        profile         TEXT NOT NULL DEFAULT 'default',
                        config_json     TEXT NOT NULL DEFAULT '{}',
                        config_hash     TEXT NOT NULL DEFAULT '',
                        status          TEXT NOT NULL DEFAULT 'pending',
                        error_category  TEXT NOT NULL DEFAULT '',
                        progress_json   TEXT NOT NULL DEFAULT '{}',
                        result_json     TEXT,
                        error           TEXT,
                        total_duration_seconds REAL DEFAULT 0,
                        created_at      TEXT NOT NULL,
                        completed_at    TEXT
                    )
                """)
                # Schema 迁移：添加旧表缺失的列
                self._migrate_add_column(conn, "task_history", "run_id", "TEXT NOT NULL DEFAULT ''")
                self._migrate_add_column(conn, "task_history", "error_category", "TEXT NOT NULL DEFAULT ''")
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_status
                        ON task_history(status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_created
                        ON task_history(created_at DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_hash
                        ON task_history(input_file_hash, config_hash)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_run_id
                        ON task_history(run_id)
                """)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _migrate_add_column(conn, table: str, column: str, column_def: str) -> None:
        """安全添加列（如果不存在）"""
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
        except sqlite3.OperationalError:
            pass  # 列已存在

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        task_id: str,
        file_name: str,
        file_hash: str,
        file_size: int,
        profile: str,
        config,
        *,
        run_id: str = "",
    ) -> None:
        """创建新任务记录（状态初始为 pending）

        Args:
            task_id: 任务唯一 ID
            file_name: 输入文件名
            file_hash: 文件内容 SHA256
            file_size: 文件大小 (bytes)
            profile: 使用的场景模板名称
            config: PipelineConfig 对象
            run_id: 运行 ID（可选，运行开始时关联）
        """
        from dataclasses import asdict

        from .file_hasher import compute_config_hash

        config_dict = asdict(config)
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = compute_config_hash(config)
        now = datetime.now().isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO task_history
                        (id, run_id, input_file_name, input_file_hash, input_file_size,
                         profile, config_json, config_hash, status,
                         created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        task_id, run_id, file_name, file_hash, file_size,
                        profile, config_json, config_hash, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        logger.debug("Task history created: %s (run: %s)", task_id, run_id or "N/A")

    def update(self, task_id: str, **fields) -> None:
        """更新任务字段（状态变更时验证转换合法性）

        Args:
            task_id: 任务 ID
            **fields: 要更新的字段名和值
        """
        if not fields:
            return

        allowed = {
            "status", "run_id", "progress_json", "result_json", "error",
            "error_category", "total_duration_seconds", "completed_at",
            "input_file_hash",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return

        # 状态转换验证
        if "status" in updates:
            new_status = updates["status"]
            if new_status not in VALID_TASK_STATUSES:
                raise ValueError(
                    f"Invalid status: {new_status!r}. Valid: {sorted(VALID_TASK_STATUSES)}"
                )
            current = self.get(task_id)
            if current is not None:
                current_status = current.get("status", "pending")
                allowed_next = ALLOWED_TRANSITIONS.get(current_status, frozenset())
                if new_status != current_status and allowed_next and new_status not in allowed_next:
                    logger.warning(
                        "Status transition %s → %s not in allowed set %s",
                        current_status, new_status, sorted(allowed_next),
                    )

            # 终态自动记录完成时间
            if new_status in TERMINAL_STATUSES and "completed_at" not in updates:
                updates["completed_at"] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    f"UPDATE task_history SET {set_clause} WHERE id = ?",
                    values,
                )
                conn.commit()
            finally:
                conn.close()

    def transition_status(
        self,
        task_id: str,
        new_status: str,
        *,
        error: str = "",
        error_category: str = "",
        run_id: str = "",
    ) -> None:
        """执行任务状态转换（强制验证）。

        Args:
            task_id: 任务 ID
            new_status: 目标状态
            error: 错误消息（failed/cancelled 时）
            error_category: 错误分类（failed 时）
            run_id: 运行 ID（preflight → running 时关联）

        Raises:
            ValueError: 状态转换不合法
        """
        if new_status not in VALID_TASK_STATUSES:
            raise ValueError(
                f"Invalid status: {new_status!r}. Valid: {sorted(VALID_TASK_STATUSES)}"
            )

        updates: dict = {"status": new_status}
        if run_id:
            updates["run_id"] = run_id
        if error:
            updates["error"] = error
        if error_category:
            updates["error_category"] = error_category

        self.update(task_id, **updates)
        logger.info("Task %s: %s → %s", task_id[:20], new_status, "")

    def set_preflight(self, task_id: str) -> None:
        """pending → preflight"""
        self.transition_status(task_id, "preflight")

    def set_running(self, task_id: str, run_id: str = "") -> None:
        """preflight → running"""
        self.transition_status(task_id, "running", run_id=run_id)

    def set_completed(self, task_id: str) -> None:
        """running → completed"""
        self.transition_status(task_id, "completed")

    def set_degraded_completed(
        self, task_id: str, *, reason: str = "", category: str = ""
    ) -> None:
        """running → degraded_completed"""
        self.transition_status(
            task_id, "degraded_completed",
            error=reason, error_category=category or ErrorCategory.RECOVERABLE_DEGRADATION,
        )

    def set_failed(
        self, task_id: str, *, reason: str, category: str = ""
    ) -> None:
        """running → failed"""
        self.transition_status(
            task_id, "failed",
            error=reason, error_category=category or ErrorCategory.UNRECOVERABLE_FAILURE,
        )

    def set_cancelled(self, task_id: str) -> None:
        """pending/preflight/running → cancelled"""
        self.transition_status(task_id, "cancelled", error="用户取消")

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务记录"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM task_history WHERE id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """分页列出任务历史

        Args:
            limit: 每页数量
            offset: 偏移量
            status: 按状态过滤（可选）

        Returns:
            任务记录列表，按创建时间倒序
        """
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM task_history
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM task_history
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete(self, task_id: str) -> bool:
        """删除任务记录

        Returns:
            是否实际删除了记录
        """
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM task_history WHERE id = ?", (task_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def clear(self, older_than_days: Optional[int] = None) -> int:
        """清除历史记录

        Args:
            older_than_days: 只删除 N 天前的记录，None 则清除全部

        Returns:
            删除的记录数
        """
        with self._lock:
            conn = self._get_conn()
            try:
                if older_than_days is not None:
                    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                    cursor = conn.execute(
                        "DELETE FROM task_history WHERE created_at < ?",
                        (cutoff,),
                    )
                else:
                    cursor = conn.execute("DELETE FROM task_history")
                conn.commit()
                count = cursor.rowcount
                logger.info("Cleared %d history records", count)
                return count
            finally:
                conn.close()

    def count(self, status: Optional[str] = None) -> int:
        """获取记录总数

        Args:
            status: 可选的状态过滤
        """
        conn = self._get_conn()
        try:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM task_history WHERE status = ?",
                    (status,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM task_history"
                ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 缓存查找
    # ------------------------------------------------------------------

    def find_by_hash(
        self,
        file_hash: str,
        config_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """通过文件哈希 + 配置哈希查找已完成的任务（缓存命中）

        只返回最近一次成功的记录。

        Args:
            file_hash: 输入文件 SHA256
            config_hash: 配置 SHA256

        Returns:
            匹配的任务记录或 None
        """
        if not file_hash or not config_hash:
            return None

        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT * FROM task_history
                WHERE input_file_hash = ?
                  AND config_hash = ?
                  AND status IN ('completed', 'degraded_completed')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (file_hash, config_hash),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def find_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """通过 run_id 查找任务。

        Args:
            run_id: 运行 ID

        Returns:
            匹配的任务记录或 None
        """
        if not run_id:
            return None

        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM task_history WHERE run_id = ? LIMIT 1",
                (run_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # 数据库信息
    # ------------------------------------------------------------------

    def get_db_size_mb(self) -> float:
        """获取数据库文件大小 (MB)"""
        if self._db_path.exists():
            return self._db_path.stat().st_size / (1024 * 1024)
        return 0.0

    def vacuum(self) -> None:
        """压缩数据库文件"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("VACUUM")
            finally:
                conn.close()

    def fixup_stale_running_tasks(self) -> int:
        """将残留的非终态任务标记为 'failed'。

        服务器重启后，任何处于 'running' 或 'preflight' 状态的任务实际已中断，
        继续保留该状态会导致前端永远显示"处理中"。

        Returns:
            被修复的任务数量
        """
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "UPDATE task_history SET status = 'failed', "
                    "error = 'Server restarted during task execution', "
                    "error_category = 'unrecoverable_failure' "
                    "WHERE status IN ('running', 'preflight')"
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
