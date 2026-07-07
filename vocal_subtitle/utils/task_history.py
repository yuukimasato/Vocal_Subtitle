"""任务历史管理器

基于 SQLite 的持久化任务历史记录，支持：
- 任务生命周期跟踪 (pending → running → completed/failed)
- 基于文件哈希 + 配置哈希的缓存查找
- 分页查询和历史清理
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
        """初始化数据库表结构"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_history (
                        id              TEXT PRIMARY KEY,
                        input_file_name TEXT NOT NULL,
                        input_file_hash TEXT NOT NULL DEFAULT '',
                        input_file_size INTEGER NOT NULL DEFAULT 0,
                        profile         TEXT NOT NULL DEFAULT 'default',
                        config_json     TEXT NOT NULL DEFAULT '{}',
                        config_hash     TEXT NOT NULL DEFAULT '',
                        status          TEXT NOT NULL DEFAULT 'pending',
                        progress_json   TEXT NOT NULL DEFAULT '{}',
                        result_json     TEXT,
                        error           TEXT,
                        total_duration_seconds REAL DEFAULT 0,
                        created_at      TEXT NOT NULL,
                        completed_at    TEXT
                    )
                """)
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
                conn.commit()
            finally:
                conn.close()

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
    ) -> None:
        """创建新任务记录

        Args:
            task_id: 任务唯一 ID
            file_name: 输入文件名
            file_hash: 文件内容 SHA256
            file_size: 文件大小 (bytes)
            profile: 使用的场景模板名称
            config: PipelineConfig 对象
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
                        (id, input_file_name, input_file_hash, input_file_size,
                         profile, config_json, config_hash, status,
                         created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        task_id, file_name, file_hash, file_size,
                        profile, config_json, config_hash, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        logger.debug("Task history created: %s", task_id)

    def update(self, task_id: str, **fields) -> None:
        """更新任务字段

        Args:
            task_id: 任务 ID
            **fields: 要更新的字段名和值
        """
        if not fields:
            return

        allowed = {
            "status", "progress_json", "result_json", "error",
            "total_duration_seconds", "completed_at", "input_file_hash",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return

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
                  AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (file_hash, config_hash),
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
