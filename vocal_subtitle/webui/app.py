"""FastAPI 应用工厂"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .websocket import router as ws_router


def _mark_stale_running_tasks():
    """将启动前残留的 "running" 任务标记为 "failed"。

    服务器重启时会中断所有正在运行的任务，历史记录中此前的
    running 状态若不清理将永远无法完成（孤儿状态）。
    """
    try:
        from ..utils.task_history import TaskHistoryManager

        history = TaskHistoryManager()
        fixed = history.fixup_stale_running_tasks()
        if fixed > 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Marked %d stale running task(s) as failed after restart", fixed)
    except Exception:
        pass


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用

    Returns:
        配置完成的 FastAPI 实例
    """
    app = FastAPI(
        title="Vocal Subtitle",
        description="人声分离 + 字幕生成全链路工具 — Web GUI",
        version="0.1.0",
    )

    # CORS 中间件（允许本地开发）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")

    # 挂载静态文件目录
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # html=True 使访问 / 时自动服务 index.html
    app.mount(
        "/",
        StaticFiles(directory=str(static_dir), html=True),
        name="static",
    )

    # 启动事件：清理因服务器重启而残留的 running 状态任务
    @app.on_event("startup")
    async def _startup_cleanup():
        _mark_stale_running_tasks()

    return app
