"""FastAPI 应用工厂"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from typing import AsyncIterator

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


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup cleanup without FastAPI's deprecated event API."""
    _mark_stale_running_tasks()
    yield


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用

    Returns:
        配置完成的 FastAPI 实例
    """
    app = FastAPI(
        title="Vocal Subtitle",
        description="人声分离 + 字幕生成全链路工具 — Web GUI",
        version="0.2.0",
        lifespan=_lifespan,
    )

    # CORS 中间件（允许本地开发）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # 编辑台(8631)跨源 fetch 字幕文件时需读取下载文件名
        expose_headers=["Content-Disposition"],
    )

    # 注册路由
    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")

    # 挂载静态文件目录
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # html=True 使访问 / 时自动服务 index.html。
    # WebSocket 请求若未匹配任何 /ws 路由会落到根挂载；StaticFiles 只支持
    # http，会抛 AssertionError 变成 500。包装一层把非 http scope 明确拒绝。
    class _HttpOnlyStatic:
        def __init__(self, asgi_app: Any):
            self._asgi_app = asgi_app

        async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
            if scope.get("type") == "websocket":
                from starlette.websockets import WebSocketClose
                await WebSocketClose(code=1008)(scope, receive, send)
                return
            await self._asgi_app(scope, receive, send)

    app.mount(
        "/",
        _HttpOnlyStatic(StaticFiles(directory=str(static_dir), html=True)),
        name="static",
    )

    return app
