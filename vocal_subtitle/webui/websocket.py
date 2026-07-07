"""WebSocket 管理器 — 实时进度推送

每个 Pipeline 任务对应一个 WebSocket 连接，
ProgressManager 的回调事件通过 WebSocket 推送到前端。
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSocketManager:
    """WebSocket 连接管理器

    管理 task_id → [WebSocket] 的映射，
    支持同一任务的多个连接和广播。
    """

    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self._task_states: Dict[str, Dict[str, Any]] = {}
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_main_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置主事件循环（FastAPI/uvicorn 的 loop）。

        必须在主线程的异步上下文中调用。
        """
        self._main_loop = loop

    def _get_main_loop(self) -> asyncio.AbstractEventLoop:
        """获取主事件循环引用"""
        if self._main_loop is not None:
            return self._main_loop
        # 降级：尝试获取调用线程的 loop（仅在主线程调用时正确）
        return asyncio.get_event_loop()

    async def connect(self, websocket: WebSocket, task_id: str) -> None:
        """接受 WebSocket 连接"""
        await websocket.accept()
        self._connections[task_id].append(websocket)
        # 捕获主事件循环（FastAPI 的 running loop），用于跨线程广播
        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()
        logger.info("WebSocket connected for task: %s (total: %d)",
                     task_id, len(self._connections[task_id]))

    def disconnect(self, websocket: WebSocket, task_id: str) -> None:
        """移除 WebSocket 连接"""
        if task_id in self._connections:
            try:
                self._connections[task_id].remove(websocket)
            except ValueError:
                pass
            if not self._connections[task_id]:
                del self._connections[task_id]
        logger.info("WebSocket disconnected for task: %s", task_id)

    async def broadcast(self, task_id: str, message: Dict[str, Any]) -> None:
        """向所有监听该任务的 WebSocket 广播消息"""
        dead: List[WebSocket] = []
        for ws in self._connections.get(task_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, task_id)

    def create_progress_callback(self, task_id: str) -> Callable:
        """创建同步回调函数，桥接到 WebSocket 广播

        Pipeline 在后台线程中运行，调用此回调时通过
        asyncio.run_coroutine_threadsafe 安全地将消息
        发送到 FastAPI 事件循环中的 WebSocket。

        Args:
            task_id: 任务 ID

        Returns:
            同步回调函数 (event: dict) -> None
        """
        # 必须使用 FastAPI 主事件循环，而非后台线程的 loop
        loop = self._get_main_loop()

        def callback(event: Dict[str, Any]) -> None:
            """同步回调 — 从后台线程调用"""
            try:
                asyncio.run_coroutine_threadsafe(
                    self.broadcast(task_id, event), loop
                )
            except Exception as e:
                logger.error("Failed to broadcast progress: %s", e)

        return callback

    def store_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """存储任务结果"""
        self._task_states[task_id] = result

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """获取任务结果"""
        return self._task_states.get(task_id, {})

    def broadcast_from_thread(self, task_id: str, message: Dict[str, Any]) -> None:
        """从任意线程向主事件循环发送广播（fire-and-forget）

        用于后台线程向 WebSocket 客户端推送消息。
        不等待广播完成，异常由 broadcast() 内部静默处理。
        """
        loop = self._get_main_loop()
        asyncio.run_coroutine_threadsafe(
            self.broadcast(task_id, message), loop
        )


# 全局单例
ws_manager = WebSocketManager()


# ---------------------------------------------------------------------------
# WebSocket 路由
# ---------------------------------------------------------------------------


@router.websocket("/tasks/{task_id}")
async def task_progress_websocket(websocket: WebSocket, task_id: str):
    """任务进度 WebSocket 端点

    前端连接到此端点以接收 Pipeline 执行的实时进度更新。
    """
    await ws_manager.connect(websocket, task_id)
    try:
        # 保持连接直到客户端断开
        while True:
            # 接收客户端消息（心跳或控制指令）
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, task_id)
    except Exception as e:
        logger.error("WebSocket error for task %s: %s", task_id, e)
        ws_manager.disconnect(websocket, task_id)
