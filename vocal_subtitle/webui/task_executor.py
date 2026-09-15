"""WebUI 任务执行器(2026-09-15 重构计划 Task 8)。

把后台任务执行与生命周期事件发布从路由/广播代码中分离:
``TaskExecutor.submit`` 在工作线程执行任务函数,按序发布
start → completion / failure / cancelled 事件;进度与降级事件由任务
函数通过 ``publish`` 上报。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .task_events import TaskEventPublisher


class TaskExecutor:
    """后台任务执行 + 有序事件发布。"""

    def __init__(
        self,
        publisher: TaskEventPublisher | None = None,
        *,
        max_workers: int = 2,
    ) -> None:
        self.publisher = publisher or TaskEventPublisher()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="webui-task",
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(
        self, task_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future:
        """提交一个任务;start 事件同步发布,结果/失败在线程内发布。"""

        def _wrapped() -> Any:
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self.publisher.publish(
                    task_id,
                    "failure",
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            self.publisher.publish(task_id, "completion", {"result": result})
            return result

        self.publisher.publish(task_id, "start")
        future = self._executor.submit(_wrapped)
        with self._lock:
            self._futures[task_id] = future
        return future

    def cancel(self, task_id: str, reason: str = "cancelled") -> bool:
        """取消排队任务;已运行任务依赖其协作取消逻辑。"""
        with self._lock:
            future = self._futures.get(task_id)
        cancelled = bool(future is not None and future.cancel())
        if cancelled:
            self.publisher.publish(task_id, "cancelled", {"reason": reason})
        return cancelled

    def events_for(self, task_id: str):
        return self.publisher.events_for(task_id)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


__all__ = ["TaskExecutor"]
