"""WebUI 任务事件总线(2026-09-15 重构计划 Task 8)。

任务生命周期事件(start / progress / degradation / failure / cancellation /
completion)的有序发布:每个事件带单调递增 sequence,可按任务检索;
订阅者按发布顺序收到事件。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

EVENT_KINDS = (
    "start",
    "progress",
    "degradation",
    "failure",
    "cancelled",
    "completion",
)


@dataclass(frozen=True)
class TaskEvent:
    """一条任务生命周期事件。"""

    task_id: str
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unsupported task event kind: {self.kind}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "sequence": self.sequence,
            **self.payload,
        }


class TaskEventPublisher:
    """线程安全的有序事件发布器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._events: List[TaskEvent] = []
        self._subscribers: List[Callable[[TaskEvent], None]] = []

    def publish(
        self,
        task_id: str,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TaskEvent:
        event = TaskEvent(
            task_id=task_id,
            kind=kind,
            payload=dict(payload or {}),
        )
        with self._lock:
            self._sequence += 1
            object.__setattr__(event, "sequence", self._sequence)
            self._events.append(event)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                # 订阅者故障不得影响事件流本身。
                pass
        return event

    def subscribe(self, callback: Callable[[TaskEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def events_for(self, task_id: str) -> Tuple[TaskEvent, ...]:
        with self._lock:
            return tuple(
                event for event in self._events if event.task_id == task_id
            )

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


__all__ = ["TaskEvent", "TaskEventPublisher", "EVENT_KINDS"]
