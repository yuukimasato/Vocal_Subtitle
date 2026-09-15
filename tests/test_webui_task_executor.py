"""WebUI 任务执行器与事件总线(2026-09-15 重构计划 Task 8)。

- 生命周期事件按 start → progress/… → completion/failure/cancelled 顺序
  发布,sequence 单调递增;
- TaskExecutor 在工作线程执行任务并发布结果事件;
- 取消的排队任务发布 cancelled 事件。
"""

import threading
import time

from vocal_subtitle.webui.task_events import TaskEventPublisher
from vocal_subtitle.webui.task_executor import TaskExecutor


def test_events_publish_in_order_with_monotonic_sequence():
    publisher = TaskEventPublisher()

    publisher.publish("t1", "start")
    publisher.publish("t1", "progress", {"percent": 50})
    publisher.publish("t1", "completion", {"result": "ok"})

    events = publisher.events_for("t1")
    assert [event.kind for event in events] == ["start", "progress", "completion"]
    assert [event.sequence for event in events] == sorted(
        event.sequence for event in events
    )


def test_events_are_isolated_per_task():
    publisher = TaskEventPublisher()

    publisher.publish("a", "start")
    publisher.publish("b", "start")
    publisher.publish("a", "completion", {"result": 1})

    assert [event.kind for event in publisher.events_for("a")] == [
        "start",
        "completion",
    ]
    assert [event.kind for event in publisher.events_for("b")] == ["start"]


def test_unknown_event_kind_rejected():
    import pytest

    publisher = TaskEventPublisher()

    with pytest.raises(ValueError):
        publisher.publish("t1", "mystery")


def test_executor_publishes_start_then_completion():
    executor = TaskExecutor(max_workers=1)
    seen = {}

    def task():
        seen["ran"] = True
        return 42

    future = executor.submit("task-1", task)
    assert future.result(timeout=5.0) == 42

    kinds = [event.kind for event in executor.events_for("task-1")]
    assert kinds == ["start", "completion"]
    assert seen["ran"] is True


def test_executor_publishes_failure_and_reraises():
    executor = TaskExecutor(max_workers=1)

    def bad_task():
        raise RuntimeError("boom")

    future = executor.submit("task-bad", bad_task)

    try:
        future.result(timeout=5.0)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

    kinds = [event.kind for event in executor.events_for("task-bad")]
    assert kinds == ["start", "failure"]
    failure = executor.events_for("task-bad")[-1]
    assert failure.payload["error_type"] == "RuntimeError"
    assert "boom" in failure.payload["error"]


def test_executor_cancels_queued_task():
    executor = TaskExecutor(max_workers=1)
    gate = threading.Event()

    def blocker():
        gate.wait(timeout=5.0)
        return "done"

    executor.submit("blocked", blocker)
    time.sleep(0.05)  # 保证 blocker 已占住唯一工作线程

    def queued():
        return "never"

    future = executor.submit("queued", queued)
    cancelled = executor.cancel("queued", reason="user_abort")

    assert cancelled is True
    assert future.cancelled()
    kinds = [event.kind for event in executor.events_for("queued")]
    assert kinds == ["start", "cancelled"]
    gate.set()
    executor.shutdown(wait=True)


def test_subscriber_receives_events_in_publish_order():
    publisher = TaskEventPublisher()
    received = []
    publisher.subscribe(lambda event: received.append(event.kind))

    publisher.publish("t", "start")
    publisher.publish("t", "completion", {"result": None})

    assert received == ["start", "completion"]
