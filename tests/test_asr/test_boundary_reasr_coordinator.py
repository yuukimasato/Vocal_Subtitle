"""边界冗余识别统一走 WindowExecutionCoordinator(重构计划 Task 4)。

- 每个窗口经 coordinator 并发执行(不再自建线程池);
- 失败窗口按边界归组为 success=False 的结果(原语义);
- coordinator 诊断暴露活动/超时/取消/失败计数。
"""

from types import SimpleNamespace

from vocal_subtitle.asr.boundary_reasr import (
    SlidingWindowConfig,
    SlidingWindowReASR,
)
from vocal_subtitle.asr.window_execution import WindowExecutionCoordinator


class StubEngine:
    """记录窗口调用的假 ASR 引擎。"""

    def __init__(self, fail_indices=()):
        self.calls = []
        self.fail_indices = set(fail_indices)

    def transcribe(self, audio, sample_rate, language=None, **kwargs):
        # SlidingWindowReASR._transcribe_window 会切片后调用引擎;
        # 这里用调用次数区分窗口。
        self.calls.append(sample_rate)
        if len(self.calls) in self.fail_indices:
            raise RuntimeError("stub engine failure")
        return []


def _make_reasr(monkeypatch, fail_call_indices=()):
    engine = StubEngine(fail_call_indices)
    reasr = SlidingWindowReASR(SlidingWindowConfig(), engine, cache=None)
    # _transcribe_window 内部会调用引擎.transcribe;失败场景由引擎抛错模拟。
    return reasr, engine


def test_process_boundaries_uses_shared_coordinator_diagnostics(monkeypatch):
    reasr, engine = _make_reasr(monkeypatch)
    segments = [
        SimpleNamespace(start=0.0, end=2.0),
        SimpleNamespace(start=2.0, end=4.0),
        SimpleNamespace(start=4.0, end=6.0),
    ]
    asr_results = [[], [], []]

    used = {}
    real_execute = WindowExecutionCoordinator.execute

    def spy_execute(self, *args, **kwargs):
        used["coordinator"] = True
        used["max_workers"] = self.max_workers
        return real_execute(self, *args, **kwargs)

    monkeypatch.setattr(WindowExecutionCoordinator, "execute", spy_execute)

    results = reasr.process_boundaries(
        low_conf_indices=[0, 1],
        segments=segments,
        asr_results=asr_results,
        audio=[0.0] * 100,
        sample_rate=10,
        total_duration=6.0,
    )

    assert used.get("coordinator") is True
    assert results.keys() >= {0, 1}
    # 诊断暴露计数。
    diagnostics = getattr(reasr, "last_execution_diagnostics", {})
    assert "failed_count" in diagnostics
    assert "cancelled_count" in diagnostics
    assert "timeout_count" in diagnostics
    assert diagnostics["window_count"] >= 1


def test_window_failure_is_scoped_and_reported(monkeypatch):
    reasr, engine = _make_reasr(monkeypatch, fail_call_indices={1})
    segments = [
        SimpleNamespace(start=0.0, end=2.0),
        SimpleNamespace(start=2.0, end=4.0),
    ]
    asr_results = [[], []]

    results = reasr.process_boundaries(
        low_conf_indices=[0],
        segments=segments,
        asr_results=asr_results,
        audio=[0.0] * 100,
        sample_rate=10,
        total_duration=4.0,
    )

    assert 0 in results
    boundary = results[0]
    window_results = boundary.windows
    assert window_results, "窗口结果必须保留(含失败)"
    assert any(not item.success for item in window_results), (
        "失败窗口必须以 success=False 报告,不得静默丢弃"
    )
    # _transcribe_window 自带异常捕获,失败在窗口级收敛为 success=False;
    # coordinator 层计数仍可用于执行器级故障观测。
    diagnostics = reasr.last_execution_diagnostics
    assert "failed_count" in diagnostics
