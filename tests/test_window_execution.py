from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from vocal_subtitle.asr.evidence import CandidateEvidence
from vocal_subtitle.asr.window_execution import WindowExecutionCoordinator


def _window(identifier: str, start: float) -> SimpleNamespace:
    return SimpleNamespace(id=identifier, start=start, end=start + 0.5)


def test_window_executor_respects_worker_limit_and_collects_candidates():
    lock = threading.Lock()
    active = 0
    peak = 0

    class Engine:
        name = "fake-secondary"

        def review(self, audio, sample_rate, window, *, language=None, cancellation_token=None):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return [CandidateEvidence(
                id=f"candidate-{window.id}",
                source="qwen",
                text="hello",
                start=window.start,
                end=window.end,
                window_id=window.id,
            )]

    candidates, diagnostics = WindowExecutionCoordinator(max_workers=2).execute(
        None,
        16000,
        [_window("a", 0.0), _window("b", 1.0), _window("c", 2.0)],
        Engine(),
    )

    assert len(candidates) == 3
    assert peak == 2
    assert diagnostics["status_counts"] == {"ok": 3}


def test_window_executor_timeout_is_window_scoped_and_cancellable():
    started = threading.Event()

    class SlowEngine:
        name = "slow-secondary"

        def review(self, audio, sample_rate, window, *, language=None, cancellation_token=None):
            started.set()
            while True:
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                time.sleep(0.005)

    coordinator = WindowExecutionCoordinator(max_workers=1, timeout_seconds=0.03)
    candidates, diagnostics = coordinator.execute(
        None,
        16000,
        [_window("timeout", 0.0)],
        SlowEngine(),
    )

    assert candidates == []
    assert started.is_set()
    assert diagnostics["status_counts"] == {"timeout": 1}
    assert diagnostics["cancelled"] is True
