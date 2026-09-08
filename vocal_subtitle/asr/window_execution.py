"""Bounded concurrent execution for review windows.

The executor deliberately keeps model adapters synchronous.  ASR SDKs are
often synchronous and some of them release the GIL, so a small thread pool is
the least invasive way to add bounded parallelism while retaining the public
port contract.
"""

from __future__ import annotations

import inspect
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .review_engines import ReviewEngineUnavailable


class CancellationToken:
    """Cooperative cancellation signal shared by one review batch."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._reason = str(reason or "cancelled")
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ReviewWindowCancelled(self.reason)


class ReviewWindowCancelled(RuntimeError):
    """Raised by an adapter that cooperatively stops a window."""


@dataclass(frozen=True)
class WindowExecutionResult:
    window_id: str
    candidates: tuple[Any, ...] = ()
    status: str = "ok"
    reason: Optional[str] = None
    error: Optional[str] = None
    wall_time_seconds: float = 0.0
    resources: dict[str, Any] = field(default_factory=dict)


class WindowExecutionCoordinator:
    """Execute bounded windows with explicit concurrency and cancellation."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        if isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        self.max_workers = int(max_workers)
        self.timeout_seconds = timeout_seconds
        self._active_token: Optional[CancellationToken] = None
        self._active_futures: set[Future[Any]] = set()
        self._lock = threading.Lock()

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancel queued windows and request cooperative stop for running ones."""
        with self._lock:
            token = self._active_token
            futures = tuple(self._active_futures)
        if token is not None:
            token.cancel(reason)
        for future in futures:
            future.cancel()

    def execute(
        self,
        audio: Any,
        sample_rate: int,
        windows: Sequence[Any],
        engine: Any,
        *,
        language: Optional[str] = None,
        token: Optional[CancellationToken] = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        selected_token = token or CancellationToken()
        executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="subtitle-review",
        )
        futures: dict[Future[Any], Any] = {}
        started = time.perf_counter()
        with self._lock:
            self._active_token = selected_token
            self._active_futures = set()
        try:
            for window in windows:
                if selected_token.cancelled:
                    break
                future = executor.submit(
                    self._invoke,
                    engine,
                    audio,
                    sample_rate,
                    window,
                    language,
                    selected_token,
                )
                futures[future] = window
                with self._lock:
                    self._active_futures.add(future)

            results: list[WindowExecutionResult] = []
            for future, window in futures.items():
                item_started = time.perf_counter()
                if selected_token.cancelled and not future.done():
                    future.cancel()
                    results.append(WindowExecutionResult(
                        window_id=window.id,
                        status="cancelled",
                        reason=selected_token.reason,
                    ))
                    continue
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    if isinstance(result, WindowExecutionResult):
                        results.append(result)
                    else:
                        results.append(WindowExecutionResult(
                            window_id=window.id,
                            candidates=tuple(result or ()),
                            wall_time_seconds=time.perf_counter() - item_started,
                        ))
                except TimeoutError:
                    selected_token.cancel("window_timeout")
                    future.cancel()
                    results.append(WindowExecutionResult(
                        window_id=window.id,
                        status="timeout",
                        reason="window_timeout",
                        wall_time_seconds=time.perf_counter() - item_started,
                    ))
                except ReviewWindowCancelled as exc:
                    results.append(WindowExecutionResult(
                        window_id=window.id,
                        status="cancelled",
                        reason=str(exc) or selected_token.reason,
                        wall_time_seconds=time.perf_counter() - item_started,
                    ))
                except ReviewEngineUnavailable as exc:
                    results.append(WindowExecutionResult(
                        window_id=window.id,
                        status="unavailable",
                        reason=exc.reason,
                        error=exc.detail or str(exc),
                        wall_time_seconds=time.perf_counter() - item_started,
                    ))
                except Exception as exc:  # adapter failures are window-scoped
                    results.append(WindowExecutionResult(
                        window_id=window.id,
                        status="failed",
                        reason="execution_failed",
                        error=str(exc),
                        wall_time_seconds=time.perf_counter() - item_started,
                    ))
                finally:
                    with self._lock:
                        self._active_futures.discard(future)
            if selected_token.cancelled and len(results) < len(windows):
                results.extend(
                    WindowExecutionResult(
                        window_id=window.id,
                        status="cancelled",
                        reason=selected_token.reason,
                    )
                    for window in windows[len(results):]
                )
            candidates = [candidate for result in results for candidate in result.candidates]
            diagnostics = {
                "engine": getattr(engine, "name", "unknown"),
                "model": getattr(engine, "model_name", None),
                "max_workers": self.max_workers,
                "timeout_seconds": self.timeout_seconds,
                "window_count": len(windows),
                "completed_window_count": len(results),
                "candidate_count": len(candidates),
                "cancelled": selected_token.cancelled,
                "cancel_reason": selected_token.reason if selected_token.cancelled else None,
                "status_counts": self._status_counts(results),
                "windows": [self._result_to_dict(item) for item in results],
                "wall_time_seconds": round(time.perf_counter() - started, 6),
                "degraded": any(item.status != "ok" for item in results),
            }
            return candidates, diagnostics
        finally:
            with self._lock:
                self._active_token = None
                self._active_futures.clear()
            # Do not wait on a timed-out SDK call.  Cancellation is cooperative
            # because Python cannot safely kill a thread executing model code.
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _invoke(
        engine: Any,
        audio: Any,
        sample_rate: int,
        window: Any,
        language: Optional[str],
        token: CancellationToken,
    ) -> WindowExecutionResult:
        token.raise_if_cancelled()
        started = time.perf_counter()
        method = getattr(engine, "review")
        kwargs: dict[str, Any] = {"language": language}
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "cancellation_token" in parameters:
            kwargs["cancellation_token"] = token
        result = method(audio, sample_rate, window, **kwargs)
        token.raise_if_cancelled()
        return WindowExecutionResult(
            window_id=window.id,
            candidates=tuple(result or ()),
            wall_time_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _status_counts(results: Sequence[WindowExecutionResult]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in results:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    @staticmethod
    def _result_to_dict(item: WindowExecutionResult) -> dict[str, Any]:
        return {
            "window_id": item.window_id,
            "status": item.status,
            "reason": item.reason,
            "error": item.error,
            "candidate_count": len(item.candidates),
            "wall_time_seconds": round(item.wall_time_seconds, 6),
            "resources": dict(item.resources),
        }


__all__ = [
    "CancellationToken",
    "ReviewWindowCancelled",
    "WindowExecutionCoordinator",
    "WindowExecutionResult",
]
