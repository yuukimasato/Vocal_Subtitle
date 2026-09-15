"""Small, dependency-light telemetry helpers for bounded ASR review stages."""

from __future__ import annotations

import resource
import time
from typing import Any


def resource_snapshot() -> dict[str, Any]:
    """Return process RSS and optional CUDA memory without requiring torch."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot: dict[str, Any] = {
        "rss_mb": round(float(usage.ru_maxrss) / 1024.0, 3),
        "gpu": {"available": False},
    }
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            snapshot["gpu"] = {
                "available": True,
                "device": device,
                "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 3),
                "reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 3),
            }
    except (ImportError, RuntimeError, AttributeError):
        pass
    return snapshot


def timed_call(
    callback, *args: Any, **kwargs: Any
) -> tuple[Any, float, dict[str, Any]]:
    """Execute one stage and return its result, wall time and resource snapshot."""
    started = time.perf_counter()
    result = callback(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return result, round(elapsed, 6), resource_snapshot()


__all__ = ["resource_snapshot", "timed_call"]
