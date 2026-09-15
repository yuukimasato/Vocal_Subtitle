"""Preflight orchestration for offline pipeline runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..application.pipeline_result import PipelineStats

logger = logging.getLogger(__name__)


def run_preflight(
    pipeline: Any,
    input_path: Path,
    output_path: Path,
    task_id: str | None,
    stats: PipelineStats,
    skip_separation: bool = False,
) -> None:
    """Run preflight checks and manage task state transitions.

    The pipeline argument is the application facade that owns configuration,
    hashes and the task-history service. Keeping that dependency explicit
    makes this lifecycle step independently testable without importing the
    full Pipeline class.
    """
    from ..application.preflight import (
        PreflightError,
        _error_category_from_checks,
        run_preflight_checks,
    )
    from ..utils.session_manager import create_run_id, create_task_id

    effective_task_id = task_id or create_task_id(input_path)
    result = run_preflight_checks(
        input_path,
        output_path,
        pipeline.config,
        skip_separation=skip_separation,
    )
    stats.quality_diagnostics["preflight"] = result.to_dict()

    history = pipeline._get_history()
    if history.get(effective_task_id) is None:
        history.create(
            effective_task_id,
            input_path.name,
            pipeline._file_hash,
            input_path.stat().st_size,
            getattr(pipeline.config, "_profile_name", "default"),
            pipeline.config,
        )
    history.set_preflight(effective_task_id)

    if result.passed:
        run_id = create_run_id(effective_task_id)
        history.set_running(effective_task_id, run_id=run_id)
        stats.run_id = run_id
        stats.task_id = effective_task_id
        pipeline._preflight_ok = True
        pipeline._effective_task_id = effective_task_id
        return

    mode = getattr(pipeline.config.degradation, "preflight_mode", "report")
    failed_critical = result.failed_critical()
    category = _error_category_from_checks(failed_critical)

    if mode == "enforce":
        history.set_failed(
            effective_task_id,
            reason="; ".join(c.label for c in failed_critical),
            category=category,
        )
        stats.status = "failed"
        stats.error_category = category
        raise PreflightError(result)

    logger.warning(
        "Preflight failed (report mode): %s",
        ", ".join(f"{c.key}={c.passed}" for c in failed_critical),
    )
    run_id = create_run_id(effective_task_id)
    history.set_running(effective_task_id, run_id=run_id)
    stats.run_id = run_id
    stats.task_id = effective_task_id
    # In report mode, preflight does NOT touch fallback_category or
    # fallback_reason.  Their default values (empty string) are the
    # expected contract when nothing has gone wrong at the ASR-routing
    # layer.  The global-ASR-path classifier writes the authoritative
    # values later.  Preflight's job is to set the task state; the
    # stats fields it leaves alone so callers get consistent
    # diagnostics after routing completes.
    pipeline._preflight_ok = False
    pipeline._effective_task_id = effective_task_id
