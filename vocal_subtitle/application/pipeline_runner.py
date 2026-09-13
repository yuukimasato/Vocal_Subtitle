"""Compatibility facade for the offline pipeline lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..application.pipeline_result import PipelineStats
from ..application.run_finalizer import finalize_task_state
from ..application.run_preflight import run_preflight
from ..application.run_reporter import generate_run_report
from .run_lifecycle import PipelineLifecycleMixin


class PipelineRunMixin(PipelineLifecycleMixin):
    """Keep the historical mixin import while delegating lifecycle work."""

    def _run_preflight(
        self,
        input_path: Path,
        output_path: Path,
        task_id: Optional[str],
        stats: "PipelineStats",
        skip_separation: bool = False,
    ) -> None:
        """Compatibility wrapper for the application preflight component."""
        run_preflight(
            self,
            input_path,
            output_path,
            task_id,
            stats,
            skip_separation=skip_separation,
        )

    def _finalize_task_state(
        self,
        stats: "PipelineStats",
        *,
        result_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Compatibility wrapper for task-history finalization."""
        finalize_task_state(self, stats, result_payload=result_payload)

    def _generate_run_report(
        self,
        input_path: Path,
        stats: "PipelineStats",
        task_id: Optional[str] = None,
        sample_rate: int = 0,
        final_subtitle_path: Optional[Path] = None,
    ) -> None:
        """Compatibility wrapper for the run-report component."""
        generate_run_report(
            self,
            input_path,
            stats,
            task_id=task_id,
            sample_rate=sample_rate,
            final_subtitle_path=final_subtitle_path,
        )
