"""Independent global-ASR service and compatibility boundary."""

from __future__ import annotations

from .contracts import ASRRuntimePorts, GlobalASRRequest, GlobalASRResult


class GlobalASRService:
    """Run global ASR through explicit runtime ports.

    The application supplies the implementation as a callable port during the
    migration.  This service never receives or introspects ``Pipeline``.
    """

    def run(
        self,
        request: GlobalASRRequest,
        ports: ASRRuntimePorts,
    ) -> GlobalASRResult:
        if ports.global_runner is None:
            raise RuntimeError("global ASR implementation port is not configured")
        raw = ports.global_runner(request)
        if isinstance(raw, GlobalASRResult):
            return raw
        events, diagnostics, transcript = raw
        return GlobalASRResult(
            events=list(events or []),
            diagnostics=dict(diagnostics or {}),
            transcript=transcript,
        )


class GlobalASRPath(GlobalASRService):
    """Backward-compatible class name for the global ASR service."""


__all__ = ["GlobalASRPath", "GlobalASRService"]
