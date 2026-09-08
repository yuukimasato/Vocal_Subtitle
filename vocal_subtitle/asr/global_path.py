"""Independent global-ASR service and compatibility boundary."""

from __future__ import annotations

from .contracts import ASRRuntimePorts, GlobalASRRequest, GlobalASRResult
from .evidence import candidate_from_subtitle_event, candidates_from_global_transcript


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
            if not raw.evidence and raw.transcript is not None:
                raw.evidence = candidates_from_global_transcript(raw.transcript)
            return raw
        if len(raw) == 4:
            events, diagnostics, transcript, evidence = raw
        else:
            events, diagnostics, transcript = raw
            evidence = candidates_from_global_transcript(transcript)
            if not evidence:
                evidence = [
                    candidate_from_subtitle_event(event, source="global")
                    for event in (events or [])
                    if hasattr(event, "start")
                    and hasattr(event, "end")
                    and hasattr(event, "text")
                ]
        if not evidence:
            evidence = candidates_from_global_transcript(transcript)
        return GlobalASRResult(
            events=list(events or []),
            evidence=list(evidence or []),
            diagnostics=dict(diagnostics or {}),
            transcript=transcript,
        )


class GlobalASRPath(GlobalASRService):
    """Backward-compatible class name for the global ASR service."""


__all__ = ["GlobalASRPath", "GlobalASRService"]
