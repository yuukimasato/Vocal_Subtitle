"""Independent segmented-ASR service and compatibility boundary."""

from __future__ import annotations

from .contracts import ASRRuntimePorts, SegmentedASRRequest, SegmentedASRResult


class SegmentedASRService:
    """Run segmented ASR through an injected implementation port."""

    def run(
        self,
        request: SegmentedASRRequest,
        ports: ASRRuntimePorts,
    ) -> SegmentedASRResult:
        if ports.segmented_runner is None:
            raise RuntimeError("segmented ASR implementation port is not configured")
        raw = ports.segmented_runner(request)
        if isinstance(raw, SegmentedASRResult):
            return raw
        events, segment_count, context = raw
        return SegmentedASRResult(
            events=list(events or []),
            segment_count=int(segment_count or 0),
            context=context,
        )


class SegmentedASRPath(SegmentedASRService):
    """Backward-compatible class name for segmented ASR."""


__all__ = ["SegmentedASRPath", "SegmentedASRService"]
