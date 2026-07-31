"""Serializable diagnostics for acoustic boundary validation."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .boundary import find_directional_boundary
from .skeleton import has_speech_in_range, is_time_in_speech


def generate_diagnostic_report(
    events: List,
    speech_skeleton: List[Tuple[float, float]],
    flag_threshold_ms: float = 200,
) -> Dict:
    """Build the stable acoustic health report used by API and WebUI."""
    report = {
        "total_events": len(events),
        "start_in_silence": 0,
        "end_in_silence": 0,
        "end_truncated": 0,
        "start_out_of_range": 0,
        "end_out_of_range": 0,
        "events_flagged": [],
    }
    threshold = flag_threshold_ms / 1000.0

    for event in events:
        if not is_time_in_speech(event.start, speech_skeleton):
            report["start_in_silence"] += 1
            _, nearest, _ = find_directional_boundary(
                event.start, speech_skeleton, "start",
            )
            if nearest is not None and abs(nearest - event.start) > threshold:
                report["start_out_of_range"] += 1

        if not is_time_in_speech(event.end, speech_skeleton):
            report["end_in_silence"] += 1
            if has_speech_in_range(event.end, event.end + 0.2, speech_skeleton):
                report["end_truncated"] += 1
                report["events_flagged"].append({
                    "id": getattr(event, "index", 0),
                    "issue": "end_truncated",
                    "current_end": event.end,
                    "text_preview": getattr(event, "text", "")[:50],
                })

            _, nearest, _ = find_directional_boundary(
                event.end, speech_skeleton, "end",
            )
            if nearest is not None and abs(nearest - event.end) > threshold:
                report["end_out_of_range"] += 1

    total_checks = len(events) * 2
    issues = report["start_in_silence"] + report["end_in_silence"]
    report["health_score"] = round(
        (1 - issues / max(total_checks, 1)) * 100, 1,
    )
    return report


__all__ = ["generate_diagnostic_report"]
