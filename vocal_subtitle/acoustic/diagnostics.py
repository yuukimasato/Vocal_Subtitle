"""Diagnostic report generation for acoustic validation.

Usage:
    report = generate_diagnostic_report(events, skeleton)            # standalone
    report = validator.generate_diagnostic_report(events, skeleton)   # as method proxy
"""

from typing import Dict, List, Tuple

from .skeleton import _has_speech_in_range, _is_time_in_speech
from .boundary import _find_directional_boundary


def generate_diagnostic_report(
    events: List,
    speech_skeleton: List[Tuple[float, float]],
    flag_threshold_ms: float = 200,
) -> Dict:
    """生成物理校验诊断报告"""
    report = {
        "total_events": len(events),
        "start_in_silence": 0,
        "end_in_silence": 0,
        "end_truncated": 0,
        "start_out_of_range": 0,
        "end_out_of_range": 0,
        "events_flagged": [],
    }

    for event in events:
        # 检查 start
        is_start_ok = _is_time_in_speech(event.start, speech_skeleton)
        if not is_start_ok:
            report["start_in_silence"] += 1
            _, nearest, _ = _find_directional_boundary(
                event.start, speech_skeleton, "start",
            )
            if nearest is not None and abs(nearest - event.start) > (
                flag_threshold_ms / 1000.0
            ):
                report["start_out_of_range"] += 1

        # 检查 end
        is_end_ok = _is_time_in_speech(event.end, speech_skeleton)
        if not is_end_ok:
            report["end_in_silence"] += 1
            # 检查是否切尾
            if _has_speech_in_range(
                event.end, event.end + 0.2, speech_skeleton,
            ):
                report["end_truncated"] += 1
                report["events_flagged"].append({
                    "id": getattr(event, "index", 0),
                    "issue": "end_truncated",
                    "current_end": event.end,
                    "text_preview": (
                        getattr(event, "text", "")[:50]
                        if hasattr(event, "text") else ""
                    ),
                })

            _, nearest, _ = _find_directional_boundary(
                event.end, speech_skeleton, "end",
            )
            if nearest is not None and abs(nearest - event.end) > (
                flag_threshold_ms / 1000.0
            ):
                report["end_out_of_range"] += 1

    # 计算健康度评分
    total_checks = len(events) * 2
    issues = report["start_in_silence"] + report["end_in_silence"]
    report["health_score"] = round(
        (1 - issues / max(total_checks, 1)) * 100, 1,
    )

    return report
