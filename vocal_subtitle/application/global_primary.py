"""Global-primary 实验路由的可行性门禁。

``routing="global_primary"`` 时,全局窗口识别结果在被当作主候选之前必须
通过本门禁;未通过时记录 ``global_primary_fallback_reason`` 并回退
segmented 路径。门禁只做合法性检查,不改变任何识别文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


DEFAULT_MIN_SPEECH_COVERAGE = 0.6
DEFAULT_MIN_CHARS_PER_SECOND = 0.5
DEFAULT_MAX_CHARS_PER_SECOND = 100.0
TIME_TOLERANCE_SECONDS = 0.01


@dataclass(frozen=True)
class GlobalPrimaryGateResult:
    """Outcome of one global-primary suitability evaluation."""

    passed: bool
    reason: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            **self.details,
        }


def _word_span(word: Any) -> Optional[tuple[float, float]]:
    start = getattr(word, "raw_start", getattr(word, "start", None))
    end = getattr(word, "raw_end", getattr(word, "end", None))
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        return None
    if start < 0.0 or end <= start:
        return None
    return start, end


def _speech_spans(physical_timeline: Any) -> list[tuple[float, float]]:
    spans = getattr(physical_timeline, "speech_evidence_spans", None) or ()
    result = []
    for span in spans:
        start = float(getattr(span, "start", 0.0))
        end = float(getattr(span, "end", 0.0))
        if end > start:
            result.append((start, end))
    return result


def _overlap(total: float, span: tuple[float, float], windows: list[tuple[float, float]]) -> float:
    covered = 0.0
    cursor = span[0]
    for start, end in sorted(windows):
        start = max(start, cursor)
        end = min(end, span[1])
        if end > start:
            covered += end - start
            cursor = end
    return min(covered, total)


def evaluate_global_primary_suitability(
    transcript: Any,
    *,
    audio_duration: Optional[float] = None,
    physical_timeline: Any = None,
    config: Any = None,
) -> GlobalPrimaryGateResult:
    """Check whether a global transcript may act as the primary candidate."""

    min_coverage = float(
        getattr(config, "global_primary_min_speech_coverage", DEFAULT_MIN_SPEECH_COVERAGE)
    )
    min_density = float(
        getattr(config, "global_primary_min_chars_per_second", DEFAULT_MIN_CHARS_PER_SECOND)
    )
    max_density = float(
        getattr(config, "global_primary_max_chars_per_second", DEFAULT_MAX_CHARS_PER_SECOND)
    )

    words = list(getattr(transcript, "words", ()) or ())
    if not words:
        return GlobalPrimaryGateResult(False, "empty_transcript")

    spans: list[tuple[float, float]] = []
    total_chars = 0
    for word in words:
        span = _word_span(word)
        if span is None:
            return GlobalPrimaryGateResult(
                False,
                "invalid_time_range",
                {"invalid_word_id": getattr(word, "id", "")},
            )
        if (
            audio_duration is not None
            and span[1] > float(audio_duration) + TIME_TOLERANCE_SECONDS
        ):
            return GlobalPrimaryGateResult(
                False,
                "invalid_time_range",
                {"invalid_word_id": getattr(word, "id", ""), "word_end": span[1]},
            )
        spans.append(span)
        total_chars += len(str(getattr(word, "text", "") or "").strip())

    speech = _speech_spans(physical_timeline)
    details: dict[str, Any] = {
        "word_count": len(words),
        "char_count": total_chars,
    }
    if speech:
        speech_total = sum(end - start for start, end in speech)
        covered = sum(
            _overlap(end - start, (start, end), spans) for start, end in speech
        )
        coverage = covered / speech_total if speech_total > 0 else 0.0
        details["speech_coverage"] = round(coverage, 6)
        if coverage < min_coverage:
            return GlobalPrimaryGateResult(
                False, "insufficient_speech_coverage", details
            )

    voiced_start = min(span[0] for span in spans)
    voiced_end = max(span[1] for span in spans)
    voiced = voiced_end - voiced_start
    density = total_chars / voiced if voiced > 0 else 0.0
    details["chars_per_second"] = round(density, 6)
    if density < min_density or density > max_density:
        return GlobalPrimaryGateResult(False, "abnormal_text_density", details)

    return GlobalPrimaryGateResult(True, None, details)
