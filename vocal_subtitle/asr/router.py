"""Task-level ASR language probing and engine routing."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from math import ceil

import numpy as np

from .base import LanguageDetection

SUPPORTED_ENGINES = ("auto", "faster-whisper", "funasr", "qwen", "whisper-cpp")


@dataclass(frozen=True)
class ASRRouteDecision:
    """The single routing decision consumed by the rest of a task."""

    requested_engine: str
    selected_engine: str
    selected_model: str
    detected_language: str
    language_probability: float
    window_evidence: tuple = field(default_factory=tuple)
    decision_reason: str = ""
    fallback_engine: str | None = None
    route_version: str = "asr-route-v1"
    quality_gate_version: str = "asr-quality-v1"

    @property
    def language(self) -> str | None:
        if self.detected_language in {"unknown", "mixed", "other"}:
            return None
        return self.detected_language or None

    def to_dict(self) -> dict:
        return {
            "requested_engine": self.requested_engine,
            "selected_engine": self.selected_engine,
            "selected_model": self.selected_model,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "window_evidence": list(self.window_evidence),
            "decision_reason": self.decision_reason,
            "fallback_engine": self.fallback_engine,
            "route_version": self.route_version,
            "quality_gate_version": self.quality_gate_version,
        }


def _normalise_detection(value, source: str = "unknown") -> LanguageDetection:
    if isinstance(value, LanguageDetection):
        return value
    language = getattr(value, "language", None)
    probability = getattr(value, "probability", None)
    if probability is None:
        probability = getattr(value, "language_probability", None)
    if language is None and isinstance(value, str):
        language = value
    return LanguageDetection(
        language=str(language or "unknown").lower(),
        probability=max(
            0.0, min(1.0, float(probability if probability is not None else 0.0))
        ),
        source=getattr(value, "source", source) or source,
    )


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    values = sorted(
        (max(0.0, float(item[0])), max(0.0, float(item[1])))
        for item in intervals
        if len(item) >= 2 and float(item[1]) > float(item[0])
    )
    merged: list[tuple[float, float]] = []
    for start, end in values:
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_probe_windows(
    duration_seconds: float,
    speech_intervals: Iterable[Sequence[float]] | None = None,
    window_seconds: float = 8.0,
    max_windows: int = 8,
) -> list[tuple[float, float]]:
    """Build deterministic windows covering the full available speech span."""
    duration = max(0.0, float(duration_seconds))
    if duration <= 0 or max_windows <= 0:
        return []
    intervals = merge_intervals(speech_intervals or [])
    if not intervals:
        intervals = [(0.0, duration)]
    window = max(0.25, float(window_seconds))
    speech_total = sum(end - start for start, end in intervals)
    if speech_total <= window:
        # Short inputs still need independent head/tail evidence.
        window = min(window, max(0.25, duration / 2.0))
        count = 2
    else:
        count = min(int(max_windows), max(2, int(ceil(speech_total / window))))
    count = max(1, count)
    positions = np.linspace(0.0, max(0.0, speech_total - 0.001), count)

    def locate(offset: float) -> float:
        remaining = offset
        for start, end in intervals:
            span = end - start
            if remaining <= span:
                return start + remaining
            remaining -= span
        return intervals[-1][1]

    result = []
    for offset in positions:
        center = locate(float(offset))
        start = max(0.0, center - window / 2.0)
        end = min(duration, start + window)
        start = max(0.0, end - window)
        if end > start:
            item = (round(start, 6), round(end, 6))
            if item not in result:
                result.append(item)
    return result


class ASRRouter:
    """Resolve one ASR engine for a complete task."""

    def __init__(self, config, engine_factory: Callable[..., object]):
        self.config = config
        self.engine_factory = engine_factory

    def decide(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speech_intervals: Iterable[Sequence[float]] | None = None,
    ) -> ASRRouteDecision:
        asr = self.config.asr
        policy = asr.auto_routing
        requested = str(asr.engine or "auto").lower()
        if requested not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Unknown ASR engine: {requested}. Options: {', '.join(SUPPORTED_ENGINES)}"
            )

        if requested != "auto" or not policy.enabled:
            selected = requested if requested != "auto" else "faster-whisper"
            language = str(asr.language or "unknown").lower()
            if selected == "funasr" and language not in {"unknown", "", "zh"}:
                raise ValueError(
                    "FunASR 为中文专用引擎，请将语言改为 zh 或选择 faster-whisper"
                )
            return ASRRouteDecision(
                requested_engine=requested,
                selected_engine=selected,
                selected_model=self._model_for(selected),
                detected_language=language or "unknown",
                language_probability=1.0 if language not in {"unknown", ""} else 0.0,
                decision_reason="explicit_engine"
                if requested != "auto"
                else "auto_routing_disabled",
                fallback_engine=self._fallback_engine(selected),
                route_version=policy.route_version,
                quality_gate_version=policy.quality_gate_version,
            )

        # An explicit language lock is stronger than automatic probing.
        if asr.language:
            language = str(asr.language).lower()
            selected = "funasr" if language == "zh" else "faster-whisper"
            return ASRRouteDecision(
                requested_engine=requested,
                selected_engine=selected,
                selected_model=self._model_for(selected),
                detected_language=language,
                language_probability=1.0,
                decision_reason="explicit_language_lock",
                fallback_engine=self._fallback_engine(selected),
                route_version=policy.route_version,
                quality_gate_version=policy.quality_gate_version,
            )

        windows = build_probe_windows(
            len(audio) / max(sample_rate, 1),
            speech_intervals,
            policy.language_probe_window_seconds,
            policy.language_probe_max_windows,
        )
        evidence = []
        try:
            probe = self.engine_factory(
                "faster-whisper", model=policy.language_probe_model, probe=True
            )
            probe.load_model()
        except Exception as exc:
            evidence.append({"status": "error", "error": str(exc)})
            return self._fallback_decision(
                requested, evidence, "language_probe_unavailable"
            )

        for index, (start, end) in enumerate(windows):
            chunk = np.asarray(audio)[int(start * sample_rate) : int(end * sample_rate)]
            try:
                detector = getattr(probe, "detect_language_info", None)
                if callable(detector):
                    detection = _normalise_detection(
                        detector(chunk, sample_rate), probe.model_name
                    )
                else:
                    detection = _normalise_detection(
                        probe.detect_language(chunk, sample_rate), probe.model_name
                    )
                evidence.append(
                    {
                        "window_id": f"probe-{index:02d}",
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 6),
                        **detection.to_dict(),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                evidence.append(
                    {
                        "window_id": f"probe-{index:02d}",
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 6),
                        "status": "error",
                        "error": str(exc),
                    }
                )

        valid = [item for item in evidence if item.get("status") == "ok"]
        zh_valid = [
            item
            for item in valid
            if item.get("language") == "zh"
            and float(item.get("probability", 0.0)) >= policy.zh_min_probability
        ]
        if valid and len(valid) == len(windows) and windows:
            ratio = len(zh_valid) / len(valid)
            if ratio >= policy.zh_required_window_ratio:
                probability = min(float(item["probability"]) for item in zh_valid)
                return ASRRouteDecision(
                    requested_engine=requested,
                    selected_engine="funasr",
                    selected_model=self._model_for("funasr"),
                    detected_language="zh",
                    language_probability=probability,
                    window_evidence=tuple(evidence),
                    decision_reason="all_probe_windows_zh",
                    fallback_engine=policy.fallback_engine,
                    route_version=policy.route_version,
                    quality_gate_version=policy.quality_gate_version,
                )

        languages = {item.get("language") for item in valid if item.get("language")}
        if "en" in languages:
            detected = "en"
        elif len(languages) > 1:
            detected = "mixed"
        elif languages:
            detected = "other"
        else:
            detected = "unknown"
        probability = min(
            (float(item.get("probability", 0.0)) for item in valid),
            default=0.0,
        )
        reason = "non_chinese_or_uncertain_probe"
        if not windows:
            reason = "insufficient_probe_windows"
        elif len(valid) != len(windows):
            reason = "language_probe_failed"
        return ASRRouteDecision(
            requested_engine=requested,
            selected_engine="faster-whisper",
            selected_model=self._model_for("faster-whisper"),
            detected_language=detected,
            language_probability=probability,
            window_evidence=tuple(evidence),
            decision_reason=reason,
            fallback_engine=None,
            route_version=policy.route_version,
            quality_gate_version=policy.quality_gate_version,
        )

    def _model_for(self, engine: str) -> str:
        if engine == "funasr":
            return self.config.asr.model
        return self.config.asr.model

    def _fallback_engine(self, selected: str) -> str | None:
        if (
            selected == "funasr"
            and self.config.asr.auto_routing.fallback_on_quality_failure
        ):
            return self.config.asr.auto_routing.fallback_engine
        return None

    def _fallback_decision(self, requested, evidence, reason):
        policy = self.config.asr.auto_routing
        return ASRRouteDecision(
            requested_engine=requested,
            selected_engine="faster-whisper",
            selected_model=self._model_for("faster-whisper"),
            detected_language="unknown",
            language_probability=0.0,
            window_evidence=tuple(evidence),
            decision_reason=reason,
            route_version=policy.route_version,
            quality_gate_version=policy.quality_gate_version,
        )
