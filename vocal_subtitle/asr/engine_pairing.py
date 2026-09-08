"""Deterministic primary/secondary ASR pairing and language policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


ENGINE_FAMILIES = {
    "faster-whisper": "whisper",
    "whisper-cpp": "whisper",
    "whisper": "whisper",
    "funasr": "funasr",
    "qwen": "qwen",
}


@dataclass(frozen=True)
class EnginePairDecision:
    primary: str
    secondary: Optional[str]
    primary_family: str
    secondary_family: Optional[str]
    language: Optional[str]
    policy: str
    decision_reason: str
    degraded: bool = False
    fallback_reason: Optional[str] = None
    route_version: str = "asr-pair-v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "primary_family": self.primary_family,
            "secondary_family": self.secondary_family,
            "language": self.language,
            "policy": self.policy,
            "decision_reason": self.decision_reason,
            "degraded": self.degraded,
            "fallback_reason": self.fallback_reason,
            "route_version": self.route_version,
        }


class EnginePairRouter:
    """Choose an explicitly explainable heterogeneous engine pair."""

    def __init__(self, route_version: str = "asr-pair-v1") -> None:
        self.route_version = route_version

    def route(
        self,
        *,
        language: Optional[str],
        primary: str = "auto",
        secondary: str = "auto",
        policy: str = "risk_only",
        selected_primary: Optional[str] = None,
        same_family_policy: str = "reject",
    ) -> EnginePairDecision:
        if policy not in {"risk_only", "full_quality"}:
            raise ValueError(f"unsupported pair policy: {policy}")
        if same_family_policy not in {"reject", "allow"}:
            raise ValueError(f"unsupported same-family policy: {same_family_policy}")

        normalized_language = (language or "").casefold() or None
        primary_name = self._normalize(primary)
        if primary_name == "auto":
            primary_name = self._default_primary(normalized_language, selected_primary)
        if primary_name not in ENGINE_FAMILIES:
            raise ValueError(f"unsupported primary engine: {primary}")

        degraded = False
        fallback_reason = None
        if primary_name == "funasr" and not self._is_chinese(normalized_language):
            primary_name = "faster-whisper"
            degraded = True
            fallback_reason = "funasr_requires_chinese_language"

        primary_family = ENGINE_FAMILIES[primary_name]
        secondary_name = self._normalize(secondary)
        if secondary_name == "auto":
            secondary_name = self._default_secondary(primary_name, normalized_language)
        if secondary_name is not None and secondary_name not in ENGINE_FAMILIES:
            raise ValueError(f"unsupported secondary engine: {secondary}")

        secondary_family = ENGINE_FAMILIES.get(secondary_name) if secondary_name else None
        if (
            secondary_name
            and secondary_family == primary_family
            and same_family_policy == "reject"
        ):
            degraded = True
            fallback_reason = "same_family_secondary_rejected"
            secondary_name = None
            secondary_family = None

        reason = "explicit_pair"
        if primary == "auto" or secondary == "auto":
            reason = "language_and_primary_route"
        if degraded and fallback_reason:
            reason = f"{reason}:{fallback_reason}"
        return EnginePairDecision(
            primary=primary_name,
            secondary=secondary_name,
            primary_family=primary_family,
            secondary_family=secondary_family,
            language=normalized_language,
            policy=policy,
            decision_reason=reason,
            degraded=degraded,
            fallback_reason=fallback_reason,
            route_version=self.route_version,
        )

    @staticmethod
    def _normalize(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        value = str(name).strip().casefold()
        aliases = {"": "auto", "whisper-family": "whisper"}
        return aliases.get(value, value)

    @staticmethod
    def _is_chinese(language: Optional[str]) -> bool:
        return language is None or language in {"zh", "cmn", "yue", "chinese"}

    @staticmethod
    def _default_primary(language: Optional[str], selected_primary: Optional[str]) -> str:
        # ``selected_primary`` is the authoritative task-level route.  The
        # pairing layer must not promote an optional review engine to the
        # primary role merely because the language is non-Chinese.
        if selected_primary and selected_primary in ENGINE_FAMILIES:
            return selected_primary
        return "faster-whisper"

    @staticmethod
    def _default_secondary(primary: str, language: Optional[str]) -> Optional[str]:
        if primary == "funasr":
            return "qwen"
        if primary == "qwen":
            return "funasr" if EnginePairRouter._is_chinese(language) else "faster-whisper"
        return "qwen"


__all__ = ["ENGINE_FAMILIES", "EnginePairDecision", "EnginePairRouter"]
