"""Independent orchestration service for ASR evidence review."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Optional

from .contracts import EvidenceReviewRequest, EvidenceReviewResult
from .evidence_cache import (
    EVIDENCE_CACHE_STAGE,
    EvidenceCacheKeyContext,
    EvidenceCachePort,
    audio_fingerprint,
)
from .evidence import (
    DECISION_POLICY_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    RISK_POLICY_VERSION,
    CandidateEvidence,
    candidate_from_subtitle_event,
)
from .evidence_decision import DecisionConfig, EvidenceDecisionEngine, decisions_to_subtitle_events
from .review_engines import (
    ForcedAlignerPort,
    ReviewEnginePort,
    SEDPort,
    SemanticReviewPort,
)
from .review_scheduler import ReviewScheduler, ReviewSchedulerConfig
from .risk_scoring import EvidenceRiskScorer, RiskAssessment, RiskScoringConfig, normalize_text
from .review_telemetry import resource_snapshot, timed_call
from .secondary_evidence import SecondaryEvidenceCollector
from .window_execution import WindowExecutionCoordinator
from .evidence import DecisionEvidenceBundle


@dataclass(frozen=True)
class EvidenceReviewRuntimePorts:
    """Optional runtime capabilities supplied by the application layer."""

    config: Any
    context_reasr: Optional[ReviewEnginePort] = None
    qwen: Optional[ReviewEnginePort] = None
    forced_aligner: Optional[ForcedAlignerPort] = None
    sed: Optional[SEDPort] = None
    semantic_review: Optional[SemanticReviewPort] = None
    language: Optional[str] = None
    cache: Optional[EvidenceCachePort] = None
    cache_ttl: Optional[int] = None
    secondary: Optional[ReviewEnginePort] = None
    secondary_name: Optional[str] = None
    window_executor: Optional[WindowExecutionCoordinator] = None


class EvidenceReviewService:
    """Score, review and decide candidates without depending on Pipeline."""

    def run(
        self,
        request: EvidenceReviewRequest,
        ports: EvidenceReviewRuntimePorts,
    ) -> EvidenceReviewResult:
        config = ports.config
        if config is None or not config.enabled or not request.events:
            return EvidenceReviewResult(
                events=list(request.events or ()),
                diagnostics={"status": "disabled"},
            )

        segmented = []
        invalid_segment_count = 0
        for event in request.events:
            if not str(getattr(event, "text", "") or "").strip():
                invalid_segment_count += 1
                continue
            segmented.append(candidate_from_subtitle_event(event, source="segmented"))
        invalid_word_timing_count = sum(
            int((candidate.diagnostics or {}).get("invalid_word_timing_count", 0))
            for candidate in segmented
        )
        global_evidence = tuple(request.global_evidence or ())
        recovery_evidence = tuple(request.recovery_evidence or ())
        primary_candidates = [*segmented, *recovery_evidence]
        scorer = EvidenceRiskScorer(RiskScoringConfig(
            medium_threshold=config.medium_threshold,
            high_threshold=config.high_threshold,
            critical_threshold=config.critical_threshold,
        ))
        assessments = scorer.score_bundle(
            primary_candidates,
            global_evidence=global_evidence,
            physical_timeline=request.physical_timeline,
        )
        scheduler = ReviewScheduler(ReviewSchedulerConfig(
            left_context=config.left_context,
            right_context=config.right_context,
            max_group_duration=config.max_group_duration,
            max_window_duration=config.max_window_duration,
        ))
        review_policy = getattr(request, "review_policy", "risk_only")
        windows = scheduler.schedule(
            primary_candidates,
            assessments,
            timeline=request.physical_timeline,
            policy=review_policy,
        )

        review_candidates = []
        review_diagnostics: dict[str, Any] = {
            "status": "skipped",
            "reason": "no_review_windows",
        }
        audio_hash = request.audio_hash
        if ports.cache is not None and not audio_hash:
            audio_hash = audio_fingerprint(request.audio)
        if windows and config.context_reasr_enabled and request.audio is not None:
            if ports.context_reasr is None:
                review_diagnostics = {
                    "status": "unavailable",
                    "reason": "context_reasr_port_missing",
                    "windows": len(windows),
                }
            else:
                review_candidates, review_diagnostics = self._review_windows(
                    scheduler,
                    request.audio,
                    request.sample_rate,
                    windows,
                    ports.context_reasr,
                    language=ports.language,
                    request=request,
                    ports=ports,
                    audio_hash=audio_hash,
                )
                assessments = self._rescore_after_context(
                    primary_candidates,
                    assessments,
                    review_candidates,
                    config,
                )
        elif windows:
            review_diagnostics = {
                "status": "unavailable",
                "reason": "context_reasr_disabled_or_audio_missing",
                "windows": len(windows),
                }

        # global 候选角色路由（优化方案 8.1）：默认只把 global 证据传给风险
        # 评分；显式准入后才生成 global_alternative 参与替换决策。缺失该字段
        # 的旧配置对象保持既有准入行为。
        alternative_admission_enabled = bool(
            getattr(config, "global_alternative_enabled", True)
        )
        if alternative_admission_enabled:
            global_alternatives, global_diagnostics = self._global_alternatives(
                primary_candidates,
                global_evidence,
                assessments,
                request.physical_timeline,
            )
            review_candidates.extend(global_alternatives)
        else:
            global_diagnostics = {
                "role": "global_signal_only",
                "alternative_admission": "disabled",
                "signal_count": len(global_evidence),
                "considered_overlap_count": 0,
                "considered_alternative_count": 0,
                "accepted_alternative_count": 0,
                "selected_global_count": 0,
                "accepted_candidate_ids": [],
                "rejected_count": 0,
                "rejected": [],
            }

        optional_diagnostics = self._optional_capabilities(config, ports)
        qwen_assessments = assessments
        if review_policy == "risk_only":
            qwen_assessments = [
                replace(
                    item,
                    review_required=(
                        (
                            item.level in {"high", "critical"}
                            and "context_reasr_agreement" not in item.evidence_codes
                        )
                        or "context_reasr_conflict" in item.evidence_codes
                    ),
                )
                for item in assessments
            ]
        qwen_windows = scheduler.schedule(
            primary_candidates,
            qwen_assessments,
            timeline=request.physical_timeline,
            policy=review_policy,
        )
        secondary_engine = ports.secondary or ports.qwen
        secondary_enabled = bool(
            getattr(config, "qwen_enabled", False)
            or ports.secondary is not None
        )
        if (
            qwen_windows
            and secondary_enabled
            and secondary_engine is not None
            and request.audio is not None
        ):
            qwen_candidates, qwen_diagnostics = self._review_windows(
                scheduler,
                request.audio,
                request.sample_rate,
                qwen_windows,
                secondary_engine,
                language=ports.language,
                request=request,
                ports=ports,
                audio_hash=audio_hash,
                phase="qwen",
            )
            review_candidates.extend(qwen_candidates)
            secondary_key = ports.secondary_name or (
                "qwen" if ports.qwen is secondary_engine
                else getattr(secondary_engine, "name", "qwen")
            )
            optional_diagnostics[secondary_key] = qwen_diagnostics
            optional_diagnostics[secondary_key]["policy"] = review_policy
        elif secondary_enabled:
            secondary_key = ports.secondary_name or (
                "qwen" if ports.qwen is secondary_engine
                else getattr(secondary_engine, "name", "qwen")
            )
            optional_diagnostics.setdefault(secondary_key, {})
            optional_diagnostics[secondary_key].update({
                "status": "unavailable",
                "reason": (
                    "secondary_port_missing"
                    if secondary_engine is None
                    else (
                        "residual_risk_gate"
                        if review_policy == "risk_only" and not qwen_windows
                        else "audio_or_review_window_missing"
                    )
                ),
                "policy": review_policy,
            })
        elif getattr(config, "qwen_enabled", False):
            optional_diagnostics["qwen"]["status"] = "unavailable"
            optional_diagnostics["qwen"]["reason"] = "qwen_port_missing"

        secondary_evidence = SecondaryEvidenceCollector().collect(
            audio=request.audio,
            sample_rate=request.sample_rate,
            windows=qwen_windows if review_policy == "full_quality" else windows,
            candidates=primary_candidates,
            language=ports.language,
            config=config,
            forced_aligner=ports.forced_aligner,
            sed=ports.sed,
            semantic_review=ports.semantic_review,
        )
        assessments = self._apply_secondary_evidence(
            assessments,
            secondary_evidence,
            windows,
            config,
        )
        decisions = EvidenceDecisionEngine(
            DecisionConfig(
                unresolved_keeps_candidate=config.unresolved_keeps_candidate,
                require_multi_source_drop=getattr(config, "require_multi_source_drop", True),
            )
        ).decide_bundle(
            primary_candidates,
            review=review_candidates,
            assessments=assessments,
            physical_timeline=request.physical_timeline,
            secondary_bundles=[
                DecisionEvidenceBundle(
                    source=item["source"],
                    status=item["status"],
                    candidate_ids=tuple(item.get("candidate_ids", ())),
                    window_id=item.get("window_id"),
                    evidence=dict(item.get("evidence") or {}),
                    diagnostics=dict(item.get("diagnostics") or {}),
                )
                for item in secondary_evidence.get("bundles", ())
            ],
        )
        candidate_by_id = {
            item.id: item for item in (
                list(primary_candidates) + list(review_candidates) + list(global_evidence)
            )
        }
        selected_global_ids = [
            item.selected_candidate_id
            for item in decisions
            if item.selected_candidate_id in candidate_by_id
            and candidate_by_id[item.selected_candidate_id].source == "global"
        ]
        global_diagnostics = {
            **global_diagnostics,
            "selected_global_count": len(selected_global_ids),
            "selected_candidate_ids": list(dict.fromkeys(selected_global_ids)),
            "signal_count": len(global_evidence),
            "considered_alternative_count": global_diagnostics.get(
                "considered_alternative_count",
                global_diagnostics.get("considered_overlap_count", 0),
            ),
        }
        diagnostics = {
            "status": "ok",
            "evidence_schema_version": request.evidence_schema_version or EVIDENCE_SCHEMA_VERSION,
            "risk_policy_version": request.risk_policy_version or RISK_POLICY_VERSION,
            "decision_policy_version": request.decision_policy_version or DECISION_POLICY_VERSION,
            "review_policy_version": request.review_policy_version,
            "pair_route_version": request.pair_route_version or request.route_version,
            "secondary_engine": request.secondary_engine or ports.secondary_name,
            "shadow_mode": config.shadow_mode,
            "candidate_count": len(primary_candidates),
            "segmented_candidate_count": len(segmented),
            "recovery_candidate_count": len(recovery_evidence),
            "invalid_segment_count": invalid_segment_count,
            "invalid_word_timing_count": invalid_word_timing_count,
            "global_evidence_count": len(global_evidence),
            "global_evidence": global_diagnostics,
            "risk": [item.to_dict() for item in assessments],
            "windows": [item.__dict__ for item in windows],
            "review": review_diagnostics,
            "review_policy": getattr(request, "review_policy", "risk_only"),
            "residual_risk": [item.to_dict() for item in assessments],
            "optional_engines": optional_diagnostics,
            "secondary_evidence": secondary_evidence,
            "cache": {
                "enabled": ports.cache is not None,
                "usable": bool(ports.cache is not None and (request.input_hash or audio_hash)),
                "stage": EVIDENCE_CACHE_STAGE,
            },
            "decisions": [item.to_dict() for item in decisions],
        }
        output_events = (
            list(request.events)
            if config.shadow_mode
            else decisions_to_subtitle_events(decisions)
        )
        return EvidenceReviewResult(
            events=output_events,
            decisions=decisions,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _global_alternatives(
        segmented: list[CandidateEvidence],
        global_evidence: tuple[CandidateEvidence, ...],
        assessments: list[RiskAssessment],
        physical_timeline: Any,
    ) -> tuple[list[CandidateEvidence], dict[str, Any]]:
        """Admit only bounded global candidates as replacement evidence."""
        assessments_by_id = {item.candidate_id: item for item in assessments}
        decision_engine = EvidenceDecisionEngine()
        alternatives: list[CandidateEvidence] = []
        accepted_ids: set[str] = set()
        rejected: list[dict[str, Any]] = []
        considered = 0

        clips = tuple(getattr(physical_timeline, "physical_clips", ()) or ())
        for candidate in segmented:
            assessment = assessments_by_id.get(candidate.id)
            if assessment is None or not (
                assessment.level in {"high", "critical"}
                or "global_text_conflict" in assessment.evidence_codes
            ):
                continue
            for observation in global_evidence:
                if observation.source != "global":
                    continue
                if min(candidate.end, observation.end) <= max(candidate.start, observation.start):
                    continue
                considered += 1
                reasons: list[str] = []
                if not observation.has_word_times:
                    reasons.append("missing_word_timestamps")
                if not observation.window_id:
                    reasons.append("missing_window_id")
                if (
                    candidate.language
                    and observation.language
                    and candidate.language.casefold() != observation.language.casefold()
                ):
                    reasons.append("language_mismatch")
                physical = decision_engine._validate_physical(
                    observation,
                    physical_timeline,
                )
                if not physical["valid"]:
                    reasons.append(str(physical.get("status", "physical_validation_failed")))
                if clips:
                    overlapping_clip_ids = {
                        clip.id
                        for clip in clips
                        if min(observation.end, clip.end)
                        > max(observation.start, clip.start)
                    }
                    if len(overlapping_clip_ids) > 1:
                        reasons.append("cross_physical_clip")
                if normalize_text(candidate.text) == normalize_text(observation.text):
                    reasons.append("no_text_change")
                if reasons:
                    rejected.append({
                        "candidate_id": observation.id,
                        "segmented_candidate_id": candidate.id,
                        "reasons": list(dict.fromkeys(reasons)),
                    })
                    continue
                if observation.id in accepted_ids:
                    continue
                accepted = replace(
                    observation,
                    candidate_role="global_alternative",
                    alternative_for=candidate.id,
                    source_id=observation.source_id or observation.id,
                    trace_context={
                        **observation.trace_context,
                        "candidate_role": "global_alternative",
                        "alternative_for": candidate.id,
                    },
                    diagnostics={
                        **observation.diagnostics,
                        "evidence_role": "global_alternative",
                        "alternative_for": candidate.id,
                        "physical_validation": physical,
                    },
                )
                alternatives.append(accepted)
                accepted_ids.add(accepted.id)

        return alternatives, {
            "role": "global_signal_and_alternative",
            "candidate_role": "global_alternative",
            "alternative_admission": "enabled",
            "signal_count": len(global_evidence),
            "considered_overlap_count": considered,
            "considered_alternative_count": considered,
            "accepted_alternative_count": len(alternatives),
            "selected_global_count": 0,
            "accepted_candidate_ids": [item.id for item in alternatives],
            "rejected_count": len(rejected),
            "rejected": rejected,
        }

    @staticmethod
    def _rescore_after_context(
        candidates: list[CandidateEvidence],
        assessments: list[RiskAssessment],
        review_candidates: list[CandidateEvidence],
        config: Any,
    ) -> list[RiskAssessment]:
        """Re-score residual risk after Context Re-ASR evidence arrives."""
        updated: list[RiskAssessment] = []
        for assessment in assessments:
            candidate = next(
                (item for item in candidates if item.id == assessment.candidate_id),
                None,
            )
            if candidate is None:
                updated.append(assessment)
                continue
            alternatives = [
                item for item in review_candidates
                if min(candidate.end, item.end) > max(candidate.start, item.start)
            ]
            if not alternatives:
                updated.append(assessment)
                continue
            similarity = max(
                _text_similarity(candidate.text, item.text)
                for item in alternatives
            )
            if similarity >= 0.75:
                delta = -0.25
                code = "context_reasr_agreement"
            else:
                delta = 0.15
                code = "context_reasr_conflict"
            score = min(1.0, max(0.0, assessment.score + delta))
            level = _risk_level(score, config)
            updated.append(replace(
                assessment,
                score=round(score, 6),
                level=level,
                evidence_codes=tuple(dict.fromkeys((*assessment.evidence_codes, code))),
                factors={**assessment.factors, "context_reasr_residual": delta},
                review_required=level in {"medium", "high", "critical"},
            ))
        return updated

    @staticmethod
    def _apply_secondary_evidence(
        assessments: list[RiskAssessment],
        secondary: dict[str, Any],
        windows: list[Any],
        config: Any,
    ) -> list[RiskAssessment]:
        """Fold structured auxiliary signals into risk without deleting text."""
        codes_by_candidate: dict[str, set[str]] = defaultdict(set)
        candidate_ids_by_window = {
            window.id: tuple(window.candidate_ids) for window in windows
        }

        for item in secondary.get("forced_aligner", {}).get("windows", ()):
            if item.get("status") == "ok" and item.get("word_count", 0) > 0:
                candidate_id = item.get("candidate_id")
                if candidate_id:
                    codes_by_candidate[candidate_id].add("forced_alignment_observed")

        for item in secondary.get("sed", {}).get("windows", ()):
            if item.get("status") != "ok":
                continue
            evidence = item.get("evidence") or {}
            label = str(evidence.get("label", evidence.get("class", ""))).casefold()
            score = float(evidence.get("score", evidence.get("confidence", 0.0)) or 0.0)
            if score < 0.7 or not any(
                token in label for token in ("breath", "music", "noise", "non_speech", "non-speech")
            ):
                continue
            for candidate_id in candidate_ids_by_window.get(item.get("window_id"), ()):
                codes_by_candidate[candidate_id].add("sed_non_speech")

        for item in secondary.get("semantic_review", {}).get("windows", ()):
            if item.get("status") != "ok":
                continue
            evidence = item.get("evidence") or {}
            risk = str(
                evidence.get("risk", evidence.get("class", evidence.get("label", "")))
            ).casefold()
            if any(token in risk for token in ("non_speech", "non-speech", "hallucination")):
                candidate_id = item.get("candidate_id")
                if candidate_id:
                    codes_by_candidate[candidate_id].add("semantic_non_speech")

        updated: list[RiskAssessment] = []
        for assessment in assessments:
            codes = codes_by_candidate.get(assessment.candidate_id, set())
            if not codes:
                updated.append(assessment)
                continue
            bump = 0.10 if "semantic_non_speech" in codes else 0.05
            score = min(1.0, assessment.score + bump)
            if score >= config.critical_threshold:
                level = "critical"
            elif score >= config.high_threshold:
                level = "high"
            elif score >= config.medium_threshold:
                level = "medium"
            else:
                level = "low"
            updated.append(replace(
                assessment,
                score=round(score, 6),
                level=level,
                evidence_codes=tuple(dict.fromkeys((*assessment.evidence_codes, *sorted(codes)))),
                factors={**assessment.factors, "secondary_evidence": bump},
                review_required=level in {"medium", "high", "critical"},
            ))
        return updated

    @staticmethod
    def _optional_capabilities(config: Any, ports: EvidenceReviewRuntimePorts) -> dict[str, dict[str, Any]]:
        capabilities = {
            "qwen": ("qwen_enabled", ports.qwen),
            "forced_aligner": ("forced_aligner_enabled", ports.forced_aligner),
            "sed": ("sed_enabled", ports.sed),
            "semantic_review": ("semantic_review_enabled", ports.semantic_review),
        }
        result = {}
        for name, (flag, port) in capabilities.items():
            enabled = bool(getattr(config, flag, False))
            availability = None
            if enabled and port is not None and hasattr(port, "availability"):
                try:
                    availability = port.availability()
                except Exception as exc:
                    availability = {
                        "status": "unavailable",
                        "reason": "availability_check_failed",
                        "error": str(exc),
                    }
            result[name] = {
                "enabled": enabled,
                "status": (
                    availability.get("status")
                    if availability is not None
                    else "ready" if enabled and port is not None else "disabled" if not enabled else "unavailable"
                ),
                "engine": getattr(port, "name", None),
                "reason": (
                    availability.get("reason")
                    if availability is not None
                    else None if enabled and port is not None else "feature_disabled" if not enabled else "port_missing"
                ),
            }
            if availability:
                result[name]["model"] = availability.get("model")
        return result

    def _review_windows(
        self,
        scheduler: ReviewScheduler,
        audio: Any,
        sample_rate: int,
        windows: list[Any],
        engine: ReviewEnginePort,
        *,
        language: Optional[str],
        request: EvidenceReviewRequest,
        ports: EvidenceReviewRuntimePorts,
        audio_hash: str,
        phase: str = "context_reasr",
    ) -> tuple[list[CandidateEvidence], dict[str, Any]]:
        """Review windows with cache policy isolated from scheduling policy."""
        all_candidates: list[CandidateEvidence] = []
        diagnostics: dict[str, Any] = {
            "engine": getattr(engine, "name", "unknown"),
            "windows": [],
            "cache_enabled": ports.cache is not None,
            "phase": phase,
        }
        stage_started = time.perf_counter()
        cache_usable = ports.cache is not None and bool(request.input_hash or audio_hash)
        pending_windows = []
        cache_diagnostics: dict[str, dict[str, Any]] = {}
        for window in windows:
            cache_key = None
            if cache_usable:
                context = EvidenceCacheKeyContext(
                    input_hash=request.input_hash,
                    audio_hash=audio_hash,
                    physical_timeline_version=request.physical_timeline_version,
                    window_start=window.start,
                    window_end=window.end,
                    phase=phase,
                    engine=(request.engine if phase == "context_reasr" else getattr(engine, "name", "")),
                    model=(request.model if phase == "context_reasr" else getattr(engine, "model_name", None) or request.model),
                    language=language,
                    route_version=request.route_version,
                    sample_rate=sample_rate,
                    physical_clip_id=window.physical_clip_id or "",
                    review_policy_version=request.review_policy_version,
                    pair_route_version=request.pair_route_version or request.route_version,
                    evidence_schema_version=request.evidence_schema_version,
                )
                cache_key = context.key()
                try:
                    cached = ports.cache.get(EVIDENCE_CACHE_STAGE, cache_key)
                    candidates = self._decode_cached_candidates(cached)
                except Exception:
                    candidates = None
                if candidates is not None:
                    all_candidates.extend(candidates)
                    cache_diagnostics[window.id] = {
                        "window_id": window.id,
                        "start": window.start,
                        "end": window.end,
                        "candidate_ids": list(window.candidate_ids),
                        "status": "cache_hit",
                        "candidate_count": len(candidates),
                        "cache_hit": True,
                        "cache_key": cache_key,
                        "wall_time_seconds": 0.0,
                        "resources": resource_snapshot(),
                    }
                    continue
            pending_windows.append((window, cache_key))
        if pending_windows:
            executor = ports.window_executor or WindowExecutionCoordinator(
                max_workers=max(1, int(getattr(ports.config, "max_workers", 2))),
                timeout_seconds=getattr(ports.config, "window_timeout_seconds", 60.0),
            )
            batch_candidates, batch_diagnostics = executor.execute(
                audio,
                sample_rate,
                [item[0] for item in pending_windows],
                engine,
                language=language,
            )
            candidates_by_window = {
                item["window_id"]: [] for item in batch_diagnostics.get("windows", ())
            }
            for candidate in batch_candidates:
                candidate_window_id = getattr(candidate, "window_id", None)
                if candidate_window_id in candidates_by_window:
                    candidates_by_window[candidate_window_id].append(candidate)
                    continue
                # Test ports and third-party adapters may omit window_id.  A
                # bounded result can still be assigned by absolute overlap.
                for pending_window, _cache_key in pending_windows:
                    if min(candidate.end, pending_window.end) > max(candidate.start, pending_window.start):
                        candidates_by_window[pending_window.id].append(candidate)
                        break
            for window, cache_key in pending_windows:
                item_diag = next(
                    (item for item in batch_diagnostics.get("windows", ()) if item["window_id"] == window.id),
                    {"window_id": window.id, "status": "cancelled", "reason": "batch_cancelled"},
                )
                candidates = candidates_by_window.get(window.id, [])
                item_diag = dict(item_diag)
                item_diag["cache_hit"] = False
                if not item_diag.get("resources"):
                    item_diag["resources"] = resource_snapshot()
                if cache_key is not None:
                    item_diag["cache_key"] = cache_key
                    if item_diag.get("status") == "ok":
                        try:
                            ports.cache.set(
                                EVIDENCE_CACHE_STAGE,
                                cache_key,
                                {
                                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                                    "candidates": [item.to_dict() for item in candidates],
                                },
                                ttl=ports.cache_ttl,
                            )
                            item_diag["cache_stored"] = True
                        except Exception as exc:
                            item_diag["cache_stored"] = False
                            item_diag["cache_error"] = str(exc)
                all_candidates.extend(candidates)
                cache_diagnostics[window.id] = item_diag
            diagnostics["executor"] = batch_diagnostics
        diagnostics["windows"] = [
            cache_diagnostics[window.id]
            for window in windows
            if window.id in cache_diagnostics
        ]
        diagnostics["reviewed_window_count"] = len(windows)
        diagnostics["candidate_count"] = len(all_candidates)
        diagnostics["wall_time_seconds"] = round(time.perf_counter() - stage_started, 6)
        diagnostics["resources"] = resource_snapshot()
        diagnostics["degraded"] = any(
            item.get("degraded") or item.get("status") not in {"ok", "cache_hit"}
            for item in diagnostics["windows"]
        )
        diagnostics["status"] = "degraded" if diagnostics["degraded"] else "ok"
        if ports.cache is not None and not cache_usable:
            diagnostics["cache_reason"] = "missing_input_or_audio_hash"
        return all_candidates, diagnostics

    @staticmethod
    def _decode_cached_candidates(value: Any) -> Optional[list[CandidateEvidence]]:
        if not isinstance(value, dict) or value.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            return None
        payload = value.get("candidates")
        if not isinstance(payload, list):
            return None
        try:
            return [CandidateEvidence.from_dict(item) for item in payload]
        except (TypeError, ValueError):
            return None


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def _risk_level(score: float, config: Any) -> str:
    if score >= config.critical_threshold:
        return "critical"
    if score >= config.high_threshold:
        return "high"
    if score >= config.medium_threshold:
        return "medium"
    return "low"


__all__ = [
    "EvidenceReviewRuntimePorts",
    "EvidenceReviewService",
]
