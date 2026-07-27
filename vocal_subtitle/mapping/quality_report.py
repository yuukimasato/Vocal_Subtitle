"""分阶段质量诊断模块

从单次 Pipeline 运行产出四维度质量报告，不需人工参考字幕。

四维度：
- 声学边界健康度 (acoustic)：物理覆盖率、边界置信度、噪声模型稳定性
- 语义断句健康度 (semantic)：分片合理性、LLM 操作合法性、投影成功/失败
- 说话人健康度 (speaker)：已确认/UNKNOWN 比例、变更检测信号类型
- 最终结构健康度 (structure)：非真实重叠、硬边界冲突、重复词、完整性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DimensionScore:
    """单个质量维度的得分与详情。"""

    name: str  # acoustic | semantic | speaker | structure
    score: float  # 0.0-1.0
    grade: str  # excellent | good | fair | poor
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """分阶段质量报告。

    四维度独立评分，不合并为单一数值。
    """

    acoustic: DimensionScore
    semantic: DimensionScore
    speaker: DimensionScore
    structure: DimensionScore
    pipeline_timing: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _dim(d: DimensionScore) -> Dict[str, Any]:
            return {
                "name": d.name,
                "score": d.score,
                "grade": d.grade,
                "issues": list(d.issues),
                "metrics": dict(d.metrics),
            }
        return {
            "acoustic": _dim(self.acoustic),
            "semantic": _dim(self.semantic),
            "speaker": _dim(self.speaker),
            "structure": _dim(self.structure),
            "pipeline_timing": dict(self.pipeline_timing),
            "resource_usage": dict(self.resource_usage),
            "diagnostics": dict(self.diagnostics),
        }


def _grade(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    elif score >= 0.70:
        return "good"
    elif score >= 0.50:
        return "fair"
    else:
        return "poor"


def build_quality_report(
    *,
    coverage_report: Optional[Dict[str, Any]] = None,
    noise_profile_diagnostics: Optional[Dict[str, Any]] = None,
    boundary_decision_stats: Optional[Dict[str, Any]] = None,
    allocation_diagnostics: Optional[Dict[str, Any]] = None,
    llm_window_diagnostics: Optional[Dict[str, Any]] = None,
    projection_diagnostics: Optional[Dict[str, Any]] = None,
    final_validation_diagnostics: Optional[Dict[str, Any]] = None,
    speaker_timeline_diagnostics: Optional[Dict[str, Any]] = None,
    pipeline_timing: Optional[Dict[str, float]] = None,
    event_count: int = 0,
) -> QualityReport:
    """从各阶段诊断数据构建质量报告。

    每个维度独立评分，缺失诊断数据将降级但不阻止报告生成。
    """
    # ── 声学边界健康度 ──────────────────────────────────────────────
    acoustic_issues: List[str] = []
    acoustic_metrics: Dict[str, Any] = {"event_count": event_count}

    if coverage_report:
        covered = coverage_report.get("covered_physical_bin_count", 0)
        total = coverage_report.get("physical_bin_count", 1)
        coverage_ratio = covered / max(total, 1)
        acoustic_metrics["physical_coverage_ratio"] = round(coverage_ratio, 3)
        acoustic_metrics["uncovered_bin_count"] = coverage_report.get(
            "uncovered_physical_bin_count", 0
        )
        if coverage_ratio < 0.5:
            acoustic_issues.append(
                f"physical_coverage_low: {coverage_ratio:.1%}"
            )
    else:
        acoustic_metrics["physical_coverage_ratio"] = None
        acoustic_issues.append("missing_coverage_report")

    if noise_profile_diagnostics:
        noise_stable = noise_profile_diagnostics.get("stable_regions", 0)
        noise_warnings = noise_profile_diagnostics.get("warning_count", 0)
        acoustic_metrics["noise_stable_regions"] = noise_stable
        acoustic_metrics["noise_warnings"] = noise_warnings
        if noise_warnings > 0:
            acoustic_issues.append(f"noise_profile_warnings: {noise_warnings}")

    if boundary_decision_stats:
        accepted = boundary_decision_stats.get("accepted_count", 0)
        rejected = boundary_decision_stats.get("rejected_count", 0)
        degraded = boundary_decision_stats.get("timing_degraded_count", 0)
        total_bd = accepted + rejected
        acceptance_rate = accepted / max(total_bd, 1)
        acoustic_metrics["boundary_acceptance_rate"] = round(acceptance_rate, 3)
        acoustic_metrics["timing_degraded_count"] = degraded
        if acceptance_rate < 0.5:
            acoustic_issues.append(
                f"boundary_acceptance_low: {acceptance_rate:.1%}"
            )
    else:
        acoustic_metrics["boundary_acceptance_rate"] = None

    acoustic_score = _dimension_score(
        issues=acoustic_issues,
        bonus=acoustic_metrics.get("physical_coverage_ratio", 0) or 0,
    )

    # ── 语义断句健康度 ──────────────────────────────────────────────
    semantic_issues: List[str] = []
    semantic_metrics: Dict[str, Any] = {}

    if allocation_diagnostics:
        rejected = allocation_diagnostics.get("rejected_count", 0)
        total = allocation_diagnostics.get("word_count", 1)
        semantic_metrics["word_allocation_ratio"] = round(
            1.0 - rejected / max(total, 1), 3
        )
        semantic_metrics["cross_boundary_count"] = allocation_diagnostics.get(
            "cross_boundary_count", 0
        )
    else:
        semantic_metrics["word_allocation_ratio"] = None

    if llm_window_diagnostics:
        failed_windows = llm_window_diagnostics.get("failed_windows", [])
        total_windows = llm_window_diagnostics.get("total_windows", 1)
        semantic_metrics["llm_window_success_rate"] = round(
            1.0 - len(failed_windows) / max(total_windows, 1), 3
        )
        semantic_metrics["llm_fallback_count"] = len(failed_windows)
        if failed_windows:
            semantic_issues.append(
                f"llm_windows_failed: {len(failed_windows)}"
            )
    else:
        semantic_metrics["llm_window_success_rate"] = None

    if projection_diagnostics:
        fallback = projection_diagnostics.get("fallback_count", 0)
        total = projection_diagnostics.get("total_groups", 1)
        semantic_metrics["projection_success_rate"] = round(
            1.0 - fallback / max(total, 1), 3
        )
        if fallback > 0:
            semantic_issues.append(f"projection_fallbacks: {fallback}")
    else:
        semantic_metrics["projection_success_rate"] = None

    semantic_score = _dimension_score(
        issues=semantic_issues,
        bonus=semantic_metrics.get("word_allocation_ratio", 0) or 0,
    )

    # ── 说话人健康度 ────────────────────────────────────────────────
    speaker_issues: List[str] = []
    speaker_metrics: Dict[str, Any] = {}

    if speaker_timeline_diagnostics:
        known = speaker_timeline_diagnostics.get("known_speaker_count", 0)
        unknown = speaker_timeline_diagnostics.get("unknown_speaker_count", 0)
        confirmed = speaker_timeline_diagnostics.get("confirmed_speaker_retention", 1.0)
        speaker_metrics["known_speaker_ratio"] = round(
            known / max(known + unknown, 1), 3
        )
        speaker_metrics["confirmed_retention"] = round(confirmed, 3)
        speaker_metrics["unknown_speaker_count"] = unknown
        if confirmed < 1.0:
            speaker_issues.append(
                f"confirmed_speaker_loss: {1.0 - confirmed:.1%}"
            )
    else:
        speaker_metrics["known_speaker_ratio"] = None
        speaker_metrics["confirmed_retention"] = None

    speaker_score = _dimension_score(
        issues=speaker_issues,
        bonus=speaker_metrics.get("confirmed_retention", 0) or 0,
    )

    # ── 结构健康度 ──────────────────────────────────────────────────
    structure_issues: List[str] = []
    structure_metrics: Dict[str, Any] = {"event_count": event_count}

    if final_validation_diagnostics:
        removed = final_validation_diagnostics.get("removed_count", 0)
        overlaps = final_validation_diagnostics.get("overlap_count", 0)
        overlap_trimmed = final_validation_diagnostics.get(
            "overlap_trimmed_count", 0
        )
        structure_metrics["validation_removed"] = removed
        structure_metrics["non_genuine_overlap_count"] = overlaps
        if overlaps > 0:
            structure_issues.append(f"non_genuine_overlaps: {overlaps}")
        if overlap_trimmed > 0:
            structure_issues.append(f"overlaps_trimmed: {overlap_trimmed}")
        if removed > 0:
            structure_issues.append(f"validation_removed: {removed}")
    else:
        structure_metrics["validation_removed"] = None

    structure_score = _dimension_score(
        issues=structure_issues,
        bonus=1.0 - 0.2 * len(structure_issues),
    )

    # ── 组装报告 ────────────────────────────────────────────────────
    return QualityReport(
        acoustic=DimensionScore(
            name="acoustic",
            score=acoustic_score,
            grade=_grade(acoustic_score),
            issues=acoustic_issues,
            metrics=acoustic_metrics,
        ),
        semantic=DimensionScore(
            name="semantic",
            score=semantic_score,
            grade=_grade(semantic_score),
            issues=semantic_issues,
            metrics=semantic_metrics,
        ),
        speaker=DimensionScore(
            name="speaker",
            score=speaker_score,
            grade=_grade(speaker_score),
            issues=speaker_issues,
            metrics=speaker_metrics,
        ),
        structure=DimensionScore(
            name="structure",
            score=structure_score,
            grade=_grade(structure_score),
            issues=structure_issues,
            metrics=structure_metrics,
        ),
        pipeline_timing=pipeline_timing or {},
        diagnostics={
            "has_coverage_report": coverage_report is not None,
            "has_noise_diagnostics": noise_profile_diagnostics is not None,
            "has_boundary_stats": boundary_decision_stats is not None,
            "has_allocation_diagnostics": allocation_diagnostics is not None,
            "has_llm_diagnostics": llm_window_diagnostics is not None,
            "has_projection_diagnostics": projection_diagnostics is not None,
            "has_validation_diagnostics": final_validation_diagnostics is not None,
        },
    )


def _dimension_score(
    *,
    issues: List[str],
    bonus: float,
) -> float:
    """计算单个维度的得分。

    基础分 1.0，每个 issue 扣 0.15，用 bonus 拉回。
    """
    base = max(0.0, 1.0 - 0.15 * len(issues))
    # bonus 填充缺失数据的影响
    if bonus > 0:
        base = max(base, bonus * 0.8)
    return round(max(0.0, min(1.0, base)), 3)
