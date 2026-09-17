"""版本趋势报告器

基于 QUALITY_OPERATIONS.md §3 的版本趋势报告模板，
生成 YAML 格式的版本对比趋势报告。

跟踪维度:
  - D0_engineering: test_pass_rate, regression_count, build_time_seconds
  - D1_reference: scenarios, coverage_avg, text_accuracy_avg, time_mae_avg_ms, regressions
  - D3_feedback: status (not_available 或实际数据)
  - D4_challenge: 场景覆盖, 退化标记
  - operations: crash_rate, degradation_rate, avg_duration, revision_rate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 指标方向性: True = 越高越好, False = 越低越好
TREND_METRIC_DIRECTIONS: dict[str, bool] = {
    "test_pass_rate": True,
    "regression_count": False,
    "build_time_seconds": False,
    "scenarios": True,
    "coverage_avg": True,
    "text_accuracy_avg": True,
    "time_mae_avg_ms": False,
    "crash_rate": False,
    "degradation_rate": False,
    "avg_duration_seconds": False,
    "p95_duration_seconds": False,
    "user_revision_rate": False,
    "long_audio_pass": True,
    "heavy_noise_degraded": False,
    "mixed_language_acceptable": True,
    "high_severity_count": False,
}


@dataclass
class VersionMetrics:
    """单版本质量指标"""

    version: str
    date: str
    baseline: str = ""
    trends: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "date": self.date,
            "baseline": self.baseline,
            "trends": self.trends,
        }

    def to_yaml(self) -> str:
        """导出为 YAML 格式字符串。"""
        import yaml

        return yaml.safe_dump(
            self.to_dict(),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


@dataclass
class MetricChange:
    """单个指标的变更"""

    group: str  # D0_engineering, D1_reference, etc.
    metric: str  # test_pass_rate, coverage_avg, etc.
    old: Any
    new: Any
    delta: Any = None
    direction: str = "neutral"  # "improved" | "regressed" | "neutral"

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "metric": self.metric,
            "old": self.old,
            "new": self.new,
            "delta": self.delta,
            "direction": self.direction,
        }


@dataclass
class TrendDiff:
    """两个版本的趋势差异"""

    version: str
    baseline: str
    changes: list[MetricChange] = field(default_factory=list)
    regressions: list[MetricChange] = field(default_factory=list)

    @property
    def regression_count(self) -> int:
        return len(self.regressions)

    @property
    def improvement_count(self) -> int:
        return sum(1 for c in self.changes if c.direction == "improved")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "baseline": self.baseline,
            "changes": [c.to_dict() for c in self.changes],
            "regressions": [c.to_dict() for c in self.regressions],
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
        }


class TrendReporter:
    """版本趋势报告器。

    使用示例:
        reporter = TrendReporter()
        report = reporter.build_report(
            version="0.3.0", baseline="0.2.0",
            date="2026-08-15",
            trends={"D0_engineering": {"test_pass_rate": 0.98, ...}}
        )
        reporter.write_report(report)
        diff = reporter.compare(prev_report, curr_report)
    """

    def __init__(self, storage_dir: Path | None = None):
        self._storage_dir = Path(
            storage_dir
            or (Path(__file__).parent.parent.parent / "cache" / "quality" / "trends")
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ---- 构建与持久化 ----

    @staticmethod
    def build_report(
        version: str,
        baseline: str,
        trends: dict,
        *,
        date: str = "",
    ) -> VersionMetrics:
        """构建版本指标报告。

        Args:
            version: 当前版本号
            baseline: 基线版本号
            trends: 趋势数据字典（D0/D1/D3/D4/operations）
            date: 报告日期，默认今天

        Returns:
            VersionMetrics 实例
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        # 确保五组都存在
        filled = {
            "D0_engineering": trends.get("D0_engineering", {}),
            "D1_reference": trends.get(
                "D1_reference", {"scenarios": 0, "status": "not_available"}
            ),
            "D3_feedback": trends.get("D3_feedback", {"status": "not_available"}),
            "D4_challenge": trends.get(
                "D4_challenge", {"scenarios": 0, "status": "not_available"}
            ),
            "operations": trends.get("operations", {}),
        }

        return VersionMetrics(
            version=version,
            date=date,
            baseline=baseline,
            trends=filled,
        )

    def write_report(self, report: VersionMetrics) -> Path:
        """将趋势报告写入 YAML 文件。

        Args:
            report: VersionMetrics 实例

        Returns:
            写入的文件路径
        """
        safe_ver = report.version.replace("/", "-").replace(" ", "_")
        path = self._storage_dir / f"trend-{safe_ver}.yaml"
        path.write_text(report.to_yaml(), encoding="utf-8")
        logger.info("Trend report written: %s", path)
        return path

    def load_report(self, path: Path) -> VersionMetrics:
        """从 YAML 文件加载趋势报告。

        Args:
            path: YAML 文件路径

        Returns:
            VersionMetrics 实例
        """
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return VersionMetrics(
            version=data.get("version", ""),
            date=data.get("date", ""),
            baseline=data.get("baseline", ""),
            trends=data.get("trends", {}),
        )

    # ---- 对比 ----

    @staticmethod
    def compare(
        previous: VersionMetrics,
        current: VersionMetrics,
    ) -> TrendDiff:
        """比较两个版本的指标趋势。

        遍历两组 trends 的叶子节点，计算方向和 delta。

        Args:
            previous: 基线版本
            current: 当前版本

        Returns:
            TrendDiff 包含所有变更和退化列表
        """
        changes: list[MetricChange] = []
        regressions: list[MetricChange] = []

        for group in (
            "D0_engineering",
            "D1_reference",
            "D3_feedback",
            "D4_challenge",
            "operations",
        ):
            prev_group = previous.trends.get(group, {})
            curr_group = current.trends.get(group, {})

            # 跳过 not_available 的 D3
            if (
                isinstance(prev_group, dict)
                and prev_group.get("status") == "not_available"
            ):
                continue
            if (
                isinstance(curr_group, dict)
                and curr_group.get("status") == "not_available"
            ):
                continue

            if not isinstance(prev_group, dict) or not isinstance(curr_group, dict):
                continue

            for metric, new_val in curr_group.items():
                if metric == "regressions" or metric == "failures":
                    continue  # 跳过嵌套列表

                old_val = prev_group.get(metric)
                if old_val is None:
                    continue

                try:
                    old_num = (
                        float(old_val) if not isinstance(old_val, bool) else old_val
                    )
                    new_num = (
                        float(new_val) if not isinstance(new_val, bool) else new_val
                    )
                except (TypeError, ValueError):
                    continue

                if old_num == new_num:
                    direction = "neutral"
                else:
                    higher_is_better = TREND_METRIC_DIRECTIONS.get(metric, True)
                    if isinstance(old_num, bool):
                        direction = (
                            "improved" if new_num and not old_num else "regressed"
                        )
                    elif higher_is_better:
                        direction = "improved" if new_num > old_num else "regressed"
                    else:
                        direction = "improved" if new_num < old_num else "regressed"

                delta_val = None
                if isinstance(old_num, (int, float)) and isinstance(
                    new_num, (int, float)
                ):
                    delta_val = round(new_num - old_num, 4)

                change = MetricChange(
                    group=group,
                    metric=metric,
                    old=old_val,
                    new=new_val,
                    delta=delta_val,
                    direction=direction,
                )
                changes.append(change)
                if direction == "regressed":
                    regressions.append(change)

        return TrendDiff(
            version=current.version,
            baseline=previous.version,
            changes=changes,
            regressions=regressions,
        )

    # ---- 月度报告 ----

    @staticmethod
    def render_monthly_report(
        report: VersionMetrics,
        diff: TrendDiff | None = None,
        *,
        known_issues: list[dict] | None = None,
        next_plans: list[str] | None = None,
    ) -> str:
        """生成 QUALITY_OPERATIONS.md §7 格式的月度质量报告（Markdown）。

        Args:
            report: 当前版本指标
            diff: 版本对比结果（可选）
            known_issues: 已知问题 Top N 列表
            next_plans: 下月计划

        Returns:
            Markdown 格式字符串
        """
        lines = [
            f"# 月度质量报告 — {report.date[:7] if len(report.date) >= 7 else report.date}",
            "",
            "## 总体状态",
            "- 生产可用性: production-usable",
        ]

        issues = known_issues or []
        critical = sum(1 for i in issues if i.get("severity") == "critical")
        high = sum(1 for i in issues if i.get("severity") == "high")
        lines.append(f"- 关键问题: {critical}")
        lines.append(f"- 高优先级问题: {high}")

        if diff:
            lines.append("")
            lines.append(f"## 版本对比 ({diff.baseline} → {diff.version})")
            for c in diff.changes:
                icon = {"improved": "📈", "regressed": "📉", "neutral": "➡️"}.get(
                    c.direction, ""
                )
                delta_str = f" ({c.delta:+.4f})" if c.delta is not None else ""
                lines.append(
                    f"- {icon} {c.group}.{c.metric}: {c.old} → {c.new}{delta_str}"
                )

        lines.append("")
        lines.append("## 用户反馈统计")
        ops = report.trends.get("operations", {})
        revision_rate = ops.get("user_revision_rate", "N/A")
        lines.append("- 本月反馈数: N/A")
        lines.append(f"- 修订率: {revision_rate}")
        lines.append("- 主要投诉: N/A")

        if known_issues:
            lines.append("")
            lines.append("## 已知问题 Top 5")
            for i, issue in enumerate(known_issues[:5], 1):
                lines.append(
                    f"{i}. [{issue.get('severity', 'N/A').upper()}] "
                    f"{issue.get('title', 'Unknown')} "
                    f"— 影响范围 {issue.get('impact_range', '?')}, "
                    f"严重度 {issue.get('user_severity', '?')}"
                )

        if next_plans:
            lines.append("")
            lines.append("## 下月计划")
            for plan in next_plans:
                lines.append(f"- {plan}")

        return "\n".join(lines)


__all__ = [
    "VersionMetrics",
    "MetricChange",
    "TrendDiff",
    "TrendReporter",
    "TREND_METRIC_DIRECTIONS",
]
