"""发布、支持与维护治理

基于 RELEASE_GOVERNANCE.md (release-governance-v1) 定义的:
  - 发布状态分类 (§1): development → production-usable → quality-improving
  - production-usable 最低标准检查单 (§2)
  - 发布前检查单 (§3): 依赖/模型/测试/D0回归/冒烟/导出/文档
  - 发布后观测 (§4): 崩溃率/耗时/降级率/导出失败/修订率
  - 回滚机制 (§5): 配置 → 引擎禁用 → 代码 → 数据结论
  - 已知限制清单 (§6)
  - 版本说明模板 (§7)
  - 支持手册 (§8)
  - 当前发布状态 (§9)

当前为 governance 模块的 CLI-only 组件，不在 pipelines 执行路径中。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReleaseStatus(str, Enum):
    """发布状态分类 (§1)"""

    DEVELOPMENT = "development"
    PRODUCTION_USABLE = "production-usable"
    QUALITY_IMPROVING = "quality-improving"


class RollbackStrategy(str, Enum):
    """回滚策略优先级 (§5): 配置优先于代码回滚"""

    CONFIG = "config"
    ENGINE_DISABLE = "engine_disable"
    CODE_REVERT = "code_revert"
    DATA_SUPERSEDE = "data_supersede"


@dataclass
class KnownLimitation:
    """已知限制 (§6)"""

    limitation_id: str
    title: str
    description: str
    mitigation: str

    def to_dict(self) -> dict:
        return {
            "limitation_id": self.limitation_id,
            "title": self.title,
            "description": self.description,
            "mitigation": self.mitigation,
        }


@dataclass
class ObservabilityMetrics:
    """发布后观测指标 (§4)"""

    crash_rate: float = 0.0  # 崩溃/失败率
    avg_duration_seconds: float = 0.0  # 任务平均耗时
    degradation_rate: float = 0.0  # 降级率
    export_failure_rate: float = 0.0  # 导出失败
    user_revision_rate: float | None = None  # 用户修订率
    high_severity_count: int = 0  # 高严重度问题数
    previous_avg_duration_seconds: float | None = None  # 上一版本耗时

    def to_dict(self) -> dict:
        return {
            "crash_rate": self.crash_rate,
            "avg_duration_seconds": self.avg_duration_seconds,
            "degradation_rate": self.degradation_rate,
            "export_failure_rate": self.export_failure_rate,
            "user_revision_rate": self.user_revision_rate,
            "high_severity_count": self.high_severity_count,
        }


@dataclass
class AlertReport:
    """告警报告 (§4 阈值评估)"""

    alerts: list[dict] = field(default_factory=list)
    # alerts: [{"metric": ..., "observed": ..., "threshold": ..., "condition": ...}]

    def ok(self) -> bool:
        return len(self.alerts) == 0

    def to_dict(self) -> dict:
        return {"ok": self.ok(), "alerts": self.alerts}


@dataclass
class PreReleaseChecklist:
    """发布前检查单 (§3)"""

    version: str
    checked_at: str = ""
    sections: dict[str, list[dict]] = field(default_factory=dict)
    # sections: {"dependencies": [{"key": ..., "description": ..., "passed": bool|None, "note": ""}], ...}

    def all_passed(self) -> bool:
        for items in self.sections.values():
            for item in items:
                if item.get("passed") is not True:
                    return False
        return True

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "checked_at": self.checked_at,
            "sections": self.sections,
            "all_passed": self.all_passed(),
        }


@dataclass
class ReleaseRecord:
    """发布记录"""

    version: str
    status: ReleaseStatus = ReleaseStatus.DEVELOPMENT
    released_at: str = ""
    blockers: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "status": self.status.value,
            "released_at": self.released_at,
            "blockers": self.blockers,
            "metrics": self.metrics,
            "notes": self.notes,
        }


class ReleaseManager:
    """发布治理管理器。

    使用示例:
        mgr = ReleaseManager()
        status, blockers = mgr.classify(production_criteria)
        checklist = mgr.check_pre_release("0.2.0")
        mgr.verify_item(checklist.version, "dependencies", "lock_files", True)
        notes = mgr.release_notes(version="0.2.0", status=ReleaseStatus.PRODUCTION_USABLE,
                                   changes={...}, upgrades=[...])
    """

    # 告警阈值 (§4 表格)
    ALERT_THRESHOLDS: dict[str, Any] = {
        "crash_rate": 0.10,
        "avg_duration_change_pct": 50.0,
        "degradation_rate": 0.20,
        "export_failure_rate": 0.05,
        "user_revision_trend": "rising",
        "high_severity_count": 0,
    }

    # production-usable 最低标准 (§2)
    PRODUCTION_CRITERIA: list[tuple[str, str, bool]] = [
        ("install_script", "安装脚本 bash install.sh --production 可成功执行", True),
        ("deps_locked", "依赖锁定 (pyproject.toml + uv.lock)", True),
        ("default_task", "默认配置任务可完成并导出字幕", True),
        ("cli_run", "CLI vocal-subtitle run input.mp3 -o output.srt 可运行", True),
        ("webui_start", "WebUI vocal-subtitle-gui 可启动并提供图形界面", True),
        ("graceful_degradation", "可选引擎缺失时主链不崩溃，正确降级", True),
        ("actionable_errors", "失败任务返回可行动的错误原因", True),
        ("revision_entry", "人工修订入口可用 (CLI feedback + WebUI 上传修订)", True),
        ("d0_regression", "D0 回归测试全部通过", False),
        ("d1_replay", "少量 D1 回放验证", False),
        ("smoke_test", "CLI/Web 冒烟测试通过", False),
    ]

    # 已知限制 (§6 表格)
    KNOWN_LIMITATIONS: list[KnownLimitation] = [
        KnownLimitation(
            "LIM-001",
            "多人重叠场景 (>3人)",
            "说话人标注可能不准确",
            "启用 pyannote 全局聚类改善",
        ),
        KnownLimitation(
            "LIM-002", "高背景噪声", "ASR 准确率下降", "启用噪声抑制预处理"
        ),
        KnownLimitation(
            "LIM-003", "中英频繁切换", "语言检测可能出错", "手动指定 --language mixed"
        ),
        KnownLimitation(
            "LIM-004",
            "长音频 (>2h)",
            "处理时间线性增长，可能触发超时",
            "启用宏观切块分治",
        ),
        KnownLimitation("LIM-005", "专业词汇", "可能识别错误", "考虑启用 LLM 后处理"),
        KnownLimitation(
            "LIM-006", "远场录音", "VAD 可能漏检远距离语音", "降低 VAD 阈值"
        ),
        KnownLimitation(
            "LIM-007", "无 GPU 设备", "CPU 推理速度慢 5-10x", "使用 whisper.cpp 引擎"
        ),
        KnownLimitation(
            "LIM-008", "1GB 以下显存", "大模型无法加载", "使用 tiny/small 模型或 CPU"
        ),
    ]

    # 支持手册常见问题 (§8)
    SUPPORT_FAQ: list[dict] = [
        {
            "issue": "模型未找到",
            "diagnosis": "检查 ~/.cache/vocal-subtitle/",
            "resolution": "vocal-subtitle download-models --all",
        },
        {
            "issue": "CUDA out of memory",
            "diagnosis": "检查 nvidia-smi",
            "resolution": "--device cpu 或 --model small",
        },
        {
            "issue": "ffmpeg not found",
            "diagnosis": "which ffmpeg",
            "resolution": "sudo apt install ffmpeg",
        },
        {
            "issue": "端口被占用 (WebUI)",
            "diagnosis": "lsof -i :7860",
            "resolution": "vocal-subtitle-gui --port 8080",
        },
        {
            "issue": "处理超时",
            "diagnosis": "检查音频长度和可用内存",
            "resolution": "启用 macro_chunking 分治",
        },
        {
            "issue": "字幕时间偏移",
            "diagnosis": "对比参考字幕",
            "resolution": "检查 acoustic_validation.report",
        },
    ]

    # 日志位置 (§8)
    LOG_LOCATIONS: dict[str, str] = {
        "管道日志": "logs/pipeline.log",
        "WebUI 日志": "stdout (uvicorn)",
        "任务历史": "cache/task_history.db",
        "运行报告": "cache/reports/{run_id}/run_report.json",
        "声学校验": "cache/reports/{run_id}/diagnostics/acoustic_report.json",
    }

    def __init__(self, storage_dir: Path | None = None):
        self._storage_dir = Path(
            storage_dir
            or (
                Path(__file__).parent.parent.parent / "cache" / "governance" / "release"
            )
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ---- §1 状态分类 ----

    @classmethod
    def classify(
        cls,
        criteria: dict[str, bool] | None = None,
    ) -> tuple[ReleaseStatus, list[str]]:
        """根据 production 标准评估发布状态。

        Args:
            criteria: {key: passed} 映射，默认使用内置的生产标准

        Returns:
            (ReleaseStatus, 未通过的检查项列表)
        """
        if criteria is None:
            criteria = {key: checked for key, _, checked in cls.PRODUCTION_CRITERIA}

        blockers = [key for key, passed in criteria.items() if not passed]

        if blockers:
            return ReleaseStatus.DEVELOPMENT, blockers
        return ReleaseStatus.PRODUCTION_USABLE, []

    # ---- §3 发布前检查单 ----

    def check_pre_release(self, version: str) -> PreReleaseChecklist:
        """为指定版本创建发布前检查单。

        Args:
            version: 版本号

        Returns:
            PreReleaseChecklist（所有项目初始为未检查）
        """
        sections = {
            "dependencies": [
                {
                    "key": "lock_files",
                    "description": "pyproject.toml 依赖版本已锁定 + uv.lock 已更新",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "fresh_install",
                    "description": "bash install.sh --production --cpu 全新安装成功",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "pip_install",
                    "description": "pip install -e '.[all]' 无冲突",
                    "passed": None,
                    "note": "",
                },
            ],
            "models": [
                {
                    "key": "fw_model",
                    "description": "faster-whisper large-v3 可正常运行",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "silero_vad",
                    "description": "Silero VAD 模型可正常加载",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "funasr_model",
                    "description": "FunASR 模型路径配置正确（可选）",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "qwen_model",
                    "description": "Qwen3-ASR 模型路径配置正确（可选）",
                    "passed": None,
                    "note": "",
                },
            ],
            "tests": [
                {
                    "key": "pytest_all",
                    "description": "pytest tests/ -x --tb=short 全部通过",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "pytest_physical",
                    "description": "pytest tests/test_physical/ -v 物理层测试通过",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "pytest_router",
                    "description": "pytest tests/test_asr/test_router.py -v 路由测试通过",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "pytest_cli",
                    "description": "pytest tests/test_cli.py -v CLI 测试通过",
                    "passed": None,
                    "note": "",
                },
            ],
            "d0_regression": [
                {
                    "key": "quality_benchmark",
                    "description": "python scripts/run_quality_benchmark.py 无可检测回归",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "coverage",
                    "description": "D0 样本无覆盖率下降 > 5%",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "hallucination",
                    "description": "D0 样本无新增幻觉",
                    "passed": None,
                    "note": "",
                },
            ],
            "smoke": [
                {
                    "key": "cli_run",
                    "description": "CLI: vocal-subtitle run test/中文多人员测试音频.wav -o /tmp/test.srt",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "cli_info",
                    "description": "CLI: vocal-subtitle info 无异常",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "webui_flow",
                    "description": "WebUI: 启动 → 上传 → 处理 → 下载 完整流程",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "webui_edit",
                    "description": "WebUI: 字幕编辑功能可用",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "webui_feedback",
                    "description": "WebUI: 反馈上传功能可用",
                    "passed": None,
                    "note": "",
                },
            ],
            "export": [
                {
                    "key": "srt",
                    "description": "SRT 格式可用标准播放器打开",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "vtt",
                    "description": "VTT 格式可用浏览器打开",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "ass",
                    "description": "ASS 格式可用 Aegisub 打开",
                    "passed": None,
                    "note": "",
                },
            ],
            "docs": [
                {
                    "key": "readme",
                    "description": "README 示例命令可执行",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "architecture",
                    "description": "ARCHITECTURE_STATE.md 已更新",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "doc_index",
                    "description": "DOCUMENT_INDEX.md 已更新",
                    "passed": None,
                    "note": "",
                },
                {
                    "key": "version",
                    "description": "版本号 pyproject.toml version 已更新",
                    "passed": None,
                    "note": "",
                },
            ],
        }

        now = datetime.now(timezone.utc).isoformat()
        checklist = PreReleaseChecklist(
            version=version,
            checked_at=now,
            sections=sections,
        )
        self._save_checklist(checklist)
        return checklist

    def verify_item(
        self,
        version: str,
        section: str,
        item_key: str,
        passed: bool,
        note: str = "",
    ) -> bool:
        """标记检查单中的一个项目。

        Args:
            version: 版本号
            section: 检查单分组 (dependencies, models, tests, ...)
            item_key: 项目 key
            passed: 是否通过
            note: 备注

        Returns:
            是否成功更新
        """
        checklist = self._load_checklist(version)
        if checklist is None:
            logger.warning("Checklist not found: %s", version)
            return False

        if section not in checklist.sections:
            return False

        for item in checklist.sections[section]:
            if item["key"] == item_key:
                item["passed"] = passed
                item["note"] = note
                checklist.checked_at = datetime.now(timezone.utc).isoformat()
                self._save_checklist(checklist)
                return True

        return False

    # ---- §4 发布后观测 ----

    @classmethod
    def observe(cls, metrics: ObservabilityMetrics) -> AlertReport:
        """对发布后指标进行告警评估。

        Args:
            metrics: 观测指标

        Returns:
            AlertReport 包含所有触发的告警
        """
        alerts: list[dict] = []

        if metrics.crash_rate > cls.ALERT_THRESHOLDS["crash_rate"]:
            alerts.append(
                {
                    "metric": "crash_rate",
                    "observed": metrics.crash_rate,
                    "threshold": cls.ALERT_THRESHOLDS["crash_rate"],
                    "condition": f"> {cls.ALERT_THRESHOLDS['crash_rate']}",
                }
            )

        if (
            metrics.previous_avg_duration_seconds
            and metrics.previous_avg_duration_seconds > 0
        ):
            change_pct = (
                (metrics.avg_duration_seconds - metrics.previous_avg_duration_seconds)
                / metrics.previous_avg_duration_seconds
                * 100
            )
            if change_pct > cls.ALERT_THRESHOLDS["avg_duration_change_pct"]:
                alerts.append(
                    {
                        "metric": "avg_duration_change_pct",
                        "observed": round(change_pct, 1),
                        "threshold": cls.ALERT_THRESHOLDS["avg_duration_change_pct"],
                        "condition": f"> +{cls.ALERT_THRESHOLDS['avg_duration_change_pct']}%",
                    }
                )

        if metrics.degradation_rate > cls.ALERT_THRESHOLDS["degradation_rate"]:
            alerts.append(
                {
                    "metric": "degradation_rate",
                    "observed": metrics.degradation_rate,
                    "threshold": cls.ALERT_THRESHOLDS["degradation_rate"],
                    "condition": f"> {cls.ALERT_THRESHOLDS['degradation_rate']}",
                }
            )

        if metrics.export_failure_rate > cls.ALERT_THRESHOLDS["export_failure_rate"]:
            alerts.append(
                {
                    "metric": "export_failure_rate",
                    "observed": metrics.export_failure_rate,
                    "threshold": cls.ALERT_THRESHOLDS["export_failure_rate"],
                    "condition": f"> {cls.ALERT_THRESHOLDS['export_failure_rate']}",
                }
            )

        if metrics.high_severity_count > cls.ALERT_THRESHOLDS["high_severity_count"]:
            alerts.append(
                {
                    "metric": "high_severity_count",
                    "observed": metrics.high_severity_count,
                    "threshold": cls.ALERT_THRESHOLDS["high_severity_count"],
                    "condition": f"> {cls.ALERT_THRESHOLDS['high_severity_count']}",
                }
            )

        return AlertReport(alerts=alerts)

    # ---- §5 回滚机制 ----

    @staticmethod
    def rollback_config(profile_name: str) -> dict:
        """生成配置回滚指令。

        Args:
            profile_name: profile 名称

        Returns:
            {"strategy": "config", "command": "...", "description": "..."}
        """
        return {
            "strategy": RollbackStrategy.CONFIG.value,
            "command": "vocal-subtitle feedback rollback",
            "description": f"回滚 profile '{profile_name}' 到上一个版本",
        }

    @staticmethod
    def disable_engine(engine_name: str) -> dict:
        """生成引擎禁用覆盖配置。

        Args:
            engine_name: 引擎名 (如 qwen, forced_aligner, sed)

        Returns:
            {"strategy": "engine_disable", "override_yaml": "...", "description": "..."}
        """
        engine_overrides = {
            "qwen": "evidence_review:\n  qwen_enabled: false",
            "forced_aligner": "evidence_review:\n  forced_aligner_enabled: false",
            "sed": "evidence_review:\n  sed_enabled: false",
            "fusion": "fusion:\n  enabled: false",
            "llm_optimize": "llm_optimize:\n  enabled: false",
            "noise_reduction": "noise_reduction:\n  enabled: false",
        }
        override_yaml = engine_overrides.get(engine_name, f"# 手动禁用 {engine_name}")

        return {
            "strategy": RollbackStrategy.ENGINE_DISABLE.value,
            "engine": engine_name,
            "override_yaml": override_yaml,
            "description": f"独立禁用 {engine_name} 引擎，无需修改代码",
        }

    @staticmethod
    def rollback_code(commit: str) -> dict:
        """生成代码回滚指令。

        Args:
            commit: 要 revert 的 commit SHA

        Returns:
            {"strategy": "code_revert", "command": "...", "description": "..."}
        """
        return {
            "strategy": RollbackStrategy.CODE_REVERT.value,
            "command": f"git revert {commit}",
            "alternative": "git checkout <previous-tag>",
            "description": f"Revert commit {commit}",
        }

    def supersede_dataset(self, version_id: str, reason: str = "") -> dict:
        """标记数据集结论为无效。

        Args:
            version_id: 数据集版本 ID
            reason: 淘汰原因

        Returns:
            操作结果
        """
        from ..quality.data_version_manager import DatasetTier, DataVersionManager

        dvm = DataVersionManager()
        for tier in DatasetTier:
            success = dvm.supersede(tier, version_id, reason=reason)
            if success:
                return {
                    "strategy": RollbackStrategy.DATA_SUPERSEDE.value,
                    "version_id": version_id,
                    "status": "superseded",
                    "description": f"数据集 {version_id} 结论已标记为无效，需重新评估",
                }

        return {
            "strategy": RollbackStrategy.DATA_SUPERSEDE.value,
            "version_id": version_id,
            "status": "not_found",
            "description": f"未找到数据集版本: {version_id}",
        }

    # ---- §7 版本说明 ----

    @classmethod
    def release_notes(
        cls,
        *,
        version: str,
        status: ReleaseStatus,
        changes: dict | None = None,
        upgrades: list[str] | None = None,
        known_issues: list[str] | None = None,
    ) -> str:
        """生成 RELEASE_GOVERNANCE.md §7 格式的版本说明（Markdown）。

        Args:
            version: 版本号
            status: 发布状态
            changes: {"新增": [...], "修复": [...], "优化": [...], "废弃": [...]}
            upgrades: 升级注意事项
            known_issues: 已知问题列表
            notes: 额外说明

        Returns:
            Markdown 格式字符串
        """
        changes = changes or {}
        upgrades = upgrades or []
        known_issues = known_issues or []

        lines = [
            f"# Vocal Subtitle v{version} 发布说明",
            "",
            "## 发布状态",
            status.value,
            "",
            "## 主要变更",
        ]

        for section in ("新增", "修复", "优化", "废弃"):
            items = changes.get(section, [])
            if items:
                lines.append(f"- **{section}**:")
                for item in items:
                    lines.append(f"  - {item}")

        if upgrades:
            lines.append("")
            lines.append("## 升级注意事项")
            for u in upgrades:
                lines.append(f"- {u}")

        if known_issues:
            lines.append("")
            lines.append("## 已知问题")
            for i, issue in enumerate(known_issues, 1):
                lines.append(f"{i}. {issue}")

        lines.append("")
        lines.append("## 安装")
        lines.append("```bash")
        lines.append("bash install.sh --production")
        lines.append("```")
        lines.append("")
        lines.append("## 验证")
        lines.append("```bash")
        lines.append("pytest tests/ -x")
        lines.append("```")

        return "\n".join(lines)

    # ---- §6 §8 查询 ----

    @classmethod
    def list_known_limitations(cls) -> list[dict]:
        """列出所有已知限制。"""
        return [lim.to_dict() for lim in cls.KNOWN_LIMITATIONS]

    @classmethod
    def support_manual(cls) -> dict:
        """返回支持手册（常见问题 + 日志位置）。"""
        return {
            "faq": cls.SUPPORT_FAQ,
            "log_locations": cls.LOG_LOCATIONS,
        }

    @classmethod
    def get_support_info(cls, issue_key: str) -> dict | None:
        """按关键词查找支持信息。

        Args:
            issue_key: 问题关键词（如 "模型未找到", "CUDA"）

        Returns:
            匹配的 FAQ 条目或 None
        """
        for faq in cls.SUPPORT_FAQ:
            if issue_key in faq["issue"]:
                return faq
        return None

    # ---- §9 当前状态 ----

    def register_release(
        self,
        version: str,
        status: ReleaseStatus,
        *,
        notes: str = "",
    ) -> ReleaseRecord:
        """注册一个发布版本。

        Args:
            version: 版本号
            status: 发布状态
            notes: 备注

        Returns:
            ReleaseRecord
        """
        now = datetime.now(timezone.utc).isoformat()
        record = ReleaseRecord(
            version=version,
            status=status,
            released_at=now,
            notes=notes,
        )
        self._save_release(record)
        return record

    def current_state(self) -> dict:
        """获取当前发布状态摘要。"""
        records = self._load_releases()
        if not records:
            return {
                "current_version": "0.2.0",
                "status": "development",
                "target": "production-usable",
                "blockers": [
                    "D0 回归测试未完全通过",
                    "D1 回放未建立",
                    "冒烟测试未完整覆盖",
                ],
            }
        latest = sorted(records, key=lambda r: r["released_at"], reverse=True)[0]
        return latest

    # ---- 持久化 ----

    def _checklist_path(self, version: str) -> Path:
        safe = version.replace("/", "-")
        return self._storage_dir / f"checklist-{safe}.json"

    def _save_checklist(self, checklist: PreReleaseChecklist) -> None:
        path = self._checklist_path(checklist.version)
        path.write_text(
            json.dumps(checklist.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_checklist(self, version: str) -> PreReleaseChecklist | None:
        path = self._checklist_path(version)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PreReleaseChecklist(
                version=data["version"],
                checked_at=data.get("checked_at", ""),
                sections=data.get("sections", {}),
            )
        except (OSError, json.JSONDecodeError):
            return None

    def _releases_path(self) -> Path:
        return self._storage_dir / "releases.json"

    def _load_releases(self) -> list[dict]:
        path = self._releases_path()
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save_release(self, record: ReleaseRecord) -> None:
        releases = self._load_releases()
        releases.append(record.to_dict())
        self._releases_path().write_text(
            json.dumps(releases, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = [
    "ReleaseStatus",
    "RollbackStrategy",
    "KnownLimitation",
    "ObservabilityMetrics",
    "AlertReport",
    "PreReleaseChecklist",
    "ReleaseRecord",
    "ReleaseManager",
]
