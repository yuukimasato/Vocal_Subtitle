"""参数冲突检测与交互引导 (Phase 5.4)

检测参数学习的震荡行为（连续多次反馈朝相反方向调整同一参数），
生成 ConflictReport 供前端渲染交互式引导界面。

关键设计：不静默处理冲突，将决策权交还给用户。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OscillationEntry:
    """单次震荡记录"""

    timestamp: str
    direction: str  # "increase" | "decrease"
    param_value_before: float
    param_value_after: float
    delta: float
    summary: str


@dataclass
class ConflictReport:
    """参数冲突检测报告"""

    param_path: str
    is_oscillating: bool = False
    oscillation_count: int = 0  # 连续翻转次数
    entries: List[OscillationEntry] = field(default_factory=list)
    total_flips: int = 0
    recommended_action: str = ""  # "lock" | "branch" | "continue"
    possible_causes: List[str] = field(default_factory=list)
    suggested_actions: List[Dict[str, str]] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if self.oscillation_count >= 3:
            return "high"
        elif self.oscillation_count >= 2:
            return "medium"
        else:
            return "low"


class ConflictDetector:
    """参数震荡检测与交互引导

    检测逻辑:
    1. 取最近 window 次该参数的调整记录
    2. 计算调整方向的符号序列（+1 = 增大, -1 = 减小）
    3. 若符号序列中出现 >= 3 次翻转 → 判定为震荡
    4. 生成 ConflictReport 供前端渲染交互式引导
    """

    def __init__(self, window: int = 5):
        self._window = window

    def detect_oscillation(
        self,
        param_path: str,
        history: List[Dict[str, Any]],
    ) -> Optional[ConflictReport]:
        """检测参数调整震荡

        Args:
            param_path: 参数路径（如 "merging.padding"）
            history: 用户配置的 history 列表

        Returns:
            ConflictReport 或 None（无震荡）
        """
        # 从历史记录中提取该参数的调整
        entries = self._extract_param_history(param_path, history)
        if len(entries) < 2:
            return None

        # 取最近 window 条
        recent = entries[-self._window:]

        # 计算方向符号序列
        signs = []
        for e in recent:
            if e.delta > 0.001:
                signs.append(1)
            elif e.delta < -0.001:
                signs.append(-1)
            else:
                signs.append(0)

        # 过滤掉无变化的记录
        signs = [s for s in signs if s != 0]
        if len(signs) < 2:
            return None

        # 计算翻转次数
        flips = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                flips += 1

        report = ConflictReport(
            param_path=param_path,
            oscillation_count=flips,
            entries=recent,
            total_flips=flips,
        )

        if flips >= 3:
            report.is_oscillating = True
            report.recommended_action = "lock"
            report.possible_causes = [
                "不同音频类型需要不同的参数值（如播客 vs 会议）",
                "用户手动修订时的偶然不一致",
                "内容类型差异：即使同属一个场景模板，具体内容特征不同",
            ]
            report.suggested_actions = [
                {
                    "id": "lock",
                    "label": "🔒 锁定当前值",
                    "description": f"暂停 {param_path} 的自动学习，保持当前值不变",
                },
                {
                    "id": "branch",
                    "label": "📋 创建分支模板",
                    "description": "为不同音频类型创建独立的用户配置文件",
                },
                {
                    "id": "continue",
                    "label": "▶️ 继续学习",
                    "description": "忽略检测到的冲突，保持最近一次调整",
                },
                {
                    "id": "view_details",
                    "label": "📊 查看详情",
                    "description": "展开完整的冲突历史记录",
                },
            ]
            logger.warning(
                "Oscillation detected for '%s': %d flips in last %d adjustments",
                param_path, flips, len(recent),
            )

        elif flips >= 2:
            report.is_oscillating = True
            report.recommended_action = "continue"  # 2次翻转可能是巧合
            logger.info(
                "Mild oscillation for '%s': %d flips — monitoring",
                param_path, flips,
            )

        return report

    def detect_all_oscillations(
        self,
        history: List[Dict[str, Any]],
    ) -> List[ConflictReport]:
        """检测所有参数中的震荡

        Args:
            history: 用户配置的 history 列表

        Returns:
            震荡的 ConflictReport 列表（按严重程度降序）
        """
        # 收集所有被调整过的参数
        all_params = set()
        for entry in history:
            adjustments = entry.get("adjustments", {})
            all_params.update(adjustments.keys())

        reports = []
        for param_path in sorted(all_params):
            report = self.detect_oscillation(param_path, history)
            if report and report.is_oscillating:
                reports.append(report)

        # 按震荡次数降序
        reports.sort(key=lambda r: r.oscillation_count, reverse=True)
        return reports

    def resolve(
        self,
        report: ConflictReport,
        user_choice: str,
    ) -> Dict[str, Any]:
        """根据用户选择执行冲突解决

        Args:
            report: 冲突报告
            user_choice: "lock" | "branch" | "continue"

        Returns:
            解决操作的结果描述
        """
        result = {
            "param_path": report.param_path,
            "action": user_choice,
            "resolved_at": datetime.now().isoformat(),
        }

        if user_choice == "lock":
            result["status"] = "locked"
            result["message"] = f"参数 '{report.param_path}' 已被锁定，不再自动学习"
            logger.info("Param '%s' locked by user — auto-learning disabled", report.param_path)

        elif user_choice == "branch":
            result["status"] = "branch_requested"
            result["message"] = f"建议为不同音频类型创建独立配置模板"
            logger.info("Branch template requested for param '%s'", report.param_path)

        elif user_choice == "continue":
            result["status"] = "continued"
            result["message"] = f"忽略冲突警告，继续学习 '{report.param_path}'"
            logger.info("User chose to continue learning '%s' despite oscillation", report.param_path)

        else:
            result["status"] = "unknown"
            result["message"] = f"未知选择: {user_choice}"

        return result

    @staticmethod
    def _extract_param_history(
        param_path: str,
        history: List[Dict[str, Any]],
    ) -> List[OscillationEntry]:
        """从历史记录中提取某参数的全部调整序列"""
        entries = []
        for h in history:
            adjustments = h.get("adjustments", {})
            if param_path not in adjustments:
                continue

            vals = adjustments[param_path]
            if not isinstance(vals, list) or len(vals) < 2:
                continue

            old_val, new_val = float(vals[0]), float(vals[1])
            delta = new_val - old_val
            direction = "increase" if delta > 0 else "decrease"

            entries.append(OscillationEntry(
                timestamp=h.get("timestamp", ""),
                direction=direction,
                param_value_before=old_val,
                param_value_after=new_val,
                delta=delta,
                summary=h.get("diff_report_summary", ""),
            ))

        return entries

    @staticmethod
    def is_param_locked(
        param_path: str,
        locked_params: List[str],
    ) -> bool:
        """检查参数是否已被锁定"""
        return param_path in locked_params
