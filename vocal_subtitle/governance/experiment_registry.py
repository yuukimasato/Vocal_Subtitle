"""实验注册表

管理所有实验性引擎、模型和配置组合的生命周期。
对应 EXPERIMENT_REGISTRY.md (experiment-registry-v1)。

实验状态机:
  proposed → shadow → review → enabled
      ↑                    ↓         ↓
      └──────────────── rolled_back ←─┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    SHADOW = "shadow"
    REVIEW = "review"
    ENABLED = "enabled"
    ROLLED_BACK = "rolled_back"


class ExperimentCategory(str, Enum):
    ENGINE = "engine"
    MODEL = "model"
    QUANTIZATION = "quantization"
    CONFIG = "config"
    COMPOSITE = "composite"


ALLOWED_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.PROPOSED: frozenset({ExperimentStatus.SHADOW}),
    ExperimentStatus.SHADOW: frozenset(
        {ExperimentStatus.REVIEW, ExperimentStatus.ROLLED_BACK}
    ),
    ExperimentStatus.REVIEW: frozenset(
        {ExperimentStatus.ENABLED, ExperimentStatus.ROLLED_BACK}
    ),
    ExperimentStatus.ENABLED: frozenset({ExperimentStatus.ROLLED_BACK}),
    ExperimentStatus.ROLLED_BACK: frozenset({ExperimentStatus.PROPOSED}),
}


@dataclass
class ExperimentRecord:
    """单个实验记录"""

    experiment_id: str
    name: str
    category: ExperimentCategory = ExperimentCategory.CONFIG
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    engines: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    expected_benefit: str = ""
    known_risks: list[str] = field(default_factory=list)
    enable_scope: str = "shadow_only"  # shadow_only | opt_in | language_zh | all
    withdraw_conditions: list[str] = field(default_factory=list)
    validation_samples: list[str] = field(default_factory=list)
    owner: str = ""
    created: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "engines": self.engines,
            "models": self.models,
            "languages": self.languages,
            "expected_benefit": self.expected_benefit,
            "known_risks": self.known_risks,
            "enable_scope": self.enable_scope,
            "withdraw_conditions": self.withdraw_conditions,
            "validation_samples": self.validation_samples,
            "owner": self.owner,
            "created": self.created,
        }


class ExperimentRegistry:
    """管理所有注册实验的生命周期。

    使用示例:
        registry = ExperimentRegistry()
        registry.list_by_status("shadow")
        registry.transition("exp-20260802-qwen-review", "review", reason="D1 passed")
    """

    def __init__(self):
        self._experiments: dict[str, ExperimentRecord] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册 EXPERIMENT_REGISTRY.md 中定义的 EXP-001 ~ EXP-006。"""
        today = "2026-08-02"

        experiments = [
            ExperimentRecord(
                experiment_id="exp-20260802-qwen-review",
                name="Qwen3-ASR 作为复核引擎从 Shadow 进入 Review",
                category=ExperimentCategory.ENGINE,
                status=ExperimentStatus.SHADOW,
                engines=["qwen-asr"],
                models=["Qwen3-ASR-1.7B"],
                languages=["zh", "en"],
                expected_benefit="为 EvidenceDecision 提供异质引擎的文本候选，提高召回率和文本准确性",
                known_risks=[
                    "GPU 内存额外占用 ~4GB",
                    "处理时间增加 1.5-2x",
                    "可能因 Qwen 错误导致替换退化",
                ],
                enable_scope="opt_in",
                withdraw_conditions=[
                    "D1 对照运行中文/英文覆盖率下降 > 5%",
                    "替换文本错误率 > 分段主候选",
                ],
                validation_samples=["D1-培训测试-双人", "D1-中文多人", "D1-英文多人"],
                owner="yuukimasato",
                created=today,
            ),
            ExperimentRecord(
                experiment_id="exp-20260802-forced-aligner",
                name="Qwen3-ForcedAligner 提供词级时间戳精修",
                category=ExperimentCategory.ENGINE,
                status=ExperimentStatus.SHADOW,
                engines=["qwen-forced-aligner"],
                models=["Qwen3-ForcedAligner-0.6B"],
                languages=["zh", "en"],
                expected_benefit="改进词级时间戳精度，减少时间 MAE",
                known_risks=[
                    "GPU 内存额外占用 ~2GB",
                    "仅在 ASR 文本准确时有效",
                    "处理时间增加 0.5-1x",
                ],
                enable_scope="shadow_only",
                withdraw_conditions=["时间 MAE 无显著改善或恶化"],
                validation_samples=["D1-中文多人", "D1-英文多人"],
                owner="yuukimasato",
                created=today,
            ),
            ExperimentRecord(
                experiment_id="exp-20260802-sed-non-speech",
                name="AST-AudioSet SED 检测非语音区域",
                category=ExperimentCategory.ENGINE,
                status=ExperimentStatus.SHADOW,
                engines=["ast-audioset"],
                models=["MIT/ast-finetuned-audioset-10-10-0.4593"],
                languages=["all"],
                expected_benefit="识别音乐、噪声等非语音区域，辅助 EvidenceDecision 的 drop 决策",
                known_risks=["可能过度标记（false positive）", "GPU 内存额外占用 ~1GB"],
                enable_scope="shadow_only",
                withdraw_conditions=[
                    "D4 过度检测率 > 15%",
                    "导致有效的语音字幕被错误 drop",
                ],
                validation_samples=["D4-音乐现场", "D4-高噪声"],
                owner="yuukimasato",
                created=today,
            ),
            ExperimentRecord(
                experiment_id="exp-20260802-vad-fusion",
                name="三方法 VAD 边界融合",
                category=ExperimentCategory.CONFIG,
                status=ExperimentStatus.SHADOW,
                engines=["silero-vad", "ffmpeg-vad", "rms-detector"],
                models=[],
                languages=["all"],
                expected_benefit="更精确的 VAD 边界，减少 ASR 漏识和过识别",
                known_risks=["三方法不一致时多数决可能选择次优边界", "CPU 开销增加"],
                enable_scope="shadow_only",
                withdraw_conditions=[
                    "D1 对照运行 VAD 边界精度无改善",
                    "引入额外的语音片段碎片化",
                ],
                validation_samples=["D1-培训测试-双人"],
                owner="yuukimasato",
                created=today,
            ),
            ExperimentRecord(
                experiment_id="exp-20260802-llm-optimize",
                name="LLM 字幕后处理优化",
                category=ExperimentCategory.COMPOSITE,
                status=ExperimentStatus.PROPOSED,
                engines=["llm-api"],
                models=["deepseek-v4-pro", "gpt-4o"],
                languages=["zh", "en"],
                expected_benefit="修正 ASR 常见错误（同音字、标点），优化断句可读性",
                known_risks=[
                    "可能改变语义",
                    "API 成本",
                    "增加处理延迟",
                    "不同 LLM 产出不一致",
                ],
                enable_scope="opt_in",
                withdraw_conditions=["语义改变率 > 5%", "用户报告不需要的文本修改"],
                validation_samples=["D1-培训测试-双人", "D1-英文多人"],
                owner="yuukimasato",
                created=today,
            ),
            ExperimentRecord(
                experiment_id="exp-20260802-noise-reduction",
                name="噪声抑制预处理",
                category=ExperimentCategory.CONFIG,
                status=ExperimentStatus.PROPOSED,
                engines=["spectral_gate"],
                models=[],
                languages=["all"],
                expected_benefit="降低背景噪声对 ASR 的影响",
                known_risks=["过度降噪可能损伤语音信号", "纯净录音场景下降质"],
                enable_scope="opt_in",
                withdraw_conditions=["D4 纯净场景质量下降"],
                validation_samples=["D4-高噪声", "D4-咖啡厅"],
                owner="yuukimasato",
                created=today,
            ),
        ]
        for exp in experiments:
            self._experiments[exp.experiment_id] = exp

    # ---- 查询 ----

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._experiments.get(experiment_id)

    def list_all(self) -> list[ExperimentRecord]:
        return list(self._experiments.values())

    def list_by_status(self, status: str) -> list[ExperimentRecord]:
        try:
            st = ExperimentStatus(status)
        except ValueError:
            return []
        return [e for e in self._experiments.values() if e.status == st]

    def list_by_category(self, category: str) -> list[ExperimentRecord]:
        try:
            cat = ExperimentCategory(category)
        except ValueError:
            return []
        return [e for e in self._experiments.values() if e.category == cat]

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self._experiments.items()}

    # ---- 状态转换 ----

    def transition(
        self,
        experiment_id: str,
        target_status: str,
        *,
        reason: str = "",
    ) -> bool:
        """转换实验状态。

        Args:
            experiment_id: 实验 ID
            target_status: 目标状态
            reason: 转换原因

        Returns:
            是否成功

        Raises:
            ValueError: 实验不存在或转换不合法
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Unknown experiment: {experiment_id!r}")

        try:
            target = ExperimentStatus(target_status)
        except ValueError:
            raise ValueError(f"Invalid status: {target_status!r}")

        if exp.status == target:
            return False

        allowed = ALLOWED_EXPERIMENT_TRANSITIONS.get(exp.status, frozenset())
        if target not in allowed:
            raise ValueError(
                f"Invalid transition for {experiment_id}: "
                f"{exp.status.value} → {target.value} not allowed. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        old = exp.status
        exp.status = target
        logger.info(
            "Experiment %s: %s → %s (reason: %s)",
            experiment_id,
            old.value,
            target.value,
            reason or "N/A",
        )
        return True

    def approve_to_shadow(self, experiment_id: str) -> bool:
        """proposed → shadow"""
        return self.transition(experiment_id, "shadow", reason="approved")

    def promote_to_review(self, experiment_id: str) -> bool:
        """shadow → review"""
        return self.transition(
            experiment_id, "review", reason="shadow validation passed"
        )

    def enable(self, experiment_id: str) -> bool:
        """review → enabled"""
        return self.transition(experiment_id, "enabled", reason="human sign-off")

    def rollback(self, experiment_id: str, reason: str = "regression detected") -> bool:
        """任意状态 → rolled_back"""
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Unknown experiment: {experiment_id!r}")
        # 回退是强制操作
        exp.status = ExperimentStatus.ROLLED_BACK
        logger.info("Experiment %s rolled_back: %s", experiment_id, reason)
        return True
