"""引擎生命周期状态机

对应 ENGINE_LIFECYCLE.md §1 定义的五态生命周期。

状态转换路径:
  unavailable ──(install deps)──► model_missing
  model_missing ──(download model)──► ready_shadow
  ready_shadow ──(enable review, pass shadow validation)──► ready_review
  ready_review ──(pass regression test + human check)──► ready_default
  ready_default ──(regression detected)──► ready_shadow (回退)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EngineLifecycle(str, Enum):
    """引擎生命周期五态枚举"""

    UNAVAILABLE = "unavailable"
    MODEL_MISSING = "model_missing"
    READY_SHADOW = "ready_shadow"
    READY_REVIEW = "ready_review"
    READY_DEFAULT = "ready_default"

    def __str__(self) -> str:
        return self.value


# 允许的状态转换
ALLOWED_LIFECYCLE_TRANSITIONS: dict[EngineLifecycle, frozenset[EngineLifecycle]] = {
    EngineLifecycle.UNAVAILABLE: frozenset({EngineLifecycle.MODEL_MISSING}),
    EngineLifecycle.MODEL_MISSING: frozenset({EngineLifecycle.READY_SHADOW}),
    EngineLifecycle.READY_SHADOW: frozenset(
        {EngineLifecycle.READY_REVIEW, EngineLifecycle.MODEL_MISSING}
    ),
    EngineLifecycle.READY_REVIEW: frozenset(
        {EngineLifecycle.READY_DEFAULT, EngineLifecycle.READY_SHADOW}
    ),
    EngineLifecycle.READY_DEFAULT: frozenset({EngineLifecycle.READY_SHADOW}),
}


@dataclass
class EngineStatus:
    """引擎状态记录"""

    engine: str
    model: str = ""
    status: EngineLifecycle = EngineLifecycle.UNAVAILABLE
    device: str = "cpu"
    model_path: str = ""
    model_hash: str = ""
    resource_requirement: str = ""  # 如 "GPU ~4GB"
    language_support: str = ""  # 如 "zh, en"
    degradation_strategy: str = ""  # 降级策略描述
    last_checked: str = ""
    last_transition: str = ""

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "model": self.model,
            "status": self.status.value,
            "device": self.device,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "resource_requirement": self.resource_requirement,
            "language_support": self.language_support,
            "degradation_strategy": self.degradation_strategy,
            "last_checked": self.last_checked,
            "last_transition": self.last_transition,
        }


class EngineRegistry:
    """引擎注册表 — 管理所有引擎及其生命周期状态。

    对应 ENGINE_LIFECYCLE.md §2 的完整引擎矩阵。
    """

    def __init__(self):
        self._engines: dict[str, EngineStatus] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册所有已知引擎的默认状态。

        当前默认状态对应 ARCHITECTURE_STATE.md §2 的配置概要。
        """
        defaults = [
            # 人声分离
            EngineStatus(
                "uvr",
                "bs_roformer",
                EngineLifecycle.READY_DEFAULT,
                resource_requirement="CPU/GPU, ~1GB RAM",
                language_support="通用",
                degradation_strategy="降级到 Open-Unmix",
            ),
            EngineStatus(
                "spleeter",
                "2stems",
                EngineLifecycle.UNAVAILABLE,
                resource_requirement="TensorFlow, ~2GB RAM",
                language_support="通用",
                degradation_strategy="N/A",
            ),
            EngineStatus(
                "open-unmix",
                "umxhq",
                EngineLifecycle.READY_SHADOW,
                resource_requirement="PyTorch, ~1GB RAM",
                language_support="通用",
                degradation_strategy="N/A",
            ),
            # VAD
            EngineStatus(
                "silero",
                "silero_vad",
                EngineLifecycle.READY_DEFAULT,
                resource_requirement="CPU, 极低",
                language_support="通用",
                degradation_strategy="降级到 WebRTC VAD",
            ),
            EngineStatus(
                "webrtc",
                "",
                EngineLifecycle.READY_SHADOW,
                resource_requirement="CPU, 极低",
                language_support="通用",
                degradation_strategy="降级到 TEN VAD",
            ),
            # ASR
            EngineStatus(
                "faster-whisper",
                "large-v3",
                EngineLifecycle.READY_DEFAULT,
                resource_requirement="GPU 推荐, CPU 可用",
                language_support="多语言 (99+)",
                degradation_strategy="降级到 tiny 模型",
            ),
            EngineStatus(
                "funasr",
                "paraformer-zh",
                EngineLifecycle.READY_SHADOW,
                resource_requirement="GPU 推荐, ~2GB",
                language_support="中文优化",
                degradation_strategy="降级到 faster-whisper",
            ),
            EngineStatus(
                "qwen-asr",
                "Qwen3-ASR-1.7B",
                EngineLifecycle.READY_SHADOW,
                resource_requirement="GPU 推荐, ~4GB",
                language_support="多语言",
                degradation_strategy="降级到 faster-whisper",
            ),
            EngineStatus(
                "whisper.cpp",
                "ggml-medium",
                EngineLifecycle.READY_SHADOW,
                resource_requirement="CPU, 低内存",
                language_support="多语言",
                degradation_strategy="降级到 tiny 模型",
            ),
            # 复核引擎
            EngineStatus(
                "global-asr-evidence",
                "large-v3",
                EngineLifecycle.READY_DEFAULT,
                resource_requirement="同 ASR",
                language_support="多语言",
                degradation_strategy="shadow 模式",
            ),
            EngineStatus(
                "context-reasr",
                "large-v3",
                EngineLifecycle.UNAVAILABLE,
                resource_requirement="同 ASR",
                language_support="多语言",
                degradation_strategy="实验阶段",
            ),
            EngineStatus(
                "qwen-review",
                "Qwen3-ASR-1.7B",
                EngineLifecycle.MODEL_MISSING,
                resource_requirement="GPU ~4GB",
                language_support="zh, en",
                degradation_strategy="回退到分段结果",
            ),
            EngineStatus(
                "forced-aligner",
                "Qwen3-ForcedAligner-0.6B",
                EngineLifecycle.MODEL_MISSING,
                resource_requirement="GPU ~2GB",
                language_support="zh, en",
                degradation_strategy="使用 ASR 原生时间戳",
            ),
            EngineStatus(
                "sed",
                "MIT/ast-finetuned-audioset",
                EngineLifecycle.MODEL_MISSING,
                resource_requirement="GPU ~1GB",
                language_support="通用",
                degradation_strategy="不使用音频分类",
            ),
            EngineStatus(
                "semantic-review",
                "",
                EngineLifecycle.UNAVAILABLE,
                resource_requirement="CPU",
                language_support="通用",
                degradation_strategy="实验阶段",
            ),
            # 说话人分离
            EngineStatus(
                "speechbrain-ecapa",
                "speechbrain/ecapa",
                EngineLifecycle.READY_DEFAULT,
                resource_requirement="CPU, ~500MB",
                language_support="通用",
                degradation_strategy="降级到纯文本聚类",
            ),
            EngineStatus(
                "pyannote",
                "speaker-diarization-3.1",
                EngineLifecycle.READY_SHADOW,
                resource_requirement="GPU 推荐, ~2GB",
                language_support="通用",
                degradation_strategy="保留 agglomerative 结果",
            ),
        ]
        for entry in defaults:
            self._engines[entry.engine] = entry

    # ---- 查询 ----

    def get(self, engine: str) -> EngineStatus | None:
        return self._engines.get(engine)

    def list_all(self) -> list[EngineStatus]:
        return list(self._engines.values())

    def list_by_status(self, status: EngineLifecycle) -> list[EngineStatus]:
        return [e for e in self._engines.values() if e.status == status]

    def list_by_category(self, category: str) -> list[EngineStatus]:
        """按类别过滤：separation, vad, asr, review, diarization"""
        category_map = {
            "separation": {"uvr", "spleeter", "open-unmix"},
            "vad": {"silero", "webrtc"},
            "asr": {"faster-whisper", "funasr", "qwen-asr", "whisper.cpp"},
            "review": {
                "global-asr-evidence",
                "context-reasr",
                "qwen-review",
                "forced-aligner",
                "sed",
                "semantic-review",
            },
            "diarization": {"speechbrain-ecapa", "pyannote"},
        }
        keys = category_map.get(category, set())
        return [e for k, e in self._engines.items() if k in keys]

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self._engines.items()}


class LifecycleManager:
    """管理引擎状态转换。

    确保转换符合 ENGINE_LIFECYCLE.md §1 定义的路径。
    """

    def __init__(self, registry: EngineRegistry | None = None):
        self.registry = registry or EngineRegistry()

    def transition(
        self,
        engine: str,
        target: EngineLifecycle,
        *,
        reason: str = "",
        force: bool = False,
    ) -> bool:
        """将引擎转换到目标状态。

        Args:
            engine: 引擎名
            target: 目标状态
            reason: 转换原因（记录日志）
            force: 强制转换（跳过合法性检查，用于回退场景）

        Returns:
            是否成功转换

        Raises:
            ValueError: 引擎不存在或转换不合法
        """
        current = self.registry.get(engine)
        if current is None:
            raise ValueError(f"Unknown engine: {engine!r}")

        if current.status == target:
            return False

        if not force:
            allowed = ALLOWED_LIFECYCLE_TRANSITIONS.get(current.status, frozenset())
            if target not in allowed:
                raise ValueError(
                    f"Invalid transition for {engine}: "
                    f"{current.status.value} → {target.value} not allowed. "
                    f"Allowed: {[s.value for s in allowed]}"
                )

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        old_status = current.status
        current.status = target
        current.last_transition = now
        current.last_checked = now

        logger.info(
            "Engine %s: %s → %s (reason: %s)",
            engine,
            old_status.value,
            target.value,
            reason or "N/A",
        )
        return True

    def promote_to_shadow(self, engine: str, *, model_path: str = "") -> bool:
        """model_missing → ready_shadow"""
        self.transition(engine, EngineLifecycle.READY_SHADOW, reason="model downloaded")
        current = self.registry.get(engine)
        if current and model_path:
            current.model_path = model_path
        return True

    def promote_to_review(self, engine: str) -> bool:
        """ready_shadow → ready_review"""
        return self.transition(
            engine, EngineLifecycle.READY_REVIEW, reason="shadow validation passed"
        )

    def promote_to_default(self, engine: str) -> bool:
        """ready_review → ready_default"""
        return self.transition(
            engine,
            EngineLifecycle.READY_DEFAULT,
            reason="regression test + human check passed",
        )

    def rollback_to_shadow(self, engine: str) -> bool:
        """ready_default → ready_shadow (回退)"""
        return self.transition(
            engine,
            EngineLifecycle.READY_SHADOW,
            reason="regression detected",
            force=True,
        )

    def mark_unavailable(self, engine: str, reason: str = "") -> bool:
        """任意状态 → unavailable（重置）"""
        return self.transition(
            engine, EngineLifecycle.UNAVAILABLE, reason=reason, force=True
        )
