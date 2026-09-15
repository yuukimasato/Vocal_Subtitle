"""Lazy runtime dependencies used by :class:`vocal_subtitle.pipeline.Pipeline`."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from ..asr.base import ASREngine
from ..config import PipelineConfig
from ..separation.base import SeparationEngine
from ..utils.cache_manager import CacheManager
from ..utils.task_history import TaskHistoryManager
from ..vad.base import VADEngine

logger = logging.getLogger(__name__)


def _torch_available() -> bool:
    """torch 是否可用（--no-torch / 无 GPU 自动安装的环境里没有 torch）。

    只做 find_spec 探测，不真正 import torch（导入本身要数秒）。
    """
    if "torch" in sys.modules:
        return True
    try:
        from importlib.util import find_spec

        return find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


# Engine name mapping: PipelineServices key → EngineRegistry key
_ENGINE_REGISTRY_ALIASES: dict[str, str] = {
    "openunmix": "open-unmix",
    "qwen": "qwen-asr",
    "whisper-cpp": "whisper.cpp",
}


class PipelineServices:
    """Own lazy engine/cache lifecycles without owning pipeline stages."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.separation_engine: SeparationEngine | None = None
        self.vad_engine: VADEngine | None = None
        self.asr_engine: ASREngine | None = None
        self.asr_engines: dict[str, ASREngine] = {}
        self.cache: CacheManager | None = None
        self.history: TaskHistoryManager | None = None

        # ---- Engine lifecycle governance (ENGINE_LIFECYCLE.md) ----
        from ..governance.engine_lifecycle import EngineRegistry, LifecycleManager

        self.engine_registry = EngineRegistry()
        self.lifecycle = LifecycleManager(self.engine_registry)
        self._lifecycle_events: list[
            dict
        ] = []  # buffered until degradation logger attached
        self._degradation_logger: object = None  # DegradationLogger, lazy-attached

    def get_separation_engine(self) -> SeparationEngine:
        if self.separation_engine is not None:
            return self.separation_engine
        name = self.config.separation.engine
        if name == "spleeter":
            if sys.version_info >= (3, 12):
                raise RuntimeError(
                    "Spleeter 不支持 Python 3.12+（已于 2022 年停止维护）。"
                    " 请改用 UVR 引擎：--separator uvr，"
                    " 或使用 UVR BS-RoFormer 模型获得更高品质："
                    " --separator uvr --uvr-model model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                )
            from ..separation.spleeter_engine import SpleeterEngine

            self.separation_engine = SpleeterEngine()
        elif name == "openunmix":
            from ..separation.openunmix_engine import OpenUnmixEngine

            self.separation_engine = OpenUnmixEngine()
        elif name == "uvr":
            from ..separation.uvr_engine import UVREngine

            self.separation_engine = UVREngine()
        else:
            raise ValueError(
                f"Unknown separation engine: {name}. Options: uvr, openunmix, spleeter"
            )
        # Track lifecycle
        model = self.get_sep_model_name()
        self._sync_lifecycle(
            name,
            "ready_default" if name == "uvr" else "ready_shadow",
            reason=f"engine instantiated (model={model})",
            model=model,
        )
        return self.separation_engine

    def get_sep_model_name(self) -> str:
        if self.config.separation.engine == "uvr":
            return self.config.separation.uvr_model
        if self.config.separation.engine == "openunmix":
            return "umxhq"
        return ""

    def get_vad_engine(self) -> VADEngine:
        if self.vad_engine is not None:
            return self.vad_engine
        name = self.config.vad.engine
        if name == "silero":
            from ..vad.silero_vad import SileroVAD

            if not _torch_available():
                # --no-torch / 无 GPU 自动安装只装了 webrtcvad，而默认配置仍是 silero：
                # 直接实例化会在加载模型时 ImportError。降级为 WebRTC 并留下告警
                logger.warning(
                    "vad.engine=silero 需要 torch，当前环境未安装；自动降级为 WebRTC VAD"
                )
                from ..vad.webrtc_vad import WebRTCVAD

                self.vad_engine = WebRTCVAD()
                name = "webrtc"
            else:
                self.vad_engine = SileroVAD()
        elif name == "ten":
            from ..vad.ten_vad import TENVAD

            self.vad_engine = TENVAD()
        elif name == "webrtc":
            from ..vad.webrtc_vad import WebRTCVAD

            self.vad_engine = WebRTCVAD()
        else:
            raise ValueError(
                f"Unknown VAD engine: {name}. Options: silero, ten, webrtc"
            )
        # Track lifecycle
        self._sync_lifecycle(
            name,
            "ready_default" if name == "silero" else "ready_shadow",
            reason="engine instantiated",
        )
        return self.vad_engine

    def get_asr_engine_for(
        self, engine_name: str, model: str | None = None, *, cache: bool = True
    ) -> ASREngine:
        if cache and engine_name in self.asr_engines:
            return self.asr_engines[engine_name]
        asr_cfg = self.config.asr
        model = model or asr_cfg.model
        device = asr_cfg.device
        if device == "auto":
            from ..utils.gpu_detector import GPUDetector

            best = GPUDetector.get_best_device()
            device = "cpu" if best.value == "mps" else best.value
            logger.info("Auto device detection: %s -> %s", best.value, device)
        if engine_name == "faster-whisper":
            from ..asr.faster_whisper_engine import FasterWhisperEngine

            engine = FasterWhisperEngine(
                model=model,
                device=device,
                compute_type=asr_cfg.compute_type,
                beam_size=asr_cfg.beam_size,
                word_timestamps=asr_cfg.word_timestamps,
                condition_on_previous_text=asr_cfg.condition_on_previous_text,
                vad_filter=asr_cfg.vad_filter,
            )
        elif engine_name == "whisper-cpp":
            from ..asr.whisper_cpp_engine import WhisperCppEngine

            engine = WhisperCppEngine(
                model=model,
                language=asr_cfg.language,
                model_path=getattr(asr_cfg, "whisper_cpp_model_path", None),
                whisper_cpp_bin=getattr(asr_cfg, "whisper_cpp_bin", None),
            )
        elif engine_name == "funasr":
            from ..asr.funasr_engine import FunASREngine

            engine = FunASREngine(model=model, device=device)
        elif engine_name == "qwen":
            from ..asr.qwen_engine import QwenASREngine

            engine = QwenASREngine(
                model_path=getattr(asr_cfg, "qwen_model_path", None),
                device=device,
                language=asr_cfg.language,
            )
        else:
            raise ValueError(
                f"Unknown ASR engine: {engine_name}. Options: auto, faster-whisper, whisper-cpp, funasr, qwen"
            )
        if cache:
            self.asr_engines[engine_name] = engine
        # Track lifecycle
        self._sync_lifecycle(
            engine_name,
            "ready_default" if engine_name == "faster-whisper" else "ready_shadow",
            reason=f"engine instantiated (model={model})",
            model=model or "",
        )
        return engine

    def get_asr_engine(self, selected_engine: str | None = None) -> ASREngine:
        if self.asr_engine is not None:
            return self.asr_engine
        name = selected_engine or self.config.asr.engine
        if name == "auto":
            name = "faster-whisper"
        self.asr_engine = self.get_asr_engine_for(name)
        return self.asr_engine

    def get_cache(self) -> CacheManager:
        if self.cache is None:
            cfg = self.config.cache
            self.cache = CacheManager(
                cache_dir=cfg.directory,
                ttl_separation=cfg.ttl_separation,
                ttl_transcription=cfg.ttl_transcription,
            )
        return self.cache

    def get_history(self) -> TaskHistoryManager:
        if self.history is None:
            self.history = TaskHistoryManager()
        return self.history

    # ------------------------------------------------------------------
    # Engine lifecycle governance
    # ------------------------------------------------------------------

    def _resolve_registry_key(self, engine_name: str) -> str:
        """Map PipelineServices engine name to EngineRegistry key."""
        return _ENGINE_REGISTRY_ALIASES.get(engine_name, engine_name)

    def _sync_lifecycle(
        self,
        engine_name: str,
        status: str,
        reason: str = "",
        model: str = "",
    ) -> None:
        """Sync an engine's lifecycle status with the EngineRegistry.

        Records lifecycle events that are later flushed to the degradation
        log when a DegradationLogger is attached.
        """
        from ..governance.engine_lifecycle import EngineLifecycle

        key = self._resolve_registry_key(engine_name)
        entry = self.engine_registry.get(key)
        if entry is None:
            return  # unknown engine (e.g. "ten" VAD)

        old_status = entry.status.value
        try:
            self.lifecycle.transition(
                key, EngineLifecycle(status), reason=reason, force=True
            )
        except ValueError as exc:
            logger.debug("Lifecycle sync skipped for %s: %s", key, exc)
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "engine_lifecycle",
            "engine": key,
            "from": old_status,
            "to": status,
            "reason": reason,
            "model": model,
        }
        self._lifecycle_events.append(event)

        # Immediately flush to degradation logger if attached
        dlog = self._degradation_logger
        if dlog is not None and hasattr(dlog, "record"):
            dlog.record(
                stage=f"engine:{key}",
                from_path=old_status,
                to_path=status,
                reason=reason,
                category="engine_lifecycle",
            )

    def attach_degradation_logger(self, dlog: object) -> None:
        """Attach a DegradationLogger and flush buffered lifecycle events."""
        self._degradation_logger = dlog
        for ev in self._lifecycle_events:
            if hasattr(dlog, "record"):
                dlog.record(
                    stage="engine:{}".format(ev["engine"]),
                    from_path=ev["from"],
                    to_path=ev["to"],
                    reason=ev["reason"],
                    category="engine_lifecycle",
                )
        self._lifecycle_events.clear()

    def record_stage_degradation(
        self, stage: str, reason: str, category: str = "recoverable_degradation"
    ) -> None:
        """Record a stage-level degradation event in real time.

        Flushes immediately if a DegradationLogger is attached.
        """
        dlog = self._degradation_logger
        if dlog is not None and hasattr(dlog, "record"):
            dlog.record(
                stage=stage,
                from_path="quality_first",
                to_path="degraded",
                reason=reason,
                category=category,
            )

    def get_engine_snapshot(self) -> dict[str, dict]:
        """Return a unified engine status snapshot for the run report."""
        snapshot: dict[str, dict] = {}
        for engine in self.engine_registry.list_all():
            snapshot[engine.engine] = {
                "status": engine.status.value,
                "model": engine.model,
                "device": engine.device,
                "model_path": engine.model_path,
            }
        return snapshot
