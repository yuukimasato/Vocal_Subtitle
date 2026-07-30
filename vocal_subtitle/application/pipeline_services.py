"""Pipeline service factories for engine, cache, and history construction.

Provides lazy-construction of separation, VAD, ASR engines, cache manager,
and task history manager. Removes factory logic from the Pipeline orchestrator
while preserving the same lazy-loading semantics.

None of these services read Pipeline private state — they receive config
explicitly and return constructed objects.
"""

import logging
import sys
from typing import Any, Dict, Optional

from ..asr.base import ASREngine
from ..asr.router import ASRRouteDecision
from ..config import PipelineConfig
from ..separation.base import SeparationEngine
from ..utils.cache_manager import CacheManager
from ..utils.gpu_detector import GPUDetector
from ..utils.task_history import TaskHistoryManager
from ..vad.base import VADEngine

logger = logging.getLogger(__name__)


class PipelineServiceFactory:
    """Holds engine/dependency construction logic away from Pipeline orchestrator.

    Each getter accepts an explicit config; storage of constructed singletons
    is owned by this factory, not by Pipeline.__init__ attribute assignment.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._separation_engine: Optional[SeparationEngine] = None
        self._vad_engine: Optional[VADEngine] = None
        self._asr_engines: Dict[str, ASREngine] = {}
        self._cache: Optional[CacheManager] = None
        self._history: Optional[TaskHistoryManager] = None
        self._asr_route_decision: Optional[ASRRouteDecision] = None
        self._resolved_language: Optional[str] = None
        self._asr_engine: Optional[ASREngine] = None

    # -- ASR route decision (read/write for Pipeline) -----------------------

    @property
    def asr_route_decision(self) -> Optional[ASRRouteDecision]:
        return self._asr_route_decision

    @asr_route_decision.setter
    def asr_route_decision(self, value: Optional[ASRRouteDecision]):
        self._asr_route_decision = value

    @property
    def resolved_language(self) -> Optional[str]:
        return self._resolved_language

    @resolved_language.setter
    def resolved_language(self, value: Optional[str]):
        self._resolved_language = value

    def resolved_language_or_config(self) -> Optional[str]:
        """Return the resolved language for the current task, falling back to config."""
        return self._resolved_language or self._config.asr.language

    # -- Separation engine -------------------------------------------------

    def get_separation_engine(self) -> SeparationEngine:
        if self._separation_engine is not None:
            return self._separation_engine

        engine_name = self._config.separation.engine
        if engine_name == "spleeter":
            if sys.version_info >= (3, 12):
                raise RuntimeError(
                    "Spleeter 不支持 Python 3.12+（已于 2022 年停止维护）。"
                    " 请改用 UVR 引擎：--separator uvr，"
                    " 或使用 UVR BS-RoFormer 模型获得更高品质："
                    " --separator uvr --uvr-model model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                )
            from ..separation.spleeter_engine import SpleeterEngine

            self._separation_engine = SpleeterEngine()
        elif engine_name == "openunmix":
            from ..separation.openunmix_engine import OpenUnmixEngine

            self._separation_engine = OpenUnmixEngine()
        elif engine_name == "uvr":
            from ..separation.uvr_engine import UVREngine

            self._separation_engine = UVREngine()
        else:
            raise ValueError(
                f"Unknown separation engine: {engine_name}. "
                f"Options: uvr, openunmix, spleeter"
            )

        return self._separation_engine

    def get_sep_model_name(self) -> str:
        """Return the model name string for the current separation engine."""
        sep = self._config.separation
        if sep.engine == "uvr":
            return sep.uvr_model
        elif sep.engine == "openunmix":
            return "umxhq"
        return ""

    # -- VAD engine --------------------------------------------------------

    def get_vad_engine(self) -> VADEngine:
        if self._vad_engine is not None:
            return self._vad_engine

        engine_name = self._config.vad.engine
        if engine_name == "silero":
            from ..vad.silero_vad import SileroVAD

            self._vad_engine = SileroVAD()
        elif engine_name == "ten":
            from ..vad.ten_vad import TENVAD

            self._vad_engine = TENVAD()
        elif engine_name == "webrtc":
            from ..vad.webrtc_vad import WebRTCVAD

            self._vad_engine = WebRTCVAD()
        else:
            raise ValueError(
                f"Unknown VAD engine: {engine_name}. "
                f"Options: silero, ten, webrtc"
            )

        return self._vad_engine

    # -- ASR engines ------------------------------------------------------

    def get_asr_engine_for(
        self,
        engine_name: str,
        model: Optional[str] = None,
        cache: bool = True,
    ) -> ASREngine:
        """Construct one concrete engine; ``auto`` never reaches this method."""
        if cache and engine_name in self._asr_engines:
            return self._asr_engines[engine_name]

        asr_cfg = self._config.asr
        model = model or asr_cfg.model

        # Auto-detect device
        device = asr_cfg.device
        if device == "auto":
            best = GPUDetector.get_best_device()
            device = best.value  # "cuda" | "mps" | "cpu"
            if device == "mps":
                # CTranslate2 / faster-whisper 不支持 MPS，回退到 CPU
                device = "cpu"
            logger.info("Auto device detection: %s → %s", best.value, device)

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

            engine = FunASREngine(
                model=model,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown ASR engine: {engine_name}. "
                "Options: auto, faster-whisper, whisper-cpp, funasr"
            )
        if cache:
            self._asr_engines[engine_name] = engine
        return engine

    def get_asr_engine(self) -> ASREngine:
        """Resolve the current ASR engine, considering 'auto' route."""
        if self._asr_engine is not None:
            return self._asr_engine
        engine_name = self._config.asr.engine
        if engine_name == "auto":
            engine_name = (
                self._asr_route_decision.selected_engine
                if self._asr_route_decision is not None
                else "faster-whisper"
            )
        self._asr_engine = self.get_asr_engine_for(engine_name)
        return self._asr_engine

    def invalidate_asr_engine(self):
        """Clear cached ASR engine so next call re-constructs."""
        self._asr_engine = None

    # -- Cache manager ----------------------------------------------------

    def get_cache(self) -> CacheManager:
        if self._cache is None:
            cache_cfg = self._config.cache
            self._cache = CacheManager(
                cache_dir=cache_cfg.directory,
                ttl_separation=cache_cfg.ttl_separation,
                ttl_transcription=cache_cfg.ttl_transcription,
            )
        return self._cache

    # -- History manager --------------------------------------------------

    def get_history(self) -> TaskHistoryManager:
        if self._history is None:
            self._history = TaskHistoryManager()
        return self._history
