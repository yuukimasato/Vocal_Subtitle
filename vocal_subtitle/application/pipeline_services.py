"""Lazy runtime dependencies used by :class:`vocal_subtitle.pipeline.Pipeline`."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from ..asr.base import ASREngine
from ..config import PipelineConfig
from ..separation.base import SeparationEngine
from ..utils.cache_manager import CacheManager
from ..utils.task_history import TaskHistoryManager
from ..vad.base import VADEngine

logger = logging.getLogger(__name__)


class PipelineServices:
    """Own lazy engine/cache lifecycles without owning pipeline stages."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.separation_engine: Optional[SeparationEngine] = None
        self.vad_engine: Optional[VADEngine] = None
        self.asr_engine: Optional[ASREngine] = None
        self.asr_engines: dict[str, ASREngine] = {}
        self.cache: Optional[CacheManager] = None
        self.history: Optional[TaskHistoryManager] = None

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

            self.vad_engine = SileroVAD()
        elif name == "ten":
            from ..vad.ten_vad import TENVAD

            self.vad_engine = TENVAD()
        elif name == "webrtc":
            from ..vad.webrtc_vad import WebRTCVAD

            self.vad_engine = WebRTCVAD()
        else:
            raise ValueError(f"Unknown VAD engine: {name}. Options: silero, ten, webrtc")
        return self.vad_engine

    def get_asr_engine_for(
        self, engine_name: str, model: Optional[str] = None, *, cache: bool = True
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
                model=model, device=device, compute_type=asr_cfg.compute_type,
                beam_size=asr_cfg.beam_size, word_timestamps=asr_cfg.word_timestamps,
                condition_on_previous_text=asr_cfg.condition_on_previous_text,
                vad_filter=asr_cfg.vad_filter,
            )
        elif engine_name == "whisper-cpp":
            from ..asr.whisper_cpp_engine import WhisperCppEngine

            engine = WhisperCppEngine(
                model=model, language=asr_cfg.language,
                model_path=getattr(asr_cfg, "whisper_cpp_model_path", None),
                whisper_cpp_bin=getattr(asr_cfg, "whisper_cpp_bin", None),
            )
        elif engine_name == "funasr":
            from ..asr.funasr_engine import FunASREngine

            engine = FunASREngine(model=model, device=device)
        else:
            raise ValueError(
                f"Unknown ASR engine: {engine_name}. Options: auto, faster-whisper, whisper-cpp, funasr"
            )
        if cache:
            self.asr_engines[engine_name] = engine
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
