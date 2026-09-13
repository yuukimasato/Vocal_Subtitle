"""管道编排器

负责调度多个处理阶段的有序执行，管理数据流传递。

处理流程:
    Stage 0: (可选) 宏观切块 → Stage 1: 人声分离 → Stage 2: VAD 检测
    → Stage 2.5: (可选) ffmpeg 并行 VAD + 三方法融合
    → Stage 3: 片段合并 → Stage 4: ASR 识别
    → Stage 4.5: (可选) ASR 边界精修 → Stage 5: 时间轴映射 + 字幕输出
    → Stage 5.5: (可选) LLM 语义合并 + 帧级无缝衔接 + 声学标尺校验
    → (可选) LLM 优化
"""

import json
import logging
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .asr.base import ASREngine, ASRInvalidResultError, TranscriptionSegment
from .application.pipeline_result import PipelineStats
from .application.pipeline_services import PipelineServices
from .application.asr_path import PipelineASRPathMixin
from .application.physical_path import PipelinePhysicalPathMixin
from .application.stage_runner import PipelineStageMixin
from .application.pipeline_runner import PipelineRunMixin
from .application.streaming_runner import PipelineStreamingMixin
from .diarization.pipeline_stage import PipelineDiarizationMixin
from .mapping.pipeline_stage import PipelineMappingMixin
from .application.chunk_runner import PipelineChunkMixin
from .application.postprocess_runner import PipelinePostprocessMixin
from .feedback.pipeline_stage import PipelineFeedbackMixin
from .asr.router import ASRRouteDecision, ASRRouter
from .config import PipelineConfig
from .mapping.subtitle_builder import SubtitleBuilder, SubtitleRule
from .mapping.time_mapper import SubtitleEvent, TimeMapper
from .merging.merge_strategy import MergeConfig, MergeStrategy
from .pipeline_context import ASRFragment, NoiseProfile, PipelineContext
from .separation.base import SeparationEngine, SeparationResult
from .utils.audio_utils import AudioUtils
from .utils.cache_manager import CacheManager
from .utils.file_hasher import compute_config_hash, compute_file_hash
from .utils.gpu_detector import GPUDetector
from .utils.logger import get_logger, setup_logging
from .utils.progress import ProgressManager
from .utils.task_history import TaskHistoryManager
from .vad.base import SpeechSegment, VADEngine

logger = logging.getLogger(__name__)


class Pipeline(PipelineASRPathMixin, PipelinePhysicalPathMixin, PipelineStageMixin, PipelineRunMixin, PipelineStreamingMixin, PipelineDiarizationMixin, PipelineMappingMixin, PipelineChunkMixin, PipelinePostprocessMixin, PipelineFeedbackMixin):
    """人声分离 + 字幕生成管道

    编排 5 个处理阶段，将原始音频转换为字幕文件。

    使用示例:
        config = ConfigLoader().load_profile("podcast")
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_path=Path("input.mp3"),
            output_path=Path("output.srt"),
        )
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Args:
            config: 管道配置，默认加载 default 配置
        """
        from .config import ConfigLoader

        self.config = config or ConfigLoader().load_profile("default")
        self._services = PipelineServices(self.config)

        self._setup_logging()

        # 初始化各阶段组件
        self._separation_engine: Optional[SeparationEngine] = None
        self._vad_engine: Optional[VADEngine] = None
        self._asr_engine: Optional[ASREngine] = None
        self._merge_strategy: Optional[MergeStrategy] = None
        self._time_mapper: Optional[TimeMapper] = None
        self._subtitle_builder: Optional[SubtitleBuilder] = None
        self._embedding_engine = None  # 说话人嵌入引擎（惰性初始化）
        self._cache: Optional[CacheManager] = None
        self._history: Optional[TaskHistoryManager] = None
        self._progress: Optional[ProgressManager] = None

        # 当前任务的输入文件哈希（用于缓存键）
        self._file_hash: str = ""
        self._config_hash: str = ""

        # ASR 路径追踪
        self._requested_asr_path: str = ""
        self._asr_route_decision: Optional[ASRRouteDecision] = None
        self._asr_engines: Dict[str, ASREngine] = {}
        self._global_evidence_attempted: bool = False
        self._global_evidence_diagnostics: Dict[str, Any] = {}

        # [层1] 说话人身份主干（early_turns）：全局 pass 结果与骨架×turns
        # 跨度，由 run() 在分离之后填充；未启用时保持 None/空。
        self._early_turns_state = None
        self._early_turn_spans: List[Any] = []

    # Direct full-audio ASR is intentionally bounded until the existing
    # GlobalTranscriber windowing path is promoted to the main route.
    GLOBAL_ASR_MAX_DURATION_SECONDS = 180.0
    TASK_LANGUAGE_MIN_PROBABILITY = 0.85

    # ------------------------------------------------------------------
    # ASR path resolution (global vs. segmented)
    # ------------------------------------------------------------------



    def _prepare_task_language(self, audio, sample_rate: int) -> str | None:
        """Detect language once from the full task audio.

        Used by downstream stages (e.g. ASR, speaker labels) to avoid
        unreliable per-segment auto-detection.
        """
        configured_language = getattr(self.config.asr, "language", None)
        if configured_language:
            self._resolved_language = configured_language
            return configured_language

        # Automatic routing has already probed complete-task windows.  Its
        # ``None`` is deliberate for uncertain/non-supported languages and
        # must not be replaced by a later local VAD chunk detection.
        decision = getattr(self, "_asr_route_decision", None)
        if decision is not None and decision.requested_engine == "auto":
            self._resolved_language = decision.language
            return decision.language

        engine = self._get_asr_engine()
        try:
            engine.load_model()
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "ASR engine '%s' is not installed; task language detection skipped. "
                "Install the required engine to enable language auto-detection.",
                engine.name,
            )
            return None
        except Exception:
            logger.warning(
                "Failed to load ASR engine '%s' for language detection. "
                "Continuing without language auto-detection.",
                engine.name,
                exc_info=True,
            )
            return None
        detect = getattr(engine, "detect_language_info", None)
        if callable(detect):
            lang_info = detect(audio, sample_rate)
            lang = getattr(lang_info, "language", None)
            probability = getattr(lang_info, "probability", 0.0)
            if (
                lang
                and str(lang).lower() not in {"unknown", "other", "mixed"}
                and float(probability) >= self.TASK_LANGUAGE_MIN_PROBABILITY
            ):
                self._resolved_language = lang
                return lang
        return None

    def _setup_logging(self) -> None:
        """初始化日志"""
        log_cfg = self.config.logging
        setup_logging(
            level=log_cfg.level,
            log_format=log_cfg.format,
            log_file=log_cfg.file,
        )

    # ------------------------------------------------------------------
    # 引擎工厂方法
    # ------------------------------------------------------------------

    def _get_separation_engine(self) -> SeparationEngine:
        if self._separation_engine is None:
            self._separation_engine = self._services.get_separation_engine()
        return self._separation_engine

    def _get_sep_model_name(self) -> str:
        return self._services.get_sep_model_name()

    def _get_vad_engine(self) -> VADEngine:
        if self._vad_engine is None:
            self._vad_engine = self._services.get_vad_engine()
        return self._vad_engine

    def _get_asr_engine_for(
        self,
        engine_name: str,
        model: Optional[str] = None,
        cache: bool = True,
    ) -> ASREngine:
        """Compatibility adapter for the application dependency service."""
        self._services.asr_engines = self._asr_engines
        return self._services.get_asr_engine_for(engine_name, model, cache=cache)

    def _get_asr_engine(self) -> ASREngine:
        if self._asr_engine is None:
            engine_name = self.config.asr.engine
            if engine_name == "auto":
                engine_name = (
                    self._asr_route_decision.selected_engine
                    if self._asr_route_decision is not None
                    else "faster-whisper"
                )
            self._asr_engine = self._get_asr_engine_for(engine_name)
        return self._asr_engine

    def _get_cache(self) -> CacheManager:
        if self._cache is None:
            self._cache = self._services.get_cache()
        return self._cache

    def _get_history(self) -> TaskHistoryManager:
        if self._history is None:
            self._history = self._services.get_history()
        return self._history
    # ------------------------------------------------------------------
    # 核心处理流程
    # ------------------------------------------------------------------








    # ------------------------------------------------------------------
    # 反馈学习通道 (Phase 5)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 字幕构建器
    # ------------------------------------------------------------------

    def _get_subtitle_builder(self) -> SubtitleBuilder:
        if self._subtitle_builder is None:
            sub_cfg = self.config.subtitle
            self._subtitle_builder = SubtitleBuilder(
                rule=SubtitleRule(
                    min_duration=sub_cfg.min_duration,
                    max_duration=sub_cfg.max_duration,
                    max_chars_cjk=sub_cfg.max_chars_cjk,
                    max_chars_latin=sub_cfg.max_chars_latin,
                    max_lines=sub_cfg.max_lines,
                )
            )
        return self._subtitle_builder
