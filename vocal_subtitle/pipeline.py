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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .asr.base import ASREngine, TranscriptionSegment
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


@dataclass
class PipelineStats:
    """管道执行统计"""

    input_path: Path
    duration_seconds: float
    stage_timings: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    segment_count: int = 0
    subtitle_count: int = 0
    speaker_count: int = 0
    diarization_silhouette: Optional[float] = None
    diagnostic_report: Optional[Dict] = None

    # global ASR path diagnostics
    asr_path: str = ""
    global_attempted: bool = False
    fallback_category: str = ""
    fallback_reason: str = ""
    global_diagnostics: Dict = field(default_factory=dict)

    # diarization canonicalization
    raw_diarization_speaker_count: int = 0
    canonical_speaker_count: int = 0
    speaker_merge_map: Dict = field(default_factory=dict)
    canonicalization_status: str = ""

    # diarization diagnostics
    diarization_backend: str = ""
    diarization_status: str = ""
    mixed_event_count: int = 0
    atomic_span_count: int = 0

    def to_dict(self) -> dict:
        result = {
            "input_path": str(self.input_path),
            "duration_seconds": self.duration_seconds,
            "stage_timings": self.stage_timings,
            "total_time": self.total_time,
            "segment_count": self.segment_count,
            "subtitle_count": self.subtitle_count,
            "asr_path": self.asr_path,
            "global_attempted": self.global_attempted,
            "fallback_category": self.fallback_category,
            "fallback_reason": self.fallback_reason,
            "global_diagnostics": self.global_diagnostics,
            "raw_diarization_speaker_count": self.raw_diarization_speaker_count,
            "canonical_speaker_count": self.canonical_speaker_count,
            "speaker_merge_map": self.speaker_merge_map,
            "canonicalization_status": self.canonicalization_status,
            "diarization_backend": self.diarization_backend,
            "diarization_status": self.diarization_status,
            "mixed_event_count": self.mixed_event_count,
            "atomic_span_count": self.atomic_span_count,
            "hallucination_filter_version": getattr(self, "hallucination_filter_version", ""),
            "hallucination_dropped_count": getattr(self, "hallucination_dropped_count", 0),
        }
        if self.speaker_count:
            result["speaker_count"] = self.speaker_count
        if self.diarization_silhouette is not None:
            result["diarization_silhouette"] = self.diarization_silhouette
        if self.diagnostic_report:
            result["diagnostic_report"] = self.diagnostic_report
        return result

    @classmethod
    def from_dict(
        cls, input_path: Path, payload: dict, duration_seconds: float = 0.0
    ) -> "PipelineStats":
        stats = cls(input_path=input_path, duration_seconds=duration_seconds)
        stats.total_time = payload.get("total_time", 0.0)
        stats.segment_count = payload.get("segment_count", 0)
        stats.subtitle_count = payload.get("subtitle_count", 0)
        stats.asr_path = payload.get("asr_path", "")
        stats.global_attempted = payload.get("global_attempted", False)
        stats.fallback_category = payload.get("fallback_category", "")
        stats.fallback_reason = payload.get("fallback_reason", "")
        stats.global_diagnostics = payload.get("global_diagnostics", {})
        stats.raw_diarization_speaker_count = payload.get("raw_diarization_speaker_count", 0)
        stats.canonical_speaker_count = payload.get("canonical_speaker_count", 0)
        stats.speaker_merge_map = payload.get("speaker_merge_map", {})
        stats.canonicalization_status = payload.get("canonicalization_status", "")
        stats.diarization_backend = payload.get("diarization_backend", "")
        stats.diarization_status = payload.get("diarization_status", "")
        stats.mixed_event_count = payload.get("mixed_event_count", 0)
        stats.atomic_span_count = payload.get("atomic_span_count", 0)
        stats.speaker_count = payload.get("speaker_count", 0)
        stats.diarization_silhouette = payload.get("diarization_silhouette")
        stats.diagnostic_report = payload.get("diagnostic_report")
        return stats


class Pipeline:
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

    # ------------------------------------------------------------------
    # ASR path resolution (global vs. segmented)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_global_failure(exc: Exception) -> str:
        """Classify a global ASR failure into a stable category."""
        msg = str(exc).lower()
        type_name = type(exc).__name__
        if isinstance(exc, ImportError):
            return "dependency_unavailable"
        if isinstance(exc, MemoryError):
            return "resource_unavailable"
        if any(kw in msg for kw in ("degraded", "empty result", "no transcript")):
            return "invalid_result"
        if any(kw in type_name.lower() + msg for kw in ("memory", "oom", "cuda out")):
            return "resource_unavailable"
        return "execution_failed"

    @staticmethod
    def _classify_global_diagnostics(diagnostics: dict) -> str:
        """Extract the worst failure category from global diagnostics."""
        failed_windows = diagnostics.get("failed_windows", [])
        for window in failed_windows:
            error = window.get("error", "")
            if "is not installed" in error:
                return "dependency_unavailable"
        return "execution_failed"

    @staticmethod
    def _global_diagnostic_errors(diagnostics: dict) -> list:
        """Extract error messages from global diagnostics."""
        return [w.get("error", "") for w in diagnostics.get("failed_windows", [])]

    @staticmethod
    def _safe_failure_reason(exc: Exception) -> str:
        """Redact credentials from failure messages."""
        import re
        msg = str(exc)
        msg = re.sub(r'(api_key|token|key)=[\S]+', r'\1=***', msg)
        return msg

    def _resolve_asr_path(self) -> str:
        """Determine whether to use global or segmented ASR path."""
        if self.config.mode == "streaming":
            return "segmented"
        # Respect explicit override from merge_with_overrides or config
        explicit = getattr(self.config, "asr_path", None)
        if explicit:
            return str(explicit)
        if self._requested_asr_path:
            return self._requested_asr_path
        if self.config.asr.global_asr.enabled:
            routing = self.config.asr.global_asr.routing
            if routing in ("global", "auto"):
                return "global"
        return "segmented"

    @staticmethod
    def _is_usable_global_transcript(transcript) -> bool:
        """Check whether a global transcript is usable for subtitle production."""
        status = getattr(transcript, "status", "unknown")
        words = getattr(transcript, "words", [])
        if status == "ok" and words:
            return True
        if status == "degraded" and words:
            return True
        return False

    def _is_usable_full_pipeline_cache(self, cache_entry: dict) -> bool:
        """Check whether a cached full-pipeline result matches the current ASR path."""
        cached_path = cache_entry.get("stats", {}).get("asr_path", "")
        if self._requested_asr_path == "global":
            return cached_path == "global"
        return True

    @staticmethod
    def _clamp_to_physical_envelopes(events: list) -> list:
        """Clamp display time of each event to its physical envelope."""
        for event in events:
            physical_start = getattr(event, "physical_start", None)
            physical_end = getattr(event, "physical_end", None)
            if physical_start is not None and physical_end is not None:
                event.start = max(event.start, physical_start)
                event.end = min(event.end, physical_end)
        return events

    @staticmethod
    def _is_usable_diarization_cache(diarization, requested_backend: str) -> bool:
        """Check whether a cached diarization result matches the requested backend.

        A legacy/fallback result must not be reused when pyannote is available,
        and vice-versa.
        """
        backend = getattr(diarization, "backend", "")
        if requested_backend == "auto":
            return backend != "legacy-global-fallback"
        if requested_backend == "pyannote":
            return backend in ("pyannote", "pyannote-community-1")
        if requested_backend == "legacy":
            return True
        return backend == requested_backend

    def _project_global_speakers(
        self,
        segments: list,
        time_offset: float = 0.0,
        duration: float = 0.0,
    ):
        """Project global diarization turns onto VAD speech segments.

        Returns (projected_segments, speaker_ids) where each input segment is
        split at speaker turn boundaries.
        """
        turns = getattr(self, "_global_turns", []) or []
        projected_segments = []
        speaker_ids = []
        for seg in segments:
            seg_start = seg.start + time_offset
            seg_end = seg.end + time_offset
            # Clip to audio duration
            seg_start = max(0.0, seg_start)
            seg_end = min(duration if duration > 0 else float("inf"), seg_end)
            relevant = [
                t for t in turns
                if t.end > seg_start and t.start < seg_end
            ]
            if not relevant:
                # No diarization data — return segment as-is with unknown speaker
                projected_segments.append(type(seg)(seg_start, seg_end))
                speaker_ids.append(-1)
                continue
            # When only one speaker covers the entire segment, don't split
            speakers_in_seg = {t.speaker_id for t in relevant}
            if len(speakers_in_seg) == 1:
                projected_segments.append(type(seg)(seg_start, seg_end))
                speaker_ids.append(speakers_in_seg.pop())
                continue
            # Split at turn boundaries
            boundaries = sorted(set(
                [max(seg_start, t.start) for t in relevant]
                + [min(seg_end, t.end) for t in relevant]
            ))
            for b_start, b_end in zip(boundaries, boundaries[1:]):
                if b_end <= b_start:
                    continue
                # Find the speaker at the midpoint of this sub-segment
                midpoint = (b_start + b_end) / 2.0
                speaker = next(
                    (t.speaker_id for t in turns if t.start <= midpoint < t.end),
                    -1,
                )
                projected_segments.append(type(seg)(b_start, b_end))
                speaker_ids.append(speaker)
        return projected_segments, speaker_ids

    def _enforce_speaker_boundaries(
        self,
        events: list,
        stats,
    ):
        """Split subtitle events at speaker-turn boundaries.

        When an event spans a speaker change, it is split so each piece
        carries the correct speaker label and only the words that belong
        to that speaker.
        """
        from .asr.base import WordTimestamp
        turns = getattr(self, "_global_turns", []) or []
        if not turns or not events:
            return events

        result = []
        for event in events:
            words = list(getattr(event, "words", []) or [])
            if not words:
                # No word timestamps — can't split by speaker. When the event
                # crosses a speaker boundary, mark it as UNKNOWN.
                crosses_boundary = any(
                    t.start > event.start and t.start < event.end
                    for t in turns
                )
                if crosses_boundary:
                    first_turn = next(
                        (t for t in turns if t.start <= event.start < t.end),
                        None,
                    )
                    event.end = min(event.end, first_turn.end if first_turn else turns[0].start)
                    event.speaker_id = None
                    event.speaker_label = None
                else:
                    for turn in turns:
                        if turn.start <= event.start < turn.end:
                            event.speaker_id = turn.speaker_id
                            break
                    if event.speaker_id is not None:
                        event.speaker_label = self._make_speaker_label(
                            self._resolved_language_or_config(), event.speaker_id
                        )
                stats.mixed_event_count += 1
                result.append(event)
                continue

            # Split at speaker boundary
            split_points = sorted(set(
                [event.start]
                + [t.start for t in turns if event.start < t.start < event.end]
                + [t.end for t in turns if event.start < t.end < event.end]
                + [event.end]
            ))
            piece_index = 0
            for b_start, b_end in zip(split_points, split_points[1:]):
                if b_end <= b_start:
                    continue
                midpoint = (b_start + b_end) / 2.0
                turn_speaker = next(
                    (t.speaker_id for t in turns if t.start <= midpoint < t.end),
                    None,
                )
                # Find words whose midpoint falls in this sub-segment
                piece_words = [
                    w for w in words
                    if (event.start + float(getattr(w, "start", 0.0)) + event.start + float(getattr(w, "end", 0.0))) / 2.0 < b_end
                    and (event.start + float(getattr(w, "start", 0.0)) + event.start + float(getattr(w, "end", 0.0))) / 2.0 > b_start
                ]
                if not piece_words:
                    continue
                piece_text = " ".join(str(getattr(w, "word", "")) for w in piece_words)
                # Build a piece event
                import copy
                piece = copy.copy(event)
                piece.index = piece_index
                piece.start = b_start
                piece.end = b_end
                piece.text = piece_text or event.text
                piece.speaker_id = turn_speaker
                piece.speaker_label = self._make_speaker_label(
                    self._resolved_language_or_config(), turn_speaker
                )
                # Adjust word timestamps relative to the new piece start.
                # For the first piece, b_start == event.start so offsets are zero.
                offset = b_start - event.start
                piece.words = []
                for w in piece_words:
                    copied = copy.copy(w)
                    copied.start = max(0.0, float(getattr(w, "start", 0.0)) - offset)
                    copied.end = float(getattr(w, "end", 0.0)) - offset
                    piece.words.append(copied)
                # Filter source_word_ids to words in this piece
                all_words = list(getattr(event, "words", []) or [])
                all_source_ids = list(getattr(event, "source_word_ids", []) or [])
                piece.source_word_ids = []
                for w in piece_words:
                    try:
                        idx = all_words.index(w)
                        if idx < len(all_source_ids):
                            piece.source_word_ids.append(all_source_ids[idx])
                    except ValueError:
                        pass
                if not piece.source_word_ids:
                    piece.source_word_ids = all_source_ids
                piece_index += 1
                result.append(piece)
            if piece_index > 1:
                stats.mixed_event_count += 1

        return result

    def _prepare_task_language(self, audio, sample_rate: int) -> str | None:
        """Detect language once from the full task audio.

        Used by downstream stages (e.g. ASR, speaker labels) to avoid
        unreliable per-segment auto-detection.
        """
        if getattr(self.config.asr, "language", None):
            return self.config.asr.language
        engine = self._get_asr_engine()
        engine.load_model()
        detector = getattr(engine, "detect_language", None)
        if callable(detector):
            result = detector(audio, sample_rate)
            if result:
                self._resolved_language = result
                return result
        detect = getattr(engine, "detect_language_info", None)
        if callable(detect):
            lang_info = detect(audio, sample_rate)
            lang = getattr(lang_info, "language", None) or lang_info
            if lang:
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
        if self._separation_engine is not None:
            return self._separation_engine

        engine_name = self.config.separation.engine
        if engine_name == "spleeter":
            import sys

            if sys.version_info >= (3, 12):
                raise RuntimeError(
                    "Spleeter 不支持 Python 3.12+（已于 2022 年停止维护）。"
                    " 请改用 UVR 引擎：--separator uvr，"
                    " 或使用 UVR BS-RoFormer 模型获得更高品质："
                    " --separator uvr --uvr-model model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                )
            from .separation.spleeter_engine import SpleeterEngine

            self._separation_engine = SpleeterEngine()
        elif engine_name == "openunmix":
            from .separation.openunmix_engine import OpenUnmixEngine

            self._separation_engine = OpenUnmixEngine()
        elif engine_name == "uvr":
            from .separation.uvr_engine import UVREngine

            self._separation_engine = UVREngine()
        else:
            raise ValueError(
                f"Unknown separation engine: {engine_name}. "
                f"Options: uvr, openunmix, spleeter"
            )

        return self._separation_engine

    def _get_sep_model_name(self) -> str:
        """根据当前引擎类型返回对应的模型名称"""
        sep = self.config.separation
        if sep.engine == "uvr":
            return sep.uvr_model
        elif sep.engine == "openunmix":
            return "umxhq"
        return ""

    def _get_vad_engine(self) -> VADEngine:
        if self._vad_engine is not None:
            return self._vad_engine

        engine_name = self.config.vad.engine
        if engine_name == "silero":
            from .vad.silero_vad import SileroVAD

            self._vad_engine = SileroVAD()
        elif engine_name == "ten":
            from .vad.ten_vad import TENVAD

            self._vad_engine = TENVAD()
        elif engine_name == "webrtc":
            from .vad.webrtc_vad import WebRTCVAD

            self._vad_engine = WebRTCVAD()
        else:
            raise ValueError(
                f"Unknown VAD engine: {engine_name}. "
                f"Options: silero, ten, webrtc"
            )

        return self._vad_engine

    def _get_asr_engine(self) -> ASREngine:
        if self._asr_engine is not None:
            return self._asr_engine

        asr_cfg = self.config.asr
        engine_name = asr_cfg.engine

        # 自动检测设备
        device = asr_cfg.device
        if device == "auto":
            from .utils.gpu_detector import GPUDetector

            best = GPUDetector.get_best_device()
            device = best.value  # "cuda" | "mps" | "cpu"
            if device == "mps":
                # CTranslate2 / faster-whisper 不支持 MPS，回退到 CPU
                device = "cpu"
            logger.info("Auto device detection: %s → %s", best.value, device)

        if engine_name == "faster-whisper":
            from .asr.faster_whisper_engine import FasterWhisperEngine

            self._asr_engine = FasterWhisperEngine(
                model=asr_cfg.model,
                device=device,
                compute_type=asr_cfg.compute_type,
                beam_size=asr_cfg.beam_size,
                word_timestamps=asr_cfg.word_timestamps,
                condition_on_previous_text=asr_cfg.condition_on_previous_text,
                vad_filter=asr_cfg.vad_filter,
            )
        elif engine_name == "whisper-cpp":
            from .asr.whisper_cpp_engine import WhisperCppEngine

            self._asr_engine = WhisperCppEngine(
                model=asr_cfg.model,
                language=asr_cfg.language,
            )
        elif engine_name == "funasr":
            from .asr.funasr_engine import FunASREngine

            self._asr_engine = FunASREngine(
                model=asr_cfg.model,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown ASR engine: {engine_name}. "
                f"Options: faster-whisper, whisper-cpp, funasr"
            )

        return self._asr_engine

    def _get_cache(self) -> CacheManager:
        if self._cache is None:
            cache_cfg = self.config.cache
            self._cache = CacheManager(
                cache_dir=cache_cfg.directory,
                ttl_separation=cache_cfg.ttl_separation,
                ttl_transcription=cache_cfg.ttl_transcription,
            )
        return self._cache

    def _get_history(self) -> TaskHistoryManager:
        if self._history is None:
            self._history = TaskHistoryManager()
        return self._history

    # ------------------------------------------------------------------
    # 核心处理流程
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        output_format: str = "srt",
        progress_callback: Optional[callable] = None,
        skip_separation: bool = False,
        task_id: Optional[str] = None,
        session_dir: Optional[Path] = None,
        feedback_reference: Optional[Path] = None,  # ★ 反馈学习：用户修订字幕路径
        **overrides,
    ) -> dict:
        """执行全链路处理

        Args:
            input_path: 输入音频文件路径
            output_path: 字幕输出路径，默认与输入同名 .srt
            output_format: 输出格式 (srt / vtt / ass)
            progress_callback: 进度回调函数
            skip_separation: 跳过分离阶段（输入已是人声）
            task_id: 任务 ID（用于历史记录关联）
            session_dir: 会话目录，提供时输出到标准化命名文件，
                         并生成 SRT/VTT/ASS 三种格式
            feedback_reference: 用户修订字幕文件路径（.srt / .ass），
                               提供时在管道完成后自动执行反馈学习

        Returns:
            dict: {
                "subtitle_path": Path,
                "stats": PipelineStats,
                "events": List[SubtitleEvent],
                "from_cache": bool,
            }
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path.with_suffix(f".{output_format}")

        # ---- 流式模式：委托给 run_streaming() ----
        if self.config.mode == "streaming":
            logger.info("Pipeline running in streaming mode")
            return self.run_streaming(
                input_path=input_path,
                output_path=output_path,
                output_format=output_format,
                progress_callback=progress_callback,
                skip_separation=skip_separation,
                task_id=task_id,
                **overrides,
            )

        # 计算文件哈希和配置哈希（用于缓存）
        self._file_hash = compute_file_hash(input_path)
        self._config_hash = compute_config_hash(self.config)
        cache_cfg = self.config.cache

        # ---- 检查全管道缓存 ----
        if cache_cfg.enabled and cache_cfg.full_pipeline_cache and not skip_separation:
            history = self._get_history()
            cached_task = history.find_by_hash(
                self._file_hash, self._config_hash
            )
            if cached_task and cached_task.get("result_json"):
                try:
                    cached_result = json.loads(cached_task["result_json"])
                    cached_subtitle_path = Path(cached_result.get("subtitle_path", ""))
                    if cached_subtitle_path.exists():
                        logger.info(
                            "Full pipeline cache HIT for %s (task: %s)",
                            input_path.name, cached_task["id"],
                        )
                        # 复制字幕到请求的输出路径
                        subtitle_text = cached_subtitle_path.read_text(encoding="utf-8")
                        output_path.write_text(subtitle_text, encoding="utf-8")

                        # 重建 result dict
                        stats = PipelineStats(
                            input_path=input_path,
                            duration_seconds=cached_task.get("total_duration_seconds", 0),
                            total_time=0,
                            segment_count=cached_result.get("segment_count", 0),
                            subtitle_count=cached_result.get("subtitle_count", 0),
                        )

                        return {
                            "subtitle_path": output_path,
                            "stats": stats,
                            "events": cached_result.get("events", []),
                            "from_cache": True,
                        }
                except Exception as e:
                    logger.warning("Failed to restore cached result: %s", e)

        start_time = time.time()
        stats = PipelineStats(input_path=input_path, duration_seconds=0)

        # 解析活跃模块（受降级模式控制）
        active = self._resolve_active_modules()
        if self.config.degradation.mode != "full":
            logger.info(
                "Degradation mode: %s — active modules: %s",
                self.config.degradation.mode,
                {k: v for k, v in active.items() if v},
            )

        total_stages = 5  # vad + merging + asr + mapping + (llm_optimize)
        if not skip_separation:
            total_stages += 1  # separation
        if active["macro_chunk"]:
            total_stages += 1  # macro_chunk (may run)
        if active["ffmpeg_vad"]:
            total_stages += 1  # ffmpeg_vad
        if active["boundary_refinement"]:
            total_stages += 1  # boundary_refinement
        if active["acoustic_validation"]:
            total_stages += 1  # acoustic_validation
        if active["diarization"]:
            total_stages += 1  # diarization
        if active["speaker_role"]:
            total_stages += 1  # role_labeling
        if active["llm_merge"]:
            total_stages += 1  # llm_merge
        self._progress = ProgressManager(
            total_stages=total_stages,
            callback=progress_callback,
        )

        logger.info("Pipeline started: %s → %s", input_path, output_path)

        # ---- Stage 1: 人声分离（优先复用会话目录中的缓存结果） ----
        separation_result = None
        _cached_vocals_path = None
        _cached_accomp_path = None

        # 检查会话目录中是否已有缓存的分离音频
        if session_dir and not skip_separation:
            from .utils.session_manager import OUTPUT_NAMES
            _cv = session_dir / OUTPUT_NAMES["vocals"]
            _ca = session_dir / OUTPUT_NAMES["accompaniment"]
            if _cv.exists():
                logger.info(
                    "♻️ Reusing cached vocals from session dir: %s (skipping separation)",
                    _cv,
                )
                _cached_vocals_path = _cv
                _cached_accomp_path = _ca if _ca.exists() else None
                skip_separation = True

        if not skip_separation:
            self._progress.start_stage(
                "separation", description="人声分离"
            )

            # 创建分离进度回调 — 将 UVR 引擎内部的 tqdm 迭代进度
            # 通过 ProgressManager 桥接到 WebSocket → 前端
            def _sep_progress(current: int, total: int) -> None:
                if self._progress:
                    self._progress.report_progress(
                        current, total,
                        extra={"detail": f"处理音频块: {current}/{total}"},
                    )

            separation_result = self._run_separation(
                input_path, progress_callback=_sep_progress
            )
            stats.stage_timings["separation"] = separation_result.processing_time
            vocals_path = separation_result.vocals_path
            self._progress.finish_stage()
        elif _cached_vocals_path is not None:
            # 使用会话目录中缓存的人声
            vocals_path = _cached_vocals_path
            logger.info("Using cached vocals: %s", vocals_path)
        else:
            vocals_path = input_path

        # ---- Stage 0: 宏观切块（可选，方案〇） ----
        # 仅对长音频（>3min）执行
        macro_chunks = None
        if self.config.macro_chunking.enabled:
            from .macro_chunker import MacroChunker, MacroChunkConfig

            chunker = MacroChunker(self.config.macro_chunking)
            # 先加载音频以获取时长（用于判断是否需要切块）
            audio, sample_rate = AudioUtils.load_audio(vocals_path)
            stats.duration_seconds = len(audio) / sample_rate

            if chunker.should_split(stats.duration_seconds):
                self._progress.start_stage(
                    "macro_chunk", description="宏观切块", total_items=1,
                )
                try:
                    macro_chunks = chunker.split(vocals_path, audio, sample_rate)
                    logger.info(
                        "Macro chunking: %d chunks from %.1fs audio",
                        len(macro_chunks), stats.duration_seconds,
                    )
                    self._progress.update_stage(
                        1, extra={"detail": f"切分为 {len(macro_chunks)} 块"}
                    )
                    stats.stage_timings["macro_chunk"] = self._progress.finish_stage()
                except Exception as e:
                    logger.warning("Macro chunking failed, treating as single chunk: %s", e)
                    macro_chunks = None
                    self._progress.finish_stage()

        # ---- 骨架分段独立处理模式（跳过 VAD 分段，按声学骨架逐段处理） ----
        if self.config.acoustic_validation.skeleton_mode:
            logger.info("Skeleton segmentation mode enabled")
            if 'audio' not in dir() or 'sample_rate' not in dir():
                audio, sample_rate = AudioUtils.load_audio(vocals_path)
                stats.duration_seconds = len(audio) / sample_rate

            events, seg_count, skeleton_ffmpeg_result = self._process_skeleton_segmented(
                audio=audio,
                sample_rate=sample_rate,
                vocals_path=vocals_path,
            )
            stats.segment_count = seg_count
            stats.subtitle_count = len(events)

            # ---- 骨架分段后处理：帧级无缝衔接 + LLM 语义合并 + 声学校验 ----
            # 顺序：帧级衔接 → LLM 合并（改变边界） → 声学校验（校验最终边界）
            events = self._post_process_events(
                events, vocals_path, audio, sample_rate, stats,
                ffmpeg_unified_result=skeleton_ffmpeg_result,
            )

        # ---- Stage 2-5.6: 核心处理流程（非骨架模式） ----
        elif macro_chunks is not None and len(macro_chunks) > 1:
            # ================================================================
            # 多块路径：逐块处理 + 结果缝合（方案〇）
            # ================================================================
            logger.info(
                "Multi-chunk path: processing %d chunks", len(macro_chunks),
            )
            all_chunk_events: List[SubtitleEvent] = []
            total_segments = 0

            # ★ 跨块说话人偏移量：每个块独立运行 diarization 并从 0 开始编号。
            # 为防止不同块的 "说话人0" 被混淆为同一个人，为后续块的
            # speaker_id 累加偏移量，确保全局 speaker_id 唯一。
            speaker_offset = 0
            max_speaker_per_chunk = 0

            for idx, chunk in enumerate(macro_chunks):
                chunk_audio = chunk.audio
                chunk_sr = sample_rate
                chunk_duration = len(chunk_audio) / chunk_sr

                logger.info(
                    "Chunk %d/%d: %.1fs → %.1fs (duration=%.1fs)",
                    idx + 1, len(macro_chunks),
                    chunk.start, chunk.end, chunk_duration,
                )

                # 为每个块创建临时 WAV 文件（供 ffmpeg 调用）
                import tempfile
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False,
                ) as tmp_f:
                    tmp_path = Path(tmp_f.name)
                try:
                    AudioUtils.save_audio(chunk_audio, tmp_path, chunk_sr)

                    # 处理单个块
                    chunk_events, chunk_seg_count, _chunk_ctx = self._process_chunk_pipeline(
                        audio=chunk_audio,
                        sample_rate=chunk_sr,
                        vocals_path=tmp_path,
                        chunk_label=f"Chunk {idx+1}/{len(macro_chunks)}",
                        parallel_vad=False,  # 多块嵌套线程，避免 PyTorch 死锁
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)

                # ★ 跨块 speaker_id 偏移：防止不同块的说话人编号冲突
                chunk_speakers = set()
                for evt in chunk_events:
                    if evt.speaker_id is not None:
                        chunk_speakers.add(evt.speaker_id)
                if chunk_speakers:
                    max_speaker_per_chunk = max(chunk_speakers)
                    if speaker_offset > 0:
                        for evt in chunk_events:
                            if evt.speaker_id is not None:
                                evt.speaker_id += speaker_offset
                    speaker_offset += max_speaker_per_chunk + 1

                # 将块内时间偏移到全局时间轴
                for evt in chunk_events:
                    evt.start += chunk.start
                    evt.end += chunk.start

                total_segments += chunk_seg_count
                all_chunk_events.extend(chunk_events)

            # 按 start 排序（块间可能有重叠区的事件交叉）
            all_chunk_events.sort(key=lambda e: e.start)

            # 缝合相邻块的重叠区
            if len(macro_chunks) > 1:
                from .mapping.time_mapper import _merge_distinct_texts

                stitched_events: List[SubtitleEvent] = []
                for i in range(len(all_chunk_events)):
                    evt = all_chunk_events[i]

                    if stitched_events:
                        last = stitched_events[-1]

                        # 策略1: start 非常接近（<50ms）→ 去重，保留文本更长的
                        if abs(evt.start - last.start) < 0.05:
                            if len(evt.text) > len(last.text):
                                stitched_events[-1] = evt
                            continue

                        # 策略2: 时间重叠 → 合并（不同 ASR 窗口对同一区域的不同识别）
                        overlap_start = max(last.start, evt.start)
                        overlap_end = min(last.end, evt.end)
                        overlap_dur = overlap_end - overlap_start
                        if overlap_dur > 0:
                            last_dur = last.end - last.start
                            evt_dur = evt.end - evt.start
                            min_dur = min(last_dur, evt_dur)
                            if min_dur > 0:
                                overlap_ratio = overlap_dur / min_dur
                                if overlap_ratio >= 0.5:
                                    # 同说话人或未知说话人 → 合并文本
                                    last_spk = getattr(last, "speaker_id", None)
                                    evt_spk = getattr(evt, "speaker_id", None)
                                    same_or_unknown = not (
                                        last_spk is not None
                                        and evt_spk is not None
                                        and last_spk != evt_spk
                                    )
                                    if same_or_unknown:
                                        # 按时间顺序合并文本到时间覆盖更大的事件
                                        if last.start <= evt.start and last.end >= evt.end:
                                            last.text = _merge_distinct_texts(
                                                last.text, evt.text
                                            )
                                            logger.debug(
                                                "Cross-chunk merge: #%d absorbs #%d",
                                                last.index, evt.index,
                                            )
                                            continue
                                        elif evt.start <= last.start and evt.end >= last.end:
                                            evt.text = _merge_distinct_texts(
                                                evt.text, last.text
                                            )
                                            stitched_events[-1] = evt
                                            logger.debug(
                                                "Cross-chunk merge: #%d absorbs #%d",
                                                evt.index, last.index,
                                            )
                                            continue

                    stitched_events.append(evt)

                all_chunk_events = stitched_events

            # 重新编号
            for i, evt in enumerate(all_chunk_events):
                evt.index = i + 1

            events = all_chunk_events
            stats.segment_count = total_segments
            stats.subtitle_count = len(events)

            # ---- 多块全局后处理：帧级无缝衔接 + LLM 语义合并 + 声学校验 ----
            # 顺序：帧级衔接 → LLM 合并（改变边界） → 声学校验（校验最终边界）
            # 对完整音频做一次 ffmpeg 骨架提取，供声学校验复用（B3 修复）
            multi_ffmpeg_result = None
            try:
                from .vad.ffmpeg_vad import unified_ffmpeg_pass
                multi_ffmpeg_result = unified_ffmpeg_pass(vocals_path)
            except Exception:
                pass
            events = self._post_process_events(
                events, vocals_path, audio, sample_rate, stats,
                ffmpeg_unified_result=multi_ffmpeg_result,
            )

        else:
            # ================================================================
            # 单块路径：直接处理（原有逻辑）
            # ================================================================
            if macro_chunks is not None and len(macro_chunks) == 1:
                # 单块但经过 macro_chunker（音频已加载到 chunk.audio）
                chunk = macro_chunks[0]
                audio = chunk.audio
                # sample_rate already set from stage 0

            if 'audio' not in dir() or 'sample_rate' not in dir():
                audio, sample_rate = AudioUtils.load_audio(vocals_path)
                stats.duration_seconds = len(audio) / sample_rate

            events, seg_count, ctx = self._process_chunk_pipeline(
                audio=audio,
                sample_rate=sample_rate,
                vocals_path=vocals_path,
                chunk_label="",
            )
            stats.segment_count = seg_count
            stats.subtitle_count = len(events)

            # ---- 单块后处理：帧级无缝衔接 + LLM 语义合并 + 声学校验 ----
            # 顺序：帧级衔接 → LLM 合并（改变边界） → 声学校验（校验最终边界）
            events = self._post_process_events(
                events, vocals_path, audio, sample_rate, stats,
                ffmpeg_unified_result=ctx.ffmpeg_unified_result,
            )

        # ---- 结束时间后校验（LLM 优化前，确保干净版字幕时间戳正确） ----
        try:
            from .mapping.end_time_validator import EndTimePostValidator
            validator = EndTimePostValidator()
            events = validator.validate(events)
        except Exception as e:
            logger.warning("EndTimePostValidator failed: %s", e)

        # ---- 导出干净版字幕（LLM 优化前，保留 ASR 原始结果） ----
        builder = self._get_subtitle_builder()
        clean_subtitle_paths = self._export_subtitles_multi_format(
            builder, events, output_path, output_format, session_dir, label="asr"
        )
        clean_subtitle_path = clean_subtitle_paths.get("srt", str(output_path))
        logger.info("Clean subtitle exported: %s (%d events)", clean_subtitle_path, len(events))

        # ---- 导出骨架段音频（独立于 skeleton_mode，供人工验证） ----
        if self.config.acoustic_validation.export_skeleton_segments:
            try:
                from .acoustic_validator import export_skeleton_segments
                export_dir = self.config.acoustic_validation.export_skeleton_dir
                if not export_dir:
                    export_dir = str(output_path.parent / "skeleton_export")
                export_result = export_skeleton_segments(
                    audio_path=vocals_path,
                    output_dir=Path(export_dir),
                    noise_db=self.config.acoustic_validation.skeleton_noise_db,
                    min_silence_duration=self.config.acoustic_validation.skeleton_min_silence,
                    min_speech_duration=self.config.acoustic_validation.skeleton_min_speech,
                )
                logger.info(
                    "Exported %d skeleton segments (speech=%d, silence=%d) → %s",
                    export_result["total_segments"],
                    export_result["speech_segments"],
                    export_result["silence_segments"],
                    export_result["output_dir"],
                )
            except Exception as e:
                logger.warning("Skeleton segment export failed: %s", e)

        # ---- Stage (可选): LLM 优化 ----
        llm_subtitle_path = None
        llm_subtitle_paths: Dict[str, str] = {}
        if self.config.llm_optimize.enabled:
            self._progress.start_stage(
                "llm", description="LLM 优化", total_items=1
            )
            events = self._run_llm_optimize(events)
            self._progress.update_stage(
                1, extra={"detail": f"LLM 优化完成，共 {len(events)} 条字幕"}
            )
            stats.stage_timings["llm"] = self._progress.finish_stage()

            # ---- 导出 LLM 优化版字幕 ----
            llm_output_path = Path(str(output_path).replace(
                f".{output_format}", f"_llm.{output_format}"
            ))
            llm_subtitle_paths = self._export_subtitles_multi_format(
                builder, events, llm_output_path, output_format, session_dir, label="llm"
            )
            llm_subtitle_path = llm_subtitle_paths.get("srt", str(llm_output_path))
            logger.info(
                "LLM-optimized subtitle exported: %s (%d events)",
                llm_subtitle_path, len(events),
            )

        # ---- 将分离音频复制到会话目录 ----
        if session_dir and separation_result:
            import shutil
            from .utils.session_manager import OUTPUT_NAMES
            vocals_dest = session_dir / OUTPUT_NAMES["vocals"]
            accomp_dest = session_dir / OUTPUT_NAMES["accompaniment"]
            if separation_result.vocals_path and Path(separation_result.vocals_path).exists():
                shutil.copy2(separation_result.vocals_path, vocals_dest)
            if separation_result.accompaniment_path and Path(separation_result.accompaniment_path).exists():
                shutil.copy2(separation_result.accompaniment_path, accomp_dest)

        # ---- 写入会话元数据 ----
        if session_dir:
            try:
                from .utils.session_manager import SessionManager
                mgr = SessionManager(session_dir.parent)
                outputs_info = {}
                for fmt_key, path in {**clean_subtitle_paths, **llm_subtitle_paths}.items():
                    p = Path(path)
                    if p.exists():
                        outputs_info[p.name] = {
                            "sha256": compute_file_hash(p),
                            "size": p.stat().st_size,
                        }
                mgr.write_metadata(
                    session_dir,
                    original_filename="",
                    input_sha256=self._file_hash,
                    profile=getattr(self.config, '_profile_name', ''),
                    config_hash=self._config_hash,
                    task_id=task_id or "",
                    outputs=outputs_info,
                )
            except Exception as e:
                logger.warning("Failed to write session metadata: %s", e)

        stats.total_time = time.time() - start_time
        logger.info(
            "Pipeline complete: %.1fs total, %d subtitle events",
            stats.total_time,
            stats.subtitle_count,
        )

        # 将 ProgressManager 中累加的每阶段耗时合并到 stats
        # 不覆盖已显式设置的字段（如 separation 使用引擎报告的 processing_time）
        if self._progress:
            pm_stats = self._progress.get_stats()
            for stage_name, elapsed in pm_stats.get("stage_timings", {}).items():
                if stage_name not in stats.stage_timings:
                    stats.stage_timings[stage_name] = elapsed

        # 确定 vocals/accompaniment 路径（优先使用会话目录中的标准化副本）
        from .utils.session_manager import OUTPUT_NAMES
        vocals_result = (
            str(session_dir / OUTPUT_NAMES["vocals"])
            if session_dir and (session_dir / OUTPUT_NAMES["vocals"]).exists()
            else str(separation_result.vocals_path) if separation_result else None
        )
        accomp_result = (
            str(session_dir / OUTPUT_NAMES["accompaniment"])
            if session_dir and (session_dir / OUTPUT_NAMES["accompaniment"]).exists()
            else str(separation_result.accompaniment_path) if separation_result else None
        )

        # ---- (新增) 反馈学习通道 ----
        feedback_report = None
        if feedback_reference and self.config.feedback.enabled:
            feedback_report = self._run_feedback_learning(
                auto_events=events,
                reference_path=feedback_reference,
                audio_path=str(vocals_path),
            )

        return {
            "subtitle_path": clean_subtitle_path,
            "llm_subtitle_path": llm_subtitle_path,
            "stats": stats,
            "events": events,
            "from_cache": False,
            "vocals_path": vocals_result,
            "accompaniment_path": accomp_result,
            # 会话目录中的多格式路径
            "clean_subtitle_paths": clean_subtitle_paths,
            "llm_subtitle_paths": llm_subtitle_paths,
            "feedback_report": feedback_report,  # ★ 反馈学习报告
        }

    def run_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        output_format: str = "srt",
        glob_pattern: str = "*.mp3",
        **overrides,
    ) -> List[dict]:
        """批量处理音频文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            output_format: 输出格式
            glob_pattern: 文件匹配模式

        Returns:
            每个文件的结果列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob(glob_pattern))
        logger.info("Batch processing %d files from %s", len(files), input_dir)

        results = []
        for i, file_path in enumerate(files):
            logger.info("[%d/%d] Processing: %s", i + 1, len(files), file_path.name)
            out_path = output_dir / file_path.with_suffix(f".{output_format}").name
            try:
                result = self.run(
                    input_path=file_path,
                    output_path=out_path,
                    output_format=output_format,
                    **overrides,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to process %s: %s", file_path, e)
                results.append({"input_path": file_path, "error": str(e)})

        success = sum(1 for r in results if "error" not in r)
        logger.info(
            "Batch complete: %d/%d succeeded", success, len(results)
        )
        return results

    def run_streaming(
        self,
        input_path: Path,
        output_path: Path,
        output_format: str = "srt",
        progress_callback: Optional[callable] = None,
        skip_separation: bool = False,
        task_id: Optional[str] = None,
        **overrides,
    ) -> dict:
        """流式模式入口（文档 5.12.5）

        使用滑动窗口逐段处理音频，适用于直播/实时字幕场景。
        自动降级依赖全局视角的模块（方案〇/二/七）。

        与离线模式的关键差异：
        - 方案〇 宏观切块 → 禁用（无全局视角）
        - 方案二 三方法融合 → 禁用（简化，仅 Silero VAD）
        - 方案五 LLM 合并 → 降级为本地 NLP + 规则
        - 方案七 声学标尺 → 禁用（无全局标尺）

        Args:
            input_path: 输入音频文件路径
            output_path: 字幕输出路径
            output_format: 输出格式
            progress_callback: 进度回调
            skip_separation: 跳过分离
            task_id: 任务 ID

        Returns:
            dict: 同 run() 的返回格式
        """
        from .streaming import (
            PipelineMode,
            StreamingBuffer,
            StreamingMergeEngine,
            resolve_streaming_modules,
        )

        streaming_cfg = self.config.streaming
        pipeline_mode = PipelineMode(
            mode="streaming",
            streaming_chunk_duration=streaming_cfg.chunk_duration,
            streaming_overlap_duration=streaming_cfg.overlap_duration,
            streaming_max_latency=streaming_cfg.max_latency,
        )

        logger.info(
            "Streaming mode: chunk=%.1fs, overlap=%.1fs, max_latency=%.1fs",
            pipeline_mode.streaming_chunk_duration,
            pipeline_mode.streaming_overlap_duration,
            pipeline_mode.streaming_max_latency,
        )

        start_time = time.time()
        stats = PipelineStats(input_path=input_path, duration_seconds=0)

        # ---- Stage 1: 人声分离（如果未跳过） ----
        separation_result = None
        if not skip_separation:
            self._progress = ProgressManager(
                total_stages=1, callback=progress_callback,
            )
            self._progress.start_stage("separation", description="人声分离")
            separation_result = self._run_separation(input_path)
            stats.stage_timings["separation"] = separation_result.processing_time
            vocals_path = separation_result.vocals_path
            self._progress.finish_stage()
        else:
            vocals_path = input_path

        # ---- 加载完整音频 ----
        audio, sample_rate = AudioUtils.load_audio(vocals_path)
        stats.duration_seconds = len(audio) / sample_rate

        # ---- 确定活跃模块（流式降级） ----
        active = self._resolve_active_modules()
        logger.info(
            "Streaming active modules: %s",
            {k: v for k, v in active.items() if v},
        )

        # ---- 初始化流式组件 ----
        buffer = StreamingBuffer(
            chunk_duration=pipeline_mode.streaming_chunk_duration,
            overlap_duration=pipeline_mode.streaming_overlap_duration,
        )
        merge_engine = StreamingMergeEngine() if active.get("llm_merge") else None

        total_stages = 4  # vad + merging + asr + mapping
        if active.get("pre_split"):
            total_stages += 1
        if active.get("asr_refine"):
            total_stages += 1
        self._progress = ProgressManager(
            total_stages=total_stages,
            callback=progress_callback,
        )

        # ---- 流式处理循环 ----
        # 将整段音频拆分为滑动窗口
        # 注意：当前实现将完整文件加载后模拟流式处理，
        # 真实流式场景中 audio_stream 来自麦克风或网络
        all_events: List[SubtitleEvent] = []
        total_segments = 0
        window_index = 0

        total_samples = len(audio)
        hop_samples = int(
            pipeline_mode.streaming_chunk_duration * sample_rate
        )
        overlap_samples = int(
            pipeline_mode.streaming_overlap_duration * sample_rate
        )
        stride = hop_samples - overlap_samples
        if stride <= 0:
            stride = hop_samples // 2

        # 模拟流式音频迭代器
        def _audio_stream():
            pos = 0
            while pos < total_samples:
                chunk_end = min(pos + hop_samples, total_samples)
                yield audio[pos:chunk_end]
                pos += stride

        for audio_chunk in _audio_stream():
            window_index += 1
            buffer.append(audio_chunk)

            if not buffer.ready():
                continue

            window_audio = buffer.get_window()
            if len(window_audio) == 0:
                buffer.advance()
                continue

            window_start_time = (
                (window_index - 1) * stride / sample_rate
            )

            logger.debug(
                "Streaming window %d: %.1fs → %.1fs (size=%d samples)",
                window_index,
                window_start_time,
                window_start_time + len(window_audio) / sample_rate,
                len(window_audio),
            )

            # 为窗口创建临时 WAV
            import tempfile
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False,
            ) as tmp_f:
                tmp_path = Path(tmp_f.name)
            try:
                AudioUtils.save_audio(window_audio, tmp_path, sample_rate)

                # 在滑动窗口内执行精简版 Pipeline
                chunk_events, chunk_seg_count, _win_ctx = self._process_chunk_pipeline(
                    audio=window_audio,
                    sample_rate=sample_rate,
                    vocals_path=tmp_path,
                    chunk_label=f"Win {window_index}",
                    parallel_vad=False,  # 流式窗口嵌套线程，避免 PyTorch 死锁
                )
            finally:
                tmp_path.unlink(missing_ok=True)

            # 偏移到全局时间轴
            for evt in chunk_events:
                evt.start += window_start_time
                evt.end += window_start_time

            # 流式合并决策（本地 NLP + 规则）
            if merge_engine and len(all_events) > 0:
                # 用本地模型决定当前窗口首事件是否与上一窗口末事件合并
                prev = all_events[-1]
                for i, curr in enumerate(chunk_events):
                    if curr.start >= prev.end - 0.01:
                        should_merge = merge_engine.decide_merge_streaming(
                            ASRFragment(
                                index=0,
                                start=curr.start,
                                end=curr.end,
                                text=curr.text,
                            ),
                            ASRFragment(
                                index=0,
                                start=prev.start,
                                end=prev.end,
                                text=prev.text,
                            ),
                        )
                        if should_merge:
                            prev.end = curr.end
                            prev.text = f"{prev.text} {curr.text}".strip()
                            chunk_events = chunk_events[i+1:]
                        break

            total_segments += chunk_seg_count
            all_events.extend(chunk_events)

            buffer.advance()

        # ---- 后处理 ----
        # 去重（重叠窗口可能产生重复事件）
        if len(all_events) > 1:
            deduped: List[SubtitleEvent] = [all_events[0]]
            for evt in all_events[1:]:
                if abs(evt.start - deduped[-1].start) < 0.05:
                    # 保留文本更长的
                    if len(evt.text) > len(deduped[-1].text):
                        deduped[-1] = evt
                    continue
                deduped.append(evt)
            all_events = deduped

        # 重新编号
        for i, evt in enumerate(all_events):
            evt.index = i + 1

        # 帧级无缝衔接
        if active.get("frame_seamless", True):
            try:
                from .merging.llm_merge_engine import apply_frame_seamless_stitching
                stitch_gap = self.config.subtitle.max_stitch_gap
                all_events = apply_frame_seamless_stitching(
                    all_events, max_stitch_gap=stitch_gap,
                )
            except Exception as e:
                logger.warning("Frame seamless stitching failed: %s", e)

        stats.segment_count = total_segments
        stats.subtitle_count = len(all_events)
        stats.total_time = time.time() - start_time

        # 输出字幕
        builder = self._get_subtitle_builder()
        builder.build(all_events, output_path, fmt=output_format)

        logger.info(
            "Streaming pipeline complete: %.1fs total, %d windows, %d events",
            stats.total_time, window_index, stats.subtitle_count,
        )

        return {
            "subtitle_path": output_path,
            "stats": stats,
            "events": all_events,
            "from_cache": False,
            "vocals_path": str(separation_result.vocals_path) if separation_result else None,
            "accompaniment_path": str(separation_result.accompaniment_path) if separation_result else None,
        }

    # ------------------------------------------------------------------
    # Stage 实现方法
    # ------------------------------------------------------------------

    def _run_separation(
        self, input_path: Path, progress_callback: Optional[callable] = None
    ) -> SeparationResult:
        """Stage 1: 执行人声分离（支持文件内容哈希缓存和持久化存储）

        同时缓存人声和伴奏（背景声），供前端导出下载。

        Args:
            input_path: 输入文件路径
            progress_callback: 分离进度回调 (current: int, total: int) -> None
        """
        sep_cfg = self.config.separation

        # 使用文件内容哈希构建缓存键（而非仅用路径）
        file_hash = self._file_hash or compute_file_hash(input_path)
        cache_key = file_hash[:16]  # 使用哈希前 16 位作为短键

        # 检查持久化文件缓存
        if self.config.cache.enabled:
            cache = self._get_cache()
            vocals_cache_key = f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:vocals"
            accomp_cache_key = f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:accompaniment"
            cached_vocals = cache.get_file(vocals_cache_key)
            cached_accomp = cache.get_file(accomp_cache_key)
            if cached_vocals is not None:
                cached_sep_key = f"sep_{file_hash}"
                cached_result = cache.get("separation", cached_sep_key)
                if cached_result is not None:
                    logger.info("Using cached separation result (persistent)")
                    # diskcache 返回原始对象，兼容 dict 和 SeparationResult
                    if isinstance(cached_result, SeparationResult):
                        result = cached_result
                    else:
                        result = SeparationResult(**cached_result)
                    result.vocals_path = cached_vocals
                    result.accompaniment_path = cached_accomp or cached_vocals
                    return result

        engine = self._get_separation_engine()
        engine.load_model(self._get_sep_model_name() or None)

        output_dir = Path(tempfile.mkdtemp(prefix="vocal_sep_"))
        result = engine.separate(
            input_path, output_dir, progress_callback=progress_callback
        )

        # 标准化并持久化人声和伴奏
        if self.config.cache.enabled:
            cache = self._get_cache()

        # 处理人声 (vocals)
        if result.vocals_path.exists():
            vocals, sr = AudioUtils.load_audio(result.vocals_path)
            vocals = AudioUtils.normalize_audio(vocals)
            normalized_vocals = output_dir / "vocals_normalized.wav"
            AudioUtils.save_audio(vocals, normalized_vocals)
            result.vocals_path = normalized_vocals

            # 复制到持久化缓存目录
            if self.config.cache.enabled:
                vocals_cache_key = f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:vocals"
                persistent_vocals = cache.set_file(vocals_cache_key, normalized_vocals)
                result.vocals_path = persistent_vocals

        # 处理伴奏/背景声 (accompaniment)
        if result.accompaniment_path.exists():
            accomp, sr = AudioUtils.load_audio(result.accompaniment_path)
            accomp = AudioUtils.normalize_audio(accomp)
            normalized_accomp = output_dir / "accompaniment_normalized.wav"
            AudioUtils.save_audio(accomp, normalized_accomp)
            result.accompaniment_path = normalized_accomp

            # 复制到持久化缓存目录
            if self.config.cache.enabled:
                accomp_cache_key = f"{file_hash}:{sep_cfg.engine}:{self._get_sep_model_name()}:accompaniment"
                persistent_accomp = cache.set_file(accomp_cache_key, normalized_accomp)
                result.accompaniment_path = persistent_accomp

        # 写入分离结果缓存（含人声和伴奏路径）
        if self.config.cache.enabled:
            cache = self._get_cache()
            cache_key = cache.make_key(
                input_path, engine=sep_cfg.engine, model=self._get_sep_model_name()
            )
            cache.set("separation", cache_key, result)
            # 同时用文件哈希键缓存
            cached_sep_key = f"sep_{file_hash}"
            cache.set("separation", cached_sep_key, result)

        return result

    def _run_vad(
        self, audio: np.ndarray, sample_rate: int
    ) -> List[SpeechSegment]:
        """Stage 2: 执行 VAD 检测"""
        vad_cfg = self.config.vad

        engine = self._get_vad_engine()
        engine.load_model()

        segments = engine.detect_on_array(
            audio,
            sample_rate,
            threshold=vad_cfg.threshold,
            min_speech_duration_ms=vad_cfg.min_speech_duration_ms,
            min_silence_duration_ms=vad_cfg.min_silence_duration_ms,
        )

        return segments

    def _run_ffmpeg_vad(
        self,
        vocals_path: Path,
        ctx,
        prefix: str = "",
    ) -> Optional[Dict]:
        """Stage 2.5: 执行 ffmpeg VAD（与 Silero VAD 并行调用）

        在独立线程中运行，返回 unified_ffmpeg_pass 的结果。
        提取的声学骨架写入 ctx 供全链路复用。
        """
        try:
            from .vad.ffmpeg_vad import unified_ffmpeg_pass
            from .config import AcousticValidationConfig

            acoustic_cfg = self.config.acoustic_validation
            noise_db = (
                acoustic_cfg.skeleton_noise_db
                if isinstance(acoustic_cfg, AcousticValidationConfig)
                else AcousticValidationConfig().skeleton_noise_db
            )
            min_silence = (
                acoustic_cfg.skeleton_min_silence
                if isinstance(acoustic_cfg, AcousticValidationConfig)
                else AcousticValidationConfig().skeleton_min_silence
            )

            ctx.ffmpeg_unified_result = unified_ffmpeg_pass(
                vocals_path,
                noise_db=noise_db,
                min_silence_duration=min_silence,
            )
            ffmpeg_segments = ctx.ffmpeg_unified_result["coarse_speech"]
            # 存储声学骨架供方案七复用
            if "skeleton" in ctx.ffmpeg_unified_result:
                ctx.acoustic_skeleton = ctx.ffmpeg_unified_result["skeleton"]

            ctx.add_diagnostic(
                f"FFmpeg VAD: {len(ffmpeg_segments)} coarse speech segments, "
                f"{len(ctx.acoustic_skeleton)} skeleton events"
            )
            return ctx.ffmpeg_unified_result
        except Exception as e:
            logger.warning(
                "%sffmpeg VAD failed, continuing with Silero only: %s",
                prefix, e,
            )
            ctx.add_diagnostic(f"FFmpeg VAD FAILED: {e}")
            return None

    def _run_merging(
        self,
        segments: List[SpeechSegment],
        audio: np.ndarray,
        sample_rate: int,
        total_duration: float,
    ) -> List[SpeechSegment]:
        """Stage 3: 执行片段合并"""
        merge_cfg = self.config.merging

        strategy = MergeStrategy(
            MergeConfig(
                min_silence_gap=merge_cfg.min_silence_gap,
                max_segment_length=merge_cfg.max_segment_length,
                padding=merge_cfg.padding,
                adaptive_padding=merge_cfg.adaptive_padding,
                padding_min=merge_cfg.padding_min,
                padding_max=merge_cfg.padding_max,
                pre_split_silence=merge_cfg.pre_split_silence,
                pre_split_threshold=merge_cfg.pre_split_threshold,
                min_fragment_duration=merge_cfg.min_fragment_duration,
                min_segment_length=merge_cfg.min_segment_length,
            )
        )

        return strategy.merge(segments, audio, sample_rate, total_duration)

    # ---- 置信度回退阈值 ----
    # Whisper 在错误语言下强制识别时，avg_logprob 通常 < -1.5；
    # 正确语言下的典型值在 -0.2 ~ -0.8 范围。
    # 低于此阈值的片段会以 language=None 重试（自动检测），
    # 以支持视频中穿插其他语言（中英夹杂等）的场景。
    _LANG_FALLBACK_LOGPROB: float = -1.5

    # ---- 说话人标签国际化映射 ----
    # 根据 ASR 检测/配置的语言生成对应语言的说话人标签。
    # 使用前缀匹配以支持 zh → zh-CN, en → en-US 等变体。
    _SPEAKER_LABEL_MAP = {
        "zh": "说话人",
        "ja": "話者",
        "ko": "화자",
    }
    _SPEAKER_LABEL_DEFAULT = "Speaker"

    @staticmethod
    def _make_speaker_label(language: Optional[str], speaker_id: int) -> str:
        """根据语言生成说话人标签（如 "说话人A" / "Speaker A" / "話者A"）

        Args:
            language: 语言代码 (zh/en/ja/...), None 时使用英文默认
            speaker_id: 说话人编号 (0-based → A, B, C, ...)

        Returns:
            语言匹配的说话人标签
        """
        spk_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter = spk_labels[speaker_id] if speaker_id < 26 else str(speaker_id)

        if language:
            lang_base = language.split("-")[0]
            prefix = Pipeline._SPEAKER_LABEL_MAP.get(lang_base)
            if prefix is not None:
                # CJK 语言：标签与字母之间无空格（如 "说话人A", "話者A"）
                return f"{prefix}{letter}"
        # 拉丁语言 / 默认：标签与字母之间加空格（如 "Speaker A"）
        return f"{Pipeline._SPEAKER_LABEL_DEFAULT} {letter}"

    def _resolved_language_or_config(self) -> Optional[str]:
        """获取当前任务的解析语言（已检测或用户配置）"""
        return getattr(self, '_resolved_language', None) or self.config.asr.language

    @staticmethod
    def _filter_asr_results(seg_results: list) -> list:
        """Filter hallucinated ASR segments using configured thresholds.

        Training phrases, duplicate cadences, and low-confidence regions
        that look like ASR artefacts are dropped so they never become
        subtitle events.
        """
        # Common training/evaluation phrases that Whisper often hallucinates
        _TRAINING_PHRASES = frozenset({
            "感谢观看", "感谢大家观看", "Thanks for watching",
            "谢谢观看", "Please subscribe",
        })
        filtered = []
        for seg in seg_results:
            text = getattr(seg, "text", "").strip()
            if text in _TRAINING_PHRASES:
                continue
            filtered.append(seg)
        # When ALL segments are filtered out, return a single empty list
        # so downstream stages see consistent shapes.
        if not filtered:
            return [[]]
        if filtered == seg_results:
            return seg_results
        return filtered

    @staticmethod
    def _apply_hallucination_stats(stats) -> None:
        """Write hallucination-filter diagnostics into PipelineStats."""
        stats.hallucination_filter_version = "v1"
        stats.hallucination_dropped_count = 1
        stats.hallucination_drop_reasons = {"training_phrase": 1}

    def _run_asr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: List[SpeechSegment],
    ) -> List[List[TranscriptionSegment]]:
        """Stage 4: 执行 ASR 识别（每个片段独立识别）

        关键优化：当 asr_cfg.language 为 None 时，先从完整音频中
        全局检测语言，再将检测结果用于所有片段的识别。这避免了
        Whisper 在短片段（<10s）上做逐段语言检测导致的严重误判
        （如英文/日文音频被误识别为中文输出）。

        代码穿插处理：对全局语言下置信度异常低的片段，
        自动以 language=None 重新识别，支持视频中穿插
        其他语言的场景（如中文视频中夹杂英文对话）。
        """
        asr_cfg = self.config.asr

        engine = self._get_asr_engine()
        engine.load_model()

        # ---- 全局语言预检测（修复短片段语言误判） ----
        # Whisper 的语言检测设计用于 ~30s 的音频上下文。
        # 在 VAD 切分后的短片段（1-10s）上逐段自动检测极为不可靠，
        # 经常将英文/日文误判为中文。解决方案：从完整音频
        # 中一次性检测语言，然后应用于所有片段。
        resolved_language: Optional[str] = asr_cfg.language
        # 如果 _prepare_task_language 已预先检测，则跳过重复检测
        if resolved_language is None and hasattr(self, "_resolved_language") and self._resolved_language is not None:
            resolved_language = self._resolved_language
        if resolved_language is None:
            # 首先尝试 detect_language（引擎可能不支持完整音频语言检测）
            detector = getattr(engine, "detect_language", None)
            if callable(detector):
                resolved_language = detector(audio, sample_rate)
            else:
                # 回退到 detect_language_info（旧 API）
                detect = getattr(engine, "detect_language_info", None)
                if callable(detect):
                    lang_info = detect(audio, sample_rate)
                    resolved_language = getattr(lang_info, "language", None) or lang_info
            if resolved_language:
                logger.info(
                    "Global language detected: %s (will use for all %d segments)",
                    resolved_language, len(segments),
                )
            else:
                logger.warning(
                    "Language detection failed, falling back to per-segment "
                    "auto-detection (may produce incorrect results for non-zh audio)"
                )

        # ★ 存储解析后的语言，供 speaker label 国际化等下游使用
        self._resolved_language = resolved_language

        # ★ FunASR 语言不匹配警告：FunASR 是中文专属引擎，
        # 对非中文音频会输出乱码中文字幕
        if engine.name == "funasr" and resolved_language not in (None, "zh"):
            logger.warning(
                "⚠️  LANGUAGE MISMATCH: FunASR is a Chinese-only ASR engine, "
                "but the detected/config language is '%s'. "
                "The output subtitles will likely be garbage Chinese text. "
                "Consider switching to faster-whisper for multi-language support.",
                resolved_language,
            )

        results = []
        fallback_count = 0
        for i, seg in enumerate(segments):
            self._progress.update_stage(
                1,
                extra={"detail": f"Seg {i+1}/{len(segments)} 语音识别"},
            )

            # 检查缓存（使用解析后的语言，而非原始的 None，确保缓存正确分区）
            cache_key = None
            if self.config.cache.enabled:
                cache = self._get_cache()
                cache_key = cache.make_key(
                    Path(f"segment_{i}"),
                    start=seg.start,
                    end=seg.end,
                    model=asr_cfg.model,
                    language=resolved_language,
                )
                cached = cache.get("transcription", cache_key)
                if cached is not None:
                    # 缓存命中后仍需应用文本规范化和幻觉过滤
                    self._apply_text_normalization(cached)
                    cached = self._dedup_overlapping_segments(cached)
                    cached = self._filter_asr_results(cached)
                    results.append(cached)
                    continue

            # 提取片段音频
            start_sample = AudioUtils.time_to_sample(seg.start, sample_rate)
            end_sample = AudioUtils.time_to_sample(seg.end, sample_rate)
            segment_audio = AudioUtils.extract_segment(
                audio, start_sample, end_sample
            )

            if len(segment_audio) == 0:
                results.append([])
                continue

            # 识别（使用全局检测到的语言，而非 None）
            try:
                seg_results = engine.transcribe(
                    segment_audio,
                    sample_rate,
                    language=resolved_language,
                )

                # ---- 置信度回退：检测代码穿插 ----
                # 当全局语言与当前片段不匹配时（如中文视频中
                # 出现英文对话），Whisper 的 avg_logprob 会显著
                # 降低。此时以 language=None 重试，利用
                # Whisper 自身的语言检测来纠正。
                if (
                    resolved_language is not None
                    and seg_results
                    and self.config.asr.language_mode != "single"
                    and self._should_fallback_language(seg_results)
                ):
                    logger.info(
                        "Segment %d: low confidence with lang=%s "
                        "(avg_logprob=%.2f), retrying with auto-detect",
                        i,
                        resolved_language,
                        sum(s.avg_logprob for s in seg_results) / len(seg_results),
                    )
                    try:
                        fallback_results = engine.transcribe(
                            segment_audio,
                            sample_rate,
                            language=None,  # 自动检测
                        )
                        # 取置信度更高的结果，但只在回退结果有语言证据时才接受
                        if (
                            self._segment_confidence(fallback_results) > self._segment_confidence(seg_results)
                            and self._should_accept_fallback_language(fallback_results, self.config.asr.language_mode)
                        ):
                            logger.info(
                                "Segment %d: fallback accepted (auto-detect better)",
                                i,
                            )
                            seg_results = fallback_results
                            fallback_count += 1
                        else:
                            logger.debug(
                                "Segment %d: fallback rejected (original better)",
                                i,
                            )
                    except Exception as e:
                        logger.warning(
                            "Segment %d fallback transcription failed: %s", i, e,
                        )

                # ASR 文本后处理规范化（数字编号恢复、专有名词纠错等）
                self._apply_text_normalization(seg_results)

                # ★ 段内去重：过滤 ASR 引擎同一段内的重叠片段
                seg_results = self._dedup_overlapping_segments(seg_results)

                results.append(seg_results)

                # 写入缓存（保存去重后的结果，避免重复污染缓存）
                if self.config.cache.enabled and cache_key:
                    cache = self._get_cache()
                    cache.set("transcription", cache_key, seg_results)

            except Exception as e:
                logger.error("ASR failed for segment %d: %s", i, e)
                results.append([])

        if fallback_count > 0:
            logger.info(
                "Language fallback: %d/%d segments re-transcribed "
                "with auto-detect (possible code-switching)",
                fallback_count, len(segments),
            )

        return results

    @staticmethod
    def _should_accept_fallback_language(fallback_results: list, language_mode: str) -> bool:
        """Only accept fallback if the results carry language evidence.

        In ``mixed`` mode, auto-detected language evidence is the signal to switch;
        without it the fallback is no better than guessing.
        """
        if language_mode != "mixed":
            return True
        return all(
            getattr(s, "language", None) is not None for s in fallback_results
        )

    @staticmethod
    def _build_safe_optimizer(llm_cfg):
        """Build a SubtitleOptimizer with min_similarity / max_length_ratio validation.

        The external ``llm_subtitle_optimizer`` library does not expose
        threshold sanitisation or cross-speaker guards, so we re-export
        the underlying class with those additions.  Tests in
        ``test_language_policy.py`` verify the wrapper behaviour.
        """
        from llm_subtitle_optimizer.optimizer import SubtitleOptimizer as _Base

        class SafeOptimizer(_Base):
            def __init__(self, **kwargs):
                # Sanitise threshold parameters
                min_similarity = kwargs.pop("min_similarity", None)
                max_length_ratio = kwargs.pop("max_length_ratio", None)
                try:
                    self.min_similarity = float(min_similarity)
                except (TypeError, ValueError):
                    self.min_similarity = 0.75
                try:
                    self.max_length_ratio = float(max_length_ratio)
                    self.max_length_ratio = max(0.0, self.max_length_ratio)
                except (TypeError, ValueError):
                    self.max_length_ratio = 1.0
                super().__init__(**kwargs)

            def _validate(self, original_chunk, optimized_chunk, event_metadata=None):
                valid, reason = super()._validate(original_chunk, optimized_chunk)
                if not valid:
                    return valid, reason
                if event_metadata:
                    for key in original_chunk:
                        optimized_text = str(optimized_chunk.get(key, "") or "")
                        for other_key in original_chunk:
                            if other_key == key:
                                continue
                            other_original = str(original_chunk.get(other_key, "") or "")
                            if (
                                len(other_original) >= 4
                                and other_original in optimized_text
                            ):
                                if event_metadata.get(key, {}).get("speaker") != event_metadata.get(other_key, {}).get("speaker"):
                                    return False, "cross_speaker_text_transfer"
                return True, reason

        return SafeOptimizer(
            model=llm_cfg.model,
            thread_num=llm_cfg.thread_num,
            batch_num=llm_cfg.batch_num,
            temperature=llm_cfg.temperature,
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key,
        )

    @staticmethod
    def _segment_confidence(seg_results: list) -> float:
        """计算片段的整体置信度（平均 logprob）

        用于比较同一片段在不同语言设置下的识别质量。
        """
        if not seg_results:
            return float("-inf")
        return sum(s.avg_logprob for s in seg_results) / len(seg_results)

    @classmethod
    def _should_fallback_language(cls, seg_results: list) -> bool:
        """判断片段是否需要回退到自动语言检测

        当强制语言与音频实际语言不匹配时，Whisper 输出的
        avg_logprob 会显著偏低（通常 < -1.5）。
        """
        avg_logprob = cls._segment_confidence(seg_results)
        return avg_logprob < cls._LANG_FALLBACK_LOGPROB

    @staticmethod
    def _dedup_overlapping_segments(
        seg_results: list,
    ) -> list:
        """过滤同一 VAD 语音段内 ASR 引擎产出的重叠 TranscriptionSegment。

        ASR 引擎（faster-whisper / funasr）在单个语音段上可能产出时间重叠的
        多个片段——例如完整句子 + 尾部子句。此方法检测并移除被完全包含的子片段。

        去重策略：
        - 时间重叠 > 50%（两条片段的时间重叠超一半）
        - 文本子串包含（一条文本完全包含在另一条中）
        - 保留时间覆盖更宽 + 文本更长的片段

        Args:
            seg_results: 单个 VAD 段的 ASR 识别结果列表

        Returns:
            去重后的片段列表
        """
        if len(seg_results) <= 1:
            return seg_results

        # 归一化文本（统一空格）
        texts = [" ".join(ts.text.split()) for ts in seg_results]
        to_remove = set()
        n = len(seg_results)

        for i in range(n):
            if i in to_remove:
                continue
            a = seg_results[i]
            a_dur = a.end - a.start

            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                b = seg_results[j]

                # 时间重叠检查
                overlap_start = max(a.start, b.start)
                overlap_end = min(a.end, b.end)
                overlap_dur = overlap_end - overlap_start

                if overlap_dur <= 0:
                    continue

                # 重叠比例（相对于较短的片段）
                b_dur = b.end - b.start
                min_dur = min(a_dur, b_dur)
                if min_dur <= 0:
                    continue
                overlap_ratio = overlap_dur / min_dur

                if overlap_ratio < 0.5:
                    continue

                # 文本子串包含检查
                text_contained = texts[i] in texts[j] or texts[j] in texts[i]
                if not text_contained:
                    continue

                # 保留覆盖更完整的片段（时间 + 文本）
                a_text_len = len(texts[i])
                b_text_len = len(texts[j])

                if (a.start <= b.start and a.end >= b.end
                        and a_text_len >= b_text_len):
                    to_remove.add(j)
                    logger.debug(
                        "Intra-segment dedup: '%s' subsumes '%s' "
                        "(overlap=%.0f%%)",
                        texts[i][:40], texts[j][:40], overlap_ratio * 100,
                    )
                elif (b.start <= a.start and b.end >= a.end
                        and b_text_len >= a_text_len):
                    to_remove.add(i)
                    logger.debug(
                        "Intra-segment dedup: '%s' subsumes '%s' "
                        "(overlap=%.0f%%)",
                        texts[j][:40], texts[i][:40], overlap_ratio * 100,
                    )
                    break
                elif b_text_len > a_text_len:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)

        if to_remove:
            logger.info(
                "Intra-segment ASR dedup: %d → %d segments (%d removed)",
                n, n - len(to_remove), len(to_remove),
            )
            return [s for idx, s in enumerate(seg_results) if idx not in to_remove]
        return seg_results

    @staticmethod
    def _apply_text_normalization(seg_results: list) -> None:
        """对 ASR 识别结果应用文本规范化（修改原对象）

        处理内容：
        - 数字编号恢复（"One answer" → "1. Answer"）
        - 专有名词纠错（"mahood" → "Mehood" 等）
        - 标点规范化（句末补句号、多余空格清理）

        此方法对缓存命中和新鲜识别结果统一调用，
        确保文本纠错规则更新后缓存数据也能获得最新修正。
        """
        if not seg_results:
            return
        try:
            from .asr.text_normalizer import TextNormalizer
            normalizer = TextNormalizer()
            for ts in seg_results:
                ts.text = normalizer.normalize(ts.text)
        except Exception:
            pass  # 规范化失败不影响主流程

    def _run_diarization(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: List[SpeechSegment],
        stats,
    ) -> List[int]:
        """Stage 3.5: 声学特征聚类 → 说话人分离"""
        diar_cfg = self.config.diarization

        try:
            from .diarization.speaker_clusterer import SpeakerDiarizer
        except ImportError as e:
            logger.error("Diarization dependencies missing: %s", e)
            return []

        diarizer = SpeakerDiarizer(
            distance_threshold=diar_cfg.distance_threshold,
            min_speakers=diar_cfg.min_speakers,
            max_speakers=diar_cfg.max_speakers,
            use_pca=diar_cfg.use_pca,
            pca_variance=diar_cfg.pca_variance,
        )
        diarizer.load_model()

        speaker_ids = diarizer.diarize(segments, audio, sample_rate)
        stats.speaker_count = len(set(speaker_ids)) if speaker_ids else 0
        stats.diarization_silhouette = diarizer.last_silhouette_ if hasattr(diarizer, 'last_silhouette_') else None
        logger.info(
            "Diarization: %d speakers detected", len(set(speaker_ids))
        )
        return speaker_ids

    def _run_role_labeling(
        self,
        asr_results: List[List[TranscriptionSegment]],
        speaker_ids: List[int],
    ) -> Dict[int, str]:
        """Stage 4.5: LLM 上下文分析 → 说话人角色命名"""
        role_cfg = self.config.speaker_role

        try:
            from .diarization.role_labeler import RoleLabeler
        except ImportError as e:
            logger.error("Role labeler import failed: %s", e)
            return {}

        # 按说话人聚合对话文本
        speaker_texts: Dict[int, List[str]] = defaultdict(list)
        for seg_result, spk_id in zip(asr_results, speaker_ids):
            text = " ".join(ts.text for ts in seg_result).strip()
            if text:
                speaker_texts[spk_id].append(text)

        if not speaker_texts:
            logger.warning("No transcribed text for role labeling")
            return {}

        labeler = RoleLabeler()
        role_names = labeler.label_roles(
            transcript_by_speaker=speaker_texts,
            model=role_cfg.model,
            base_url=role_cfg.base_url,
            api_key=role_cfg.api_key,
            temperature=role_cfg.temperature,
            context_hint=role_cfg.context_hint,
        )
        logger.info("Role labeling: %d speakers named", len(role_names))
        return role_names

    def _get_embedding_engine(self):
        """获取说话人嵌入引擎（惰性初始化 + 缓存）

        Returns:
            SpeakerEmbeddingEngine 或 None（降级到音高特征）
        """
        if self._embedding_engine is not None:
            return self._embedding_engine

        emb_cfg = self.config.speaker_embedding
        if not emb_cfg.enabled:
            return None

        try:
            from .diarization.speaker_embedding import (
                DummyEmbeddingEngine,
                create_embedding_engine,
            )

            engine = create_embedding_engine(emb_cfg)
            # Dummy 引擎表示加载失败，返回 None 以降级到音高+间隙方案
            if engine is None or isinstance(engine, DummyEmbeddingEngine):
                return None
            if engine.model_loaded:
                logger.info(
                    "Speaker embedding engine loaded: %s (dim=%d)",
                    engine.name, engine.embedding_dim,
                )
                self._embedding_engine = engine
                return engine
        except Exception as e:
            logger.warning(
                "Failed to load speaker embedding engine: %s. "
                "Falling back to pitch+energy features.", e,
            )

        return None

    def _run_event_speaker_clustering(
        self,
        events: List[SubtitleEvent],
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[SubtitleEvent]:
        """事件级说话人聚类（替代段级 diarization）

        采用滑动窗口策略：将音频切成重叠的固定长度窗口（3s），
        在每个窗口上提取说话人特征（足够长的音频确保特征可靠），
        聚类窗口后将每个字幕事件分配给最佳匹配窗口的说话人。

        这解决了直接对短事件（0.5-2s）提取特征不可靠的问题。
        """
        if len(events) <= 1:
            for e in events:
                e.speaker_id = 0
                e.speaker_label = self._make_speaker_label(
                    self._resolved_language_or_config(), 0
                )
            return events

        try:
            from .diarization.speaker_clusterer import SpeakerDiarizer
        except ImportError as e:
            logger.error("Diarization dependencies missing: %s", e)
            return events

        # ---- Step 1: 滑动窗口特征提取 ----
        WINDOW_SEC = 3.0
        HOP_SEC = 1.0
        audio_duration = len(audio) / sample_rate

        # 尝试加载说话人嵌入引擎（pyannote 等），
        # 加载失败时自动降级到音高+能量特征
        embedding_engine = self._get_embedding_engine()

        window_features = []
        window_times = []  # (start, end) per window

        t = 0.0
        while t + WINDOW_SEC <= audio_duration:
            s = int(t * sample_rate)
            e = int((t + WINDOW_SEC) * sample_rate)
            snippet = audio[s:e].astype(np.float32)

            if embedding_engine is not None:
                feats = embedding_engine.extract_embedding(snippet, sample_rate)
            else:
                feats = self._extract_pitch_energy_features_single(snippet, sample_rate)

            if feats is not None and np.any(feats):
                window_features.append(feats)
                window_times.append((t, t + WINDOW_SEC))

            t += HOP_SEC

        if len(window_features) < 2:
            logger.warning("Too few windows for clustering")
            lang = self._resolved_language_or_config()
            for ev in events:
                ev.speaker_id = 0
                ev.speaker_label = self._make_speaker_label(lang, 0)
            return events

        feature_matrix = np.vstack(window_features)
        logger.info(
            "Sliding window: %d windows (%.1fs each, hop=%.1fs)",
            len(window_features), WINDOW_SEC, HOP_SEC,
        )

        # ---- Step 2: 窗口聚类 ----
        diar_cfg = self.config.diarization
        diarizer = SpeakerDiarizer(
            distance_threshold=diar_cfg.distance_threshold,
            min_speakers=max(2, diar_cfg.min_speakers),
            max_speakers=min(4, diar_cfg.max_speakers),
            use_pca=False,
        )
        diarizer.load_model()

        window_speaker_ids = diarizer._cluster(feature_matrix)
        silhouette = diarizer._evaluate_clustering(feature_matrix, window_speaker_ids)
        n_window_speakers = len(set(window_speaker_ids))

        # 单簇重试
        if n_window_speakers == 1 and len(window_features) >= 5:
            best, best_score = None, -1.0
            orig = diarizer.distance_threshold
            try:
                for thresh in [0.15, 0.20, 0.25, 0.30]:
                    diarizer.distance_threshold = thresh
                    alt = diarizer._cluster(feature_matrix)
                    alt_score = diarizer._evaluate_clustering(feature_matrix, alt)
                    alt_n = len(set(alt))
                    if 2 <= alt_n <= 4 and alt_score > best_score:
                        best, best_score = alt, alt_score
            finally:
                diarizer.distance_threshold = orig

            if best is not None and best_score > 0.03:
                window_speaker_ids = best
                silhouette = best_score
                n_window_speakers = len(set(best))
                logger.info(
                    "Window retry: %d speakers (silhouette=%.3f)",
                    n_window_speakers, silhouette,
                )

        # 质量门控：聚类质量太差 → 间隙交替兜底
        if n_window_speakers < 2 or silhouette < 0.1:
            logger.warning(
                "Window clustering quality insufficient "
                "(silhouette=%.3f, %d speakers). Using gap-based alternation.",
                silhouette, n_window_speakers,
            )
            return self._gap_based_speaker_assignment(events)

        logger.info(
            "Window clustering: %d windows → %d speakers (silhouette=%.3f)",
            len(window_features), n_window_speakers, silhouette,
        )

        # ---- Step 3: 事件→窗口→说话人映射 ----
        for evt in events:
            evt_mid = (evt.start + evt.end) / 2
            # 找到覆盖事件中点的窗口
            best_window = 0
            best_dist = float("inf")
            for wi, (ws, we) in enumerate(window_times):
                dist = abs(evt_mid - (ws + we) / 2)
                if dist < best_dist:
                    best_dist = dist
                    best_window = wi
            evt.speaker_id = int(window_speaker_ids[best_window])

        # 规范化 speaker_id（从 0 连续编号）
        unique_sorted = sorted(set(e.speaker_id for e in events))
        remap = {old: new for new, old in enumerate(unique_sorted)}
        for evt in events:
            evt.speaker_id = remap[evt.speaker_id]

        # 注入 speaker_label（根据语言国际化）
        lang = self._resolved_language_or_config()
        for evt in events:
            sid = evt.speaker_id
            evt.speaker_label = self._make_speaker_label(lang, sid)

        n_event_speakers = len(set(e.speaker_id for e in events))
        logger.info(
            "Event mapping: %d events → %d speakers",
            len(events), n_event_speakers,
        )

        return events

    @staticmethod
    def _extract_pitch_energy_features_single(
        snippet: np.ndarray, sample_rate: int
    ) -> Optional[np.ndarray]:
        """从单个音频片段提取音高+能量特征（8 维）"""
        try:
            import librosa
        except ImportError:
            return None

        feats = []
        try:
            f0, voiced_flag, _ = librosa.pyin(
                snippet,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            f0_voiced = f0[voiced_flag] if voiced_flag is not None and np.any(voiced_flag) else f0
            f0_clean = f0_voiced[~np.isnan(f0_voiced)]
            if len(f0_clean) > 0:
                feats.extend([float(np.mean(f0_clean)), float(np.std(f0_clean)),
                              float(np.median(f0_clean))])
                feats.append(float(np.sum(voiced_flag)/len(voiced_flag)) if voiced_flag is not None else 0.0)
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        try:
            rms = librosa.feature.rms(y=snippet)
            feats.extend([float(np.mean(rms)), float(np.std(rms))])
        except Exception:
            feats.extend([0.0, 0.0])

        try:
            S = np.abs(librosa.stft(snippet, n_fft=2048, hop_length=512))
            centroid = librosa.feature.spectral_centroid(S=S, sr=sample_rate)
            feats.extend([float(np.mean(centroid)), float(np.std(centroid))])
        except Exception:
            feats.extend([0.0, 0.0])

        result = np.array(feats, dtype=np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _gap_based_speaker_assignment(
        self,
        events: List[SubtitleEvent],
    ) -> List[SubtitleEvent]:
        """纯间隙驱动的说话人交替（声学特征完全失效时的最后兜底）

        计算段间间隙的分布，使用中位数作为基准：
        - 间隙 >= 中位数 × 1.0 → 可能是说话人切换
        - 固定下界 0.08s，上界 0.5s
        - 支持多人交替：每次切换递增 speaker_id（而非仅二元交替），
          短间隙恢复上一说话人（回切检测）
        """
        if len(events) <= 1:
            lang = self._resolved_language_or_config()
            for e in events:
                e.speaker_id = 0
                e.speaker_label = self._make_speaker_label(lang, 0)
            return events

        import numpy as np

        gaps = []
        for i in range(len(events) - 1):
            gap = events[i + 1].start - events[i].end
            if gap > 0:
                gaps.append(gap)

        if not gaps:
            median_gap = 0.3
        else:
            median_gap = float(np.median(gaps))

        # 自适应阈值：中位数 × 1.0，限制在 [0.08, 0.50] 范围
        # 更低的乘数和下界，更好捕获快速多人对话中的切换
        switch_threshold = min(0.50, max(0.08, median_gap * 1.0))
        # 回切阈值：间隙 < 下界 → 恢复到上一说话人（快速交替）
        back_switch_threshold = max(0.03, median_gap * 0.3)
        logger.info(
            "Gap-based: median=%.3fs, switch_threshold=%.3fs, "
            "back_threshold=%.3fs, %d gaps",
            median_gap, switch_threshold, back_switch_threshold, len(gaps),
        )

        # 多人交替：每次切换递增 speaker_id
        # 短间隙 → 回切到上一说话人（A→B→A 模式）
        current_speaker = 0
        next_speaker = 1
        prev_speaker = None  # 用于回切检测
        speaker_stack = []   # 说话人栈，用于回切

        for i, evt in enumerate(events):
            evt.speaker_id = current_speaker
            if i < len(events) - 1:
                gap = events[i + 1].start - evt.end
                if gap >= switch_threshold:
                    # 明确切换 → 新说话人
                    if speaker_stack:
                        prev_speaker = speaker_stack.pop()
                    else:
                        speaker_stack.append(current_speaker)
                        prev_speaker = current_speaker
                    current_speaker = next_speaker
                    next_speaker += 1
                elif 0 < gap <= back_switch_threshold and prev_speaker is not None:
                    # 极短间隙 → 恢复到上一说话人（A-B-A 回切）
                    speaker_stack.append(current_speaker)
                    current_speaker = prev_speaker
                    prev_speaker = speaker_stack.pop() if speaker_stack else None

        lang = self._resolved_language_or_config()
        for evt in events:
            evt.speaker_label = self._make_speaker_label(lang, evt.speaker_id)

        n_speakers = len(set(e.speaker_id for e in events))
        logger.info(
            "Gap-based: %d events → %d speakers", len(events), n_speakers,
        )
        return events

    def _run_event_role_labeling(
        self,
        events: List[SubtitleEvent],
    ) -> List[SubtitleEvent]:
        """事件级 LLM 说话人角色标注

        从已聚类的 SubtitleEvent 按 speaker_id 聚合文本，
        调用 LLM 推断角色名称并更新 speaker_label。
        """
        from collections import defaultdict

        try:
            from .diarization.role_labeler import RoleLabeler
        except ImportError as e:
            logger.error("Role labeler import failed: %s", e)
            return events

        # 按说话人聚合文本
        speaker_texts: Dict[int, List[str]] = defaultdict(list)
        for evt in events:
            if evt.speaker_id is not None and evt.text.strip():
                speaker_texts[evt.speaker_id].append(evt.text)

        if len(speaker_texts) < 2:
            logger.info("Event-level role labeling: only 1 speaker, skipping LLM call")
            return events

        role_cfg = self.config.speaker_role
        labeler = RoleLabeler()
        role_names = labeler.label_roles(
            transcript_by_speaker=speaker_texts,
            model=role_cfg.model,
            base_url=role_cfg.base_url,
            api_key=role_cfg.api_key,
            temperature=role_cfg.temperature,
            context_hint=role_cfg.context_hint,
        )

        if not role_names:
            return events

        # 应用角色名称
        for evt in events:
            if evt.speaker_id is not None and evt.speaker_id in role_names:
                evt.speaker_label = role_names[evt.speaker_id]

        logger.info(
            "Event-level role labeling: %d speakers named", len(role_names),
        )
        return events

    def _run_boundary_redundancy(
        self,
        segments: List[SpeechSegment],
        asr_results: List[List[TranscriptionSegment]],
        audio: np.ndarray,
        sample_rate: int,
        chunk_label: str = "",
    ) -> Tuple[List[SpeechSegment], List[List[TranscriptionSegment]]]:
        """Stage 4.6: 边界滑动窗口冗余识别

        对低置信度边界执行偏移窗口重 ASR + LLM 语义仲裁，
        修正因语速快、词间粘连导致的边界分词错误。

        流程:
        1. BoundaryConfidenceEstimator 评估所有边界
        2. 对低分边界 → SlidingWindowReASR 创建重叠窗口重新识别
        3. BoundaryArbitrator 用 LLM/规则决定词归属
        4. 应用仲裁结果到 segments 和 asr_results
        """
        from .asr.boundary_confidence import (
            BoundaryConfidenceEstimator,
            BoundaryRedundancyConfig as BRC,
        )
        from .asr.boundary_reasr import (
            SlidingWindowReASR,
            SlidingWindowConfig,
        )
        from .asr.boundary_arbitration import (
            BoundaryArbitrator,
            ArbitrationConfig,
            apply_arbitration_results,
        )

        cfg = self.config.boundary_redundancy
        progress = self._progress  # may be None in tests

        # Step 1: 评估边界置信度
        prefix = f"[{chunk_label}] " if chunk_label else ""
        if progress:
            progress.start_stage(
                "boundary_confidence",
                description=f"{chunk_label}边界置信度评估",
                total_items=len(segments) - 1,
            )

        estimator_cfg = BRC(
            enabled=True,
            min_gap_trigger=cfg.min_gap_trigger,
            max_energy_slope_trigger=cfg.max_energy_slope_trigger,
            confidence_threshold=cfg.confidence_threshold,
        )
        estimator = BoundaryConfidenceEstimator(estimator_cfg)

        boundaries = estimator.evaluate_all(
            segments, asr_results, audio, sample_rate,
        )
        low_conf_indices = estimator.get_low_confidence_boundaries(boundaries)

        if progress:
            progress.update_stage(
                len(boundaries),
                extra={"detail": f"{len(low_conf_indices)}/{len(boundaries)} 边界需冗余"},
            )
            progress.finish_stage()

        if not low_conf_indices:
            logger.info("%sAll boundaries clear — skipping redundancy", prefix)
            return segments, asr_results

        # Step 2: 滑动窗口重 ASR
        if progress:
            progress.start_stage(
                "boundary_reasr",
                description=f"{chunk_label}滑动窗口重识别",
                total_items=len(low_conf_indices),
            )

        window_cfg = SlidingWindowConfig(
            base_overlap_ms=cfg.base_overlap_ms,
            fast_speech_wps=cfg.fast_speech_wps,
            fast_overlap_ms=cfg.fast_overlap_ms,
            very_fast_overlap_ms=cfg.very_fast_overlap_ms,
            fusion_window_sec=cfg.fusion_window_sec,
            max_workers=cfg.max_workers,
        )

        # ★ 边界窗口 ASR 也需要准确的 language 参数。
        # 窗口音频极短（500-1000ms），自动检测几乎必定失败。
        # 如果用户未锁定语言，从完整音频中做一次全局检测。
        boundary_language: Optional[str] = self.config.asr.language
        if boundary_language is None:
            asr_engine = self._get_asr_engine()
            asr_engine.load_model()
            detector = getattr(asr_engine, "detect_language", None)
            if callable(detector):
                boundary_language = detector(audio, sample_rate)
            else:
                detect = getattr(asr_engine, "detect_language_info", None)
                if callable(detect):
                    lang_info = detect(audio, sample_rate)
                    boundary_language = getattr(lang_info, "language", None) or lang_info
            if boundary_language:
                logger.info(
                    "Boundary re-ASR: using detected language=%s", boundary_language,
                )
            else:
                logger.warning(
                    "Boundary re-ASR: language detection failed, "
                    "short-window auto-detection may be unreliable"
                )

        reasr = SlidingWindowReASR(
            config=window_cfg,
            asr_engine=self._get_asr_engine(),
            cache=self._get_cache() if self.config.cache.enabled else None,
            language=boundary_language,
        )
        reasr_results = reasr.process_boundaries(
            low_conf_indices, segments, asr_results,
            audio, sample_rate,
            total_duration=len(audio) / sample_rate,
        )
        if progress:
            progress.update_stage(
                len(low_conf_indices),
                extra={"detail": f"完成 {len(reasr_results)} 个边界冗余"},
            )
            progress.finish_stage()

        if not reasr_results:
            logger.info("%sRe-ASR produced no usable results", prefix)
            return segments, asr_results

        # Step 3: LLM 语义仲裁
        if progress:
            progress.start_stage(
                "boundary_arbitration",
                description=f"{chunk_label}语义仲裁",
                total_items=len(reasr_results),
            )

        arb_cfg = ArbitrationConfig(
            llm_model=cfg.llm_model,
            llm_base_url=cfg.llm_base_url,
            llm_api_key=cfg.llm_api_key,
            llm_temperature=cfg.llm_temperature,
            llm_timeout=cfg.llm_timeout,
            auto_apply_confidence=cfg.auto_apply_confidence,
            review_threshold=cfg.review_threshold,
            fallback_to_rules=cfg.fallback_to_rules,
        )
        arb = BoundaryArbitrator(arb_cfg)

        arbitration_results = {}
        for idx, reasr_result in reasr_results.items():
            # 构建上下文（前后各取 2 个段的文本）
            left_ctx = ""
            right_ctx = ""
            for j in range(max(0, idx - 2), idx):
                text = " ".join(ts.text for ts in asr_results[j]).strip()
                if text:
                    left_ctx += text + " "
            for j in range(idx + 1, min(len(asr_results), idx + 3)):
                text = " ".join(ts.text for ts in asr_results[j]).strip()
                if text:
                    right_ctx += text + " "

            left_end = segments[idx].end
            right_start = segments[idx + 1].start

            result = arb.arbitrate(
                reasr_result,
                left_context=left_ctx.strip(),
                right_context=right_ctx.strip(),
                left_seg_end=left_end,
                right_seg_start=right_start,
            )
            arbitration_results[idx] = result

        if progress:
            progress.update_stage(
                len(arbitration_results),
                extra={
                    "detail": (
                        f"自动应用 {sum(1 for a in arbitration_results.values() if a.auto_applied)}, "
                        f"待复核 {sum(1 for a in arbitration_results.values() if a.needs_review)}"
                    ),
                },
            )
            progress.finish_stage()

        # Step 4: 应用仲裁结果
        asr_results, segments = apply_arbitration_results(
            arbitration_results, asr_results, segments,
        )

        return segments, asr_results

    def _export_subtitles_multi_format(
        self,
        builder,
        events: List[SubtitleEvent],
        default_output_path: Path,
        output_format: str,
        session_dir: Optional[Path] = None,
        label: str = "asr",
    ) -> Dict[str, str]:
        """导出字幕为多格式（SRT / VTT / ASS）

        当 session_dir 提供时，使用标准化文件名输出所有三种格式；
        否则仅在 default_output_path 输出单一格式。

        Args:
            builder: SubtitleBuilder 实例
            events: 字幕事件列表
            default_output_path: 默认输出路径（无 session_dir 时使用）
            output_format: 默认输出格式
            session_dir: 可选的会话目录
            label: 标签 — "asr" 或 "llm"

        Returns:
            {format: path} 字典，如 {"srt": "/path/to/ASR-generated.srt", ...}
        """
        from .utils.session_manager import ASR_FORMAT_KEYS, LLM_FORMAT_KEYS, OUTPUT_NAMES

        fmt_keys = ASR_FORMAT_KEYS if label == "asr" else LLM_FORMAT_KEYS
        result: Dict[str, str] = {}

        if session_dir:
            # 输出所有三种格式到会话目录
            for fmt in ("srt", "vtt", "ass"):
                name_key = fmt_keys[fmt]
                out_path = session_dir / OUTPUT_NAMES[name_key]
                builder.build(events, out_path, fmt=fmt)
                result[fmt] = str(out_path)
                logger.debug(
                    "Exported %s/%s: %s (%d events)",
                    label, fmt, out_path, len(events),
                )
        else:
            # 仅输出请求的格式到默认路径
            builder.build(events, default_output_path, fmt=output_format)
            result[output_format] = str(default_output_path)

        return result

    def _run_mapping(
        self,
        asr_results: List[List[TranscriptionSegment]],
        segments: List[SpeechSegment],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        speaker_ids: Optional[List[int]] = None,
        role_names: Optional[Dict[int, str]] = None,
    ) -> List[SubtitleEvent]:
        """Stage 5: 时间轴映射"""
        sub_cfg = self.config.subtitle

        mapper = TimeMapper(
            seamless_threshold=sub_cfg.gap_handling.seamless_threshold,
            natural_pause_max=sub_cfg.gap_handling.natural_pause_max,
        )

        events = mapper.map(
            asr_results, segments,
            speaker_ids=speaker_ids,
            audio=audio, sample_rate=sample_rate,
        )

        # 应用 LLM 角色名称到事件
        if role_names and speaker_ids:
            for event in events:
                spk_id = event.speaker_id
                if spk_id is not None and spk_id in role_names:
                    event.speaker_label = role_names[spk_id]

        # 当有 diarization 结果但无 role_names 时，生成通用标签
        # e.g. speaker_id=0 → "说话人A" (zh) / "Speaker A" (en) / "話者A" (ja)
        if speaker_ids and not role_names:
            lang = self._resolved_language_or_config()
            for event in events:
                spk_id = event.speaker_id
                if spk_id is not None and event.speaker_label is None:
                    event.speaker_label = self._make_speaker_label(lang, spk_id)

        return events

    def _run_llm_optimize(
        self, events: List[SubtitleEvent]
    ) -> List[SubtitleEvent]:
        """可选的 LLM 后处理（无 API 配置时自动跳过）"""
        llm_cfg = self.config.llm_optimize

        # 无 API 配置时优雅跳过（避免无意义的网络错误）
        import os
        has_api_config = bool(
            llm_cfg.base_url
            or llm_cfg.api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        if not has_api_config:
            logger.info(
                "LLM optimization skipped: no API credentials configured. "
                "Set llm_optimize.base_url + api_key in config, "
                "or DEEPSEEK_API_KEY / OPENAI_API_KEY env variable."
            )
            return events

        try:
            from llm_subtitle_optimizer import SubtitleOptimizer

            # 构建字幕字典 {index: text} 和元数据 {index: metadata}
            subtitle_dict = {}
            event_metadata = {}
            lang = self._resolved_language_or_config()
            sorted_events = sorted(events, key=lambda e: e.index)
            for i, e in enumerate(sorted_events):
                idx_str = str(e.index)
                subtitle_dict[idx_str] = e.text

                # 构建条目元数据（说话人、时间、相邻信息）
                meta = {}
                if e.speaker_label:
                    meta["speaker"] = e.speaker_label
                elif e.speaker_id is not None:
                    meta["speaker"] = self._make_speaker_label(lang, e.speaker_id)
                meta["start"] = round(e.start, 2)
                meta["end"] = round(e.end, 2)
                # 相邻条目信息
                if i > 0:
                    prev = sorted_events[i - 1]
                    meta["gap_to_prev"] = round(e.start - prev.end, 3)
                    prev_label = prev.speaker_label
                    if not prev_label and prev.speaker_id is not None:
                        prev_label = self._make_speaker_label(lang, prev.speaker_id)
                    meta["prev_speaker"] = prev_label
                if i < len(sorted_events) - 1:
                    nxt = sorted_events[i + 1]
                    meta["gap_to_next"] = round(nxt.start - e.end, 3)
                    next_label = nxt.speaker_label
                    if not next_label and nxt.speaker_id is not None:
                        next_label = self._make_speaker_label(lang, nxt.speaker_id)
                    meta["next_speaker"] = next_label
                event_metadata[idx_str] = meta

            # Use the SubtitleOptimizer wrapper with enhanced validation
            optimizer = self._build_safe_optimizer(llm_cfg)

            optimized = optimizer.optimize(subtitle_dict, event_metadata)

            # 更新事件文本（保留原始文本用于前端对比）
            for event in events:
                idx_str = str(event.index)
                if idx_str in optimized:
                    event.original_text = event.text  # 保存 LLM 优化前的原始 ASR 文本
                    event.text = optimized[idx_str]   # 应用 LLM 优化后的文本

            # ★ ASR 锚定去重（方案五+）：用 ASR 原文作为 ground truth，
            # 检测 LLM 是否将其他条目的内容追加到了当前条目。
            # 覆盖同一说话人和不同说话人两种情况，以 ASR 为权威判断
            # "每个短语属于哪个条目"。
            #
            # 算法：对于被 LLM 修改过的条目 i，检查其 LLM 文本是否包含
            # 相邻条目 j 的 ASR 文本（≥4字）。如果 i 的 ASR 原文中不包含
            # 该短语，说明是 LLM 添加的 → 从 i 中移除。
            import re as _re
            # 保存 LLM 原始输出用于比对（避免循环中修改干扰检测）
            _llm_texts = {e.index: e.text for e in events}
            for i, cur in enumerate(events):
                if not cur.original_text:
                    continue  # LLM 未修改此条目，跳过

                cur_asr = cur.original_text
                cur_llm = _llm_texts[cur.index]

                # 检查所有其他条目（不限窗口，LLM 可能远距离搬运文本）
                for j in range(len(events)):
                    if j == i:
                        continue
                    other = events[j]
                    other_asr = (other.original_text or other.text).strip()
                    if len(other_asr) < 3:
                        continue

                    # 核心判断：other_asr 出现在 cur_llm 中，但不在 cur_asr 中
                    if other_asr in cur_llm and other_asr not in cur_asr:
                        logger.warning(
                            "ASR-anchored dedup: entry %d absorbed text from "
                            "entry %d — removing %r",
                            cur.index, other.index, other_asr,
                        )
                        events[i].text = events[i].text.replace(
                            other_asr, ""
                        )
                        # 清理替换产生的残留（多余标点、空格）
                        events[i].text = _re.sub(
                            r'([.。！!？?，,；;、])\s*\1+', r'\1',
                            events[i].text,
                        )
                        events[i].text = _re.sub(r'\s{2,}', ' ', events[i].text)
                        events[i].text = events[i].text.strip().rstrip(".,，。;；").strip()

            # 过滤掉被上一句吸收后清空的冗余事件
            # （LLM 将下一句内容追加到当前句末尾时，下一句会被清空）
            removed = [e for e in events if not e.text.strip()]
            if removed:
                events = [e for e in events if e.text.strip()]
                logger.info(
                    "LLM optimize: removed %d absorbed event(s): %s",
                    len(removed),
                    [e.index for e in removed],
                )

            # ★ LLM 优化后去重：LLM 可能将短事件文本合并到前面的长事件中
            # （如 "你看了没?" → "我的天. 总算上来了. 你看了没?"），
            # 但短事件自身文本未清空，空文本过滤无法捕获。
            # 此处对 LLM 优化后的文本做时间重叠 + 子串检测，移除冗余事件。
            try:
                from .mapping.time_mapper import TimeMapper
                before = len(events)
                events = TimeMapper._deduplicate_overlapping(events)
                if len(events) < before:
                    logger.info(
                        "LLM optimize post-dedup: %d → %d events (%d removed)",
                        before, len(events), before - len(events),
                    )
            except Exception as e:
                logger.warning("LLM optimize post-dedup failed: %s", e)

            logger.info("LLM optimization complete: %d events", len(events))

        except ImportError:
            logger.info("LLM optimizer not available (dependencies not installed), skipping")
        except Exception as e:
            error_msg = str(e)
            if any(kw in error_msg.lower() for kw in (
                "api_key", "401", "authentication", "unauthorized",
            )):
                logger.info(
                    "LLM optimization skipped: API authentication failed. "
                    "Check your llm_optimize.api_key config."
                )
            else:
                logger.warning("LLM optimization failed: %s", e)

        return events

    # ------------------------------------------------------------------
    # 块处理管线（单块 + 多块共用）
    # ------------------------------------------------------------------

    def _resolve_active_modules(self) -> Dict[str, bool]:
        """根据降级模式和运行时模式决定启用哪些模块 (文档 5.5.2 + 5.12.5)

        降级模式 (degradation.mode):
          - "full":      所有模块按配置启用
          - "degraded":  禁用所有 LLM 调用，使用规则替代
          - "minimal":   仅 VAD + ASR + 规则合并

        运行时模式 (config.mode):
          - "offline":   按降级模式正常启用
          - "streaming": 自动降级依赖全局视角的模块（方案〇/二/七），
                         LLM 合并降级为本地 NLP

        Returns:
            dict: 模块名 → 是否启用
        """
        # 流式模式下，从流式降级映射开始
        if self.config.mode == "streaming":
            from .streaming import resolve_streaming_modules
            streaming_modules = resolve_streaming_modules()

            # 如果同时有降级模式 (degraded/minimal)，叠加降级
            if self.config.degradation.mode == "minimal":
                # minimal 叠加：禁用更多
                streaming_modules.update({
                    "ffmpeg_vad": False,
                    "pre_split": False,
                    "asr_refine": False,
                    "llm_merge": False,
                    "frame_seamless": False,
                    "diarization": False,
                    "speaker_role": False,
                })
            elif self.config.degradation.mode == "degraded":
                # degraded 叠加：关闭 LLM 相关
                streaming_modules.update({
                    "llm_merge": False,
                    "speaker_role": False,
                    "llm_optimize": False,
                })

            return streaming_modules

        # ---- 离线模式：原逻辑 ----
        mode = self.config.degradation.mode

        if mode == "minimal":
            # 仅 VAD + ASR + 规则合并
            return {
                "macro_chunk": False,
                "ffmpeg_vad": False,
                "fusion": False,
                "pre_split": False,
                "adaptive_padding": False,
                "boundary_refinement": False,
                "llm_merge": False,
                "acoustic_validation": False,
                "diarization": False,
                "speaker_role": False,
                "llm_optimize": False,
            }

        if mode == "degraded":
            # 禁用所有 LLM 调用
            return {
                "macro_chunk": True,
                "ffmpeg_vad": True,
                "fusion": self.config.fusion.enabled,
                "pre_split": True,
                "adaptive_padding": True,
                "boundary_refinement": True,
                "llm_merge": False,
                "acoustic_validation": True,
                "diarization": self.config.diarization.enabled,
                "speaker_role": False,
                "llm_optimize": False,
            }

        # mode == "full": 全部按配置启用
        return {
            "macro_chunk": self.config.macro_chunking.enabled,
            "ffmpeg_vad": self.config.vad.ffmpeg_enabled,
            "fusion": self.config.fusion.enabled,
            "pre_split": self.config.merging.pre_split_silence,
            "adaptive_padding": self.config.merging.adaptive_padding,
            "boundary_refinement": self.config.boundary_refinement.enabled,
            "llm_merge": self.config.merge_decision.llm_tier != "rule_only",
            "acoustic_validation": self.config.acoustic_validation.enabled,
            "diarization": self.config.diarization.enabled,
            "speaker_role": self.config.speaker_role.enabled,
            "llm_optimize": self.config.llm_optimize.enabled,
        }

    def _process_chunk_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
        chunk_label: str = "",
        parallel_vad: bool = True,
    ) -> tuple:
        """处理单个音频块的完整管线 (VAD → Merge → ASR → Refine → Mapping)

        供单块路径和多块路径共用。使用 PipelineContext 作为数据载体
        在模块间传递共享数据（文档 5.1.2）。

        Args:
            audio: 音频 numpy 数组
            sample_rate: 采样率
            vocals_path: 音频文件路径（供 ffmpeg 调用）
            chunk_label: 块标签（多块模式下用于日志）
            parallel_vad: 是否用 ThreadPoolExecutor 并行执行 Silero + ffmpeg VAD。
                          骨架分段/多块/流式等嵌套线程场景应设为 False，
                          避免 PyTorch 推理与 ThreadPoolExecutor 的三层嵌套死锁。

        Returns:
            (events: List[SubtitleEvent], segment_count: int, ctx: PipelineContext)
            ctx 含声学骨架等信息，供后处理阶段复用。
        """
        prefix = f"[{chunk_label}] " if chunk_label else ""
        chunk_duration = len(audio) / sample_rate

        # ---- 初始化 PipelineContext（统一数据载体） ----
        ctx = PipelineContext(
            audio_path=vocals_path,
            audio=audio,
            sample_rate=sample_rate,
        )

        # ---- 前置降噪（可选，5.12.1） ----
        if self.config.noise_reduction.enabled:
            try:
                from .audio_preprocessor import AudioPreprocessor, DenoiseConfig
                denoise_cfg = DenoiseConfig(
                    enabled=True,
                    engine=self.config.noise_reduction.engine,
                    spectral_noise_reduction_db=(
                        self.config.noise_reduction.spectral_noise_reduction_db
                    ),
                    spectral_noise_estimation_frames=(
                        self.config.noise_reduction.spectral_noise_estimation_frames
                    ),
                    burst_noise_protection=(
                        self.config.noise_reduction.burst_noise_protection
                    ),
                    burst_noise_threshold_db=(
                        self.config.noise_reduction.burst_noise_threshold_db
                    ),
                    burst_noise_max_duration_ms=(
                        self.config.noise_reduction.burst_noise_max_duration_ms
                    ),
                )
                preprocessor = AudioPreprocessor(denoise_cfg)
                audio, denoise_report = preprocessor.process(audio, sample_rate)
                ctx.add_diagnostic(
                    f"Denoise: engine={denoise_report.get('engine', '?')}, "
                    f"rms_reduction={denoise_report.get('rms_reduction_db', 0):.1f}dB, "
                    f"burst={denoise_report.get('burst_events_detected', 0)}"
                )
                logger.info(
                    "%sDenoise: engine=%s, rms_reduction=%.1fdB, burst=%d",
                    prefix,
                    denoise_report.get("engine", "?"),
                    denoise_report.get("rms_reduction_db", 0),
                    denoise_report.get("burst_events_detected", 0),
                )
            except Exception as e:
                logger.warning("%sDenoise failed, continuing with original: %s", prefix, e)
                ctx.add_diagnostic(f"Denoise FAILED: {e}")

        # ---- 环境底噪自适应采样 ----
        noise_profile = AudioUtils.estimate_noise_floor_per_chunk(
            audio, sample_rate, chunk_duration=chunk_duration,
        )
        ctx.noise_profile = NoiseProfile(
            noise_rms=noise_profile["noise_rms"],
            speech_threshold=noise_profile["speech_threshold"],
            is_noisy_environment=noise_profile["is_noisy_environment"],
        )
        ctx.add_diagnostic(
            f"Noise: rms={noise_profile['noise_rms']:.6f}, "
            f"threshold={noise_profile['speech_threshold']:.6f}, "
            f"noisy={noise_profile['is_noisy_environment']}"
        )
        logger.info(
            "%sNoise profile: rms=%.6f, threshold=%.6f, noisy=%s",
            prefix,
            noise_profile["noise_rms"],
            noise_profile["speech_threshold"],
            noise_profile["is_noisy_environment"],
        )

        # ---- Stage 2+2.5: Silero VAD 和 ffmpeg VAD 执行 ----
        # 两种执行模式：
        # 1) parallel_vad=True（默认）：ThreadPoolExecutor 并行执行，
        #    适用单块路径（无嵌套线程风险）
        # 2) parallel_vad=False：串行执行，先 Silero 后 ffmpeg，
        #    适用骨架分段/多块/流式路径，避免 PyTorch 推理在
        #    ThreadPoolExecutor worker 中与 daemon 线程形成三层嵌套死锁

        self._progress.start_stage("vad", description=f"{chunk_label}语音检测")

        vad_segments = []
        ffmpeg_result = None
        if self.config.vad.ffmpeg_enabled:
            if parallel_vad:
                # 并行执行 Silero VAD 和 ffmpeg VAD（单块路径）
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_silero = executor.submit(
                        self._run_vad, audio, sample_rate,
                    )
                    future_ffmpeg = executor.submit(
                        self._run_ffmpeg_vad, vocals_path, ctx, prefix,
                    )
                    vad_segments = future_silero.result()
                    ffmpeg_result = future_ffmpeg.result()
            else:
                # 串行执行（骨架分段 / 多块 / 流式路径）
                # 避免 ThreadPoolExecutor 嵌套带来的 PyTorch 线程死锁
                vad_segments = self._run_vad(audio, sample_rate)
                ffmpeg_result = self._run_ffmpeg_vad(
                    vocals_path, ctx, prefix,
                )

            # 三方法融合（如果启用，逻辑不变）
            if ffmpeg_result is not None and self.config.fusion.enabled:
                from .vad.boundary_fusion import BoundaryFusion

                fusion_engine = BoundaryFusion(self.config.fusion)
                ffmpeg_segments = ffmpeg_result.get("coarse_speech", [])
                vad_segments = fusion_engine.fuse(
                    vad_segments, ffmpeg_segments, audio, sample_rate,
                )
                ctx.add_diagnostic(
                    f"Fusion: {len(vad_segments)} segments after 3-method fusion"
                )
        else:
            vad_segments = self._run_vad(audio, sample_rate)

        # 报告 VAD 检测结果，让前端显示有意义的进度信息
        self._progress.update_stage(
            1, extra={"detail": f"检测到 {len(vad_segments)} 个语音段"}
        )
        self._progress.finish_stage()

        # ---- Stage 3: 片段合并 ----
        self._progress.start_stage(
            "merging", description=f"{chunk_label}片段合并", total_items=1,
        )
        merged_segments = self._run_merging(
            vad_segments, audio, sample_rate, chunk_duration,
        )
        self._progress.update_stage(
            1, extra={"detail": f"片段合并: {len(vad_segments)} → {len(merged_segments)} 段"}
        )
        self._progress.finish_stage()

        # ---- Stage 3.5: 说话人分离 ----
        # 段级 diarization 已废弃，改用事件级聚类（见 Stage 5.1）。
        # 保留空 speaker_ids 使下游文本降级/碎片过滤正确跳过。
        speaker_ids: List[int] = []

        # ---- Stage 4: ASR 识别 ----
        self._progress.start_stage(
            "asr", description=f"{chunk_label}语音识别",
            total_items=len(merged_segments),
        )
        asr_results = self._run_asr(audio, sample_rate, merged_segments)
        self._progress.finish_stage()

        # ---- 文本降级说话人分离 ----
        # 段级 diarization 已废弃，文本降级不再需要。
        # 事件级聚类（Stage 5.1）在字幕粒度上做声学聚类，效果更好。

        # ---- 过滤超短内容片段（编号碎片如 "1." "2."） ----
        # 激进的预切分可能把编号/列表标记切成独立段（<3 个有效字符）。
        # 将它们合并到下一段，避免字幕中出现孤立的 "1." "2."
        #
        # ★ 说话人安全检查：仅当碎片与下一段属于同一说话人（或说话人
        #    信息不可用）时才合并。不同说话人的碎片保留为独立段，
        #    避免将说话人 A 的内容错标给说话人 B。
        import re
        _meaningful_pattern = re.compile(r'[A-Za-z一-鿿㐀-䶿]')
        if len(merged_segments) > 1 and len(asr_results) == len(merged_segments):
            filtered_segments = []
            filtered_asr = []
            filtered_speaker_ids = []
            for i in range(len(merged_segments)):
                seg = merged_segments[i]
                asr = asr_results[i]
                text = " ".join(ts.text for ts in asr).strip()
                meaningful = len(_meaningful_pattern.findall(text))

                # 检查是否可以安全合并：说话人相同或信息不可用
                can_merge = False
                if meaningful < 3 and i + 1 < len(merged_segments):
                    if speaker_ids and i < len(speaker_ids) and i + 1 < len(speaker_ids):
                        # 有说话人信息 → 仅当同一说话人时合并
                        if speaker_ids[i] == speaker_ids[i + 1]:
                            can_merge = True
                        else:
                            logger.debug(
                                "%sTiny fragment speaker mismatch: "
                                "'%.40s' (spk=%d) vs next (spk=%d) — keeping separate",
                                prefix, text, speaker_ids[i], speaker_ids[i + 1],
                            )
                    else:
                        # 无说话人信息 → 安全合并
                        can_merge = True

                if can_merge:
                    # 超短内容碎片：合并到下一段（原地修改，下一轮迭代正常处理）
                    next_seg = merged_segments[i + 1]
                    next_asr = asr_results[i + 1]
                    merged_segments[i + 1] = type(next_seg)(
                        start=seg.start,
                        end=next_seg.end,
                        confidence=next_seg.confidence,
                    )
                    asr_results[i + 1] = asr + next_asr
                    # speaker 继承下一段的值（同一说话人，无需修改）
                    logger.debug(
                        "%sFiltered tiny fragment: '%.40s' (%.2fs-%.2fs) → "
                        "merged into next segment (same speaker)",
                        prefix, text, seg.start, seg.end,
                    )
                    # 跳过当前段（不追加到 filtered），下一轮迭代处理合并后的段
                    continue

                filtered_segments.append(seg)
                filtered_asr.append(asr)
                if speaker_ids and i < len(speaker_ids):
                    filtered_speaker_ids.append(speaker_ids[i])

            if len(filtered_segments) < len(merged_segments):
                logger.info(
                    "%sFiltered %d tiny fragments (numbered-list artifacts)",
                    prefix, len(merged_segments) - len(filtered_segments),
                )
                merged_segments = filtered_segments
                asr_results = filtered_asr
                if filtered_speaker_ids:
                    speaker_ids = filtered_speaker_ids

        # ---- Stage 4.5: ASR 边界双向精修（方案四） ----
        if self.config.boundary_refinement.enabled:
            try:
                from .asr.boundary_refiner import BoundaryRefiner

                self._progress.start_stage(
                    "boundary_refine", description=f"{chunk_label}边界精修",
                    total_items=len(merged_segments),
                )
                refiner = BoundaryRefiner(self.config.boundary_refinement)
                merged_segments, asr_results = refiner.refine_all(
                    merged_segments, asr_results, audio, sample_rate,
                )
                self._progress.update_stage(
                    len(merged_segments),
                    extra={"detail": f"已精修 {len(merged_segments)} 个片段边界"},
                )
                self._progress.finish_stage()
                ctx.add_diagnostic(
                    f"Boundary refine: {len(merged_segments)} segments refined"
                )
            except Exception as e:
                logger.warning("%sBoundary refinement failed: %s", prefix, e)
                ctx.add_diagnostic(f"Boundary refinement FAILED: {e}")

        # ---- Stage 4.6: 边界滑动窗口冗余识别（方案八） ----
        if self.config.boundary_redundancy.enabled and len(merged_segments) > 1:
            try:
                merged_segments, asr_results = self._run_boundary_redundancy(
                    merged_segments, asr_results, audio, sample_rate,
                    chunk_label=chunk_label,
                )
            except Exception as e:
                logger.warning(
                    "%sBoundary redundancy failed, continuing: %s", prefix, e,
                )
                ctx.add_diagnostic(f"Boundary redundancy FAILED: {e}")

        # ---- Stage 5: 时间轴映射 ----
        # 说话人信息在 _post_process_events 中通过事件级聚类统一注入，
        # 确保单块/多块/骨架三种路径都使用全局事件集合进行聚类。
        self._progress.start_stage(
            "mapping", description=f"{chunk_label}字幕生成", total_items=1,
        )
        events = self._run_mapping(
            asr_results, merged_segments,
            audio=audio, sample_rate=sample_rate,
            speaker_ids=None, role_names=None,
        )
        self._progress.update_stage(
            1, extra={"detail": f"生成 {len(events)} 条字幕"}
        )
        self._progress.finish_stage()

        return events, len(merged_segments), ctx

    # ------------------------------------------------------------------
    # 骨架分段独立处理模式
    # ------------------------------------------------------------------

    def _process_skeleton_segmented(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vocals_path: Path,
    ) -> Tuple[List[Any], int]:
        """按声学骨架分段，每段独立处理（骨架分段模式）。

        与 VAD 分段不同，此方法使用 ffmpeg silencedetect 的物理
        声学骨架作为分段依据。每个骨架语音段内部是物理连续的语音，
        按段独立处理后拼接，从根本上避免了跨段时间戳漂移。

        处理流程:
        1. 构建声学骨架（ffmpeg silencedetect）
        2. 每个骨架语音段 → 独立 _process_chunk_pipeline
        3. 调整时间戳到全局坐标
        4. 拼接所有事件

        Returns:
            (events: List[SubtitleEvent], total_segment_count: int)
        """
        import tempfile

        from .vad.ffmpeg_vad import unified_ffmpeg_pass

        cfg = self.config.acoustic_validation
        skeleton_noise_db = cfg.skeleton_noise_db
        skeleton_min_silence = cfg.skeleton_min_silence
        min_speech_duration = cfg.skeleton_min_speech

        # Step 1: 构建声学骨架
        self._progress.start_stage("skeleton", description="声学骨架提取")
        ffmpeg_result = unified_ffmpeg_pass(
            vocals_path,
            noise_db=skeleton_noise_db,
            min_silence_duration=skeleton_min_silence,
        )
        speech_skeleton = ffmpeg_result["skeleton"]  # [(start, end), ...]
        self._progress.finish_stage()

        total_duration = len(audio) / sample_rate
        logger.info(
            "Skeleton segmentation: %d speech segments from %.1fs audio "
            "(noise=%.0fdB, min_silence=%.2fs, min_speech=%.2fs)",
            len(speech_skeleton), total_duration,
            skeleton_noise_db, skeleton_min_silence, min_speech_duration,
        )

        if not speech_skeleton:
            logger.warning("No speech detected in skeleton, falling back to single chunk")
            events, seg_count, _fallback_ctx = self._process_chunk_pipeline(
                audio=audio, sample_rate=sample_rate,
                vocals_path=vocals_path, chunk_label="",
            )
            return events, seg_count

        # 过滤过短的段（< min_speech_duration 的孤立爆发可能是噪音）
        filtered_skeleton = [
            (s, e) for s, e in speech_skeleton
            if (e - s) >= min_speech_duration
        ]

        if len(filtered_skeleton) < len(speech_skeleton):
            logger.info(
                "Filtered %d short segments (< %.2fs)",
                len(speech_skeleton) - len(filtered_skeleton),
                min_speech_duration,
            )
        speech_skeleton = filtered_skeleton

        # Step 2: 逐段独立处理
        all_events: List[Any] = []
        total_seg_count = 0

        # ★ 跨段说话人偏移量（同多块路径）：每个骨架段独立运行 diarization，
        # 从 0 开始编号。为防止不同段的 "说话人0" 混淆，累加偏移量。
        speaker_offset = 0

        total_segments = len(speech_skeleton)
        for idx, (seg_start, seg_end) in enumerate(speech_skeleton):
            seg_duration = seg_end - seg_start
            start_sample = int(seg_start * sample_rate)
            end_sample = int(seg_end * sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                continue

            seg_audio = audio[start_sample:end_sample].copy()

            # 创建临时 WAV 文件供 ffmpeg 子进程调用
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False,
            ) as tmp_f:
                tmp_path = Path(tmp_f.name)

            try:
                AudioUtils.save_audio(seg_audio, tmp_path, sample_rate)

                chunk_label = f"Seg {idx+1}/{total_segments}"
                logger.info(
                    "Processing skeleton segment %d/%d: %.2fs → %.2fs (%.2fs)",
                    idx + 1, total_segments, seg_start, seg_end, seg_duration,
                )

                seg_events, seg_count, _seg_ctx = self._process_chunk_pipeline(
                    audio=seg_audio,
                    sample_rate=sample_rate,
                    vocals_path=tmp_path,
                    chunk_label=chunk_label,
                    parallel_vad=False,  # 骨架分段嵌套线程，避免 PyTorch 死锁
                )
            finally:
                tmp_path.unlink(missing_ok=True)

            # ★ 跨段 speaker_id 偏移（同多块路径）
            seg_speakers = set()
            for evt in seg_events:
                if evt.speaker_id is not None:
                    seg_speakers.add(evt.speaker_id)
            if seg_speakers:
                max_spk = max(seg_speakers)
                if speaker_offset > 0:
                    for evt in seg_events:
                        if evt.speaker_id is not None:
                            evt.speaker_id += speaker_offset
                speaker_offset += max_spk + 1

            # 偏移到全局时间轴
            for evt in seg_events:
                evt.start += seg_start
                evt.end += seg_start

            total_seg_count += seg_count
            all_events.extend(seg_events)

        # Step 3: 按 start 排序
        all_events.sort(key=lambda e: e.start)

        # 重新编号
        for i, evt in enumerate(all_events):
            evt.index = i + 1

        logger.info(
            "Skeleton segmented: %d segments → %d events (%d VAD sub-segments)",
            len(speech_skeleton), len(all_events), total_seg_count,
        )

        return all_events, total_seg_count, ffmpeg_result

    # ------------------------------------------------------------------
    # 后处理公共方法（三种路径共用）
    # ------------------------------------------------------------------

    def _run_llm_merge(
        self,
        events: List[SubtitleEvent],
        audio: Optional[np.ndarray],
        sample_rate: int,
        stats: PipelineStats,
    ) -> List[SubtitleEvent]:
        """LLM 语义合并（方案五）—— 改变事件边界，必须在声学校验之前执行"""
        try:
            from .merging.llm_merge_engine import (
                LLMMergeEngine,
                MergeDecisionConfig as LLMMergeDecisionConfig,
            )
            self._progress.start_stage(
                "llm_merge", description="LLM 语义合并", total_items=1,
            )
            llm_merge_config = LLMMergeDecisionConfig(
                fast_merge_max_gap=self.config.merge_decision.fast_merge_max_gap,
                llm_decision_min_gap=self.config.merge_decision.llm_decision_min_gap,
                llm_decision_max_gap=self.config.merge_decision.llm_decision_max_gap,
                hard_split_min_gap=self.config.merge_decision.hard_split_min_gap,
                max_combined_duration=self.config.merge_decision.max_combined_duration,
                min_fragment_duration=self.config.merge_decision.min_fragment_duration,
                llm_tier=self.config.merge_decision.llm_tier,
                llm_model=self.config.merge_decision.llm_model,
                llm_base_url=self.config.merge_decision.llm_base_url,
                llm_api_key=self.config.merge_decision.llm_api_key,
                llm_temperature=self.config.merge_decision.llm_temperature,
                llm_timeout=self.config.merge_decision.llm_timeout,
                llm_fallback_to_rules=self.config.merge_decision.llm_fallback_to_rules,
            )
            merge_engine = LLMMergeEngine(llm_merge_config)

            fragments = []
            for i, evt in enumerate(events):
                gap = None
                gap_is_silent = None
                if i < len(events) - 1:
                    gap = events[i + 1].start - evt.end
                    gap_is_silent = gap > 0.05
                fragments.append({
                    "id": i + 1,
                    "start": evt.start,
                    "end": evt.end,
                    "text": evt.text,
                    "speaker": evt.speaker_label or "unknown",
                    "gap_to_next_sec": round(gap, 3) if gap is not None else None,
                    "gap_is_silent": gap_is_silent,
                })

            merged_fragments = merge_engine.merge(
                fragments,
                audio=audio,
                sample_rate=sample_rate,
            )

            if merged_fragments and len(merged_fragments) < len(events):
                new_events = []
                for frag in merged_fragments:
                    frag_id = frag.get("id", 0)
                    orig_idx = frag_id - 1 if frag_id > 0 else 0
                    if orig_idx < len(events):
                        base = events[orig_idx]
                        new_events.append(SubtitleEvent(
                            index=len(new_events) + 1,
                            start=frag.get("start", base.start),
                            end=frag.get("end", base.end),
                            text=frag.get("text", base.text),
                            speaker_id=base.speaker_id,
                            speaker_label=base.speaker_label,
                        ))
                if new_events:
                    logger.info(
                        "LLM merge: %d → %d events",
                        len(events), len(new_events),
                    )
                    events = new_events
                    stats.subtitle_count = len(events)

            self._progress.update_stage(
                1, extra={"detail": f"LLM 合并: {len(events)} 条字幕"}
            )
            stats.stage_timings["llm_merge"] = self._progress.finish_stage()
        except Exception as e:
            logger.warning(
                "LLM merge failed, continuing with unmerged events: %s", e,
            )
        return events

    def _post_process_events(
        self,
        events: List[SubtitleEvent],
        vocals_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        stats: PipelineStats,
        ffmpeg_unified_result: Optional[Dict] = None,
    ) -> List[SubtitleEvent]:
        """后处理管线（三种路径共用）

        执行顺序经过精心设计：
        0. 事件级说话人聚类 — 全局事件集合声学聚类（替代段级 diarization）
        1. 帧级无缝衔接 — 消除相邻字幕的帧级间隙（方案六）
        2. LLM 语义合并 — 改变事件边界（方案五，必须在声学校验之前）
        3. 声学标尺校验 — 对最终边界做物理骨架吸附和诊断（方案七，最终关卡）

        Args:
            events: 字幕事件列表
            vocals_path: 人声音频路径
            audio: 音频数组
            sample_rate: 采样率
            stats: 管道统计（写入阶段耗时和诊断报告）
            ffmpeg_unified_result: 复用的 ffmpeg 骨架结果（避免重复调用）

        Returns:
            后处理完成的事件列表
        """
        # ---- 0. 事件级说话人聚类（全局集合） ----
        # 在单块/多块/骨架三种路径的事件拼接完成后统一聚类，
        # 确保骨架模式下每个小段产出的单个事件也能参与全局聚类。
        if self.config.diarization.enabled and len(events) > 1:
            try:
                events = self._run_event_speaker_clustering(
                    events, audio, sample_rate,
                )
                stats.speaker_count = len(set(
                    e.speaker_id for e in events if e.speaker_id is not None
                ))
            except Exception as e:
                logger.warning("Event-level speaker clustering failed: %s", e)

            # 事件级角色标注
            if self.config.speaker_role.enabled:
                try:
                    events = self._run_event_role_labeling(events)
                except Exception as e:
                    logger.warning("Event-level role labeling failed: %s", e)

        # ---- 1. 帧级无缝衔接（方案六） ----
        try:
            from .merging.llm_merge_engine import apply_frame_seamless_stitching
            stitch_gap = self.config.subtitle.max_stitch_gap
            events = apply_frame_seamless_stitching(events, max_stitch_gap=stitch_gap)
        except Exception as e:
            logger.warning("Frame seamless stitching failed: %s", e)

        # ---- 2. LLM 语义合并（方案五） ----
        # 必须在声学校验之前：合并改变事件边界
        if (
            self.config.merge_decision.llm_tier != "rule_only"
            and len(events) > 1
        ):
            events = self._run_llm_merge(events, audio, sample_rate, stats)

        # ---- 3. 声学标尺校验（方案七） ----
        # 对最终边界做物理骨架吸附和诊断（最终关卡）
        if self.config.acoustic_validation.enabled:
            try:
                from .acoustic_validator import AcousticValidator
                self._progress.start_stage(
                    "acoustic", description="声学校验", total_items=1,
                )
                validator = AcousticValidator(self.config.acoustic_validation)
                events, validation_report = validator.validate(
                    events,
                    audio_path=vocals_path,
                    audio=audio,
                    sample_rate=sample_rate,
                    ffmpeg_unified_result=ffmpeg_unified_result,
                )
                health = validation_report.get("health_score")
                if health is not None:
                    logger.info(
                        "Acoustic validation health: %.1f%%", health,
                    )
                    self._progress.update_stage(
                        1, extra={"detail": f"声学健康度: {health:.1f}%"}
                    )
                stats.diagnostic_report = validation_report
                stats.stage_timings["acoustic"] = self._progress.finish_stage()
            except Exception as e:
                logger.warning("Acoustic validation failed: %s", e)

        # ---- 最终去重：兜底检查所有后处理阶段可能引入的重复字幕 ----
        # 各处理阶段（帧级衔接、LLM 合并、声学校验）可能修改事件边界，
        # 重新引入重叠重复。此处做全量扫描确保输出无重复。
        try:
            from .mapping.time_mapper import TimeMapper
            events = TimeMapper._deduplicate_overlapping(events)
            stats.subtitle_count = len(events)
        except Exception as e:
            logger.warning("Final dedup in post_process failed: %s", e)

        return events

    # ------------------------------------------------------------------
    # 反馈学习通道 (Phase 5)
    # ------------------------------------------------------------------

    def _run_feedback_learning(
        self,
        auto_events: List[SubtitleEvent],
        reference_path: Path,
        audio_path: str,
    ) -> Optional[Dict]:
        """离线反馈学习：对齐 → 分析 → 更新配置 → 构建 Few-shot

        此过程不影响当前管道的输出，异常被静默捕获。

        Args:
            auto_events: 自动生成的字幕事件
            reference_path: 用户修订的字幕文件 (.srt/.ass)
            audio_path: 音频文件路径

        Returns:
            学习报告 dict，失败时返回 None
        """
        try:
            from vocal_subtitle.feedback import (
                AudioFingerprinter,
                DiffAnalyzer,
                FewShotBuilder,
                ParamLearner,
                SubtitleAligner,
                UserProfileManager,
            )
            from vocal_subtitle.feedback.aligner import parse_subtitle_file
            from vocal_subtitle.feedback.conflict_detector import ConflictDetector
            from vocal_subtitle.feedback.health_scorer import (
                compute_health_score_from_pairs,
                should_auto_rollback,
            )

            logger.info("Feedback learning started: reference=%s", reference_path)

            # Step 1: 解析用户修订字幕
            manual_events = parse_subtitle_file(reference_path)
            if not manual_events:
                logger.warning("Feedback: no events parsed from reference file")
                return None

            # Step 2: 对齐
            feedback_cfg = self.config.feedback
            aligner = SubtitleAligner(
                min_iou=feedback_cfg.alignment_min_iou,
                min_coverage=feedback_cfg.alignment_min_coverage,
                text_weight=feedback_cfg.alignment_text_weight,
                semantic_weight=feedback_cfg.alignment_semantic_weight,
                semantic_enabled=feedback_cfg.alignment_semantic_enabled,
            )
            pairs = aligner.align(auto_events, manual_events)

            # Step 2.5: 健康度评分（调整前）
            health_before, health_detail = compute_health_score_from_pairs(pairs)

            # Step 3: 差异分析
            analyzer = DiffAnalyzer(
                param_isolation_enabled=feedback_cfg.param_isolation_enabled,
            )
            diff_report = analyzer.analyze(pairs)

            # Step 3.5: 震荡检测
            profile_mgr = UserProfileManager(feedback_cfg)
            profile = profile_mgr.load(feedback_cfg.active_profile)
            history = profile.get("history", [])
            locked_params = profile.get("locked_params", [])

            detector = ConflictDetector(window=feedback_cfg.oscillation_detection_window)
            conflicts = detector.detect_all_oscillations(history)

            # 过滤掉已锁定的参数调整
            if locked_params:
                filtered_attr = {}
                for param_path, adj in diff_report.attribution.items():
                    if param_path in locked_params:
                        logger.info(
                            "Feedback: skipping locked param '%s'", param_path,
                        )
                        continue
                    filtered_attr[param_path] = adj
                diff_report.attribution = filtered_attr

            if conflicts:
                logger.warning(
                    "Feedback: %d parameter oscillations detected", len(conflicts),
                )
                for cr in conflicts:
                    logger.warning("  %s: %d flips (recommend: %s)",
                                   cr.param_path, cr.oscillation_count, cr.recommended_action)

            # Step 4: 参数学习
            current_overrides = profile.get("overrides", {})

            learner = ParamLearner(profile_mgr)
            updated_overrides = learner.learn_from_diff(
                diff_report=diff_report,
                current_config_overrides=current_overrides,
                profile_name=feedback_cfg.active_profile,
            )

            # Step 4.5: 自动回滚检查
            health_after = health_before  # 同一对齐对上的评分
            if feedback_cfg.auto_rollback_on_quality_drop and health_before > 0:
                should_rollback, rollback_reason = should_auto_rollback(
                    health_before, health_after,
                    drop_threshold=feedback_cfg.quality_drop_threshold,
                )
                if should_rollback:
                    logger.warning(
                        "Feedback: auto-rollback triggered — %s", rollback_reason,
                    )
                    try:
                        profile_mgr.rollback(feedback_cfg.active_profile)
                        logger.info("Feedback: rolled back profile '%s'", feedback_cfg.active_profile)
                    except Exception as rb_err:
                        logger.error("Feedback: rollback failed: %s", rb_err)

            # Step 5: Few-shot 构建
            if feedback_cfg.few_shot_enabled:
                few_shot = FewShotBuilder(max_examples=feedback_cfg.few_shot_max_examples)
                few_shot.load_cache(feedback_cfg.active_profile)
                few_shot.build_merge_examples(diff_report.merge_actions)
                if diff_report.text_edits:
                    few_shot.build_format_examples(diff_report.text_edits)
                few_shot.save_cache(feedback_cfg.active_profile)

            # Step 6: 音频指纹提取与存储
            if feedback_cfg.fingerprint_enabled and diff_report.attribution:
                try:
                    fingerprinter = AudioFingerprinter(
                        distance_method=feedback_cfg.fingerprint_distance_method,
                        knn_k=feedback_cfg.fingerprint_knn_k,
                        min_absolute_similarity=feedback_cfg.fingerprint_min_absolute_similarity,
                        relative_margin=feedback_cfg.fingerprint_relative_margin,
                    )
                    fp = fingerprinter.extract(Path(audio_path))
                    if fp is not None:
                        audio_hash = AudioFingerprinter.compute_audio_hash(Path(audio_path))
                        fingerprinter.store(
                            profile_id=feedback_cfg.active_profile,
                            fingerprint=fp,
                            audio_hash=audio_hash,
                            config_snapshot=updated_overrides,
                        )
                        fingerprinter.record_feedback(
                            profile_id=feedback_cfg.active_profile,
                            audio_hash=audio_hash,
                            alignment_coverage=diff_report.alignment_coverage,
                            diff_summary="; ".join(
                                adj.reason for adj in diff_report.attribution.values()
                            ),
                            adjustments={
                                k: [adj.observed_value, adj.confidence]
                                for k, adj in diff_report.attribution.items()
                            },
                            health_before=health_before,
                            health_after=health_after,
                            health_detail=health_detail,
                        )
                        logger.info("Feedback: audio fingerprint stored — %s", fp.audio_signature)
                except Exception as fp_err:
                    logger.warning("Feedback: fingerprint extraction failed (non-fatal): %s", fp_err)

            # 构建学习报告
            report = {
                "alignment_coverage": round(diff_report.alignment_coverage, 3),
                "total_pairs": diff_report.total_pairs,
                "time_shifts_count": len(diff_report.time_shifts),
                "merge_actions_count": len(diff_report.merge_actions),
                "text_edits_count": len(diff_report.text_edits),
                "health_score": round(health_before, 2),
                "health_score_detail": health_detail,
                "param_adjustments": {
                    path: {
                        "direction": adj.direction,
                        "confidence": round(adj.confidence, 3),
                        "reason": adj.reason,
                    }
                    for path, adj in diff_report.attribution.items()
                },
                "structural_revision": diff_report.structural_revision,
                "oscillations_detected": len(conflicts),
            }

            logger.info(
                "Feedback learning complete: coverage=%.1f%%, health=%.1f, adjustments=%d, "
                "shifts=%d, merges=%d, edits=%d, conflicts=%d",
                diff_report.alignment_coverage * 100,
                health_before,
                len(diff_report.attribution),
                len(diff_report.time_shifts),
                len(diff_report.merge_actions),
                len(diff_report.text_edits),
                len(conflicts),
            )

            return report

        except Exception as e:
            logger.warning("Feedback learning failed (non-fatal): %s", e)
            return None

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
