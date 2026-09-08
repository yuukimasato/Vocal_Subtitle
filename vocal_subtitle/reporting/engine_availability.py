"""引擎可用性检查器

每次运行前预检所有引擎状态，生成 engine_availability.json 快照。
对应 ENGINE_LIFECYCLE.md 和 RUN_REPORT_SCHEMA.md。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .run_report_schema import EngineStatusEntry

logger = logging.getLogger(__name__)


@dataclass
class EngineAvailabilitySnapshot:
    """引擎可用性快照"""
    entries: dict[str, EngineStatusEntry] = field(default_factory=dict)
    host_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "engines": {k: v.to_dict() for k, v in self.entries.items()},
            "host": self.host_info,
        }


class EngineAvailabilityChecker:
    """预检所有引擎状态，生成可用性快照。

    状态优先级:
      unavailable → model_missing → ready_shadow → ready_review → ready_default
    """

    # 引擎检查注册表
    _CHECKS: list[dict] = [
        # ---- 人声分离 ----
        {"key": "separation_uvr", "label": "UVR (BS-RoFormer)", "engine": "uvr",
         "model": "bs_roformer", "category": "separation"},
        {"key": "separation_spleeter", "label": "Spleeter", "engine": "spleeter",
         "model": "2stems", "category": "separation"},
        {"key": "separation_open_unmix", "label": "Open-Unmix", "engine": "open-unmix",
         "model": "umxhq", "category": "separation"},

        # ---- VAD ----
        {"key": "vad_silero", "label": "Silero VAD", "engine": "silero",
         "model": "silero_vad", "category": "vad"},
        {"key": "vad_webrtc", "label": "WebRTC VAD", "engine": "webrtc",
         "model": "", "category": "vad"},

        # ---- ASR 主引擎 ----
        {"key": "asr_faster_whisper", "label": "faster-whisper", "engine": "faster-whisper",
         "model": "large-v3", "category": "asr"},
        {"key": "asr_funasr", "label": "FunASR", "engine": "funasr",
         "model": "paraformer-zh", "category": "asr"},
        {"key": "asr_qwen", "label": "Qwen3-ASR", "engine": "qwen-asr",
         "model": "Qwen3-ASR-1.7B", "category": "asr"},
        {"key": "asr_whisper_cpp", "label": "whisper.cpp", "engine": "whisper.cpp",
         "model": "ggml-medium", "category": "asr"},

        # ---- 复核引擎 ----
        {"key": "review_global_evidence", "label": "Global ASR Evidence", "engine": "faster-whisper",
         "model": "large-v3", "category": "review"},
        {"key": "review_context_reasr", "label": "Context Re-ASR", "engine": "faster-whisper",
         "model": "large-v3", "category": "review"},
        {"key": "review_qwen", "label": "Qwen3-ASR Review", "engine": "qwen-asr",
         "model": "Qwen3-ASR-1.7B", "category": "review"},
        {"key": "review_forced_aligner", "label": "ForcedAligner", "engine": "qwen-forced-aligner",
         "model": "Qwen3-ForcedAligner-0.6B", "category": "review"},
        {"key": "review_sed", "label": "SED (AST-AudioSet)", "engine": "ast-audioset",
         "model": "MIT/ast-finetuned-audioset", "category": "review"},
        {"key": "review_semantic", "label": "Semantic Review", "engine": "semantic",
         "model": "", "category": "review"},

        # ---- 说话人分离 ----
        {"key": "diarization_speechbrain", "label": "ECAPA (SpeechBrain)", "engine": "agglomerative",
         "model": "speechbrain/ecapa", "category": "diarization"},
        {"key": "diarization_pyannote", "label": "pyannote 全局聚类", "engine": "pyannote",
         "model": "speaker-diarization-3.1", "category": "diarization"},
    ]

    def __init__(self, config=None):
        self._config = config
        self._cache_root = Path.home() / ".cache" / "vocal-subtitle"

    def check_all(self) -> EngineAvailabilitySnapshot:
        """预检所有引擎，返回完整快照。"""
        host = self._collect_host_info()
        entries: dict[str, EngineStatusEntry] = {}

        for check in self._CHECKS:
            key = check["key"]
            status, reason, device, model_path = self._check_engine(check)
            entries[key] = EngineStatusEntry(
                engine=check["engine"],
                model=check.get("model", ""),
                status=status,
                device=device,
                model_path=model_path,
                reason=reason,
                lifecycle=check["category"],
                available=status.startswith("ready"),
            )

        logger.info(
            "Engine availability check complete: %d engines, %d ready",
            len(entries),
            sum(1 for e in entries.values() if e.status.startswith("ready")),
        )
        return EngineAvailabilitySnapshot(entries=entries, host_info=host)

    def check_critical(self) -> dict[str, bool]:
        """检查关键引擎（VAD + 至少一个 ASR），返回通过状态。"""
        snapshot = self.check_all()
        vad_ready = any(
            snapshot.entries[k].status.startswith("ready")
            for k in ("vad_silero", "vad_webrtc")
        )
        asr_ready = any(
            snapshot.entries[k].status.startswith("ready")
            for k in ("asr_faster_whisper", "asr_funasr", "asr_qwen", "asr_whisper_cpp")
        )
        return {
            "vad_available": vad_ready,
            "asr_available": asr_ready,
            "all_critical_ready": vad_ready and asr_ready,
        }

    # ---- 内部方法 ----

    def _check_engine(self, check: dict) -> tuple[str, str, str, str]:
        """检查单个引擎状态。"""
        category = check["category"]
        engine = check["engine"]
        key = check["key"]

        if category == "separation":
            return self._check_separation(engine)
        elif category == "vad":
            return self._check_vad(engine)
        elif category == "asr":
            return self._check_asr(engine, key)
        elif category == "review":
            return self._check_review(engine, key)
        elif category == "diarization":
            return self._check_diarization(engine)
        return "unavailable", "unknown category", "", ""

    def _check_separation(self, engine: str) -> tuple[str, str, str, str]:
        if engine == "uvr":
            try:
                # Import the configured BS-RoFormer architecture as well as
                # the package.  Importing only ``audio_separator`` is a
                # shallow check: the package root can be present while the
                # runtime dependencies used by MDXC are missing.
                from audio_separator.separator import Separator  # noqa: F401
                from audio_separator.separator.architectures import (  # noqa: F401
                    mdxc_separator,
                )
                return "ready_default", "", "cpu", ""
            except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
                missing = getattr(exc, "name", None) or str(exc)
                return (
                    "unavailable",
                    f"audio-separator UVR runtime unavailable ({missing})",
                    "",
                    "",
                )
        elif engine == "spleeter":
            return "unavailable", "Python < 3.12 requirement", "", ""
        elif engine == "open-unmix":
            try:
                import openunmix  # noqa: F401
                return "ready_shadow", "", "cpu", ""
            except (ImportError, ModuleNotFoundError) as exc:
                missing = getattr(exc, "name", None) or "openunmix"
                return "unavailable", f"{missing} not installed", "", ""
        return "unavailable", "unknown separation engine", "", ""

    def _check_vad(self, engine: str) -> tuple[str, str, str, str]:
        if engine == "silero":
            try:
                import torch  # noqa: F401
                import torchaudio  # noqa: F401
            except (ImportError, OSError) as exc:
                missing = getattr(exc, "name", None) or "torch/torchaudio"
                return (
                    "unavailable",
                    f"torch and torchaudio are required for Silero VAD ({missing})",
                    "",
                    "",
                )
            model_path = self._cache_root / "silero_vad.onnx"
            if model_path.exists():
                return "ready_default", "", "cpu", str(model_path)
            # Silero VAD 模型小，会运行时自动下载
            return "ready_default", "", "cpu", ""
        elif engine == "webrtc":
            try:
                import webrtcvad  # noqa: F401
                return "ready_shadow", "", "cpu", ""
            except ImportError:
                return "unavailable", "webrtcvad not installed", "", ""
        return "unavailable", "unknown VAD engine", "", ""

    def _check_asr(self, engine: str, key: str) -> tuple[str, str, str, str]:
        if key == "asr_faster_whisper":
            try:
                import faster_whisper  # noqa: F401
                import ctranslate2  # noqa: F401
                device = self._detect_device()
                return "ready_default", "", device, ""
            except ImportError:
                return "unavailable", "faster-whisper not installed", "", ""
        elif key == "asr_funasr":
            try:
                import funasr  # noqa: F401
                return "ready_shadow", "", "cpu", ""
            except ImportError:
                return "unavailable", "funasr not installed", "", ""
        elif key == "asr_qwen":
            try:
                import qwen_asr  # noqa: F401
                model_dir = self._cache_root / "review-models" / "qwen3-asr-1.7b"
                if model_dir.exists():
                    return "ready_shadow", "", self._detect_device(), str(model_dir)
                return "model_missing", "Qwen3-ASR model not downloaded", self._detect_device(), ""
            except ImportError:
                return "unavailable", "qwen-asr not installed", "", ""
        elif key == "asr_whisper_cpp":
            whisper_dir = self._cache_root / "whisper_cpp"
            ggml = list(whisper_dir.glob("ggml-*.bin")) if whisper_dir.exists() else []
            if ggml:
                return "ready_shadow", "", "cpu", str(ggml[0])
            return "model_missing", "whisper.cpp model not downloaded", "cpu", ""
        return "unavailable", "unknown ASR engine", "", ""

    def _check_review(self, engine: str, key: str) -> tuple[str, str, str, str]:
        if key == "review_global_evidence":
            return "ready_default", "", self._detect_device(), ""
        elif key == "review_context_reasr":
            return "unavailable", "实验阶段，需显式启用", "", ""
        elif key == "review_qwen":
            try:
                import qwen_asr  # noqa: F401
                model_dir = self._cache_root / "review-models" / "qwen3-asr-1.7b"
                if model_dir.exists():
                    return "ready_shadow", "", self._detect_device(), str(model_dir)
                return "model_missing", "model not downloaded", "", ""
            except ImportError:
                return "unavailable", "qwen-asr not installed", "", ""
        elif key == "review_forced_aligner":
            try:
                import qwen_asr  # noqa: F401
                model_dir = self._cache_root / "review-models" / "qwen3-forced-aligner-0.6b"
                if model_dir.exists():
                    return "ready_shadow", "", self._detect_device(), str(model_dir)
                return "model_missing", "model not downloaded", "", ""
            except ImportError:
                return "unavailable", "qwen-asr not installed", "", ""
        elif key == "review_sed":
            try:
                import transformers  # noqa: F401
                model_path = self._cache_root / "review-models" / "ast-audioset"
                if model_path.exists():
                    return "ready_shadow", "", self._detect_device(), str(model_path)
                return "model_missing", "model not downloaded", "", ""
            except ImportError:
                return "unavailable", "transformers not installed", "", ""
        elif key == "review_semantic":
            return "unavailable", "实验阶段", "", ""
        return "unavailable", "unknown review engine", "", ""

    def _check_diarization(self, engine: str) -> tuple[str, str, str, str]:
        if engine == "agglomerative":
            try:
                import speechbrain  # noqa: F401
                return "ready_default", "", "cpu", ""
            except ImportError:
                return "unavailable", "speechbrain not installed", "", ""
        elif engine == "pyannote":
            try:
                import pyannote.audio  # noqa: F401
                return "ready_shadow", "", self._detect_device(), ""
            except ImportError:
                return "unavailable", "pyannote not installed", "", ""
        return "unavailable", "unknown diarization engine", "", ""

    @staticmethod
    def _detect_device() -> str:
        """检测可用设备。"""
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _collect_host_info() -> dict:
        """收集主机信息。"""
        import platform
        info = {
            "platform": platform.system(),
            "python": platform.python_version(),
        }
        try:
            info["cpu_count"] = os.cpu_count() or 0
        except Exception:
            info["cpu_count"] = 0
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1)
        except Exception:
            pass
        return info
