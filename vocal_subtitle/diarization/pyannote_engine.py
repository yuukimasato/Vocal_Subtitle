"""pyannote community-1 全局说话人分离后端。

该模块只负责从完整音频得到全局 speaker turns，不负责字幕合并或物理
边界修正。pyannote.audio 为可选依赖，导入保持延迟，以便基础安装和
embedding fallback 仍可工作。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import DiarizationResult, SpeakerTurn
from .turn_reconciler import normalize_turns

logger = logging.getLogger(__name__)


class PyannoteDiarizationEngine:
    """使用官方 community-1 pipeline 进行一次全局 diarization。"""

    def __init__(
        self,
        model_ref: str = "pyannote/speaker-diarization-community-1",
        token: Optional[str] = None,
        device: str = "auto",
        offline: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_ref = model_ref
        self.token = token if token and token != "***" else None
        self.token = self.token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if not self.token:
            try:
                from ..utils.hf_token_store import load_hf_token

                self.token = load_hf_token()
            except (ImportError, OSError, ValueError):
                self.token = None
        self.device = device
        self.offline = offline
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._pipeline = None

    @property
    def name(self) -> str:
        return "pyannote-community-1" if "community" in self.model_ref else "pyannote-3.1"

    # 预置全局说话人分离模型
    PRESET_MODELS = {
        "pyannote/speaker-diarization-community-1": {
            "name": "pyannote/speaker-diarization-community-1",
            "description": "Community-1 全局说话人分离 pipeline (社区版，默认)",
            "requires_token": True,
            "license_url": "https://huggingface.co/pyannote/speaker-diarization-community-1",
        },
        "pyannote/speaker-diarization-3.1": {
            "name": "pyannote/speaker-diarization-3.1",
            "description": "Speaker Diarization 3.1 pipeline (最新版，需签署协议)",
            "requires_token": True,
            "license_url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
        },
    }

    @staticmethod
    def _restore_env(
        previous_offline: Optional[str],
        previous_hf_home: Optional[str],
        previous_telemetry: Optional[str],
        previous_timeout: Optional[str],
    ) -> None:
        """Restore environment variables to their previous values."""
        for key, prev in (
            ("HF_HUB_OFFLINE", previous_offline),
            ("HF_HOME", previous_hf_home),
            ("HF_HUB_DISABLE_TELEMETRY", previous_telemetry),
            ("HF_HUB_DOWNLOAD_TIMEOUT", previous_timeout),
        ):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev

    def license_info(self) -> dict:
        return {
            "engine": "pyannote.audio",
            "code_license": "MIT",
            "model": self.model_ref,
            "model_access": "Hugging Face user conditions and token may be required",
        }

    def _local_model_config(self) -> Optional[Path]:
        """Locate a complete local Hugging Face snapshot before using the network."""
        model_path = Path(self.model_ref).expanduser()
        if model_path.is_file():
            return model_path

        if "/" not in self.model_ref:
            return None
        hf_home = self.cache_dir or Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        model_dir = Path(hf_home) / "hub" / (
            "models--" + self.model_ref.replace("/", "--")
        )
        refs_main = model_dir / "refs" / "main"
        revisions = []
        if refs_main.is_file():
            revision = refs_main.read_text(encoding="utf-8").strip()
            if revision:
                revisions.append(revision)
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.is_dir():
            revisions.extend(
                item.name
                for item in snapshots_dir.iterdir()
                if item.is_dir() and item.name not in revisions
            )

        for revision in revisions:
            config_path = snapshots_dir / revision / "config.yaml"
            if config_path.is_file():
                return config_path
        return None

    def load_model(self) -> None:
        """延迟加载 pipeline，并在可用时移动到 GPU。

        优先从本地缓存加载，仅在模型未缓存且未显式设置 offline 时
        才临时关闭离线模式以允许从 Hugging Face 下载。
        """
        if self._pipeline is not None:
            return

        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_hf_home = os.environ.get("HF_HOME")
        previous_telemetry = os.environ.get("HF_HUB_DISABLE_TELEMETRY")
        previous_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")

        # pyannote uses Hugging Face Hub for model resolution. Keep optional
        # usage telemetry disabled for this local processing application.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(self.cache_dir)

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            self._restore_env(
                previous_offline, previous_hf_home,
                previous_telemetry, previous_timeout,
            )
            raise ImportError(
                "pyannote.audio is required for the pyannote diarization backend. "
                "Install the optional pyannote diarization dependency."
            ) from exc

        # ── 第一步：优先尝试用本地离线模式加载 ──
        # 先保持离线模式开启，尝试本地快照。
        if not self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        local_config = self._local_model_config()
        _need_network = (local_config is None and not self.offline)

        # ── 刷新 huggingface_hub 模块级离线常量 ──
        # vocal_subtitle.__init__ 在导入时设置 HF_HUB_OFFLINE=1，
        # huggingface_hub 将其缓存为模块常量，仅修改 os.environ 无效。
        _saved_hf_offline = None
        _saved_tf_offline = None
        if _need_network:
            # 仅在确实需要网络时才关闭离线模式
            try:
                import huggingface_hub.constants as _hf_constants
                _saved_hf_offline = _hf_constants.HF_HUB_OFFLINE
                _hf_constants.HF_HUB_OFFLINE = False
            except ImportError:
                pass
            try:
                import transformers.utils.hub as _tf_hub
                _saved_tf_offline = _tf_hub._is_offline_mode
                _tf_hub._is_offline_mode = False
            except (ImportError, ValueError):
                pass
            os.environ["HF_HUB_OFFLINE"] = "0"

        # 给予充足的下载时间（全局 diarization 模型较大）
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

        kwargs = {}
        if self.token:
            import inspect

            parameters = inspect.signature(Pipeline.from_pretrained).parameters
            if "token" in parameters:
                kwargs["token"] = self.token
            elif "use_auth_token" in parameters:
                kwargs["use_auth_token"] = self.token

        try:
            if local_config is not None:
                try:
                    logger.info(
                        "Loading global diarization model from local snapshot: %s",
                        local_config,
                    )
                    self._pipeline = Pipeline.from_pretrained(
                        str(local_config), **kwargs
                    )
                except Exception as exc:
                    self._pipeline = None
                    if self.offline or not _need_network:
                        raise RuntimeError(
                            f"Local diarization model is incomplete: {local_config}"
                        ) from exc
                    logger.warning(
                        "Local diarization snapshot failed to load (%s); "
                        "falling back to Hugging Face: %s",
                        local_config,
                        type(exc).__name__,
                    )

            if self._pipeline is None:
                if self.offline:
                    raise RuntimeError(
                        f"Local diarization model not found: {self.model_ref}"
                    )
                logger.info(
                    "Loading global diarization model from Hugging Face: %s",
                    self.model_ref,
                )
                self._pipeline = Pipeline.from_pretrained(self.model_ref, **kwargs)
        finally:
            self._restore_env(
                previous_offline, previous_hf_home,
                previous_telemetry, previous_timeout,
            )
            # 恢复离线模式常量
            if _saved_hf_offline is not None:
                try:
                    import huggingface_hub.constants as _hf_c
                    _hf_c.HF_HUB_OFFLINE = _saved_hf_offline
                except ImportError:
                    pass
            if _saved_tf_offline is not None:
                try:
                    import transformers.utils.hub as _tf_h
                    _tf_h._is_offline_mode = _saved_tf_offline
                except ImportError:
                    pass

        if self._pipeline is None:
            raise RuntimeError(f"Failed to load diarization model: {self.model_ref}")

        if self.device != "cpu":
            try:
                # 解析 "auto" → 实际可用设备
                _raw = self.device
                if _raw == "auto":
                    _raw = "cuda" if torch.cuda.is_available() else "cpu"
                _resolved = torch.device(_raw)
                if _resolved.type == "cuda":
                    self._pipeline.to(_resolved)
            except Exception as exc:
                if self.device == "cuda":
                    raise
                logger.warning("Could not move pyannote pipeline to GPU: %s", exc)

        logger.info(
            "Loaded global diarization backend: %s (model=%s)",
            self.name,
            self.model_ref,
        )

    @staticmethod
    def _get_annotation(output: Any, attribute: str, fallback: Any) -> Any:
        annotation = getattr(output, attribute, None)
        return annotation if annotation is not None else fallback

    @staticmethod
    def _annotation_to_turns(annotation: Any) -> list[tuple[float, float, str]]:
        if annotation is None:
            return []
        if hasattr(annotation, "itertracks"):
            return [
                (float(segment.start), float(segment.end), str(label))
                for segment, _, label in annotation.itertracks(yield_label=True)
                if segment.end > segment.start
            ]
        raise TypeError(
            "Unsupported pyannote diarization output; expected an Annotation"
        )

    @staticmethod
    def _overlap_duration(turns: list[tuple[float, float, str]]) -> float:
        boundaries = sorted({
            point
            for start, end, _ in turns
            for point in (start, end)
        })
        total = 0.0
        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            active_speakers = {
                label
                for turn_start, turn_end, label in turns
                if turn_start < end and turn_end > start
            }
            if len(active_speakers) > 1:
                total += end - start
        return total

    def diarize(
        self,
        audio_path: Optional[Path] = None,
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        *,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """对完整音频运行 diarization，并返回稳定的全局 speaker id。"""
        if audio_path is None and audio is None:
            raise ValueError("audio_path or audio must be provided")
        self.load_model()

        source: Any = audio_path
        if audio is not None:
            import torch

            array = np.asarray(audio, dtype=np.float32)
            if array.ndim > 1:
                array = array.mean(axis=1)
            source = {
                "waveform": torch.from_numpy(array).unsqueeze(0),
                "sample_rate": sample_rate,
            }

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kwargs["max_speakers"] = int(max_speakers)
        output = self._pipeline(source, **kwargs)

        regular_annotation = self._get_annotation(
            output, "speaker_diarization", output
        )
        exclusive_annotation = self._get_annotation(
            output, "exclusive_speaker_diarization", regular_annotation
        )
        regular_raw = self._annotation_to_turns(regular_annotation)
        exclusive_raw = self._annotation_to_turns(exclusive_annotation)

        # Use regular output order as the canonical identity order. This keeps
        # labels stable across all later chunk/skeleton processing.
        labels: list[str] = []
        for _, _, label in sorted(regular_raw, key=lambda item: (item[0], item[1])):
            if label not in labels:
                labels.append(label)
        for _, _, label in sorted(exclusive_raw, key=lambda item: (item[0], item[1])):
            if label not in labels:
                labels.append(label)
        label_to_id = {label: index for index, label in enumerate(labels)}

        overlap_duration = self._overlap_duration(regular_raw)
        regular_turns = [
            SpeakerTurn(
                start=start,
                end=end,
                speaker_id=label_to_id[label],
                overlapped=any(
                    other_label != label
                    and min(end, other_end) > max(start, other_start)
                    for other_start, other_end, other_label in regular_raw
                ),
            )
            for start, end, label in regular_raw
        ]
        exclusive_turns = [
            SpeakerTurn(
                start=start,
                end=end,
                speaker_id=label_to_id[label],
            )
            for start, end, label in exclusive_raw
        ]

        duration = None
        if audio is not None:
            duration = len(audio) / max(sample_rate, 1)
        regular_turns = normalize_turns(regular_turns, duration=duration)
        exclusive_turns = normalize_turns(exclusive_turns, duration=duration)
        return DiarizationResult(
            turns=regular_turns,
            exclusive_turns=exclusive_turns,
            speaker_count=len(labels),
            backend=self.name,
            status="ok",
            overlap_duration=overlap_duration,
            diagnostics={
                "model_ref": self.model_ref,
                "regular_turn_count": len(regular_turns),
                "exclusive_turn_count": len(exclusive_turns),
            },
        )
