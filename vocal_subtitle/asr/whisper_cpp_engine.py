"""whisper.cpp 语音识别引擎

基于 whisper.cpp 的高性能 CPU 推理引擎。
代码协议: MIT | 模型权重协议: MIT

使用 pywhispercpp 或 subprocess 调用 whisper.cpp CLI。
需要在系统上安装 whisper.cpp 或通过 pip 安装 pywhispercpp。

要求: pip install pywhispercpp (推荐)

或通过 CLI 方式:
  whisper.cpp 可执行文件路径通过 WHISPER_CPP_BIN 环境变量指定。
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import ASREngine, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)


class WhisperCppEngine(ASREngine):
    """whisper.cpp 引擎

    适用于 CPU 推理（含 Apple Silicon GPU via Metal/CoreML）。
    模型文件需为 GGML/GGUF 格式。

    使用示例:
        engine = WhisperCppEngine(model="medium")
        engine.load_model()
        results = engine.transcribe(audio, language="zh")
    """

    def __init__(
        self,
        model: str = "medium",
        whisper_cpp_bin: Optional[str] = None,
        n_threads: int = 4,
        language: Optional[str] = None,
    ):
        """
        Args:
            model: 模型大小 (tiny / base / small / medium / large-v3)
            whisper_cpp_bin: whisper.cpp 可执行文件路径
            n_threads: CPU 线程数
            language: 默认语言
        """
        self._model_size = model
        self._bin = whisper_cpp_bin or os.getenv(
            "WHISPER_CPP_BIN", "whisper-cli"
        )
        self._n_threads = n_threads
        self._language = language
        self._model_path: Optional[Path] = None

    @property
    def name(self) -> str:
        return "whisper-cpp"

    @property
    def model_name(self) -> str:
        return self._model_size

    def load_model(self, model_path: Optional[str] = None) -> None:
        """配置模型路径

        Args:
            model_path: GGML/GGUF 模型文件路径，默认从标准位置查找
        """
        if model_path:
            self._model_path = Path(model_path)
        else:
            # 从默认位置查找
            default_dirs = [
                Path.home() / ".cache" / "whisper",
                Path.home() / ".cache" / "vocal-subtitle" / "whisper-cpp",
                Path("models"),
            ]

            model_filename = f"ggml-{self._model_size}.bin"
            for d in default_dirs:
                candidate = d / model_filename
                if candidate.exists():
                    self._model_path = candidate
                    break

            if self._model_path is None:
                logger.warning(
                    "Whisper model file not found. "
                    "Please download it first, e.g.: "
                    "bash scripts/download-ggml-model.sh %s",
                    self._model_size,
                )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs,
    ) -> List[TranscriptionSegment]:
        """使用 whisper.cpp CLI 进行识别

        将音频写入临时 WAV 文件，调用 whisper.cpp 可执行文件，
        解析输出结果。
        """
        lang = language or self._language
        model_path = str(self._model_path) if self._model_path else self._model_size

        # 写入临时 WAV 文件
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            from ..utils.audio_utils import AudioUtils

            AudioUtils.save_audio(audio, tmp_path, sample_rate)

        try:
            # 构建命令行参数
            cmd = [
                self._bin,
                "-m",
                model_path,
                "-f",
                str(tmp_path),
                "-t",
                str(self._n_threads),
                "-oj",  # JSON 输出
                "-of",
                str(tmp_path.with_suffix("")),  # 输出文件前缀
                "-ml",
                "1",  # 最大长度为1，有利于分段
            ]

            if lang:
                cmd.extend(["-l", lang])

            # 输出文件
            output_json = tmp_path.with_suffix(".json")

            logger.info("Running whisper.cpp: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error("whisper.cpp stderr: %s", result.stderr)
                raise RuntimeError(
                    f"whisper.cpp failed with code {result.returncode}"
                )

            # 解析 JSON 输出
            segments = self._parse_output(output_json)

        finally:
            # 清理临时文件
            tmp_path.unlink(missing_ok=True)
            output_json = tmp_path.with_suffix(".json")
            output_json.unlink(missing_ok=True)

        return segments

    def _parse_output(self, json_path: Path) -> List[TranscriptionSegment]:
        """解析 whisper.cpp JSON 输出"""
        import json

        if not json_path.exists():
            logger.warning("Output JSON not found: %s", json_path)
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        transcriptions = data.get("transcription", [])

        for seg in transcriptions:
            words = []
            if "tokens" in seg:
                # 解析词级时间戳（如果有）
                for token in seg.get("tokens", []):
                    words.append(
                        WordTimestamp(
                            word=token.get("text", ""),
                            start=token.get("t0", 0) / 100.0,  # 10ms → seconds
                            end=token.get("t1", 0) / 100.0,
                            confidence=token.get("p", 1.0),
                        )
                    )

            timestamps = seg.get("timestamps", {})
            start_str = timestamps.get("from", "00:00:00,000")
            end_str = timestamps.get("to", "00:00:00,000")

            results.append(
                TranscriptionSegment(
                    text=seg.get("text", "").strip(),
                    start=self._parse_timestamp(start_str),
                    end=self._parse_timestamp(end_str),
                    words=words,
                )
            )

        return results

    @staticmethod
    def _parse_timestamp(ts: str) -> float:
        """解析时间戳字符串 HH:MM:SS,mmm 为秒"""
        parts = ts.replace(",", ":").split(":")
        h, m, s, ms = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
        )
        return h * 3600 + m * 60 + s + ms / 1000.0
