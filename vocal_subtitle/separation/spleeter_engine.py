"""Spleeter 人声分离引擎

使用 Deezer Spleeter 进行人声与伴奏分离。
Spleeter 代码协议: MIT | 模型权重协议: MIT

要求: pip install spleeter
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .base import LicenseInfo, SeparationEngine, SeparationResult

logger = logging.getLogger(__name__)


class SpleeterEngine(SeparationEngine):
    """Spleeter 分离引擎

    基于 TensorFlow 的预训练人声分离模型。
    品质: ★★★☆☆ | 速度: ★★★★★ | 适用: 默认/快速验证

    使用示例:
        engine = SpleeterEngine()
        engine.load_model("2stems")
        result = engine.separate(Path("input.mp3"), Path("output/"))
    """

    DEFAULT_MODEL = "2stems"

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None

    @property
    def name(self) -> str:
        return "spleeter"

    @property
    def license_info(self) -> LicenseInfo:
        return LicenseInfo(
            code_license="MIT",
            model_license="MIT",
            source_url="https://github.com/deezer/spleeter",
        )

    def load_model(self, model_name: Optional[str] = None) -> None:
        """加载 Spleeter 模型

        Args:
            model_name: 模型配置名 (2stems / 4stems / 5stems)，
                       默认 "2stems"
        """
        model_name = model_name or self.DEFAULT_MODEL

        if self._model is not None and self._model_name == model_name:
            return

        logger.info("Loading Spleeter model: %s", model_name)

        try:
            from spleeter.separator import Separator

            self._model = Separator(f"spleeter:{model_name}")
            self._model_name = model_name
        except ImportError:
            raise ImportError(
                "spleeter is required. Install it with: pip install spleeter"
            )
        except Exception as e:
            logger.error("Failed to load Spleeter model: %s", e)
            raise

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        **kwargs,
    ) -> SeparationResult:
        """执行人声分离

        Args:
            input_path: 输入音频文件路径
            output_dir: 输出目录

        Returns:
            SeparationResult
        """
        if self._model is None:
            self.load_model()

        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        logger.info("Separating: %s → %s", input_path, output_dir)

        try:
            self._model.separate_to_file(
                str(input_path),
                str(output_dir),
            )
        except Exception as e:
            logger.error("Spleeter separation failed: %s", e)
            raise

        elapsed = time.time() - start_time

        # Spleeter 输出命名规则：input_name/vocals.wav, input_name/accompaniment.wav
        stem = input_path.stem
        vocals_path = output_dir / stem / "vocals.wav"
        accompaniment_path = output_dir / stem / "accompaniment.wav"

        return SeparationResult(
            vocals_path=vocals_path,
            accompaniment_path=accompaniment_path,
            engine_name=self.name,
            processing_time=elapsed,
        )
