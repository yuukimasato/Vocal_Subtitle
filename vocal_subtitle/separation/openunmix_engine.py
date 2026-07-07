"""Open-Unmix 人声分离引擎

使用 Open-Unmix (UMX) 进行高品质人声分离。
代码协议: MIT | 模型权重协议: MIT

要求: pip install openunmix
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import LicenseInfo, SeparationEngine, SeparationResult

logger = logging.getLogger(__name__)


class OpenUnmixEngine(SeparationEngine):
    """Open-Unmix 分离引擎

    基于 PyTorch 的开放人声分离模型。
    品质: ★★★★☆ | 速度: ★★★☆☆ | 适用: 品质优先场景

    使用示例:
        engine = OpenUnmixEngine()
        engine.load_model("umxhq")
        result = engine.separate(Path("input.wav"), Path("output/"))
    """

    DEFAULT_MODEL = "umxhq"

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None

    @property
    def name(self) -> str:
        return "openunmix"

    @property
    def license_info(self) -> LicenseInfo:
        return LicenseInfo(
            code_license="MIT",
            model_license="MIT",
            source_url="https://github.com/sigsep/open-unmix-pytorch",
        )

    def load_model(self, model_name: Optional[str] = None) -> None:
        """加载 Open-Unmix 模型

        Args:
            model_name: 模型名称 (umxhq / umx / umxl)，默认 "umxhq"
        """
        model_name = model_name or self.DEFAULT_MODEL

        if self._model is not None and self._model_name == model_name:
            return

        logger.info("Loading Open-Unmix model: %s", model_name)

        try:
            import openunmix

            self._model = openunmix
            self._model_name = model_name
        except ImportError:
            raise ImportError(
                "openunmix is required. Install it with: "
                "pip install openunmix"
            )

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        **kwargs,
    ) -> SeparationResult:
        """执行人声分离

        Args:
            input_path: 输入音频文件路径 (WAV 格式)
            output_dir: 输出目录

        Returns:
            SeparationResult
        """
        if self._model is None:
            self.load_model()

        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        logger.info("Separating: %s → %s", input_path, output_dir)

        # 预定义输出路径
        vocals_path = output_dir / "vocals.wav"
        accompaniment_path = output_dir / "accompaniment.wav"

        try:
            import soundfile as sf
            import torch

            # 加载音频
            audio, sr = sf.read(str(input_path))
            audio_tensor = torch.tensor(audio.T, dtype=torch.float32)

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # 估计并分离
            estimates = self._model.separate(
                audio_tensor,
                sample_rate=sr,
                targets=["vocals"],
                model_str_or_path=self._model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # 获取人声
            vocals = estimates["vocals"].squeeze().cpu().numpy()
            if vocals.ndim > 1:
                vocals = vocals.mean(axis=0)

            # 保存人声
            sf.write(str(vocals_path), vocals, sr)

            # 计算伴奏（原始信号 - 人声）
            accompaniment = audio - vocals
            sf.write(
                str(accompaniment_path),
                accompaniment,
                sr,
            )

        except ImportError as e:
            logger.error("Missing dependency: %s", e)
            raise
        except Exception as e:
            logger.error("Open-Unmix separation failed: %s", e)
            raise

        elapsed = time.time() - start_time

        return SeparationResult(
            vocals_path=vocals_path,
            accompaniment_path=accompaniment_path,
            engine_name=self.name,
            processing_time=elapsed,
        )
