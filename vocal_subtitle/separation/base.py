"""人声分离引擎抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LicenseInfo:
    """协议信息"""

    code_license: str
    model_license: str
    source_url: str


@dataclass
class SeparationResult:
    """分离结果"""

    vocals_path: Path
    accompaniment_path: Path
    engine_name: str = ""
    processing_time: float = 0.0


class SeparationEngine(ABC):
    """人声分离引擎抽象基类

    所有分离引擎必须实现此接口。

    使用示例:
        engine = SpleeterEngine()
        engine.load_model("2stems")
        result = engine.separate(Path("input.mp3"), Path("output/"))
    """

    @abstractmethod
    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        **kwargs,
    ) -> SeparationResult:
        """分离人声

        Args:
            input_path: 输入音频文件路径
            output_dir: 输出目录

        Returns:
            SeparationResult: 包含人声路径和伴奏路径
        """
        ...

    @abstractmethod
    def load_model(self, model_name: Optional[str] = None) -> None:
        """加载模型（延迟加载）

        Args:
            model_name: 模型名称，默认使用引擎内置默认值
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        ...

    @property
    @abstractmethod
    def license_info(self) -> LicenseInfo:
        """返回代码和模型权重的协议信息"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
