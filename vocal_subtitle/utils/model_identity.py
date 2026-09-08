"""模型标识格式化工具

遵循 NAMING_CONVENTIONS.md 第 4 节约定：
  {engine}:{model_name}@{quantization}

示例:
  faster-whisper:large-v3@float16
  qwen-asr:Qwen3-ASR-1.7B@bf16
  funasr:paraformer-zh@fp32
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentity:
    """模型标识的解析结果"""

    engine: str
    model: str
    quantization: str | None = None

    def __str__(self) -> str:
        return format_model_id(self.engine, self.model, self.quantization)


def format_model_id(
    engine: str,
    model: str,
    quantization: str | None = None,
) -> str:
    """格式化为标准模型标识。

    Args:
        engine: 引擎名，如 faster-whisper, qwen-asr, funasr
        model: 模型名，如 large-v3, Qwen3-ASR-1.7B, paraformer-zh
        quantization: 量化方式，如 float16, int8, bf16, fp32（可选）

    Returns:
        标准模型标识字符串
    """
    base = f"{engine}:{model}"
    if quantization:
        return f"{base}@{quantization}"
    return base


def parse_model_id(model_id: str) -> ModelIdentity:
    """从标准模型标识解析出引擎、模型和量化方式。

    Args:
        model_id: 格式 {engine}:{model}[@{quantization}]

    Returns:
        ModelIdentity 对象

    Raises:
        ValueError: 格式不合法
    """
    if not model_id or ":" not in model_id:
        raise ValueError(f"Invalid model_id format: {model_id!r}")

    quant = None
    remaining = model_id

    if "@" in remaining:
        remaining, quant = remaining.rsplit("@", 1)
        if not quant:
            quant = None

    parts = remaining.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid model_id format: {model_id!r}")

    return ModelIdentity(engine=parts[0], model=parts[1], quantization=quant)
