"""场景切片器

基于 QUALITY_OPERATIONS.md §2 定义的 7 个切片维度对音频任务进行场景标签分配。
同时提供 §8 的维度平衡检查（任何单一维度占比不超过 60%）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# ------------------------------------------------------------------
# 维度值常量 (§2 场景切片表)
# ------------------------------------------------------------------

LANGUAGES = ("zh", "en", "mixed", "other")
SPEAKER_COUNTS = ("single", "dual", "multi")
BACKGROUND_NOISES = ("clean", "light_noise", "heavy_noise", "music")
SPEECH_RATES = ("slow", "normal", "fast")
AUDIO_LENGTHS = ("short", "medium", "long", "very_long")
DEVICES = ("cpu", "gpu_8gb", "gpu_12gb_plus", "mac_mps")
SCENE_TYPES = (
    "podcast",
    "education",
    "variety_show",
    "music_live",
    "meeting",
    "outdoor",
)

# 所有七维度的名称列表
DIMENSION_NAMES = (
    "language",
    "speaker_count",
    "background_noise",
    "speech_rate",
    "audio_length",
    "device",
    "scene_type",
)

# 语速阈值 (词/秒)
SPEED_WPS_SLOW = 3.0
SPEED_WPS_FAST = 5.0

# 音频长度阈值 (秒)
LENGTH_SHORT_MAX = 180  # < 3 min
LENGTH_MEDIUM_MAX = 1800  # 3-30 min
LENGTH_LONG_MAX = 7200  # 30-120 min (> 7200 = very_long)

# 维度平衡告警阈值 (§8)
MAX_SINGLE_DIMENSION_RATIO = 0.60


@dataclass
class SceneTag:
    """七维场景标签"""

    language: str = "unknown"
    speaker_count: str = "unknown"
    background_noise: str = "unknown"
    speech_rate: str = "unknown"
    audio_length: str = "unknown"
    device: str = "unknown"
    scene_type: str = "unknown"

    def tag_id(self) -> str:
        """生成稳定的场景标签 ID（非 unknown 维度的 SHA256 前 12 位）。"""
        parts = []
        for dim in DIMENSION_NAMES:
            val = getattr(self, dim, "unknown")
            if val != "unknown":
                parts.append(f"{dim}={val}")
        raw = "|".join(sorted(parts)) if parts else "unknown"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def as_key(self) -> str:
        """完整七维 key，用于分组。"""
        return "|".join(getattr(self, dim, "unknown") for dim in DIMENSION_NAMES)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "speaker_count": self.speaker_count,
            "background_noise": self.background_noise,
            "speech_rate": self.speech_rate,
            "audio_length": self.audio_length,
            "device": self.device,
            "scene_type": self.scene_type,
        }


class SceneSlicer:
    """场景切片器 — 对音频元数据进行七维标签分配和平衡检查。

    使用示例:
        slicer = SceneSlicer()
        tag = slicer.tag({"language": "zh", "speaker_count": 2,
                          "duration_seconds": 600, "device_name": "cuda"})
        # tag.speaker_count = "dual", tag.audio_length = "medium"
    """

    @staticmethod
    def tag(metadata: Mapping[str, Any]) -> SceneTag:
        """从音频元数据生成 SceneTag。

        Args:
            metadata: 包含 language, speaker_count, duration_seconds,
                      device_name, words_per_second, noise_level, scene_type 等字段

        Returns:
            SceneTag 实例
        """
        # 语言
        language = str(metadata.get("language", "")).lower()
        if language not in LANGUAGES:
            language = "unknown"

        # 说话人数
        spk = metadata.get("speaker_count", 0)
        try:
            spk = int(spk)
        except (TypeError, ValueError):
            spk = 0
        if spk <= 0:
            speaker_count = "unknown"
        elif spk == 1:
            speaker_count = "single"
        elif spk == 2:
            speaker_count = "dual"
        else:
            speaker_count = "multi"

        # 背景噪声
        noise = str(
            metadata.get("background_noise", metadata.get("noise_level", ""))
        ).lower()
        if noise in BACKGROUND_NOISES:
            background_noise = noise
        elif noise:
            # 尝试模糊匹配
            for candidate in BACKGROUND_NOISES:
                if candidate in noise:
                    background_noise = candidate
                    break
            else:
                background_noise = "unknown"
        else:
            background_noise = "unknown"

        # 语速
        wps = metadata.get("words_per_second")
        try:
            wps = float(wps) if wps is not None else 0.0
        except (TypeError, ValueError):
            wps = 0.0
        if wps <= 0:
            speech_rate = "unknown"
        elif wps < SPEED_WPS_SLOW:
            speech_rate = "slow"
        elif wps <= SPEED_WPS_FAST:
            speech_rate = "normal"
        else:
            speech_rate = "fast"

        # 音频长度
        duration = metadata.get("duration_seconds", 0)
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0:
            audio_length = "unknown"
        elif duration < LENGTH_SHORT_MAX:
            audio_length = "short"
        elif duration < LENGTH_MEDIUM_MAX:
            audio_length = "medium"
        elif duration < LENGTH_LONG_MAX:
            audio_length = "long"
        else:
            audio_length = "very_long"

        # 设备
        device_raw = str(
            metadata.get("device_name", metadata.get("device", ""))
        ).lower()
        if "gpu" in device_raw or "cuda" in device_raw:
            vram = metadata.get("vram_gb", 0)
            try:
                vram = float(vram)
            except (TypeError, ValueError):
                vram = 0
            if vram >= 12:
                device = "gpu_12gb_plus"
            elif vram >= 8:
                device = "gpu_8gb"
            else:
                device = "gpu_8gb"  # 默认归类
        elif "mps" in device_raw or "mac" in device_raw:
            device = "mac_mps"
        elif "cpu" in device_raw:
            device = "cpu"
        else:
            device = "unknown"

        # 场景类型
        scene = str(metadata.get("scene_type", metadata.get("scene", ""))).lower()
        if scene in SCENE_TYPES:
            scene_type = scene
        elif scene:
            for candidate in SCENE_TYPES:
                if candidate in scene:
                    scene_type = candidate
                    break
            else:
                scene_type = "unknown"
        else:
            scene_type = "unknown"

        return SceneTag(
            language=language,
            speaker_count=speaker_count,
            background_noise=background_noise,
            speech_rate=speech_rate,
            audio_length=audio_length,
            device=device,
            scene_type=scene_type,
        )

    @staticmethod
    def slice(
        samples: Iterable[Mapping[str, Any]],
    ) -> dict[str, list[dict]]:
        """按七维 key 对样本进行分组。

        Args:
            samples: 样本列表，每项需包含 metadata 字段（dict）

        Returns:
            {scene_key: [sample, ...]} 分组结果
        """
        groups: dict[str, list[dict]] = {}
        for sample in samples:
            meta = sample.get("metadata", sample)
            tag = SceneSlicer.tag(meta)
            key = tag.as_key()
            groups.setdefault(key, []).append(dict(sample))
        return groups

    @staticmethod
    def balance_report(
        samples: Iterable[Mapping[str, Any]],
    ) -> dict:
        """检查样本在各维度上的分布平衡性。

        Args:
            samples: 样本列表，每项需包含 metadata 字段

        Returns:
            {
                "total": N,
                "dimensions": {
                    "language": {"zh": {"count": 45, "ratio": 0.75}, ...},
                    ...
                },
                "warnings": [{"dimension": "language", "value": "zh", "ratio": 0.75, "message": "..."}],
            }
        """
        tags = []
        for sample in samples:
            meta = sample.get("metadata", sample)
            tags.append(SceneSlicer.tag(meta))

        total = len(tags)
        if total == 0:
            return {"total": 0, "dimensions": {}, "warnings": []}

        dimensions: dict[str, dict[str, dict]] = {}
        warnings: list[dict] = []

        for dim in DIMENSION_NAMES:
            counts: dict[str, int] = {}
            for tag in tags:
                val = getattr(tag, dim, "unknown")
                counts[val] = counts.get(val, 0) + 1

            dim_report: dict[str, dict] = {}
            for val, count in sorted(counts.items()):
                ratio = count / total
                dim_report[val] = {"count": count, "ratio": round(ratio, 4)}
                if ratio > MAX_SINGLE_DIMENSION_RATIO:
                    warnings.append(
                        {
                            "dimension": dim,
                            "value": val,
                            "ratio": round(ratio, 4),
                            "message": (
                                f"维度 {dim}={val} 占比 {ratio:.1%}，"
                                f"超过 {MAX_SINGLE_DIMENSION_RATIO:.0%} 上限"
                            ),
                        }
                    )

            dimensions[dim] = dim_report

        return {
            "total": total,
            "dimensions": dimensions,
            "warnings": warnings,
        }


__all__ = [
    "SceneTag",
    "SceneSlicer",
    "DIMENSION_NAMES",
    "LANGUAGES",
    "SPEAKER_COUNTS",
    "BACKGROUND_NOISES",
    "SPEECH_RATES",
    "AUDIO_LENGTHS",
    "DEVICES",
    "SCENE_TYPES",
    "MAX_SINGLE_DIMENSION_RATIO",
]
