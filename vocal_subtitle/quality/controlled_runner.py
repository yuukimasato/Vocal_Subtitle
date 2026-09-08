"""对照运行管理器

基于 QUALITY_OPERATIONS.md §5 的对照运行规范，
实现 baseline → candidate → shadow 对照运行和结果比较。
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RunKind(str, Enum):
    """运行类型"""
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    SHADOW = "shadow"


@dataclass
class ControlledRunSpec:
    """单次对照运行规格"""

    run_id: str
    kind: RunKind
    audio_path: str
    output_dir: str
    config_path: str = ""
    overrides: dict = field(default_factory=dict)
    ground_truth: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "kind": self.kind.value,
            "audio_path": self.audio_path,
            "output_dir": self.output_dir,
            "config_path": self.config_path,
            "overrides": self.overrides,
            "ground_truth": self.ground_truth,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ControlledRunSpec:
        return cls(
            run_id=data.get("run_id", ""),
            kind=RunKind(data.get("kind", "baseline")),
            audio_path=data.get("audio_path", ""),
            output_dir=data.get("output_dir", ""),
            config_path=data.get("config_path", ""),
            overrides=data.get("overrides", {}),
            ground_truth=data.get("ground_truth", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class ComparisonRecommendation:
    """对照运行建议"""

    recommend_enable: bool = False
    benefited_scenes: list[dict] = field(default_factory=list)
    degraded_scenes: list[dict] = field(default_factory=list)
    uncertain_areas: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "recommend_enable": self.recommend_enable,
            "benefited_scenes": self.benefited_scenes,
            "degraded_scenes": self.degraded_scenes,
            "uncertain_areas": self.uncertain_areas,
            "rationale": self.rationale,
        }


class ControlledRunManager:
    """对照运行管理器。

    管理 baseline/candidate/shadow 三种运行的创建、执行和对比。

    使用示例:
        mgr = ControlledRunManager()
        baseline = mgr.create_run(RunKind.BASELINE, audio_path, config_path)
        mgr.execute(baseline)

        candidate = mgr.create_run(RunKind.CANDIDATE, audio_path, config_path,
                                   overrides={"asr.model": "large-v3-turbo"})
        mgr.execute(candidate)

        comparison = mgr.compare(baseline, candidate)
        rec = mgr.recommend(comparison)
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._storage_dir = Path(storage_dir or (
            Path(__file__).parent.parent.parent / "cache" / "controlled_runs"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        kind: RunKind,
        audio_path: Path,
        config_path: Path,
        overrides: Optional[dict] = None,
        *,
        output_dir: Optional[Path] = None,
        ground_truth: Optional[Path] = None,
    ) -> ControlledRunSpec:
        """创建一个对照运行规格。

        Args:
            kind: baseline / candidate / shadow
            audio_path: 输入音频路径
            config_path: 配置文件路径
            overrides: 配置覆盖项
            output_dir: 输出目录（默认自动生成）
            ground_truth: 参考字幕路径（可选）

        Returns:
            ControlledRunSpec
        """
        from ..utils.session_manager import create_run_id as _create_run_id

        task_id = f"controlled-{kind.value}"
        run_id = _create_run_id(task_id)

        od = output_dir or (self._storage_dir / run_id / "output")
        od.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        spec = ControlledRunSpec(
            run_id=run_id,
            kind=kind,
            audio_path=str(audio_path),
            output_dir=str(od),
            config_path=str(config_path),
            overrides=overrides or {},
            ground_truth=str(ground_truth) if ground_truth else "",
            created_at=now,
        )
        self._save_spec(spec)
        return spec

    def execute(self, spec: ControlledRunSpec) -> Path:
        """执行对照运行（调用 Pipeline）。

        Args:
            spec: 运行规格

        Returns:
            输出字幕文件路径
        """
        from ..config import ConfigLoader
        from ..pipeline import Pipeline

        loader = ConfigLoader()
        config = loader.load_profile("default")

        # 应用覆盖项
        for key_path, value in spec.overrides.items():
            parts = key_path.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    setattr(obj, parts[-1], value)
                except (AttributeError, TypeError):
                    logger.warning("Failed to apply override: %s = %s", key_path, value)

        pipeline = Pipeline(config)
        result = pipeline.run(
            input_path=Path(spec.audio_path),
            output_path=Path(spec.output_dir),
        )

        subtitle_path = result.get("subtitle_path", "")
        if subtitle_path:
            subtitle_path = Path(subtitle_path)
            logger.info("Controlled run complete: %s (%s)", spec.run_id, subtitle_path)
            return subtitle_path

        raise RuntimeError(f"Controlled run {spec.run_id} produced no subtitle output")

    def compare(
        self,
        baseline: ControlledRunSpec,
        candidate: ControlledRunSpec,
        *,
        ground_truth: Optional[Path] = None,
    ) -> dict:
        """比较 baseline 和 candidate 的输出。

        使用 scripts/compare_timeline.py 进行比较。
        优先尝试 Python import 路径，失败时回退到 subprocess。

        Args:
            baseline: 基线运行
            candidate: 候选运行
            ground_truth: 参考字幕（可选，优先于 baseline 输出）

        Returns:
            比较报告 dict
        """
        auto_path = Path(candidate.output_dir) / "subtitle.srt"
        gt_path = ground_truth or Path(baseline.output_dir) / "subtitle.srt"

        if not auto_path.exists():
            return {"error": f"candidate subtitle not found: {auto_path}"}
        if not gt_path.exists():
            return {"error": f"baseline/reference subtitle not found: {gt_path}"}

        # 尝试 import 路径
        try:
            from scripts.compare_timeline import compare as _compare, report_to_dict
            report = _compare(str(gt_path), str(auto_path))
            return report_to_dict(report)
        except ImportError:
            pass

        # 回退到 subprocess
        try:
            result = subprocess.run(
                [
                    "python", "scripts/compare_timeline.py",
                    "--ground-truth", str(gt_path),
                    "--auto", str(auto_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return {"status": "ok", "stdout": result.stdout}
            return {"error": result.stderr or "comparison failed"}
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {"error": str(e)}

    @staticmethod
    def recommend(comparison: dict) -> ComparisonRecommendation:
        """从比较报告生成启用建议。

        Args:
            comparison: compare() 返回的比较报告

        Returns:
            ComparisonRecommendation
        """
        if "error" in comparison:
            return ComparisonRecommendation(
                recommend_enable=False,
                uncertain_areas=[comparison["error"]],
                rationale="对比失败，无法给出建议",
            )

        benefited: list[dict] = comparison.get("benefited_scenes", [])
        degraded: list[dict] = comparison.get("degraded_scenes", [])
        uncertain: list[str] = comparison.get("uncertain_areas", [])

        recommend = len(benefited) > 0 and len(degraded) == 0 and len(uncertain) == 0

        rationale_parts = []
        if benefited:
            rationale_parts.append(f"{len(benefited)} 个场景受益")
        if degraded:
            rationale_parts.append(f"{len(degraded)} 个场景退化")
        if uncertain:
            rationale_parts.append(f"{len(uncertain)} 个不确定项")
        if not rationale_parts:
            rationale_parts.append("无明显差异")

        return ComparisonRecommendation(
            recommend_enable=recommend,
            benefited_scenes=benefited,
            degraded_scenes=degraded,
            uncertain_areas=uncertain,
            rationale="; ".join(rationale_parts),
        )

    # ---- 持久化 ----

    def _spec_path(self, run_id: str) -> Path:
        return self._storage_dir / run_id / "spec.json"

    def _save_spec(self, spec: ControlledRunSpec) -> None:
        path = self._spec_path(spec.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_spec(self, run_id: str) -> Optional[ControlledRunSpec]:
        path = self._spec_path(run_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ControlledRunSpec.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None


__all__ = [
    "RunKind",
    "ControlledRunSpec",
    "ComparisonRecommendation",
    "ControlledRunManager",
]
