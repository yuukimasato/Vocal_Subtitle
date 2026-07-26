#!/usr/bin/env python3
"""Run subtitle quality comparisons against the repository's real fixtures.

This runner deliberately keeps real-material quality separate from the empty
release benchmark directory. It records whether a successful run used global
ASR, legacy ASR, or a global-to-legacy fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

# Allow direct execution via ``python scripts/run_quality_benchmark.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_timeline import match_events, parse_subtitle, report_to_dict


def load_manifest(path: Path, repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate a real-material manifest without touching models."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain a non-empty scenes list")

    root = repo_root or Path(__file__).resolve().parents[1]
    validated: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in scenes:
        if not isinstance(item, dict):
            raise ValueError("each manifest scene must be a mapping")
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"duplicate or empty scene name: {name!r}")
        audio = root / str(item.get("audio", ""))
        ground_truth = root / str(item.get("ground_truth", ""))
        if not audio.is_file():
            raise FileNotFoundError(f"{name}: audio not found: {audio}")
        if not ground_truth.is_file():
            raise FileNotFoundError(
                f"{name}: ground truth not found: {ground_truth}"
            )
        category = str(item.get("category", "")).strip()
        if not category:
            raise ValueError(f"{name}: category is required")
        names.add(name)
        validated.append(
            {
                **item,
                "name": name,
                "audio_path": audio,
                "ground_truth_path": ground_truth,
            }
        )
    return validated


def _normalized_text(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


def _reference_content_coverage(auto_events, ground_truth_events) -> float:
    """Measure reference-content recall while allowing legitimate extra words.

    Human subtitles often omit fillers or compress spoken phrasing. The
    benchmark therefore measures how much reference text is present in the
    ordered system transcript; extra text is not penalized here and is
    checked separately through physical/source-word diagnostics.
    """
    auto_text = _normalized_text("".join(event.text for event in auto_events))
    reference_text = _normalized_text("".join(event.text for event in ground_truth_events))
    if not reference_text:
        return 1.0
    matcher = SequenceMatcher(None, reference_text, auto_text, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return round(min(1.0, matched / len(reference_text)), 4)


def _display_path(path: Path, repo_root: Path) -> str:
    """Use repository-relative paths when possible, absolute otherwise."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _text_metrics(auto_path: Path, gt_path: Path) -> dict[str, Any]:
    auto_events = parse_subtitle(auto_path)
    gt_events = parse_subtitle(gt_path)
    matches = match_events(auto_events, gt_events)
    similarities = [
        SequenceMatcher(
            None,
            _normalized_text(auto.text),
            _normalized_text(gt.text),
        ).ratio()
        for auto, gt in matches
    ]
    return {
        "auto_event_count": len(auto_events),
        "ground_truth_event_count": len(gt_events),
        "matched_event_count": len(matches),
        "unmatched_auto_event_count": len(auto_events) - len(matches),
        "unmatched_ground_truth_event_count": len(gt_events) - len(matches),
        "mean_text_similarity": round(
            sum(similarities) / len(similarities), 4
        ) if similarities else 0.0,
        "char_error_rate_proxy": round(
            1.0 - sum(similarities) / len(similarities), 4
        ) if similarities else 1.0,
        "reference_content_coverage": _reference_content_coverage(auto_events, gt_events),
    }


def _run_scene(
    scene: dict[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
    asr_path: str,
    output_root: Path,
    skip_separation: bool,
    no_cache: bool,
    asr_model: str | None,
) -> dict[str, Any]:
    from vocal_subtitle.config import ConfigLoader
    from vocal_subtitle.pipeline import Pipeline

    name = scene["name"]
    scene_dir = output_root / asr_path / name
    scene_dir.mkdir(parents=True, exist_ok=True)
    subtitle_path = scene_dir / "subtitle.ass"
    started = time.monotonic()
    result: dict[str, Any] = {
        "scene": name,
        "requested_asr_path": asr_path,
        "audio": str(scene["audio_path"].relative_to(repo_root)),
        "ground_truth": str(scene["ground_truth_path"].relative_to(repo_root)),
        "category": scene["category"],
        "expected_speaker_count": scene.get("speaker_count"),
        "success": False,
        "elapsed_sec": 0.0,
        "output": _display_path(subtitle_path, repo_root),
    }
    try:
        loader = ConfigLoader()
        config = loader.load_file(config_path)
        overrides = {"asr_path": asr_path}
        if asr_model:
            overrides["asr_model"] = asr_model
        config = loader.merge_with_overrides(config, **overrides)
        if no_cache:
            config.cache.enabled = False
        pipeline_result = Pipeline(config).run(
            input_path=scene["audio_path"],
            output_path=subtitle_path,
            output_format="ass",
            skip_separation=skip_separation,
        )
        stats = pipeline_result["stats"]
        stats_dict = stats.to_dict()
        produced = Path(pipeline_result.get("subtitle_path", subtitle_path))
        if not produced.is_file():
            produced = subtitle_path
        comparison_path = scene_dir / "comparison_report.json"
        from scripts.compare_timeline import compare

        comparison = compare(
            parse_subtitle(produced),
            parse_subtitle(scene["ground_truth_path"]),
            auto_file=str(produced),
            gt_file=str(scene["ground_truth_path"]),
        )
        comparison_payload = report_to_dict(comparison)
        comparison_payload["text_metrics"] = _text_metrics(
            produced, scene["ground_truth_path"]
        )
        comparison_path.write_text(
            json.dumps(comparison_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (scene_dir / "diagnostic_report.json").write_text(
            json.dumps(stats_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result.update(
            {
                "success": True,
                "actual_asr_path": stats_dict.get("asr_path"),
                "global_attempted": stats_dict.get("global_attempted", False),
                "fallback_category": stats_dict.get("fallback_category"),
                "fallback_reason": stats_dict.get("fallback_reason"),
                "comparison": comparison_payload,
                "diagnostic": {
                    "speaker_count": stats_dict.get("speaker_count"),
                    "physical_violation_count": (
                        stats_dict.get("final_validation", {}).get(
                            "physical_violation_count", 0
                        )
                    ),
                    "alignment_warning_count": (
                        stats_dict.get("global_statistics", {}).get(
                            "cross_boundary_count", 0
                        )
                    ),
                },
            }
        )
    except Exception as exc:  # benchmark must report every scene
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    result["elapsed_sec"] = round(time.monotonic() - started, 2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 test/ 中真实音频和人工字幕进行质量对比"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("test/quality_manifest.yaml")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--asr-path",
        choices=("auto", "global", "segmented"),
        default="auto",
        help="实际请求的离线 ASR 路径",
    )
    parser.add_argument(
        "--asr-model",
        choices=("tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"),
        default=None,
        help="覆盖配置中的 ASR 模型，用于高精度真实素材验收",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("test/benchmark_results/real_material")
    )
    parser.add_argument(
        "--scene", action="append", help="只运行指定场景，可重复传入"
    )
    parser.add_argument(
        "--with-separation",
        action="store_true",
        help="不跳过人声分离；默认将 test 音频视为人声输入",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="忽略已有缓存，强制重新执行流水线",
    )
    parser.add_argument("--ci", action="store_true", help="质量门失败时返回非零")
    parser.add_argument(
        "--require-global",
        action="store_true",
        help="要求每个场景实际使用 global，不接受 fallback",
    )
    parser.add_argument("--target-end-mae", type=float, default=None)
    parser.add_argument("--target-start-mae", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    config_path = (repo_root / args.config).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    try:
        scenes = load_manifest(manifest_path, repo_root)
    except Exception as exc:
        print(f"manifest 错误: {exc}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        return 2
    if args.scene:
        selected = set(args.scene)
        scenes = [scene for scene in scenes if scene["name"] in selected]
        missing = selected - {scene["name"] for scene in scenes}
        if missing:
            print(f"场景不存在: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    results = [
        _run_scene(
            scene,
            repo_root=repo_root,
            config_path=config_path,
            asr_path=args.asr_path,
            output_root=output_root,
            skip_separation=not args.with_separation,
            no_cache=args.no_cache,
            asr_model=args.asr_model,
        )
        for scene in scenes
    ]
    successful = [item for item in results if item["success"]]
    comparisons = [item["comparison"] for item in successful if item.get("comparison")]
    start_mae = [item["statistics"]["start"]["mae_ms"] for item in comparisons]
    end_mae = [item["statistics"]["end"]["mae_ms"] for item in comparisons]
    content_coverages = [
        item["comparison"].get("text_metrics", {}).get("reference_content_coverage", 0.0)
        for item in successful
        if item.get("comparison")
    ]
    actual_paths = sorted({item.get("actual_asr_path") for item in successful})
    reasons: list[str] = []
    if not results:
        reasons.append("no_scenes")
    if len(successful) != len(results):
        reasons.append("scene_failure")
    categories = {str(scene["category"]) for scene in scenes}
    for scene in scenes:
        categories.update(str(tag) for tag in scene.get("tags", []))
    if "single_speaker" not in categories:
        reasons.append("missing_category:single_speaker")
    if "multi_speaker" not in categories:
        reasons.append("missing_category:multi_speaker")
    if "overlap_or_interruption" not in categories:
        reasons.append("missing_category:overlap_or_interruption")
    if successful and any(path != "global" for path in actual_paths):
        reasons.append("global_path_not_used")
    if args.require_global and any(path != "global" for path in actual_paths):
        reasons.append("global_path_not_used_for_every_scene")
    if args.target_start_mae is not None and start_mae:
        if sum(start_mae) / len(start_mae) > args.target_start_mae:
            reasons.append("start_mae_target_failed")
    if args.target_end_mae is not None and end_mae:
        if sum(end_mae) / len(end_mae) > args.target_end_mae:
            reasons.append("end_mae_target_failed")
    if content_coverages and min(content_coverages) < 0.99:
        reasons.append("reference_content_coverage_below_99pct")
    for item in successful:
        diagnostic = item.get("diagnostic", {})
        if diagnostic.get("physical_violation_count", 0):
            reasons.append(f"physical_violation:{item['scene']}")
        coverage = item.get("comparison", {}).get("text_metrics", {}).get(
            "reference_content_coverage", 0.0
        )
        if coverage < 0.99:
            reasons.append(f"reference_content_coverage:{item['scene']}")

    summary = {
        "manifest": str(manifest_path.relative_to(repo_root)),
        "config": str(config_path.relative_to(repo_root)),
        "requested_asr_path": args.asr_path,
        "scene_count": len(results),
        "success_count": len(successful),
        "actual_asr_paths": actual_paths,
        "aggregate": {
            "start_mae_ms": round(sum(start_mae) / len(start_mae), 1) if start_mae else None,
            "end_mae_ms": round(sum(end_mae) / len(end_mae), 1) if end_mae else None,
            "min_reference_content_coverage": min(content_coverages) if content_coverages else None,
        },
        "publishable": not reasons,
        "reasons": reasons,
        "scenes": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"summary-{args.asr_path}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.ci and reasons:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
