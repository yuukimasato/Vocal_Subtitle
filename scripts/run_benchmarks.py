#!/usr/bin/env python3
"""自动化基准回归测试

在多个场景的基准数据集上运行 Pipeline，对比 ground truth 字幕，
量化各方案的时间轴精度改善效果。

用法:
    # 全量基准测试
    python scripts/run_benchmarks.py \
        --benchmark-dir tests/benchmarks/ \
        --config configs/default.yaml \
        --output-dir test/benchmark_results/ \
        --metrics start_mae,end_mae,end_error_gt_300ms,health_score

    # 单场景测试
    python scripts/run_benchmarks.py \
        --benchmark-dir tests/benchmarks/ \
        --scene hotel_front_desk \
        --config configs/default.yaml

输出:
    benchmark_results/
    ├── summary.json                  # 全场景汇总
    ├── hotel_front_desk/
    │   ├── subtitle.ass              # 生成的字幕
    │   ├── comparison_report.json    # 与 ground truth 的对比
    │   └── diagnostic_report.json    # 声学校验诊断（如有）
    ├── meeting_3_speakers/
    │   └── ...
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------


@dataclass
class SceneResult:
    """单个场景的基准测试结果"""

    scene: str
    success: bool
    output_dir: Path
    auto_subtitle: Optional[Path] = None
    comparison: Optional[dict] = None
    diagnostic: Optional[dict] = None
    elapsed_sec: float = 0.0
    error: str = ""
    asr_path: str = ""
    fallback_category: str = ""


@dataclass
class BenchmarkSummary:
    """全场景汇总"""

    timestamp: str = ""
    config_file: str = ""
    total_scenes: int = 0
    success_scenes: int = 0
    results: List[SceneResult] = field(default_factory=list)

    # 聚合指标（所有成功场景的平均值）
    avg_start_mae_ms: float = 0.0
    avg_end_mae_ms: float = 0.0
    avg_health_score: float = 0.0
    total_elapsed_sec: float = 0.0

    # rollout 状态
    rollout_eligible: bool = False
    eligibility_reasons: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# 基准扫描
# ------------------------------------------------------------------


def discover_scenes(benchmark_dir: Path) -> Dict[str, Path]:
    """扫描基准目录，发现所有场景

    每个场景目录需包含 audio.wav 或 vocals.wav，
    且包含 ground_truth.ass。

    Returns:
        {scene_name: scene_directory}
    """
    scenes = {}
    if not benchmark_dir.exists():
        return scenes

    for entry in sorted(benchmark_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue

        has_audio = (entry / "audio.wav").exists()
        has_vocals = (entry / "vocals.wav").exists()
        has_gt = (entry / "ground_truth.ass").exists()

        if (has_audio or has_vocals) and has_gt:
            scenes[entry.name] = entry
        elif has_gt:
            print(f"  跳过 '{entry.name}': 缺少 audio.wav 或 vocals.wav")
        else:
            print(f"  跳过 '{entry.name}': 缺少 ground_truth.ass")

    return scenes


# ------------------------------------------------------------------
# Pipeline 执行
# ------------------------------------------------------------------


def run_pipeline(
    audio_path: Path,
    output_path: Path,
    config_path: Path,
    skip_separation: bool = False,
    profile: Optional[str] = None,
) -> Tuple[bool, str]:
    """运行 Pipeline 生成字幕

    Args:
        audio_path: 输入音频路径
        output_path: 输出 ASS 文件路径
        config_path: 配置 YAML 路径
        skip_separation: 是否跳过人声分离（音频已是 vocals）
        profile: 可选的场景配置模板名称

    Returns:
        (success, error_message)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "vocal_subtitle.cli", "process",
        str(audio_path),
        "--output", str(output_path),
        "--format", "ass",
        "--config", str(config_path),
    ]

    if skip_separation:
        cmd.append("--skip-separation")

    if profile:
        cmd.extend(["--profile", profile])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15分钟超时
        )
        if result.returncode != 0:
            return False, f"Pipeline 退出码 {result.returncode}:\n{result.stderr[-1000:]}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Pipeline 超时（15分钟）"
    except Exception as e:
        return False, str(e)


def run_comparison(
    auto_path: Path,
    gt_path: Path,
    output_path: Path,
    health_score: Optional[float] = None,
) -> Optional[dict]:
    """运行时间轴对比分析

    Args:
        auto_path: 自动生成的字幕文件
        gt_path: ground truth 字幕文件
        output_path: 对比报告输出路径
        health_score: 可选的声学校验健康度

    Returns:
        对比报告字典，失败返回 None
    """
    compare_script = Path(__file__).parent / "compare_timeline.py"
    if not compare_script.exists():
        print(f"    警告: compare_timeline.py 未找到，跳过对比")
        return None

    cmd = [
        sys.executable, str(compare_script),
        "--auto", str(auto_path),
        "--ground-truth", str(gt_path),
        "--output", str(output_path),
        "--quiet",
    ]

    if health_score is not None:
        cmd.extend(["--health-score", str(health_score)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (subprocess.TimeoutExpired, Exception):
        pass

    return None


# ------------------------------------------------------------------
# 汇总
# ------------------------------------------------------------------


def aggregate_summary(results: List[SceneResult]) -> BenchmarkSummary:
    """汇总所有场景结果"""
    import datetime

    summary = BenchmarkSummary(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_scenes=len(results),
        success_scenes=sum(1 for r in results if r.success),
        results=results,
    )

    success_results = [r for r in results if r.success and r.comparison]

    if success_results:
        summary.avg_start_mae_ms = sum(
            r.comparison["statistics"]["start"]["mae_ms"]
            for r in success_results
        ) / len(success_results) if success_results else 0

        summary.avg_end_mae_ms = sum(
            r.comparison["statistics"]["end"]["mae_ms"]
            for r in success_results
        ) / len(success_results) if success_results else 0

        health_scores = [
            r.comparison.get("health_score", 0) or 0
            for r in success_results
        ]
        summary.avg_health_score = (
            sum(health_scores) / len(health_scores) if health_scores else 0
        )

        summary.total_elapsed_sec = sum(r.elapsed_sec for r in results)

    return summary


def summary_to_dict(summary: BenchmarkSummary) -> dict:
    """汇总序列化"""
    return {
        "timestamp": summary.timestamp,
        "config_file": summary.config_file,
        "total_scenes": summary.total_scenes,
        "success_scenes": summary.success_scenes,
        "failure_scenes": summary.total_scenes - summary.success_scenes,
        "rollout_eligible": summary.rollout_eligible,
        "eligibility_reasons": list(summary.eligibility_reasons),
        "aggregate": {
            "avg_start_mae_ms": round(summary.avg_start_mae_ms, 1),
            "avg_end_mae_ms": round(summary.avg_end_mae_ms, 1),
            "avg_health_score": round(summary.avg_health_score, 1),
            "total_elapsed_sec": round(summary.total_elapsed_sec, 1),
        },
        "scenes": [
            {
                "scene": r.scene,
                "success": r.success,
                "elapsed_sec": round(r.elapsed_sec, 1),
                "error": r.error,
                "asr_path": r.asr_path,
                "fallback_category": r.fallback_category,
                "statistics": r.comparison.get("statistics") if r.comparison else None,
                "health_score": r.comparison.get("health_score") if r.comparison else None,
                "flagged_count": len(r.comparison.get("flagged_events", [])) if r.comparison else 0,
                "files": {
                    "subtitle": str(r.auto_subtitle) if r.auto_subtitle else None,
                    "comparison": str(r.output_dir / "comparison_report.json"),
                    "diagnostic": str(r.output_dir / "diagnostic_report.json") if (r.output_dir / "diagnostic_report.json").exists() else None,
                },
            }
            for r in summary.results
        ],
    }


def evaluate_rollout_eligibility(
    scene_dirs: dict,
    results: List[SceneResult],
    summary: BenchmarkSummary,
    min_scenes: int = 3,
    target_metrics: Optional[dict] = None,
) -> tuple:
    """Evaluate whether a benchmark rollout is eligible for publication.

    Args:
        scene_dirs: Dict of {scene_name: scene_directory}
        results: List of per-scene benchmark results
        summary: Aggregated benchmark summary
        min_scenes: Minimum number of valid scenes required
        target_metrics: Dict of metric targets, e.g. {"end_mae": 300}

    Returns:
        (eligible: bool, reasons: List[str])
    """
    reasons = []

    if summary.total_scenes < min_scenes:
        reasons.append(f"至少 {min_scenes} 个场景 (当前 {summary.total_scenes})")

    success_results = [r for r in results if r.success]
    non_global = [r for r in success_results if getattr(r, "asr_path", "") != "global"]
    if non_global:
        reasons.append("没有使用 global ASR 路径的场景")

    if target_metrics:
        end_target = target_metrics.get("end_mae")
        start_target = target_metrics.get("start_mae")
        if end_target and summary.avg_end_mae_ms > end_target:
            reasons.append(f"End MAE {summary.avg_end_mae_ms:.0f}ms 超过目标 {end_target}ms")
        if start_target and summary.avg_start_mae_ms > start_target:
            reasons.append(f"Start MAE {summary.avg_start_mae_ms:.0f}ms 超过目标 {start_target}ms")

    eligible = len(reasons) == 0
    return eligible, reasons


def print_summary(summary: BenchmarkSummary) -> None:
    """终端打印汇总"""
    print()
    print("=" * 70)
    print("  基准测试汇总报告")
    print("=" * 70)
    print(f"  时间: {summary.timestamp}")
    print(f"  配置: {summary.config_file}")
    print(f"  场景: {summary.success_scenes}/{summary.total_scenes} 成功")
    print()

    if summary.success_scenes > 0:
        print("  ┌──────────────────────────┬──────────┬──────────┬──────────┐")
        print("  │ 场景                      │ End MAE  │ St MAE   │ 健康度    │")
        print("  ├──────────────────────────┼──────────┼──────────┼──────────┤")
        for r in summary.results:
            if r.success and r.comparison:
                scene = r.scene[:24]
                end_mae = r.comparison["statistics"]["end"]["mae_ms"]
                start_mae = r.comparison["statistics"]["start"]["mae_ms"]
                health = r.comparison.get("health_score", 0) or 0
                print(f"  │ {scene:<24} │ {end_mae:>6.0f}ms │ {start_mae:>6.0f}ms │ {health:>5.1f}%   │")
            elif not r.success:
                scene = r.scene[:24]
                print(f"  │ {scene:<24} │ {'FAIL':>6} │ {'--':>6} │ {'--':>5}   │")
        print("  ├──────────────────────────┼──────────┼──────────┼──────────┤")
        print(f"  │ {'平均':>24} │ {summary.avg_end_mae_ms:>6.0f}ms │ {summary.avg_start_mae_ms:>6.0f}ms │ {summary.avg_health_score:>5.1f}%   │")
        print("  └──────────────────────────┴──────────┴──────────┴──────────┘")
        print()

    # 失败场景
    failed = [r for r in summary.results if not r.success]
    if failed:
        print(f"  ⚠ 失败场景 ({len(failed)}):")
        for r in failed:
            print(f"    - {r.scene}: {r.error[:100]}")
        print()

    print(f"  总耗时: {summary.total_elapsed_sec:.0f}s")
    print("=" * 70)


# ------------------------------------------------------------------
# 指标检查
# ------------------------------------------------------------------


def check_metrics(summary: BenchmarkSummary, target_metrics: dict) -> dict:
    """检查聚合指标是否达标

    Args:
        summary: 汇总结果
        target_metrics: 目标值字典，如 {"end_mae_ms": 300, "start_mae_ms": 200}

    Returns:
        {metric: {actual, target, pass: bool}}
    """
    checks = {}
    metric_map = {
        "start_mae": ("avg_start_mae_ms", summary.avg_start_mae_ms),
        "end_mae": ("avg_end_mae_ms", summary.avg_end_mae_ms),
        "health_score": ("avg_health_score", summary.avg_health_score),
    }

    for metric, target in target_metrics.items():
        if metric in metric_map:
            name, actual = metric_map[metric]
            checks[metric] = {
                "actual": round(actual, 1),
                "target": target,
                "pass": actual <= target if metric != "health_score" else actual >= target,
            }

    return checks


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动化基准回归测试 — 多场景字幕时间轴精度评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 全量基准测试
    %(prog)s --benchmark-dir tests/benchmarks/ --config configs/default.yaml

    # 单场景测试
    %(prog)s --benchmark-dir tests/benchmarks/ --scene hotel_front_desk

    # 带指标检查的 CI 模式
    %(prog)s --benchmark-dir tests/benchmarks/ --ci --target-end-mae 300
        """,
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("tests/benchmarks/"),
        help="基准数据集目录（默认 tests/benchmarks/）",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="仅测试指定场景（默认测试全部）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Pipeline 配置文件路径",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="可选的场景配置模板（如 podcast、education）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test/benchmark_results/"),
        help="基准测试结果输出目录",
    )
    parser.add_argument(
        "--skip-separation",
        action="store_true",
        help="跳过人声分离步骤（基准音频已是 vocals）",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI 模式：指标不达标时非零退出",
    )
    parser.add_argument(
        "--target-end-mae",
        type=float,
        default=300,
        help="目标 End MAE (ms)，CI 模式下超过此值则失败",
    )
    parser.add_argument(
        "--target-start-mae",
        type=float,
        default=200,
        help="目标 Start MAE (ms)，CI 模式下超过此值则失败",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"错误: 基准目录不存在: {benchmark_dir}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    # ---- 1. 发现场景 ----
    print("🔍 扫描基准场景...")
    all_scenes = discover_scenes(benchmark_dir)

    if args.scene:
        if args.scene not in all_scenes:
            print(f"错误: 场景 '{args.scene}' 不存在于 {benchmark_dir}")
            print(f"可用场景: {', '.join(sorted(all_scenes.keys())) or '(无)'}")
            sys.exit(1)
        scenes = {args.scene: all_scenes[args.scene]}
    else:
        scenes = all_scenes

    if not scenes:
        print("错误: 未找到任何有效的基准场景")
        print(f"请确保 {benchmark_dir}/<scene>/ 包含 audio.wav 和 ground_truth.ass")
        sys.exit(1)

    print(f"找到 {len(scenes)} 个场景: {', '.join(sorted(scenes.keys()))}")
    print()

    # ---- 2. 逐场景测试 ----
    output_dir = Path(args.output_dir)
    results: List[SceneResult] = []
    total_start = time.time()

    for i, (scene_name, scene_dir) in enumerate(sorted(scenes.items()), 1):
        print(f"[{i}/{len(scenes)}] 测试场景: {scene_name}")
        print("-" * 50)

        scene_output_dir = output_dir / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        result = SceneResult(scene=scene_name, success=False, output_dir=scene_output_dir)

        # 选择输入音频
        vocals_path = scene_dir / "vocals.wav"
        audio_path = scene_dir / "audio.wav"
        if vocals_path.exists():
            input_audio = vocals_path
            skip_sep = True
        else:
            input_audio = audio_path
            skip_sep = args.skip_separation

        auto_subtitle = scene_output_dir / "subtitle.ass"
        gt_subtitle = scene_dir / "ground_truth.ass"

        t0 = time.time()

        # 运行 Pipeline
        success, error = run_pipeline(
            input_audio, auto_subtitle, config_path,
            skip_separation=skip_sep,
            profile=args.profile,
        )

        result.elapsed_sec = time.time() - t0

        if not success:
            result.error = error
            print(f"  ❌ Pipeline 失败: {error[:200]}")
            results.append(result)
            print()
            continue

        result.auto_subtitle = auto_subtitle
        print(f"  ✓ 字幕已生成: {auto_subtitle} ({result.elapsed_sec:.0f}s)")

        # 运行对比分析
        comparison_path = scene_output_dir / "comparison_report.json"
        comparison = run_comparison(
            auto_subtitle, gt_subtitle, comparison_path,
            health_score=None,  # 声学校验健康度从诊断中提取
        )

        if comparison:
            result.comparison = comparison
            end_mae = comparison["statistics"]["end"]["mae_ms"]
            start_mae = comparison["statistics"]["start"]["mae_ms"]
            flagged = len(comparison.get("flagged_events", []))
            print(f"  ✓ 对比完成: Start MAE={start_mae:.0f}ms, "
                  f"End MAE={end_mae:.0f}ms, 标记={flagged}条")

        # 尝试加载诊断报告
        diag_path = scene_output_dir / "diagnostic_report.json"
        if diag_path.exists():
            try:
                with open(diag_path, "r", encoding="utf-8") as f:
                    result.diagnostic = json.load(f)
                print(f"  ✓ 诊断报告: 健康度={result.diagnostic.get('health_score', 'N/A')}%")
            except Exception:
                pass

        result.success = True
        results.append(result)
        print()

    # ---- 3. 汇总 ----
    summary = aggregate_summary(results)
    summary.config_file = str(config_path)
    summary.total_elapsed_sec = time.time() - total_start

    print_summary(summary)

    # 保存汇总
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_to_dict(summary), f, ensure_ascii=False, indent=2)
    print(f"汇总报告: {summary_path}")

    # ---- 4. CI 模式指标检查 ----
    if args.ci and summary.success_scenes > 0:
        targets = {
            "end_mae": args.target_end_mae,
            "start_mae": args.target_start_mae,
        }
        checks = check_metrics(summary, targets)

        all_pass = True
        print()
        print("--- CI 指标检查 ---")
        for metric, check in checks.items():
            status = "✓" if check["pass"] else "✗"
            print(f"  {status} {metric}: {check['actual']:.0f}ms "
                  f"(目标: {check['target']:.0f}ms)")

        if not all(checks.values()):
            print()
            print("❌ 指标未达标！")
            sys.exit(1)

    if summary.success_scenes == 0:
        print()
        print("❌ 所有场景均失败！")
        sys.exit(2)

    print()
    print("✅ 基准测试完成")


if __name__ == "__main__":
    main()
