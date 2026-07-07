#!/usr/bin/env python3
"""字幕时间轴对比分析工具

将自动生成的字幕（auto.ass）与手动校正的 ground truth（ground_truth.ass）
逐条对比，量化时间轴偏差。

用法:
    python scripts/compare_timeline.py \
        --auto test/subtitle_optimized.ass \
        --ground-truth test/60.ass \
        --output test/comparison_report.json

输出包含:
    - 逐条偏差明细
    - 聚合统计（MAE、标准差、误差分布）
    - 标记异常事件（偏差 > 300ms）
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------


@dataclass
class TimelineEvent:
    """字幕时间轴事件"""

    index: int
    start: float  # 秒
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EventComparison:
    """单条字幕对比结果"""

    index: int
    auto_start: float
    auto_end: float
    gt_start: float
    gt_end: float
    auto_text: str
    gt_text: str
    start_error_ms: float
    end_error_ms: float
    start_error_abs_ms: float
    end_error_abs_ms: float


@dataclass
class ComparisonReport:
    """对比分析报告"""

    auto_file: str
    ground_truth_file: str
    total_events: int
    matched_events: int
    comparisons: List[EventComparison] = field(default_factory=list)

    # 聚合统计
    start_mae_ms: float = 0.0
    end_mae_ms: float = 0.0
    start_mean_error_ms: float = 0.0
    end_mean_error_ms: float = 0.0
    start_std_ms: float = 0.0
    end_std_ms: float = 0.0
    max_start_error_ms: float = 0.0
    max_end_error_ms: float = 0.0

    # 分布统计
    start_gt_300ms_pct: float = 0.0
    end_gt_300ms_pct: float = 0.0
    start_gt_500ms_pct: float = 0.0
    end_gt_500ms_pct: float = 0.0

    # 异常事件
    flagged_events: List[dict] = field(default_factory=list)

    # 声学校验（可选，如提供了 Pipeline 诊断输出）
    health_score: Optional[float] = None


# ------------------------------------------------------------------
# ASS 解析
# ------------------------------------------------------------------


def parse_ass(file_path: Path) -> List[TimelineEvent]:
    """解析 ASS 字幕文件中的 Dialogue 事件

    提取 start, end, text 字段。忽略 Format/Comment/Style 等头部行。
    """
    events = []
    dialogue_pattern = re.compile(
        r"^Dialogue:\s*"
        r"(?:Marked=\d+,)?\s*"          # 可选的 Marked 字段
        r"[^,]*,\s*"                     # Layer
        r"(\d+:\d+:\d+\.?\d*),\s*"       # Start
        r"(\d+:\d+:\d+\.?\d*),\s*"       # End
        r"(?:[^,]*,\s*){"                # Style, Name, MarginL, MarginR, MarginV, Effect
        r"(.+)$"                         # Text
    )

    ass_time_pattern = re.compile(
        r"(\d+):(\d+):(\d+)\.?(\d*)"
    )

    text_lines = []
    idx = 0

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("Dialogue:"):
            continue

        # 尝试正则匹配
        match = dialogue_pattern.match(line)
        if not match:
            # 手动解析（更鲁棒的方式）
            try:
                parts = line[len("Dialogue:"):].split(",")
                # 跳过 Layer
                start_str = parts[1].strip() if len(parts) > 1 else ""
                end_str = parts[2].strip() if len(parts) > 2 else ""
                # Text 是最后一个逗号之后（跳过中间的样式字段）
                text_start = 0
                # ASS Dialogue 格式: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
                # Text 在第10个字段(索引9)
                text_parts = line[len("Dialogue:"):].split(",", 9)
                text = text_parts[9] if len(text_parts) > 9 else ""
            except Exception:
                continue
        else:
            start_str = match.group(1)
            end_str = match.group(2)
            text = match.group(3)

        start_sec = _ass_time_to_seconds(start_str)
        end_sec = _ass_time_to_seconds(end_str)

        if start_sec is None or end_sec is None:
            continue

        # 清理文本（去除 ASS 特效标签 {\xxx}）
        clean_text = re.sub(r"\{[^}]*\}", "", text).strip()
        # 去除 \N 换行标记（替换为空格）
        clean_text = clean_text.replace("\\N", " ").replace("\n", " ").strip()

        if not clean_text:
            continue

        idx += 1
        events.append(TimelineEvent(
            index=idx,
            start=start_sec,
            end=end_sec,
            text=clean_text,
        ))

    return events


def parse_srt(file_path: Path) -> List[TimelineEvent]:
    """解析 SRT 字幕文件（使用 pysubs2）

    提取 start, end, text 字段。自动处理 speaker 标签。
    """
    import pysubs2

    subs = pysubs2.load(str(file_path), encoding="utf-8")
    events = []
    for idx, evt in enumerate(subs.events):
        text = evt.text.replace("\\N", " ").replace("\n", " ").strip()
        if not text:
            continue
        events.append(TimelineEvent(
            index=idx + 1,
            start=evt.start / 1000.0,  # pysubs2 uses ms
            end=evt.end / 1000.0,
            text=text,
        ))
    return events


def parse_subtitle(file_path: Path) -> List[TimelineEvent]:
    """自动检测格式并解析字幕文件（ASS/SRT）"""
    suffix = file_path.suffix.lower()
    if suffix == ".srt":
        return parse_srt(file_path)
    elif suffix in (".ass", ".ssa"):
        return parse_ass(file_path)
    else:
        # 尝试先用 pysubs2 自动检测
        try:
            return parse_srt(file_path)
        except Exception:
            return parse_ass(file_path)
    """秒数 → ASS 时间格式 (H:MM:SS.cs)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _ass_time_to_seconds(time_str: str) -> Optional[float]:
    """ASS 时间格式 → 秒数"""
    time_str = time_str.strip()
    # 支持 H:MM:SS.cs 和 H:MM:SS.xx 格式
    match = re.match(r"(\d+):(\d+):(\d+)\.?(\d*)", time_str)
    if not match:
        return None
    h, m, s, cs = match.groups()
    centiseconds = float(cs or "0")
    # 处理可能的 centiseconds (2位) vs milliseconds (3位)
    if len(cs) <= 2:
        centiseconds = centiseconds / (10 ** len(cs)) if cs else 0
        # centiseconds → seconds
        seconds = int(h) * 3600 + int(m) * 60 + int(s) + centiseconds / 100
    else:
        # 三位是毫秒
        ms = float(cs)
        seconds = int(h) * 3600 + int(m) * 60 + int(s) + ms / (10 ** len(cs))
    return seconds


# ------------------------------------------------------------------
# 事件匹配
# ------------------------------------------------------------------


def match_events(
    auto_events: List[TimelineEvent],
    gt_events: List[TimelineEvent],
    max_time_diff: float = 3.0,
) -> List[Tuple[TimelineEvent, TimelineEvent]]:
    """将自动生成的字幕事件与 ground truth 事件匹配

    优先按索引匹配，如果数量不同则用小的时间窗口 + 文本相似度匹配。

    Args:
        auto_events: 自动生成的字幕事件
        gt_events: ground truth 字幕事件
        max_time_diff: 最大容许时间偏差（秒），超过则不匹配

    Returns:
        [(auto_event, gt_event), ...] 匹配对列表
    """
    from difflib import SequenceMatcher

    matches = []

    if len(auto_events) == len(gt_events):
        # 数量相同：按索引直接匹配
        for a, g in zip(auto_events, gt_events):
            matches.append((a, g))
        return matches

    # 数量不同：贪心匹配
    gt_used = set()

    for a in auto_events:
        best_match = None
        best_score = 0.0

        for g in gt_events:
            if g.index in gt_used:
                continue

            # 时间窗口过滤
            time_diff = abs(a.start - g.start) + abs(a.end - g.end)
            if time_diff > max_time_diff * 2:
                continue

            # 文本相似度 + 时间接近度
            text_sim = SequenceMatcher(None, a.text, g.text).ratio()
            time_score = max(0, 1 - time_diff / (max_time_diff * 2))
            total_score = text_sim * 0.6 + time_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_match = g

        if best_match and best_score > 0.3:
            matches.append((a, best_match))
            gt_used.add(best_match.index)

    return matches


# ------------------------------------------------------------------
# 对比计算
# ------------------------------------------------------------------


def compute_statistics(
    comparisons: List[EventComparison],
) -> dict:
    """从对比列表计算聚合统计"""
    if not comparisons:
        return {
            "start_mae_ms": 0, "end_mae_ms": 0,
            "start_mean_error_ms": 0, "end_mean_error_ms": 0,
            "start_std_ms": 0, "end_std_ms": 0,
            "max_start_error_ms": 0, "max_end_error_ms": 0,
            "start_gt_300ms_pct": 0, "end_gt_300ms_pct": 0,
            "start_gt_500ms_pct": 0, "end_gt_500ms_pct": 0,
        }

    n = len(comparisons)

    start_errors = [c.start_error_ms for c in comparisons]
    end_errors = [c.end_error_ms for c in comparisons]
    start_abs = [c.start_error_abs_ms for c in comparisons]
    end_abs = [c.end_error_abs_ms for c in comparisons]

    import math

    # 均值
    start_mean = sum(start_errors) / n
    end_mean = sum(end_errors) / n
    start_mae = sum(start_abs) / n
    end_mae = sum(end_abs) / n

    # 标准差
    start_var = sum((x - start_mean) ** 2 for x in start_errors) / n
    end_var = sum((x - end_mean) ** 2 for x in end_errors) / n
    start_std = math.sqrt(start_var)
    end_std = math.sqrt(end_var)

    # 最大值
    max_start_abs = max(start_abs)
    max_end_abs = max(end_abs)

    # 超出 300ms/500ms 占比
    start_gt_300 = sum(1 for x in start_abs if x > 300) / n * 100
    end_gt_300 = sum(1 for x in end_abs if x > 300) / n * 100
    start_gt_500 = sum(1 for x in start_abs if x > 500) / n * 100
    end_gt_500 = sum(1 for x in end_abs if x > 500) / n * 100

    return {
        "start_mae_ms": round(start_mae, 1),
        "end_mae_ms": round(end_mae, 1),
        "start_mean_error_ms": round(start_mean, 1),
        "end_mean_error_ms": round(end_mean, 1),
        "start_std_ms": round(start_std, 1),
        "end_std_ms": round(end_std, 1),
        "max_start_error_ms": round(max_start_abs, 1),
        "max_end_error_ms": round(max_end_abs, 1),
        "start_gt_300ms_pct": round(start_gt_300, 1),
        "end_gt_300ms_pct": round(end_gt_300, 1),
        "start_gt_500ms_pct": round(start_gt_500, 1),
        "end_gt_500ms_pct": round(end_gt_500, 1),
    }


def compare(
    auto_events: List[TimelineEvent],
    gt_events: List[TimelineEvent],
    auto_file: str = "",
    gt_file: str = "",
    health_score: Optional[float] = None,
) -> ComparisonReport:
    """执行完整的对比分析

    Args:
        auto_events: 自动生成的字幕事件
        gt_events: ground truth 字幕事件
        auto_file: 自动字幕文件路径（用于报告）
        gt_file: ground truth 文件路径（用于报告）
        health_score: 可选声学校验健康度分数

    Returns:
        ComparisonReport 对象
    """
    report = ComparisonReport(
        auto_file=auto_file,
        ground_truth_file=gt_file,
        total_events=len(gt_events),
        matched_events=0,
        health_score=health_score,
    )

    # 匹配事件对
    matches = match_events(auto_events, gt_events)
    report.matched_events = len(matches)

    for auto_ev, gt_ev in matches:
        start_error = (auto_ev.start - gt_ev.start) * 1000  # 转为 ms
        end_error = (auto_ev.end - gt_ev.end) * 1000

        comp = EventComparison(
            index=gt_ev.index,
            auto_start=auto_ev.start,
            auto_end=auto_ev.end,
            gt_start=gt_ev.start,
            gt_end=gt_ev.end,
            auto_text=auto_ev.text,
            gt_text=gt_ev.text,
            start_error_ms=round(start_error, 1),
            end_error_ms=round(end_error, 1),
            start_error_abs_ms=round(abs(start_error), 1),
            end_error_abs_ms=round(abs(end_error), 1),
        )
        report.comparisons.append(comp)

    # 聚合统计
    stats = compute_statistics(report.comparisons)
    for key, value in stats.items():
        setattr(report, key, value)

    # 标记异常事件
    for comp in report.comparisons:
        flags = []
        if comp.end_error_abs_ms > 300:
            flags.append(f"end_error_{comp.end_error_abs_ms:.0f}ms")
        if comp.start_error_abs_ms > 300:
            flags.append(f"start_error_{comp.start_error_abs_ms:.0f}ms")
        if flags:
            report.flagged_events.append({
                "id": comp.index,
                "issues": flags,
                "auto_start": round(comp.auto_start, 2),
                "auto_end": round(comp.auto_end, 2),
                "gt_start": round(comp.gt_start, 2),
                "gt_end": round(comp.gt_end, 2),
                "text_preview": comp.auto_text[:50],
                "start_error_ms": comp.start_error_ms,
                "end_error_ms": comp.end_error_ms,
            })

    return report


# ------------------------------------------------------------------
# 输出
# ------------------------------------------------------------------


def print_report(report: ComparisonReport) -> None:
    """在终端打印对比报告"""
    total = max(report.total_events, 1)
    matched_pct = report.matched_events / total * 100

    print()
    print("=" * 65)
    print("  字幕时间轴精度对比分析报告")
    print("=" * 65)
    print(f"  自动生成: {report.auto_file}")
    print(f"  手动校正: {report.ground_truth_file}")
    print(f"  匹配事件: {report.matched_events}/{report.total_events} ({matched_pct:.0f}%)")
    print()

    # 偏差统计表
    print("  ┌────────────┬──────────┬──────────┐")
    print("  │  指标       │ Start    │ End      │")
    print("  ├────────────┼──────────┼──────────┤")
    print(f"  │ 平均偏差    │ {report.start_mean_error_ms:>6.0f}ms │ {report.end_mean_error_ms:>6.0f}ms │")
    print(f"  │ 平均绝对误差│ {report.start_mae_ms:>6.0f}ms │ {report.end_mae_ms:>6.0f}ms │")
    print(f"  │ 标准差      │ {report.start_std_ms:>6.0f}ms │ {report.end_std_ms:>6.0f}ms │")
    print(f"  │ 最大误差    │ {report.max_start_error_ms:>6.0f}ms │ {report.max_end_error_ms:>6.0f}ms │")
    print(f"  │ >300ms 占比 │ {report.start_gt_300ms_pct:>5.1f}%  │ {report.end_gt_300ms_pct:>5.1f}%  │")
    print(f"  │ >500ms 占比 │ {report.start_gt_500ms_pct:>5.1f}%  │ {report.end_gt_500ms_pct:>5.1f}%  │")
    print("  └────────────┴──────────┴──────────┘")

    if report.health_score is not None:
        score = report.health_score
        color = "绿色" if score >= 90 else ("黄色" if score >= 70 else "红色")
        print(f"  声学校验健康度: {score:.1f}% ({color})")

    print(f"  标记异常事件: {len(report.flagged_events)} 条")
    if report.flagged_events:
        print()
        print("  --- 异常事件列表 ---")
        for f in report.flagged_events[:20]:  # 最多显示20条
            issues_str = ", ".join(f["issues"])
            print(f"  #{f['id']:>3d} [{issues_str}] "
                  f"auto:{f['auto_start']:.2f}-{f['auto_end']:.2f} "
                  f"gt:{f['gt_start']:.2f}-{f['gt_end']:.2f} "
                  f"\"{f['text_preview']}\"")
        if len(report.flagged_events) > 20:
            print(f"  ... 及其他 {len(report.flagged_events) - 20} 条")

    print()
    print("=" * 65)


def report_to_dict(report: ComparisonReport) -> dict:
    """将报告序列化为字典"""
    return {
        "meta": {
            "auto_file": report.auto_file,
            "ground_truth_file": report.ground_truth_file,
            "total_events": report.total_events,
            "matched_events": report.matched_events,
        },
        "statistics": {
            "start": {
                "mean_error_ms": report.start_mean_error_ms,
                "mae_ms": report.start_mae_ms,
                "std_ms": report.start_std_ms,
                "max_error_ms": report.max_start_error_ms,
                "gt_300ms_pct": report.start_gt_300ms_pct,
                "gt_500ms_pct": report.start_gt_500ms_pct,
            },
            "end": {
                "mean_error_ms": report.end_mean_error_ms,
                "mae_ms": report.end_mae_ms,
                "std_ms": report.end_std_ms,
                "max_error_ms": report.max_end_error_ms,
                "gt_300ms_pct": report.end_gt_300ms_pct,
                "gt_500ms_pct": report.end_gt_500ms_pct,
            },
        },
        "health_score": report.health_score,
        "flagged_events": report.flagged_events,
        "detail": [
            {
                "index": c.index,
                "auto": {"start": round(c.auto_start, 3), "end": round(c.auto_end, 3)},
                "ground_truth": {"start": round(c.gt_start, 3), "end": round(c.gt_end, 3)},
                "error_ms": {"start": c.start_error_ms, "end": c.end_error_ms},
                "text": {"auto": c.auto_text, "ground_truth": c.gt_text},
            }
            for c in report.comparisons
        ],
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="字幕时间轴对比分析工具 — 量化自动生成字幕的时间轴精度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    %(prog)s --auto test/subtitle_optimized.ass --ground-truth test/60.ass
    %(prog)s -a out.ass -g gt.ass -o report.json --quiet
    %(prog)s -a out.srt -g gt.srt --health-score 87.5
        """,
    )
    parser.add_argument(
        "-a", "--auto",
        required=True,
        type=Path,
        help="自动生成的字幕文件路径 (ASS/SRT)",
    )
    parser.add_argument(
        "-g", "--ground-truth",
        required=True,
        type=Path,
        help="手动校正的 ground truth 字幕文件路径 (ASS/SRT)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 JSON 报告路径（默认打印到 stdout）",
    )
    parser.add_argument(
        "--health-score",
        type=float,
        default=None,
        help="声学校验健康度分数（可选，0-100）",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安静模式，不打印摘要到终端",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 验证输入文件
    if not args.auto.exists():
        print(f"错误: 自动字幕文件不存在: {args.auto}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth.exists():
        print(f"错误: Ground truth 文件不存在: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    # 解析字幕文件（自动检测 ASS/SRT 格式）
    auto_events = parse_subtitle(args.auto)
    gt_events = parse_subtitle(args.ground_truth)

    if not auto_events:
        print(f"错误: 未从 {args.auto} 解析到任何字幕事件", file=sys.stderr)
        sys.exit(1)
    if not gt_events:
        print(f"错误: 未从 {args.ground_truth} 解析到任何字幕事件", file=sys.stderr)
        sys.exit(1)

    # 执行对比
    report = compare(
        auto_events, gt_events,
        auto_file=str(args.auto),
        gt_file=str(args.ground_truth),
        health_score=args.health_score,
    )

    # 终端输出
    if not args.quiet:
        print_report(report)

    # 保存 JSON 报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = report_to_dict(report)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"详细报告已保存: {output_path}")

    if not args.quiet:
        print()

    # 根据精度返回退出码
    if report.end_mae_ms > 567:  # 当前基线
        sys.exit(2)
    elif report.end_mae_ms > 300:
        sys.exit(1)


if __name__ == "__main__":
    main()
