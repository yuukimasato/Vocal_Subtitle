#!/usr/bin/env python3
"""S1 冒烟:分段 baseline vs 全程识别(global evidence)词级差异对照.

跑一次完整管线(骨架分段主识别 + shadow 证据评审),再用同一管线实例的
global backend 引擎对完整音频做一次整段识别,按规范化字符序列做 diff,
产出定案文档 §4.1 的 M1-M5 指标。

零管线改动:诊断取自 ``stats.quality_diagnostics["evidence_review"]``;
全程识别通过 ``pipeline.get_asr_engine_for()`` 复用管线已加载的引擎实例。

用法:
    .venv-production/bin/python scripts/run_global_evidence_diff_smoke.py \
        --audio test/中文多人员测试音频.wav
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.utils.audio_utils import AudioUtils

# 时间上相邻(≤1.5s)的差异字符聚类为同一个差异区域
_REGION_MERGE_GAP_SECONDS = 1.5
# 参与 diff 的字符:中英数字,丢弃标点/空白
_KEEP_CHAR = re.compile(r"[\w\u4e00-\u9fff]")


def _char_tokens(
    text: str, times: list[float] | None = None
) -> list[tuple[str, float]]:
    """文本 → [(规范化字符, 时间)]。无时间轴时用 -1 占位。"""
    tokens: list[tuple[str, float]] = []
    for index, char in enumerate(text):
        if not _KEEP_CHAR.match(char):
            continue
        moment = times[index] if times and index < len(times) else -1.0
        tokens.append((char.lower(), moment))
    return tokens


def _event_tokens(event: Any) -> list[tuple[str, float]]:
    """字幕事件 → 字符时间序列。词级时间戳相对 event.start(offset_subtitle 契约)。"""
    words = getattr(event, "words", None) or []
    tokens: list[tuple[str, float]] = []
    for word in words:
        chars = [c for c in word.word if not c.isspace()]
        if not chars:
            continue
        span = max(word.end - word.start, 1e-3) / len(chars)
        for offset, char in enumerate(chars):
            if _KEEP_CHAR.match(char):
                tokens.append((char.lower(), event.start + word.start + span * offset))
    if tokens:
        return tokens
    # 兜底:无词级时间戳时在事件区间内线性插值
    text = "".join(c for c in (event.text or "") if not c.isspace())
    span = max(event.end - event.start, 0.2)
    return [
        (char.lower(), event.start + span * i / max(len(text), 1))
        for i, char in enumerate(text)
        if _KEEP_CHAR.match(char)
    ]


def _segment_tokens(segment: Any) -> list[tuple[str, float]]:
    """全局转录段 → 字符时间序列(词时间已是绝对轴)。"""
    tokens: list[tuple[str, float]] = []
    for word in segment.words or []:
        chars = [c for c in word.word if not c.isspace()]
        if not chars:
            continue
        span = max(word.end - word.start, 1e-3) / len(chars)
        for offset, char in enumerate(chars):
            if _KEEP_CHAR.match(char):
                tokens.append((char.lower(), word.start + span * offset))
    if tokens:
        return tokens
    return _char_tokens(segment.text or [])


def _cluster_regions(
    tokens: list[tuple[str, float]], other_events: list[Any]
) -> list[dict[str, Any]]:
    """把差异字符按时间相邻性聚类为差异区域,并标记是否与对方事件重叠。"""
    regions: list[dict[str, Any]] = []
    for char, moment in tokens:
        if (
            regions
            and moment >= 0
            and moment - regions[-1]["end"] <= _REGION_MERGE_GAP_SECONDS
        ):
            regions[-1]["text"] += char
            regions[-1]["end"] = moment
            continue
        regions.append({"text": char, "start": moment, "end": moment})
    for region in regions:
        region["overlaps_baseline_events"] = any(
            event.start < region["end"] and event.end > region["start"]
            for event in other_events
            if region["start"] >= 0
        )
    return regions


def _timing_deltas(
    baseline: list[tuple[str, float]], global_: list[tuple[str, float]], matcher
) -> list[float]:
    """相等块内采样双方向字符映射的时间差(global - baseline)。"""
    deltas: list[float] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        step = max(1, (i2 - i1) // 20)
        for i, j in zip(range(i1, i2, step), range(j1, j2, step)):
            t_base, t_glob = baseline[i][1], global_[j][1]
            if t_base >= 0 and t_glob >= 0:
                deltas.append(t_glob - t_base)
    return deltas


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * pct / 100))], 3)


def _global_hallucination_metrics(segments: list[Any]) -> dict[str, Any]:
    """M4:全程识别自身的幻觉指标触发(阈值与 risk_scoring.py 一致)。"""
    return {
        "segment_count": len(segments),
        "high_no_speech_prob": sum(1 for s in segments if s.no_speech_prob >= 0.6),
        "low_avg_logprob": sum(1 for s in segments if s.avg_logprob < -1.0),
        "high_compression_ratio": sum(
            1 for s in segments if s.compression_ratio >= 2.4
        ),
    }


def _evidence_review_summary(diag: dict[str, Any]) -> dict[str, Any]:
    """M3:evidence 评审自身的决定/风险聚计(防御式读取)。"""
    decisions = diag.get("decisions") or []
    actions: dict[str, int] = {}
    for decision in decisions:
        action = str(decision.get("decision", "unknown"))
        actions[action] = actions.get(action, 0) + 1
    codes: dict[str, int] = {}
    for risk in diag.get("risk") or []:
        for code in risk.get("evidence_codes") or []:
            codes[str(code)] = codes.get(str(code), 0) + 1
    global_diag = diag.get("global_evidence") or {}
    return {
        "decision_count": len(decisions),
        "actions": actions,
        "risk_code_histogram": codes,
        "global_evidence_counts": {
            key: global_diag.get(key)
            for key in (
                "accepted_count",
                "rejected_count",
                "considered_count",
                "considered_alternative_count",
                "considered_overlap_count",
            )
            if key in global_diag
        },
        "review_policy": diag.get("review_policy"),
        "shadow_mode": diag.get("shadow_mode"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="已分离人声的 WAV(可 --skip-separation 语义,本脚本固定跳过分离)",
    )
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".scratch/evidence_diff"
    )
    parser.add_argument(
        "--language", default=None, help="覆盖语言(默认沿用 profile 配置,空则自动检测)"
    )
    args = parser.parse_args(argv)

    audio_path = args.audio if args.audio.is_absolute() else ROOT / args.audio
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_srt = output_dir / f"{audio_path.stem}.baseline.srt"

    print(f"[1/5] 加载音频: {audio_path}", flush=True)
    audio, sample_rate = AudioUtils.load_audio(audio_path)
    print(f"      时长 {len(audio) / sample_rate:.1f}s @ {sample_rate}Hz", flush=True)

    print(f"[2/5] 加载 profile: {args.profile}", flush=True)
    config = ConfigLoader().load_profile(args.profile)
    if args.language:
        config.asr.language = args.language

    print("[3/5] 跑完整管线(骨架分段主识别 + shadow 证据评审)...", flush=True)
    started = time.time()
    pipeline = Pipeline(config)
    result = pipeline.run(
        input_path=audio_path,
        output_path=output_srt,
        output_format="srt",
        skip_separation=True,
    )
    stats = result["stats"]
    events = result["events"]
    print(
        f"      完成: status={stats.status}, {len(events)} 条字幕, 耗时 {time.time() - started:.1f}s",
        flush=True,
    )

    backend = "faster-whisper"
    global_cfg = getattr(config.asr, "global_asr", None)
    if global_cfg is not None and getattr(global_cfg, "backend", None):
        backend = global_cfg.backend
    print(f"[4/5] 全程识别(global backend={backend},整段一次)...", flush=True)
    started = time.time()
    # 与 asr_path 的 global evidence 同一取引擎入口(_get_asr_engine_for),复用管线已加载实例
    engine = pipeline._get_asr_engine_for(backend)
    language = args.language or getattr(config.asr, "language", None) or None
    global_segments = engine.transcribe(audio, sample_rate, language=language)
    print(
        f"      完成: {len(global_segments)} 段, 耗时 {time.time() - started:.1f}s",
        flush=True,
    )

    print("[5/5] 词级 diff 与指标汇总...", flush=True)
    baseline_tokens: list[tuple[str, float]] = []
    for event in events:
        baseline_tokens.extend(_event_tokens(event))
    global_tokens: list[tuple[str, float]] = []
    for segment in global_segments:
        global_tokens.extend(_segment_tokens(segment))

    matcher = SequenceMatcher(
        a=[c for c, _ in baseline_tokens],
        b=[c for c, _ in global_tokens],
        autojunk=False,
    )
    baseline_only: list[tuple[str, float]] = []
    global_only: list[tuple[str, float]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            baseline_only.extend(baseline_tokens[i1:i2])
        elif tag == "insert":
            global_only.extend(global_tokens[j1:j2])

    m1_regions = _cluster_regions(global_only, events)
    m2_regions = _cluster_regions(baseline_only, events)
    deltas = _timing_deltas(baseline_tokens, global_tokens, matcher)
    m5 = {
        "sample_count": len(deltas),
        "mean_abs_seconds": round(statistics.mean(abs(d) for d in deltas), 3)
        if deltas
        else None,
        "median_seconds": statistics.median(deltas) if deltas else None,
        "p90_abs_seconds": _percentile([abs(d) for d in deltas], 90),
        "max_abs_seconds": max((abs(d) for d in deltas), default=None),
    }

    payload = {
        "audio": str(audio_path),
        "audio_duration_seconds": round(len(audio) / sample_rate, 2),
        "pipeline_status": stats.status,
        "baseline_event_count": len(events),
        "global_segment_count": len(global_segments),
        "baseline_char_count": len(baseline_tokens),
        "global_char_count": len(global_tokens),
        "M1_global_only_regions": m1_regions,
        "M2_baseline_only_regions": m2_regions,
        "M3_evidence_review": _evidence_review_summary(
            stats.quality_diagnostics.get("evidence_review", {})
        ),
        "M4_global_hallucination": _global_hallucination_metrics(global_segments),
        "M5_timing_delta_seconds": m5,
        "coverage_audit": stats.global_diagnostics.get("physical_coverage", {}),
    }
    report_path = output_dir / f"{audio_path.stem}.evidence_diff.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n===== 冒烟结果: {audio_path.name} =====")
    print(
        f"基线字幕 {len(events)} 条 / {len(baseline_tokens)} 字符;全程识别 {len(global_segments)} 段 / {len(global_tokens)} 字符"
    )
    print(
        f"M1 全程独有区域: {len(m1_regions)} 处(其中与基线事件时间重叠 {sum(1 for r in m1_regions if r['overlaps_baseline_events'])} 处)"
    )
    for region in m1_regions[:10]:
        tag = "重叠" if region["overlaps_baseline_events"] else "空白"
        print(
            f"    [{tag}] {region['start']:.2f}-{region['end']:.2f}s: {region['text']}"
        )
    print(
        f"M2 分段独有区域: {len(m2_regions)} 处(其中与全程识别时间重叠 {sum(1 for r in m2_regions if r['overlaps_baseline_events'])} 处)"
    )
    for region in m2_regions[:10]:
        print(f"    {region['start']:.2f}-{region['end']:.2f}s: {region['text']}")
    print(
        f"M3 evidence 评审: decisions={payload['M3_evidence_review']['decision_count']}, actions={payload['M3_evidence_review']['actions']}"
    )
    print(f"     风险码直方图: {payload['M3_evidence_review']['risk_code_histogram']}")
    print(f"M4 全程识别幻觉触发: {payload['M4_global_hallucination']}")
    print(f"M5 时间轴偏差(global-baseline): {m5}")
    print(f"\n报告: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
