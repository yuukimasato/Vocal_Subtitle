"""编辑日志（edit-journal-v1）摄取与 V1 统计。

对应 docs/全栈架构定型与审核学习整合方案-2026-09-10.md §3.2「学习侧摄取」：
  解析 → schema 校验 → (session_id, seq) 去重 → 与原始字幕对齐重放校验
  → 事件级统计（V1 报告）→ D2 候选样本入库 → 编辑器维度偏好写入 user_profile。

journal 事件本身即过程对齐，不依赖 DTW（§4）；重放还原「原始字幕 + 日志 = 最终字幕」，
同时是校验工具与 V3 训练的 (state, action) 序列来源。
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..mapping.time_mapper import SubtitleEvent
from .aligner import parse_subtitle_file

logger = logging.getLogger(__name__)

JOURNAL_SCHEMA = "edit-journal-v1"

# 事件必填字段（session_id 为契约字段；早期编辑器导出用 session，读取时双读）
_EVENT_REQUIRED_FIELDS = {"schema", "type", "seq", "command"}

# 重放时 id 与原始事件的时间匹配容差（秒）：cue 初值与原事件 start 的最大距离
_MATCH_TOLERANCE_SECONDS = 1.0

# 偏好判定阈值：时间偏移中位数超过此值（秒）才视为系统性偏移
_SYSTEMATIC_SHIFT_THRESHOLD = 0.03


class JournalIngestError(ValueError):
    """日志文件解析或校验失败"""


def event_session_id(record: dict[str, Any]) -> str | None:
    """事件/头部的会话 ID：session_id 为契约字段，session 为早期字段名（双读）"""
    value = record.get("session_id") or record.get("session")
    return str(value) if value else None


@dataclass
class JournalFile:
    """一份 NDJSON 日志的解析结果"""

    path: Path
    header: dict[str, Any] | None
    events: list[dict[str, Any]]
    skipped: int = 0  # 被拒绝的行数


@dataclass
class ReplayResult:
    """重放还原结果"""

    final_events: list[SubtitleEvent]
    matched_ids: int  # 日志 id 与原始事件成功对应的数量
    unmatched_ids: int  # 未能对应（新增行等）
    exact: bool  # 重放是否「精确」（全部修改行都能对应回原始事件）


@dataclass
class JournalStats:
    """V1 统计报告（均值/中位数/偏好画像的原料）"""

    event_count: int = 0
    session_count: int = 0
    actors: dict[str, int] = field(default_factory=dict)
    commands: dict[str, int] = field(default_factory=dict)
    structural: dict[str, int] = field(
        default_factory=dict
    )  # remove/split/merge/insert 计数
    start_delta_median: float | None = None
    start_delta_mean: float | None = None
    end_delta_median: float | None = None
    end_delta_mean: float | None = None
    gap_after_median: float | None = None
    cps_p90: float | None = None
    boundary_snap_rate: float | None = None  # 终点微调贴近最近语音边界的比例
    provenance_coverage: float = 0.0  # 带 manifest 出处的事件占比
    by_scenario: dict[str, JournalStats] = field(
        default_factory=dict
    )  # 按场景分层（D27）

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "session_count": self.session_count,
            "actors": self.actors,
            "commands": self.commands,
            "structural": self.structural,
            "start_delta_median": self.start_delta_median,
            "start_delta_mean": self.start_delta_mean,
            "end_delta_median": self.end_delta_median,
            "end_delta_mean": self.end_delta_mean,
            "gap_after_median": self.gap_after_median,
            "cps_p90": self.cps_p90,
            "boundary_snap_rate": self.boundary_snap_rate,
            "provenance_coverage": self.provenance_coverage,
            "by_scenario": {
                name: sub.to_dict() for name, sub in self.by_scenario.items()
            },
        }


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------


def parse_journal_text(
    text: str, source: Path | None = None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    """解析 NDJSON 文本 → (header, events, rejected_lines)。

    契约：首行 session header，其后每行一个事件；只追加可选字段，
    未知字段保留原样（向前兼容）。
    """
    header: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    rejected = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Journal %s line %d is not valid JSON: %s", source or "?", line_no, exc
            )
            rejected += 1
            continue
        if not isinstance(record, dict) or record.get("schema") != JOURNAL_SCHEMA:
            rejected += 1
            continue
        if record.get("type") == "header":
            header = record
            continue
        if (
            record.get("type") == "event"
            and _EVENT_REQUIRED_FIELDS <= set(record)
            and event_session_id(record)
        ):
            events.append(record)
            continue
        rejected += 1
    return header, events, rejected


def load_journal_files(paths: Iterable[Path | str]) -> list[JournalFile]:
    """读取并校验多份日志文件；跨文件按 (session_id, seq) 去重（重复导出幂等）。"""
    files: list[JournalFile] = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise JournalIngestError(f"Journal file not found: {path}")
        text = path.read_text(encoding="utf-8")
        header, events, rejected = parse_journal_text(text, source=path)
        deduped: list[dict[str, Any]] = []
        for event in events:
            key = (
                event_session_id(event) or "unknown-session",
                int(event.get("seq", -1)),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)
        files.append(
            JournalFile(path=path, header=header, events=deduped, skipped=rejected)
        )
    return files


# ---------------------------------------------------------------------------
# 重放（原始字幕 + 日志 = 最终字幕）
# ---------------------------------------------------------------------------


def _cue_state_from_diff(diff: list[dict[str, Any]]) -> None:
    """规范化 diff 条目（就地）：确保 modify.changes 存在"""
    for entry in diff:
        if entry.get("op") == "modify" and not isinstance(entry.get("changes"), list):
            entry["changes"] = []


def replay_journal(
    original_events: list[SubtitleEvent], events: list[dict[str, Any]]
) -> ReplayResult:
    """按事件序重放日志，还原最终字幕。

    id ↔ 原始事件对应：编辑器加载字幕时 cue 初值即原事件；首个 modify 的 before 值
    即原事件字段值，按 start 最近匹配回原事件。未被触碰的行原样保留，
    新增行按快照插入，删除行剔除。返回按 (start, end) 稳定排序的事件列表。
    """
    # id → 最终状态与首次出现时的原值（before）
    final_state: dict[str, dict[str, Any]] = {}
    initial_state: dict[str, dict[str, Any]] = {}
    first_change_index: dict[str, int] = {}
    removed: set[str] = set()
    added: dict[str, dict[str, Any]] = {}
    add_order: list[str] = []

    for seq_no, event in enumerate(events):
        diff = event.get("diff") or []
        _cue_state_from_diff(diff)
        for entry in diff:
            op = entry.get("op")
            cue_id = str(entry.get("id"))
            if op == "modify":
                state = final_state.setdefault(cue_id, {})
                initial = initial_state.setdefault(cue_id, {})
                first_change_index.setdefault(cue_id, seq_no)
                for change in entry["changes"]:
                    field_name = change.get("field")
                    if field_name in ("start", "end", "text"):
                        initial.setdefault(field_name, change.get("before"))
                        state[field_name] = change.get("after")
            elif op == "remove":
                snapshot = entry.get("cue") or {}
                initial = initial_state.setdefault(cue_id, {})
                for field_name in ("start", "end", "text"):
                    initial.setdefault(field_name, snapshot.get(field_name))
                final_state.setdefault(cue_id, {})
                removed.add(cue_id)
            elif op == "add":
                snapshot = entry.get("cue") or {}
                final_state[cue_id] = {
                    "start": snapshot.get("start"),
                    "end": snapshot.get("end"),
                    "text": snapshot.get("text", ""),
                }
                added[cue_id] = final_state[cue_id]
                if cue_id not in add_order:
                    add_order.append(cue_id)

    # id → 原始事件：按初值最近匹配（start 优先，其次 end，最后文本），容差内唯一占用
    by_id: dict[str, SubtitleEvent] = {}
    taken: set[int] = set()

    def _nearest(key_fn, value):
        best_index = -1
        best_distance = _MATCH_TOLERANCE_SECONDS
        for index, original in enumerate(original_events):
            if index in taken:
                continue
            distance = abs(key_fn(original) - float(value))
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    candidates = [
        (cue_id, initial)
        for cue_id, initial in initial_state.items()
        if cue_id not in added
    ]
    # Pass 1: 有初值 start 的 id 按 start 匹配
    for cue_id, initial in sorted(
        (item for item in candidates if item[1].get("start") is not None),
        key=lambda item: float(item[1]["start"]),
    ):
        index = _nearest(lambda e: e.start, initial["start"])
        if index == -1:
            continue
        taken.add(index)
        by_id[cue_id] = original_events[index]
    # Pass 2: 只有初值 end 的 id（首改只动 end，如合并句尾）按 end 匹配
    for cue_id, initial in sorted(
        (
            item
            for item in candidates
            if item[1].get("start") is None and item[1].get("end") is not None
        ),
        key=lambda item: float(item[1]["end"]),
    ):
        index = _nearest(lambda e: e.end, initial["end"])
        if index == -1:
            continue
        taken.add(index)
        by_id[cue_id] = original_events[index]
    # Pass 3: 只有初值文本的 id 按文本精确匹配
    for cue_id, initial in candidates:
        if cue_id in by_id or initial.get("text") is None:
            continue
        for index, original in enumerate(original_events):
            if index in taken:
                continue
            if original.text == initial["text"]:
                taken.add(index)
                by_id[cue_id] = original_events[index]
                break
    unmatched_ids = len(candidates) - len(by_id)

    # 组装最终事件列表：原事件 − 删除 + 重放终态 + 新增
    owner_by_object: dict[int, str] = {id(cue): cid for cid, cue in by_id.items()}
    rebuilt: list[SubtitleEvent] = []
    for original in original_events:
        owner_id = owner_by_object.get(id(original))
        if owner_id is None:
            rebuilt.append(original)
            continue
        if owner_id in removed:
            continue
        state = final_state.get(owner_id, {})
        rebuilt.append(
            SubtitleEvent(
                index=original.index,
                start=float(state.get("start", original.start)),
                end=float(state.get("end", original.end)),
                text=str(state.get("text", original.text)),
                original_text=original.original_text,
                speaker_id=original.speaker_id,
                speaker_label=original.speaker_label,
            )
        )
    for cue_id in add_order:
        if cue_id in removed:
            continue
        state = added[cue_id]
        rebuilt.append(
            SubtitleEvent(
                index=0,
                start=float(state["start"]),
                end=float(state["end"]),
                text=str(state.get("text", "")),
            )
        )

    rebuilt.sort(key=lambda e: (e.start, e.end))
    for new_index, event in enumerate(rebuilt, 1):
        event.index = new_index

    modified_ids = [cid for cid in first_change_index if cid not in added]
    return ReplayResult(
        final_events=rebuilt,
        matched_ids=len(by_id),
        unmatched_ids=len(added) + unmatched_ids,
        exact=len(by_id) >= len(modified_ids),
    )


# ---------------------------------------------------------------------------
# V1 统计
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


_STRUCTURAL_COMMANDS = {
    "removeCues": "remove",
    "removeCue": "remove",
    "splitCue": "split",
    "mergeWithNext": "merge",
    "insertAtTime": "insert",
    "insertRelativeTo": "insert",
    "pasteCues": "insert",
    "duplicateCues": "insert",
}


def journal_scenario(journal: JournalFile) -> str:
    """日志会话的场景标签（D27）：header 可选 scenario 字段，旧日志无此字段返回空串"""
    header = journal.header or {}
    return str(header.get("scenario") or "")


def _aggregate_journal_stats(files: list[JournalFile]) -> JournalStats:
    """聚合单层统计：命令分布、时间偏移、留白、CPS、边界吸附、出处覆盖。"""
    stats = JournalStats(session_count=len(files))
    start_deltas: list[float] = []
    end_deltas: list[float] = []
    gap_afters: list[float] = []
    cps_values: list[float] = []
    boundary_hits = 0
    boundary_total = 0
    with_provenance = 0

    for journal in files:
        for event in journal.events:
            stats.event_count += 1
            actor = str(event.get("actor") or "human")
            stats.actors[actor] = stats.actors.get(actor, 0) + 1
            command = str(event.get("command") or "unknown")
            stats.commands[command] = stats.commands.get(command, 0) + 1
            structural = _STRUCTURAL_COMMANDS.get(command)
            if structural:
                stats.structural[structural] = stats.structural.get(structural, 0) + 1
            if event.get("provenance") or (event.get("context") or {}).get(
                "provenance"
            ):
                with_provenance += 1

            context = event.get("context") or {}
            cue_context = context.get("cue") or {}
            if cue_context.get("gap_after") is not None:
                gap_afters.append(float(cue_context["gap_after"]))
            if cue_context.get("cps") is not None:
                cps_values.append(float(cue_context["cps"]))

            audio = context.get("audio") or {}
            boundary = audio.get("nearest_speech_boundary")
            if boundary and boundary.get("dist") is not None:
                distance = abs(float(boundary["dist"]))
                boundary_total += 1
                if distance <= 0.05:
                    boundary_hits += 1

            for entry in event.get("diff") or []:
                if entry.get("op") != "modify":
                    continue
                for change in entry.get("changes") or []:
                    if change.get("before") is None or change.get("after") is None:
                        continue
                    if change.get("field") == "start":
                        start_deltas.append(
                            float(change["after"]) - float(change["before"])
                        )
                    elif change.get("field") == "end":
                        end_deltas.append(
                            float(change["after"]) - float(change["before"])
                        )

    stats.start_delta_median = _median(start_deltas)
    stats.start_delta_mean = (
        sum(start_deltas) / len(start_deltas) if start_deltas else None
    )
    stats.end_delta_median = _median(end_deltas)
    stats.end_delta_mean = sum(end_deltas) / len(end_deltas) if end_deltas else None
    stats.gap_after_median = _median(gap_afters)
    stats.cps_p90 = _percentile(cps_values, 0.9)
    stats.boundary_snap_rate = (
        (boundary_hits / boundary_total) if boundary_total else None
    )
    stats.provenance_coverage = (
        with_provenance / stats.event_count if stats.event_count else 0.0
    )
    return stats


def journal_statistics(files: list[JournalFile]) -> JournalStats:
    """汇总 V1 统计：命令分布、时间偏移、留白、CPS、边界吸附、出处覆盖。

    header 携带可选 scenario（D27）时按场景分层（by_scenario）；
    无该字段的旧日志照常计入总体统计，不产生场景分层条目。
    """
    stats = _aggregate_journal_stats(files)
    grouped: dict[str, list[JournalFile]] = {}
    for journal in files:
        scenario = journal_scenario(journal)
        if scenario:
            grouped.setdefault(scenario, []).append(journal)
    for scenario, scenario_files in sorted(grouped.items()):
        stats.by_scenario[scenario] = _aggregate_journal_stats(scenario_files)
    return stats


def derive_editor_preferences(stats: JournalStats) -> dict[str, float]:
    """从统计中推导编辑器维度偏好（写入 user_profile，独立于管线参数）。

    只在样本量足以体现系统性偏移时产出（|中位数| 超阈值）。
    """
    preferences: dict[str, float] = {}
    if (
        stats.start_delta_median is not None
        and abs(stats.start_delta_median) >= _SYSTEMATIC_SHIFT_THRESHOLD
    ):
        preferences["editor.offset_start_ms"] = round(
            stats.start_delta_median * 1000, 1
        )
    if (
        stats.end_delta_median is not None
        and abs(stats.end_delta_median) >= _SYSTEMATIC_SHIFT_THRESHOLD
    ):
        preferences["editor.offset_end_ms"] = round(stats.end_delta_median * 1000, 1)
    if stats.gap_after_median is not None:
        preferences["editor.gap_preference_ms"] = round(
            stats.gap_after_median * 1000, 1
        )
    if stats.cps_p90 is not None:
        preferences["editor.cps_ceiling"] = round(stats.cps_p90, 2)
    if stats.boundary_snap_rate is not None and stats.boundary_snap_rate >= 0.5:
        preferences["editor.boundary_snap_to_speech"] = 1.0
    return preferences


# ---------------------------------------------------------------------------
# V3 触发机制（D16：只定机制，不定数值；阈值可配置，未配置则不提示）
# ---------------------------------------------------------------------------


def check_v3_trigger(
    sample_count: int,
    coverage: float,
    conflict_rate: float,
    *,
    min_samples: int | None = None,
    min_coverage: float | None = None,
    max_conflict_rate: float | None = None,
) -> tuple[bool, str]:
    """判断数据是否达到训练轻量序列模型（V3）的门槛。

    任一阈值未配置（None）即跳过该项检查；全部未配置则不触发。
    """
    checks: list[tuple[bool, str]] = []
    if min_samples is not None:
        checks.append(
            (sample_count >= min_samples, f"样本数 {sample_count}/{min_samples}")
        )
    if min_coverage is not None:
        checks.append(
            (coverage >= min_coverage, f"覆盖率 {coverage:.2f}/{min_coverage:.2f}")
        )
    if max_conflict_rate is not None:
        checks.append(
            (
                conflict_rate <= max_conflict_rate,
                f"冲突率 {conflict_rate:.2f}/{max_conflict_rate:.2f}",
            )
        )
    if not checks:
        return False, "V3 触发阈值未配置（feedback.v3_trigger_*），跳过检查"
    ok = all(passed for passed, _ in checks)
    detail = "；".join(detail for _, detail in checks)
    if ok:
        return True, f"数据达标，可训练轻量序列模型（{detail}）"
    return False, f"尚未达标（{detail}）"


def journal_original_stem(journal_path: Path | str) -> str:
    """日志文件对应的原始字幕名（<字幕同名>.journal.jsonl → <字幕同名>）"""
    name = Path(journal_path).name
    return re.sub(r"\.journal\.jsonl$", "", name, flags=re.IGNORECASE)


def find_original_subtitle(journal_path: Path | str) -> Path | None:
    """在日志同目录寻找原始字幕文件（与导出时搭车的 <同名>.journal.jsonl 配对）"""
    path = Path(journal_path)
    stem = journal_original_stem(path)
    for suffix in (".srt", ".ass", ".vtt"):
        candidate = path.parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_original_events(subtitle_path: Path | str) -> list[SubtitleEvent]:
    return parse_subtitle_file(Path(subtitle_path))


def journal_to_text_sample(
    auto_events: list[SubtitleEvent], final_events: list[SubtitleEvent]
) -> tuple[str, str]:
    """事件列表 → D2 入库用的 SRT 风格文本（复用 learn 命令的格式约定）"""

    def to_text(events: list[SubtitleEvent]) -> str:
        lines = []
        for index, event in enumerate(events, 1):
            lines.append(
                f"{index}\n{event.start:.3f} --> {event.end:.3f}\n{event.text}\n"
            )
        return "\n".join(lines)

    return to_text(auto_events), to_text(final_events)


def summarize_commands(files: list[JournalFile]) -> Counter:
    counter: Counter = Counter()
    for journal in files:
        counter.update(
            str(event.get("command") or "unknown") for event in journal.events
        )
    return counter


__all__ = [
    "JOURNAL_SCHEMA",
    "JournalFile",
    "JournalIngestError",
    "JournalStats",
    "ReplayResult",
    "check_v3_trigger",
    "derive_editor_preferences",
    "event_session_id",
    "find_original_subtitle",
    "journal_original_stem",
    "journal_scenario",
    "journal_statistics",
    "journal_to_text_sample",
    "load_journal_files",
    "load_original_events",
    "parse_journal_text",
    "replay_journal",
    "summarize_commands",
]
