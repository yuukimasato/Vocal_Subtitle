"""[层1] 说话人身份主干:全局 turns 前置与事件词级后切分。

设计文档:docs/说话人身份主干与时间轴仲裁定案-2026-09-11.md §4.2 / §7(P1/P2)。

核心解耦原则(同 ``group_speech_intervals`` + ``member_projection`` 模式):
**ASR 窗口 ≠ 字幕事件。** 音频窗口保持物理连续、上下文完整(决策 D3);
字幕事件在识别完成后按 turn 翻转点做词级后切分,标签先天正确。

- :func:`run_early_global_pass` — 分离之后对完整人声音频跑一次全局
  diarization(复用 ``speaker_fusion._run_global_pass`` 的引擎加载与
  模型选择逻辑,模型缓存沿用现有机制),turns 归一到全局时间轴。
- :class:`EarlyTurnsState` — 全局 pass 结果,由管线保存在
  ``self._early_turns_state``,贯通 chunk_runner(tiny-fragment 合并
  硬约束)与 postprocess_runner(事件标签来源)。
- :func:`assign_event_speakers` — early_turns 成功时的事件标签注入
  (D5:事件级聚类退役);``word_split`` 开启时按 turn 翻转点吸附
  最近词间隙做词级切分,重叠区诚实标注 overlapped。

所有行为由配置开关门控(diarization.early_turns / word_split_on_turn /
single_speaker_shortcut);early_turns=false 时本模块全部调用立即返回
disabled,全链路行为与现状一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import SpeakerTurn
from .speaker_fusion import _speaker_label, _split_event
from .turn_reconciler import (
    merge_same_speaker_spans,
    normalize_turns,
    reconcile_regions,
    split_event_intervals,
)

logger = logging.getLogger(__name__)


@dataclass
class EarlyTurnsState:
    """early_turns 全局 pass 的结果与状态。

    Attributes:
        turns: 归一到全局时间轴的 speaker turns(已排序、已裁剪)。
        speaker_count: turns 中的说话人数。
        backend: 诊断用后端标识(成功时 "pyannote")。
        status: "ok" | "unavailable" | "failed" | "disabled"。
        model_ref: 全局模型引用(诊断/溯源)。
        attempted: 是否真正运行过全局 pass(区别于开关直接关闭)。
        single_speaker: 全局 pass 成功且说话人 ≤1(TTS/口播短路场景)。
        diagnostics: 附加诊断(global_status / expected_speakers 等)。
    """

    turns: List[SpeakerTurn] = field(default_factory=list)
    speaker_count: int = 0
    backend: str = "unknown"
    status: str = "disabled"
    model_ref: str = ""
    attempted: bool = False
    single_speaker: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """全局 turns 可用作事件标签来源(early_turns 成功)。"""
        return self.status == "ok"


def run_early_global_pass(
    audio: Any,
    sample_rate: int,
    config: Any,
    *,
    duration: Optional[float] = None,
) -> EarlyTurnsState:
    """[P1] 分离之后对完整人声音频跑一次全局 diarization。

    复用 ``speaker_fusion._run_global_pass``:后端/范围门控、模型选择
    (auto → community-1 → diarization-3.1)与 pyannote 引擎加载全部
    沿用现有逻辑。pyannote 不可用时返回非 ok 状态(日志记录原因),
    后处理自动回退到事件级聚类 —— 降级链 global→embedding→MFCC→unknown
    保持不变。
    """
    diar_cfg = getattr(config, "diarization", None)
    if diar_cfg is None or not getattr(diar_cfg, "enabled", True):
        return EarlyTurnsState(
            status="disabled",
            diagnostics={"early_turns_status": "disabled", "reason": "diarization_disabled"},
        )
    if not getattr(diar_cfg, "early_turns", False):
        return EarlyTurnsState(
            status="disabled",
            diagnostics={"early_turns_status": "disabled", "reason": "early_turns_disabled"},
        )

    from .speaker_fusion import _run_global_pass

    result, model_ref, global_status = _run_global_pass(audio, sample_rate, config)
    # 优先取 regular turns（携带 pyannote 的 overlapped 标记并覆盖两人
    # 重叠区）；后处理 fusion 用 exclusive_turns 做事件中点归属，而层1
    # 需要把重叠区诚实标注为 overlapped，不能提前丢弃重叠段。
    raw_turns = (
        list(result.turns or result.exclusive_turns) if result is not None else []
    )
    turns = normalize_turns(raw_turns, duration=duration)
    speakers = sorted({turn.speaker_id for turn in turns})
    state = EarlyTurnsState(
        turns=turns,
        speaker_count=len(speakers),
        backend="pyannote" if result is not None else "unknown",
        status="ok" if result is not None else global_status,
        model_ref=model_ref,
        attempted=True,
        single_speaker=result is not None and len(speakers) <= 1,
        diagnostics={
            "global_model": model_ref,
            "global_status": global_status,
            "global_turn_count": len(turns),
            "expected_speakers": getattr(diar_cfg, "expected_speakers", None),
        },
    )
    if result is None:
        # pyannote 不可用/被禁用:early_turns 自动回退,记录原因,
        # 后处理的事件级聚类照常运行(现状行为)。
        logger.info(
            "Early turns pass unavailable (status=%s, model=%s) — "
            "postprocess event-level fusion stays authoritative",
            global_status, model_ref or "none",
        )
    else:
        logger.info(
            "Early turns pass: %d turns / %d speakers (model=%s, single=%s)",
            len(turns), len(speakers), model_ref, state.single_speaker,
        )
    return state


def spans_from_skeleton(
    skeleton_intervals: Sequence,
    turns: Sequence[SpeakerTurn],
    *,
    duration: Optional[float] = None,
    boundary_collar_ms: float = 80.0,
) -> List[Any]:
    """[P1] 骨架物理区间 × 全局 turns 求交,再做同说话人合并。

    物理区间只提供时间信息;speaker identity 一律来自 turns(停顿不
    推断身份)。相邻且同 speaker 的跨度按 max_gap=0.12s 合并,不同
    speaker 永不合并(见 turn_reconciler.merge_same_speaker_spans)。
    """
    spans = reconcile_regions(
        skeleton_intervals,
        turns,
        physical_source="skeleton",
        boundary_collar_ms=boundary_collar_ms,
        duration=duration,
    )
    return merge_same_speaker_spans(spans)


def dominant_turn_at(
    turns: Sequence[SpeakerTurn],
    start: float,
    end: float,
) -> Optional[SpeakerTurn]:
    """返回与区间 [start, end] 重叠时间最长的 turn;无覆盖为 None。"""
    best: Optional[SpeakerTurn] = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = turn, overlap
    return best


def dominant_speaker_at(
    turns: Sequence[SpeakerTurn],
    start: float,
    end: float,
) -> Optional[int]:
    """区间的主说话人(与区间重叠最长的 turn);无覆盖为 None。"""
    turn = dominant_turn_at(turns, start, end)
    return turn.speaker_id if turn is not None else None


def dominant_span_speaker(
    spans: Sequence,
    start: float,
    end: float,
) -> Optional[int]:
    """区间的主说话人(按 AtomicSpeechSpan 跨度求主覆盖)。

    只在已知身份的跨度中取主覆盖;仅 unknown 跨度覆盖时返回 None
    (无信息,调用方按"安全合并"处理)。两人重叠跨度不参与归属。
    """
    best_id: Optional[int] = None
    best_overlap = 0.0
    for span in spans:
        if span.speaker_id is None:
            continue
        overlap = min(end, span.end) - max(start, span.start)
        if overlap > best_overlap:
            best_id, best_overlap = span.speaker_id, overlap
    return best_id


def _turn_at(turns: Sequence[SpeakerTurn], point: float) -> Optional[SpeakerTurn]:
    return next(
        (turn for turn in turns if turn.start <= point < turn.end),
        None,
    )


def _apply_identity(
    event: Any,
    speaker_id: Optional[int],
    turn: Optional[SpeakerTurn],
    model_ref: str,
) -> bool:
    """把单个 turn 的身份写入事件(字段契约与 run_speaker_fusion 一致)。

    返回是否标记了两人重叠。标签在 ID 压缩后统一生成。
    """
    event.speaker_id = int(speaker_id) if speaker_id is not None else None
    event.speaker_source = "global" if speaker_id is not None else "unknown"
    event.speaker_status = "confirmed" if speaker_id is not None else "unknown"
    event.speaker_repair_reason = (
        "" if speaker_id is not None else "speaker_evidence_unavailable"
    )
    event.speaker_model = model_ref or None
    # 引擎目前没有校准过的后验概率,保留 None 而不是伪造数值置信度。
    event.speaker_confidence = None
    if turn is not None and getattr(turn, "overlapped", False):
        # 两人重叠区:诚实标注,不强判归属,进复核队列。
        event.genuine_overlap = True
        return True
    return False


def _clamp_event_bounds(event: Any, start: float, end: float) -> None:
    """无词级时间戳 fallback:把事件范围钳制到第一个说话人片段。"""
    event.start = max(float(event.start), start)
    event.end = min(float(event.end), end)
    if getattr(event, "physical_start", None) is not None:
        event.physical_start = max(event.physical_start, start)
    if getattr(event, "physical_end", None) is not None:
        event.physical_end = min(event.physical_end, end)


def assign_event_speakers(
    events: List[Any],
    turns: Sequence[SpeakerTurn],
    *,
    word_split: bool = False,
    min_part_duration: float = 0.0,
    language: Optional[str] = None,
    model_ref: str = "",
) -> Tuple[List[Any], Dict[str, Any]]:
    """[P1/P2] early_turns 成功时的事件说话人注入(标签先天正确)。

    - 单一说话人覆盖(常见快路径):事件链整段继承该标签,零切分开销;
    - 跨 turn 翻转点且 ``word_split=True``:翻转点按词中点吸附最近词间隙
      (与 ``speaker_fusion._split_event`` 同源逻辑),两段各自继承 turn
      标签;重叠区标 ``genuine_overlap``(overlapped);
    - ``word_split=False`` 或切分未发生:整事件继承主说话人;
    - 无词级时间戳 fallback:用 ``turn_reconciler.split_event_intervals``
      拆区间,整段文本保留在第一个说话人片段(文本无法按词归属)。

    返回 ``(events, diagnostics)``,diagnostics 键与 ``run_speaker_fusion``
    对齐,供 stats 字段契约复用。
    """
    normalized = normalize_turns(turns)
    output: List[Any] = []
    word_split_count = 0
    fallback_split_count = 0
    overlapped_count = 0

    for event in events:
        relevant = [
            turn for turn in normalized
            if turn.end > event.start and turn.start < event.end
        ]
        speakers = {turn.speaker_id for turn in relevant}
        if len(speakers) <= 1:
            # 单一说话人(常见快路径)/无 turns 覆盖:整段继承,零切分开销
            speaker_id = next(iter(speakers), None)
            turn = (
                next(t for t in relevant if t.speaker_id == speaker_id)
                if speaker_id is not None
                else None
            )
            if _apply_identity(event, speaker_id, turn, model_ref):
                overlapped_count += 1
            output.append(event)
            continue

        words = list(getattr(event, "words", []) or [])
        if not word_split:
            # word_split=false:事件不切分(配置语义),整事件继承主说话人
            turn = dominant_turn_at(normalized, event.start, event.end)
            if _apply_identity(
                event,
                turn.speaker_id if turn is not None else None,
                turn, model_ref,
            ):
                overlapped_count += 1
            output.append(event)
            continue

        if len(words) < 2:
            # 无词级时间戳 fallback:turn_reconciler.split_event_intervals
            # 拆区间。文本无法按词归属,整段文本保留在第一个说话人片段,
            # 其余片段无法归属即丢弃(计数进诊断)。
            pieces = [
                (start, end, speaker_id)
                for start, end, speaker_id in split_event_intervals(
                    event.start, event.end, normalized,
                )
                if speaker_id is not None
            ]
            if len(pieces) >= 2:
                fallback_split_count += len(pieces) - 1
            if pieces:
                start, end, speaker_id = pieces[0]
                turn = dominant_turn_at(
                    [t for t in relevant if t.speaker_id == speaker_id],
                    start, end,
                )
                _clamp_event_bounds(event, start, end)
                if _apply_identity(event, speaker_id, turn, model_ref):
                    overlapped_count += 1
            else:
                if _apply_identity(event, None, None, model_ref):
                    overlapped_count += 1
            output.append(event)
            continue

        # 词级切分:turn 翻转点吸附最近词间隙(词中点归属,与 fusion 同源)
        points = sorted({
            point
            for turn in relevant
            for point in (turn.start, turn.end)
            if event.start + 1e-4 < point < event.end - 1e-4
        })
        parts = _split_event(event, points, min_part_duration)
        if len(parts) <= 1:
            # 切分未发生(如切出片段短于 min_part_duration):
            # 整事件继承主说话人,不强行归属。
            turn = dominant_turn_at(normalized, event.start, event.end)
            if _apply_identity(
                event,
                turn.speaker_id if turn is not None else None,
                turn, model_ref,
            ):
                overlapped_count += 1
            output.append(event)
            continue

        word_split_count += len(parts) - 1
        for part in parts:
            midpoint = (part.start + part.end) / 2.0
            turn = _turn_at(normalized, midpoint)
            if _apply_identity(
                part,
                turn.speaker_id if turn is not None else None,
                turn, model_ref,
            ):
                overlapped_count += 1
            output.append(part)

    # 压缩 ID 并保持标签稳定(与 run_speaker_fusion 相同的契约)
    unique = sorted({e.speaker_id for e in output if e.speaker_id is not None})
    remap = {old: new for new, old in enumerate(unique)}
    for index, event in enumerate(output, start=1):
        if event.speaker_id is not None:
            event.speaker_id = remap[event.speaker_id]
            event.speaker_label = _speaker_label(language, event.speaker_id)
        else:
            event.speaker_label = None
        event.index = index

    diagnostics = {
        "speaker_count": len(unique),
        "local_split_count": word_split_count,
        "fallback_split_count": fallback_split_count,
        "overlapped_count": overlapped_count,
        "conflict_count": 0,
        "unknown_count": sum(e.speaker_id is None for e in output),
    }
    logger.info(
        "Early turns labeling: %d events / %d speakers "
        "(word_split=%d, fallback_split=%d, overlapped=%d, unknown=%d)",
        len(output), len(unique), word_split_count,
        fallback_split_count, overlapped_count, diagnostics["unknown_count"],
    )
    return output, diagnostics
