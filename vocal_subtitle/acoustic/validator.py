"""全局声学标尺校验 (方案七)

将 ffmpeg silencedetect 的输出作为物理基准线 (Acoustic Ground Truth)，
对最终字幕时间轴进行校验和兜底修正。

核心功能:
1. build_global_acoustic_skeleton(): 构建物理声学骨架
2. physical_snap_validation(): 字幕端点向物理骨架吸附
3. validate_with_arbitration(): 带冲突仲裁的校验
4. generate_diagnostic_report(): 输出诊断报告

执行时机: LLM 语义合并完成后、最终字幕输出前。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..merging.merge_engine import _physical_owner_compatible_for_events
from ..utils.audio_utils import AudioUtils
from .export import export_skeleton_segments
from . import arbitration as arbitration_policy
from . import boundary as boundary_policy
from .diagnostics import generate_diagnostic_report as build_diagnostic_report
from . import event_checks
from . import skeleton as skeleton_queries

logger = logging.getLogger(__name__)


@dataclass
class AcousticValidationConfig:
    """声学标尺校验配置"""

    enabled: bool = True
    skeleton_noise_db: float = -40.0       # 骨架提取阈值（敏感模式）
    skeleton_min_silence: float = 0.1      # 最小静音段（秒）
    skeleton_min_speech: float = 0.05      # 最小语音爆发（秒）

    # 吸附参数
    max_snap_distance: float = 0.25        # 最多吸附 250ms（覆盖更多边界偏差）
    snap_start_margin: float = 0.03        # start 吸附后保留 30ms 前导余量
    snap_end_margin: float = 0.01          # end 吸附后保留 10ms 尾随余量

    # 冲突仲裁
    confidence_threshold: float = 0.6      # 低于此置信度不强制吸附
    rms_override_threshold: float = 0.15   # RMS vs ffmpeg 冲突时的阈值

    # 诊断
    generate_report: bool = True
    flag_threshold_ms: float = 200         # 偏差超过此值标记为"需复核"

    # 统一调用
    unified_ffmpeg_pass: bool = True       # 与方案一共用 ffmpeg 调用

    # 双向修正
    allow_end_shorten: bool = True         # ★ 允许声学标尺缩短结束时间（默认开启）
    allow_start_pull_earlier: bool = True  # ★ 允许声学标尺将 start 向前吸附
    # ★ 截尾修复：end 落在连续语音骨架段内部时延长到该段语音终点
    #   （钳制到下一事件 start 之前，绝不跨静音/吞下一句）
    allow_end_extend: bool = True
    # ★ 吞静音修复：start 后向吸附（吸附到下一个真实语音起点）的限幅；
    #   faster-whisper 词起点在换人/换句边界普遍偏早 200~300ms
    max_start_snap_distance: float = 0.45

    # 骨架优先（高精度方案 Task 8 / 优化方案 §10）：TTS、配音、干净单人
    # 播报场景下骨架段即 cue 的硬物理范围，词级 ASR 只做段内细化。
    skeleton_priority: bool = False


class AcousticValidator:
    """全局声学标尺校验器

    使用示例:
        validator = AcousticValidator()
        validated, report = validator.validate(
            events, audio_path, audio, sample_rate
        )
    """

    def __init__(self, config: Optional[AcousticValidationConfig] = None):
        self.config = config or AcousticValidationConfig()

    def validate(
        self,
        events: List,
        audio_path: Optional[Path] = None,
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        ffmpeg_unified_result: Optional[Dict] = None,
        evidence_candidates: Optional[Sequence] = None,
    ) -> Tuple[List, Dict]:
        """校验并修正字幕时间轴

        Args:
            events: SubtitleEvent 列表
            audio_path: 音频文件路径（用于 ffmpeg 调用）
            audio: 音频数组（用于 RMS 双确认）
            sample_rate: 采样率
            ffmpeg_unified_result: 复用的统一 ffmpeg 调用结果
            evidence_candidates: 全程识别 evidence 候选（时间轴仲裁层 R1
                的参照文本）；缺省时回退读配置上的运行期注入字段
                `arbitration_evidence_regions`（见 asr_path 的发布逻辑）

        Returns:
            (validated_events, diagnostic_report)
        """
        cfg = self.config
        if not cfg.enabled:
            return events, {"skipped": True, "reason": "acoustic_validation disabled"}

        if not events:
            return events, {"skipped": True, "reason": "no events"}

        # Step 1: 获取声学骨架
        speech_skeleton = self._get_skeleton(
            audio_path, ffmpeg_unified_result,
            audio=audio, sample_rate=sample_rate,
        )

        if not speech_skeleton:
            logger.warning("Failed to build acoustic skeleton, skipping validation")
            return events, {"skipped": True, "reason": "skeleton build failed"}

        # Step 1.5: 时间轴仲裁层（2026-09-11 定案,层2）
        # timeline_arbitration=False 时返回 None,吸附行为与现状完全一致。
        arbitration = self._build_arbitration(
            events, speech_skeleton, evidence_candidates,
        )

        # Step 2: 吸附修正
        if getattr(cfg, "skeleton_priority", False):
            # 骨架优先模式（优化方案 §10）：TTS/干净单人场景,骨架段即
            # cue 的硬物理范围,不走 ASR 词驱动吸附。
            validated, report = self._apply_skeleton_priority(
                events, speech_skeleton,
            )
        else:
            validated, report = self._physical_snap_validation(
                events, speech_skeleton, audio, sample_rate,
                arbitration=arbitration,
            )

        # Step 2.5: 微间隙合并（同说话人 + gap < 50ms → 合并）
        validated, gap_merged = self._merge_micro_gaps(validated, max_gap=0.05)
        if gap_merged > 0:
            report["gap_merged"] = gap_merged
            report["merge_trace"] = [
                item for event in validated
                for item in (getattr(event, "revision_trace", ()) or ())
                if item.get("stage") == "acoustic_micro_gap_merge"
            ]

        # Step 3: 生成诊断报告
        if cfg.generate_report:
            diagnostic = self.generate_diagnostic_report(validated, speech_skeleton)
            snap_events_flagged = report.get("events_flagged", [])
            report.update(diagnostic)
            report["events_flagged"] = snap_events_flagged + diagnostic.get(
                "events_flagged", []
            )

        logger.info(
            "Acoustic validation: %d events, %d snapped (start=%d, end=%d), "
            "health=%.1f%%",
            len(events),
            report.get("snapped_starts", 0) + report.get("snapped_ends", 0),
            report.get("snapped_starts", 0),
            report.get("snapped_ends", 0),
            report.get("health_score", 100.0),
        )
        return validated, report

    # ------------------------------------------------------------------
    # 骨架优先边界策略（高精度方案 Task 8 / 优化方案 §10）
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_skeleton_segment(
        event: object,
        speech_skeleton: List[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """返回与事件重叠最长的骨架段;无重叠为 None。"""
        best: Optional[Tuple[float, float]] = None
        best_overlap = 0.0
        for segment in speech_skeleton:
            overlap = min(float(event.end), segment[1]) - max(
                float(event.start), segment[0],
            )
            if overlap > best_overlap:
                best, best_overlap = segment, overlap
        return best

    def _apply_skeleton_priority(
        self,
        events: List,
        speech_skeleton: List[Tuple[float, float]],
    ) -> Tuple[List, Dict]:
        """骨架段决定 cue 的合法 start/end;词级只做段内细化。

        - cue start 取主骨架段起点(允许把 ASR 偏晚的起点拉回真实起音);
        - cue end 取主骨架段终点并向内收缩 snap_end_margin,不进入静音;
        - 骨架间静音是硬边界:跨越 ≥2 骨架段的事件钳制回主骨架段,
          不做跨段合并;
        - 完全无骨架覆盖的事件保持原状并计数;
        - 相邻 cue 钳制后保持原有顺序且互不重叠。
        """
        cfg = self.config
        end_margin = float(getattr(cfg, "snap_end_margin", 0.003))
        ordered = sorted(
            events,
            key=lambda item: (
                float(getattr(item, "start", 0.0)),
                float(getattr(item, "end", 0.0)),
            ),
        )

        report: Dict[str, Any] = {
            "skeleton_priority": True,
            "skeleton_start_delta_ms": 0.0,
            "skeleton_end_delta_ms": 0.0,
            "cross_skeleton_merge_count": 0,
            "micro_pause_split_count": 0,
            "skeleton_uncovered_count": 0,
            "snapped_starts": 0,
            "snapped_ends": 0,
            "events_flagged": [],
        }

        # 骨架间静音是硬边界:统计相邻骨架段之间的静音间隙。
        hard_gaps: List[Tuple[float, float]] = [
            (speech_skeleton[i][1], speech_skeleton[i + 1][0])
            for i in range(len(speech_skeleton) - 1)
            if speech_skeleton[i + 1][0] > speech_skeleton[i][1]
        ]

        # Pass 1: 主骨架段归属与基础边界。
        bounds: List[Optional[Tuple[float, float, object]]] = []
        for event in ordered:
            segment = self._dominant_skeleton_segment(event, speech_skeleton)
            if segment is None:
                report["skeleton_uncovered_count"] += 1
                bounds.append(None)
                continue
            crossed_gap = any(
                min(float(event.end), gap_end) - max(float(event.start), gap_start) > 1e-9
                for gap_start, gap_end in hard_gaps
            )
            if crossed_gap:
                # 事件原范围跨越骨架间硬静音:钳制回主骨架段,
                # 不跨静音合并、不拆文本。
                report["cross_skeleton_merge_count"] += 1
            base_start = float(segment[0])
            base_end = float(segment[1]) - end_margin
            report["skeleton_start_delta_ms"] = max(
                report["skeleton_start_delta_ms"],
                abs(float(event.start) - base_start) * 1000.0,
            )
            report["skeleton_end_delta_ms"] = max(
                report["skeleton_end_delta_ms"],
                abs(float(event.end) - base_end) * 1000.0,
            )
            bounds.append((base_start, base_end, segment))

        # Pass 2/3: 同段内多 cue 保持原顺序且互不重叠。
        adjusted: List[Optional[Tuple[float, float]]] = [None] * len(ordered)
        previous_end: Optional[float] = None
        for index, bound in enumerate(bounds):
            if bound is None:
                continue
            new_start = max(bound[0], previous_end) if previous_end is not None else bound[0]
            adjusted[index] = (new_start, bound[1])
            previous_end = bound[1]
        next_start: Optional[float] = None
        for index in range(len(ordered) - 1, -1, -1):
            if adjusted[index] is None:
                continue
            new_start, new_end = adjusted[index]
            if next_start is not None:
                new_end = min(new_end, next_start)
            adjusted[index] = (new_start, new_end)
            next_start = new_start

        # 骨架覆盖率:被 ≥1 条 cue 映射的骨架段时长 / 骨架总时长。
        total_skeleton = sum(
            seg_end - seg_start for seg_start, seg_end in speech_skeleton
        )
        covered_skeleton = sum(
            bound[2][1] - bound[2][0]
            for bound in bounds
            if bound is not None
        )
        report["skeleton_coverage_rate"] = (
            round(covered_skeleton / total_skeleton, 6) if total_skeleton > 0 else None
        )

        # 同段内相邻 cue 对计数(上游微停顿拆分,TTS 模式保留不合并)。
        for index in range(len(ordered) - 1):
            if adjusted[index] is None or adjusted[index + 1] is None:
                continue
            if bounds[index][2] == bounds[index + 1][2]:
                report["micro_pause_split_count"] += 1

        for index, (event, bound) in enumerate(zip(ordered, bounds)):
            if bound is None:
                continue
            new_start, new_end = adjusted[index]
            if new_start != float(event.start):
                report["snapped_starts"] += 1
            if new_end != float(event.end):
                report["snapped_ends"] += 1
            trace = getattr(event, "revision_trace", None)
            if hasattr(trace, "append"):
                trace.append({
                    "stage": "skeleton_priority",
                    "segment": list(bound[2]),
                    "applied_start": round(new_start, 6),
                    "applied_end": round(new_end, 6),
                })
            event.start = new_start
            event.end = new_end
            if getattr(event, "physical_start", None) is not None:
                event.physical_start = new_start
            if getattr(event, "physical_end", None) is not None:
                event.physical_end = new_end

        return ordered, report

    # ------------------------------------------------------------------
    # 骨架构建
    # ------------------------------------------------------------------

    def _get_skeleton(
        self,
        audio_path: Optional[Path],
        ffmpeg_unified_result: Optional[Dict],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float]]:
        """获取声学骨架（优先复用统一 ffmpeg 调用结果）"""
        if (
            self.config.unified_ffmpeg_pass
            and ffmpeg_unified_result is not None
            and "skeleton" in ffmpeg_unified_result
        ):
            return ffmpeg_unified_result["skeleton"]

        # 降级：独立调用 ffmpeg
        if audio_path is None:
            logger.warning("No audio_path for skeleton extraction")
            return []

        from ..vad.ffmpeg_vad import FFmpegSilenceVAD
        from .skeleton import adaptive_silence_threshold_db

        cfg = self.config
        noise_db = adaptive_silence_threshold_db(
            audio,
            sample_rate,
            enabled=cfg.skeleton_adaptive_noise_db,
            fallback_db=cfg.skeleton_noise_db,
            margin_db=cfg.skeleton_noise_margin_db,
        )
        silence_intervals = FFmpegSilenceVAD._detect_silence(
            audio_path,
            noise_db=noise_db,
            min_silence_duration=cfg.skeleton_min_silence,
        )
        total_duration = FFmpegSilenceVAD._get_duration(audio_path)
        return FFmpegSilenceVAD._invert_intervals(
            silence_intervals, total_duration,
            min_speech_duration=cfg.skeleton_min_speech,
        )

    # ------------------------------------------------------------------
    # 时间轴仲裁层（2026-09-11 定案,层2）
    # ------------------------------------------------------------------

    def _build_arbitration(
        self,
        events: List,
        speech_skeleton: List[Tuple[float, float]],
        evidence_candidates: Optional[Sequence],
    ) -> Optional[Dict]:
        """构建时间轴仲裁上下文；timeline_arbitration 关闭时返回 None。

        R1 在此完成区域级判定（吸附/校验环节之前）：分段基线文本与全程
        evidence 文本按区域字符对齐,达标事件记录共识,并按"整段信骨架"
        预先指派骨架成员段。同段多个共识事件只钳组的外边界,防止互相
        钳成同一区间后被末端去重误删文本。
        """
        cfg = self.config
        if not getattr(cfg, "timeline_arbitration", False):
            return None

        if evidence_candidates is None:
            # 运行期注入通道：ASR 管线把全程 evidence 发布到配置对象上
            # （postprocess_runner 的 validate 调用点无法传参,见 asr_path）。
            evidence_candidates = getattr(cfg, "arbitration_evidence_regions", None)
        regions = arbitration_policy.coerce_evidence_regions(evidence_candidates)
        decisions = arbitration_policy.r1_consensus_decisions(
            events,
            regions,
            min_overlap_chars=int(
                getattr(cfg, "arbitration_r1_min_overlap_chars", 6)
            ),
            min_similarity=float(
                getattr(cfg, "arbitration_r1_min_similarity", 0.85)
            ),
        )

        # 为共识事件指派目标骨架成员段（最大重叠段）,并按组分配钳制角色。
        groups: Dict[Tuple[float, float], List] = {}
        targets: Dict[int, Tuple[Tuple[float, float], bool, bool]] = {}
        for event in events:
            decision = decisions.get(id(event))
            if decision is None:
                continue
            segment = arbitration_policy.best_skeleton_segment(
                float(getattr(event, "start", 0.0)),
                float(getattr(event, "end", 0.0)),
                speech_skeleton,
            )
            if segment is None:
                continue
            groups.setdefault(segment, []).append(event)
        for segment, members in groups.items():
            members.sort(key=lambda item: float(getattr(item, "start", 0.0)))
            if len(members) == 1:
                targets[id(members[0])] = (segment, True, True)
                continue
            for position, member in enumerate(members):
                targets[id(member)] = (
                    segment,
                    position == 0,
                    position == len(members) - 1,
                )

        return {
            "decisions": decisions,
            "targets": targets,
            "r2_local_noise": bool(
                getattr(cfg, "arbitration_r2_local_noise", True)
            ),
        }

    def _apply_r1_clamp(
        self,
        event,
        target: Tuple[Tuple[float, float], bool, bool],
        events: List,
        report: Dict,
        arbitration: Dict,
    ) -> None:
        """R1 共识应用：事件边界解除 max_snap_distance 限幅,直接钳到
        骨架成员段端点（越过 reliable-boundary 跳过逻辑——R1 信任级别
        更高）；词内时刻仍用 ASR,不修改 event.words。

        端点不得越过相邻事件领地（防止与未共识邻居重叠后被去重误删）,
        也不得把事件钳成空/反转区间。
        """
        segment, clamp_start, clamp_end = target
        decision = arbitration["decisions"].get(id(event)) or {}
        original_start = float(getattr(event, "start", 0.0))
        original_end = float(getattr(event, "end", 0.0))
        adjusted: Dict[str, float] = {}

        # 领地保护：前一个事件的终点 / 后一个事件的起点
        previous_end = max(
            (
                float(getattr(item, "end", 0.0))
                for item in events
                if item is not event
                and float(getattr(item, "end", 0.0)) <= original_start + 1e-9
            ),
            default=None,
        )
        next_start = min(
            (
                float(getattr(item, "start", 0.0))
                for item in events
                if item is not event
                and float(getattr(item, "start", 0.0)) >= original_end - 1e-9
            ),
            default=None,
        )

        if clamp_start:
            candidate_start = segment[0] + self.config.snap_start_margin
            if previous_end is not None:
                candidate_start = max(candidate_start, previous_end + 0.001)
            if (
                abs(candidate_start - original_start) > 1e-6
                and candidate_start < original_end - 0.05
            ):
                event.start = candidate_start
                adjusted["start"] = candidate_start
                _record_boundary_diagnostic(
                    report, event, "start", "snapped",
                    reason="r1_consensus_skeleton_start",
                    original_time=original_start,
                    candidate_time=candidate_start,
                    distance=abs(candidate_start - original_start),
                )

        if clamp_end:
            candidate_end = segment[1] - self.config.snap_end_margin
            if next_start is not None:
                candidate_end = min(candidate_end, next_start - 0.001)
            if (
                abs(candidate_end - original_end) > 1e-6
                and candidate_end > float(getattr(event, "start", 0.0)) + 0.05
            ):
                event.end = candidate_end
                adjusted["end"] = candidate_end
                _record_boundary_diagnostic(
                    report, event, "end", "snapped",
                    reason="r1_consensus_skeleton_end",
                    original_time=original_end,
                    candidate_time=candidate_end,
                    distance=abs(candidate_end - original_end),
                )

        if adjusted:
            report["r1_applied"] = report.get("r1_applied", 0) + 1
        report.setdefault("r1_regions", []).append({
            "event_id": f"event:index:{int(getattr(event, 'index', 0) or 0):06d}",
            "similarity": getattr(decision, "similarity", None),
            "overlap_chars": getattr(decision, "overlap_chars", None),
            "member_segment": [round(segment[0], 6), round(segment[1], 6)],
            "original_start": round(original_start, 6),
            "original_end": round(original_end, 6),
            "adjusted": {key: round(value, 6) for key, value in adjusted.items()},
        })

    def _r2_evaluate_blind_words(
        self,
        event,
        boundary: float,
        side: str,
        audio: Optional[np.ndarray],
        sample_rate: int,
        report: Dict,
        arbitration: Dict,
        *,
        distance: float,
    ) -> str:
        """R2 盲区评估：候选端点移动会裁掉骨架"静音区"的 ASR 词时的仲裁。

        - 词能量确认为真语音（local_noise=True 用词周边局部噪声底）→
          禁止吸附,保留 ASR 词时间,事件已覆盖该词,标 skeleton_blind,
          返回 "kept"；
        - 无法确认 → 按幻觉裁剪处理,交由既有吸附路径执行,返回
          "trimmed"（实际裁掉后由调用方计数 r2_trimmed）；
        - 没有会被裁的词 → "no_blind",调用方走现行逻辑。

        音频缺失时无法确认（保守）,但不改变现行无音频回退语义。
        """
        spans = arbitration_policy.event_word_spans(event)
        if not spans:
            return "no_blind"
        cut_words = arbitration_policy.words_beyond(spans, boundary, side)
        if not cut_words:
            return "no_blind"

        confirmed = any(
            arbitration_policy.word_speech_confirmed(
                audio, sample_rate, word_span,
                local_noise=arbitration["r2_local_noise"],
            )
            for word_span in cut_words
        )
        if confirmed:
            word_start = min(span[0] for span in cut_words)
            word_end = max(span[1] for span in cut_words)
            report["r2_blind_kept"] = report.get("r2_blind_kept", 0) + 1
            report["skeleton_blind"] = report.get("skeleton_blind", 0) + 1
            report["events_flagged"].append({
                "id": getattr(event, "index", 0),
                "issue": "skeleton_blind",
                "boundary": side,
                "word_start": round(word_start, 6),
                "word_end": round(word_end, 6),
                "text_preview": str(getattr(event, "text", ""))[:50],
            })
            _record_boundary_diagnostic(
                report, event, side, "skipped",
                reason="r2_blind_kept",
                original_time=(
                    float(getattr(event, "end", 0.0)) if side == "end"
                    else float(getattr(event, "start", 0.0))
                ),
                candidate_time=boundary,
                distance=distance,
            )
            return "kept"
        return "trimmed"

    @staticmethod
    def _count_r2_trimmed(report: Dict, event, side: str) -> None:
        """R2 幻觉裁剪计数与复核标记（词能量无法确认为真语音被裁掉）。"""
        report["r2_trimmed"] = report.get("r2_trimmed", 0) + 1
        report["events_flagged"].append({
            "id": getattr(event, "index", 0),
            "issue": "r2_hallucination_trim",
            "boundary": side,
            "text_preview": str(getattr(event, "text", ""))[:50],
        })

    # ------------------------------------------------------------------
    # 物理吸附
    # ------------------------------------------------------------------

    def _physical_snap_validation(
        self,
        events: List,
        speech_skeleton: List[Tuple[float, float]],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        arbitration: Optional[Dict] = None,
    ) -> Tuple[List, Dict]:
        """字幕时间轴向物理声学骨架吸附"""
        cfg = self.config
        report = {
            "snapped_starts": 0,
            "snapped_ends": 0,
            "rms_overrides": 0,
            "skipped_low_confidence": 0,
            "skipped_high_confidence": 0,
            "ends_extended": 0,
            "events_flagged": [],
            "boundary_diagnostics": [],
        }
        if arbitration is not None:
            # 时间轴仲裁层诊断（r1_/r2_ 前缀）仅在开关开启时出现在报告里,
            # 关闭时保持 report 键与现状逐字段一致。
            report["r1_applied"] = 0
            report["r2_blind_kept"] = 0
            report["r2_trimmed"] = 0

        # 邻居索引（按 start 排序）：end 延长需要钳制到下一事件 start 之前
        ordered = sorted(events, key=lambda e: float(getattr(e, "start", 0.0)))
        next_of = {
            id(ordered[i]): ordered[i + 1] for i in range(len(ordered) - 1)
        }

        for event in events:
            # ---- R1 共识（层2）：整段信骨架,跳过常规吸附 ----
            if arbitration is not None:
                target = arbitration["targets"].get(id(event))
                if target is not None:
                    self._apply_r1_clamp(event, target, events, report, arbitration)
                    continue

            # ---- Start 校验（后向吸附：裁掉 start 吞并的前导静音） ----
            is_start_in_speech, next_start, _ = _find_directional_boundary(
                event.start, speech_skeleton, "start",
            )
            # 吸附目标 = 下一个真实语音起点。骨架包含判断对帧级无缝衔接
            # 产生的共享边界会误判（上一句尾巴把 start 顶进了语音段），
            # 有音频时统一改用能量口径：start 之后 120ms 基本静音才视为
            # 吞并，吸附目标取严格晚于 start 的下一个骨架段起点。
            anchor = next_start if not is_start_in_speech else None
            if audio is not None:
                if _leading_silence_ahead(audio, sample_rate, event.start):
                    # 优先用能量扫描找局部真实起点：句内停顿后的重新开口
                    # 在骨架上没有边界（<min_silence 的停顿被并进同一
                    # 语音段），只有能量能定位；无能量命中再退回骨架。
                    anchor = _next_energy_onset_after(
                        audio, sample_rate, event.start,
                        cfg.max_start_snap_distance,
                    )
                    if anchor is None:
                        anchor = _next_speech_onset_after(
                            event.start, speech_skeleton,
                        )
                else:
                    anchor = None
            if anchor is not None:
                distance = abs(anchor - event.start)

                # R2 盲区（层2）：先于吸附评估会被前移裁掉的静音区词头——
                # 能量确认为真语音则禁吸附保留 ASR 词时间。
                r2_status = "no_blind"
                start_kept = False
                if (
                    arbitration is not None
                    and distance <= cfg.max_start_snap_distance
                ):
                    r2_status = self._r2_evaluate_blind_words(
                        event,
                        anchor + cfg.snap_start_margin, "start",
                        audio, sample_rate, report, arbitration,
                        distance=distance,
                    )
                    # start_kept 只豁免 start 校验,end 校验照常进行。
                    start_kept = r2_status == "kept"

                # start 后向吸附只裁静音、不移动语音内容。ASR 词起点在
                # 换人/换句边界普遍偏早 200~300ms,文本置信度不能为时间戳
                # 背书,因此这里不做 reliable-boundary 豁免,统一用能量
                # 确认兜底（静音已由 _leading_silence_ahead 确认,起点由
                # _rms_energy_check 确认）。
                if not start_kept and distance <= cfg.max_start_snap_distance:
                    candidate_start = anchor + cfg.snap_start_margin
                    if candidate_start < event.end:
                        rms_confirmed = True
                        if audio is None:
                            rms_confirmed = distance <= 0.03
                        elif distance > 0.03:
                            rms_confirmed = _rms_energy_check(
                                audio, sample_rate, anchor,
                                window_ms=50, threshold_ratio=2.0,
                            )
                        if rms_confirmed:
                            original_time = event.start
                            event.start = candidate_start
                            report["snapped_starts"] += 1
                            if r2_status == "trimmed":
                                self._count_r2_trimmed(report, event, "start")
                            _record_boundary_diagnostic(
                                report, event, "start", "snapped",
                                reason="next_speech_start",
                                original_time=original_time,
                                candidate_time=candidate_start,
                                distance=distance,
                            )
                        else:
                            report["rms_overrides"] += 1
                            _record_boundary_diagnostic(
                                report, event, "start", "skipped",
                                reason="rms_not_confirmed",
                                original_time=event.start,
                                candidate_time=candidate_start,
                                distance=distance,
                            )
                    else:
                        _record_boundary_diagnostic(
                            report, event, "start", "skipped",
                            reason="candidate_would_invalidate_event",
                            original_time=event.start,
                            candidate_time=candidate_start,
                            distance=distance,
                        )
                elif not start_kept and distance <= 0.5:
                    report["events_flagged"].append({
                        "id": getattr(event, "index", 0),
                        "issue": "start_deviation",
                        "deviation_ms": round(distance * 1000),
                    })

            # ---- End 校验（只向前寻找上一个语音终点） ----
            (
                is_end_in_speech,
                previous_end,
                containing_speech,
            ) = _find_directional_boundary(event.end, speech_skeleton, "end")
            if is_end_in_speech:
                # End 在连续语音骨架段内部：这是 ASR 词尾时间戳偏早造成
                # 的截尾。骨架段是物理连续语音，把 end 延长到该段语音终点
                # 不会跨静音；同时钳制到下一事件 start 之前，绝不吞下一句
                # 的词（骨架段内两个 cue 共享一段连续语音时以此让界）。
                if containing_speech is not None:
                    speech_end = containing_speech[1]
                    remaining = speech_end - event.end
                    if remaining > 0.02:
                        candidate_end = speech_end - cfg.snap_end_margin
                        next_event = next_of.get(id(event))
                        if (
                            next_event is not None
                            and next_event.start > event.end
                        ):
                            candidate_end = min(
                                candidate_end, next_event.start - 0.02,
                            )
                        extended = False
                        if (
                            cfg.allow_end_extend
                            and candidate_end > event.end + 0.02
                        ):
                            original_time = event.end
                            event.end = candidate_end
                            report["ends_extended"] = (
                                report.get("ends_extended", 0) + 1
                            )
                            extended = True
                            _record_boundary_diagnostic(
                                report, event, "end", "snapped",
                                reason="containing_speech_end",
                                original_time=original_time,
                                candidate_time=candidate_end,
                                distance=remaining,
                            )
                        if (
                            not extended
                            and 0.02 < remaining <= cfg.max_snap_distance
                        ):
                            _record_boundary_diagnostic(
                                report, event, "end", "flagged",
                                reason="possible_truncation_inside_speech",
                                original_time=event.end,
                                candidate_time=speech_end,
                                distance=remaining,
                            )
                            report["events_flagged"].append({
                                "id": getattr(event, "index", 0),
                                "issue": "possible_truncation",
                                "deviation_ms": round(remaining * 1000),
                                "text_preview": getattr(event, "text", "")[:50],
                            })
                continue

            if previous_end is not None:
                distance = abs(event.end - previous_end)
                candidate_end = previous_end - cfg.snap_end_margin

                # R2 盲区（层2）：先于 reliable-boundary 跳过评估会被回缩
                # 裁掉的静音区词——词能量确认为真语音则禁吸附并标
                # skeleton_blind;无法确认则交给现行路径(reliable 跳过或
                # 回缩=按幻觉裁剪)。
                r2_status = "no_blind"
                if (
                    arbitration is not None
                    and distance <= cfg.max_snap_distance
                ):
                    r2_status = self._r2_evaluate_blind_words(
                        event, candidate_end, "end",
                        audio, sample_rate, report, arbitration,
                        distance=distance,
                    )
                    if r2_status == "kept":
                        continue

                if _preserve_reliable_asr_boundary(
                    event, "end", self.config,
                ):
                    report["skipped_high_confidence"] += 1
                    _record_boundary_diagnostic(
                        report, event, "end", "skipped",
                        reason="reliable_word_boundary",
                        original_time=event.end,
                        candidate_time=candidate_end,
                        distance=distance,
                    )
                    continue

                if distance <= cfg.max_snap_distance:
                    # End 只能回缩到前一个语音终点，禁止跨静音延长到后续语音。
                    if (
                        cfg.allow_end_shorten
                        and candidate_end < event.end - 0.03
                        and candidate_end > event.start
                    ):
                        rms_confirmed = _silence_confirmed(
                            audio,
                            sample_rate,
                            (candidate_end + event.end) / 2,
                            distance=distance,
                        )
                        if rms_confirmed:
                            original_time = event.end
                            event.end = candidate_end
                            report["snapped_ends"] += 1
                            report["ends_shortened"] = (
                                report.get("ends_shortened", 0) + 1
                            )
                            if r2_status == "trimmed":
                                self._count_r2_trimmed(report, event, "end")
                            _record_boundary_diagnostic(
                                report, event, "end", "snapped",
                                reason="previous_speech_end",
                                original_time=original_time,
                                candidate_time=candidate_end,
                                distance=distance,
                            )
                        else:
                            report["rms_overrides"] += 1
                            _record_boundary_diagnostic(
                                report, event, "end", "skipped",
                                reason="silence_not_confirmed",
                                original_time=event.end,
                                candidate_time=candidate_end,
                                distance=distance,
                            )
                    else:
                        _record_boundary_diagnostic(
                            report, event, "end", "skipped",
                            reason="end_extension_forbidden",
                            original_time=event.end,
                            candidate_time=candidate_end,
                            distance=distance,
                        )
                elif distance <= 0.5:
                    report["events_flagged"].append({
                        "id": getattr(event, "index", 0),
                        "issue": "end_deviation",
                        "deviation_ms": round(distance * 1000),
                        "text_preview": (
                            getattr(event, "text", "")[:50]
                            if hasattr(event, "text") else ""
                        ),
                    })

        return events, report

    # ------------------------------------------------------------------
    # 微间隙合并
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_micro_gaps(
        events: List,
        max_gap: float = 0.05,
    ) -> tuple:
        """合并同说话人的极近邻事件（gap < max_gap）。

        两个相邻事件间距 < 50ms 且同说话人时，极可能是 VAD
        边界精度不足导致同一句话被错误切分。合并它们以减少
        碎片化事件，提升声学校验健康度。

        Args:
            events: SubtitleEvent 列表（按 start 排序）
            max_gap: 最大合并间隙（秒）

        Returns:
            (merged_events, num_merged)
        """
        if len(events) <= 1:
            return events, 0

        merged = []
        num_merged = 0

        for event in events:
            if not merged:
                merged.append(event)
                continue

            prev = merged[-1]
            gap = event.start - prev.end

            # 同说话人检查
            same_speaker = (
                prev.speaker_id is not None
                and event.speaker_id is not None
                and prev.speaker_id == event.speaker_id
            )

            if 0 < gap <= max_gap and same_speaker:
                # 物理所有权保护：不同 physical clip 的事件不合并
                if not _physical_owner_compatible_for_events(prev, event):
                    merged.append(event)
                    continue
                # 合并：延长 prev 覆盖当前事件
                prev.end = event.end
                prev.text = f"{prev.text} {event.text}".strip()
                # Merge provenance: source_word_ids and physical_spans
                prev.source_word_ids = list(dict.fromkeys(
                    (prev.source_word_ids or []) + (event.source_word_ids or [])
                ))
                prev.physical_spans = list((prev.physical_spans or []) + (event.physical_spans or []))
                merge_trace = {
                    "op": "merge",
                    "stage": "acoustic_micro_gap_merge",
                    "reason": "micro_gap_same_speaker_same_physical_owner",
                    "event_ids": [
                        f"event:index:{int(getattr(prev, 'index', 0) or 0):06d}",
                        f"event:index:{int(getattr(event, 'index', 0) or 0):06d}",
                    ],
                    "merge_count": 1,
                    "physical_owner": {
                        "region_ids": list(dict.fromkeys(
                            item for item in (
                                getattr(prev, "physical_region_id", None),
                                getattr(event, "physical_region_id", None),
                            ) if item is not None
                        )),
                        "bin_ids": list(dict.fromkeys(
                            item for item in (
                                getattr(prev, "physical_bin_id", None),
                                getattr(event, "physical_bin_id", None),
                            ) if item is not None
                        )),
                        "speaker_ids": [prev.speaker_id],
                    },
                    "gap_ms": round(gap * 1000.0, 3),
                }
                prev.revision_trace = list(getattr(prev, "revision_trace", []) or []) + [merge_trace]
                # Keep the report local to the validator result; callers that
                # need this trace receive it through the event revision trace.
                num_merged += 1
                logger.debug(
                    "Micro-gap merge: %.0fms gap, same speaker → "
                    "merged events",
                    gap * 1000,
                )
            else:
                merged.append(event)

        # 重新编号
        for i, evt in enumerate(merged):
            evt.index = i + 1

        if num_merged > 0:
            logger.info(
                "Micro-gap merge: %d → %d events (%d merged, gap < %.0fms)",
                len(events), len(merged), num_merged, max_gap * 1000,
            )

        return merged, num_merged

    # ------------------------------------------------------------------
    # 诊断报告
    # ------------------------------------------------------------------

    def generate_diagnostic_report(
        self,
        events: List,
        speech_skeleton: List[Tuple[float, float]],
    ) -> Dict:
        """生成物理校验诊断报告，保留旧实例方法入口。"""
        return build_diagnostic_report(
            events,
            speech_skeleton,
            flag_threshold_ms=self.config.flag_threshold_ms,
        )


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------


def _find_boundary_in_skeleton(
    t: float, skeleton: List[Tuple[float, float]],
) -> Tuple[bool, float]:
    """判断时间点 t 是否在语音段内，并返回最近的边界

    Returns:
        (is_in_speech, nearest_boundary)
    """
    for s_start, s_end in skeleton:
        if s_start <= t <= s_end:
            return True, t
        if t < s_start:
            return False, s_start

    # t 在所有语音段之后
    return False, skeleton[-1][1] if skeleton else t


def _find_directional_boundary(
    t: float,
    skeleton: List[Tuple[float, float]],
    boundary_type: str,
) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]]]:
    """Find the boundary appropriate for a start or end endpoint.

    The legacy helper above returns the next boundary in a gap for both
    endpoint types. That is valid for a start, but an end must use the
    previous speech end or it can be extended across an entire silence gap.

    Returns ``(inside_speech, candidate_boundary, containing_speech)``.
    ``candidate_boundary`` is ``None`` when no boundary exists in the
    direction that is safe for this endpoint.
    """
    if boundary_type not in {"start", "end"}:
        raise ValueError("boundary_type must be 'start' or 'end'")

    previous_end: Optional[float] = None
    for speech_start, speech_end in skeleton:
        if speech_start <= t <= speech_end:
            return True, t, (speech_start, speech_end)
        if t < speech_start:
            if boundary_type == "start":
                return False, speech_start, None
            return False, previous_end, None
        previous_end = speech_end

    if boundary_type == "end":
        return False, previous_end, None
    return False, None, None


def _next_speech_onset_after(
    t: float, skeleton: List[Tuple[float, float]],
) -> Optional[float]:
    """返回严格晚于 t 的下一个骨架段起点（跨过当前所在语音段）。"""
    best: Optional[float] = None
    for s_start, _s_end in skeleton:
        if s_start > t + 1e-6 and (best is None or s_start < best):
            best = s_start
    return best


def _next_energy_onset_after(
    audio: np.ndarray,
    sample_rate: int,
    t: float,
    horizon: float,
) -> Optional[float]:
    """在 [t, t+horizon] 内扫描第一段持续语音爆发（≥2 帧超噪声底）。

    骨架段会把 <min_silence 的停顿并进同一段连续语音，句内停顿后的
    重新开口在骨架上没有边界；这里直接看能量，找到局部真实起点。
    """
    mask = _speech_frame_mask(audio, sample_rate, t, t + horizon)
    if mask is None or mask.size < 2:
        return None
    for i in range(len(mask) - 1):
        if mask[i] and mask[i + 1]:
            return t + i * 0.02
    return None


def _speech_frame_mask(
    audio: np.ndarray,
    sample_rate: int,
    t0: float,
    t1: float,
) -> Optional[np.ndarray]:
    """返回 [t0,t1] 内 20ms 帧是否含语音能量的布尔掩码。

    判据 = 帧峰值 > max(4% 全局峰值, 2× 底噪 RMS)。只用底噪倍数会把
    TTS/录音残留噪声（底噪的 2~2.5 倍）误判为语音；4% 峰值门限与
    噪声影子线一致，在干净素材上能干净地区分句内停顿与语音。
    """
    win = max(1, int(0.02 * sample_rate))
    i0 = int(t0 * sample_rate)
    i1 = min(len(audio), int(t1 * sample_rate))
    seg = audio[i0:i1].astype(np.float32)
    if seg.size < win:
        return None
    frames = seg[: seg.size // win * win].reshape(-1, win)
    frame_peak = np.abs(frames).max(axis=1)
    threshold = max(
        0.04 * float(np.abs(audio).max()) + 1e-9,
        2.0 * AudioUtils.estimate_silence_rms(audio, sample_rate),
    )
    return frame_peak > threshold


def _leading_silence_ahead(
    audio: Optional[np.ndarray],
    sample_rate: int,
    t: float,
    window_sec: float = 0.12,
    speech_frame_ratio: float = 0.2,
) -> bool:
    """判断 [t, t+window] 是否基本无语音能量（用于 start 后向吸附）。

    skeleton 的"点是否在语音段内"对帧级无缝衔接产生的共享边界会误判
    （上一句尾巴把下一句 start 顶进了语音段里），这里直接看能量：
    窗口内含语音帧占比 ≤ 阈值视为静音。
    """
    if audio is None:
        return False
    mask = _speech_frame_mask(audio, sample_rate, t, t + window_sec)
    if mask is None:
        return True
    return float(mask.mean()) <= speech_frame_ratio


def _boundary_confidence(event: object, boundary_type: str) -> Optional[float]:
    """Return confidence for the word anchoring one event endpoint."""
    words = list(getattr(event, "words", []) or [])
    if words:
        word = words[0] if boundary_type == "start" else words[-1]
        value = getattr(word, "confidence", None)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass

    for name in (f"{boundary_type}_confidence", "boundary_confidence"):
        value = getattr(event, name, None)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass
    return None


def _preserve_reliable_asr_boundary(
    event: object,
    boundary_type: str,
    config: AcousticValidationConfig,
) -> bool:
    """Keep a reliable word-level endpoint ahead of physical snapping."""
    words = list(getattr(event, "words", []) or [])
    confidence = _boundary_confidence(event, boundary_type)
    return bool(
        words
        and confidence is not None
        and confidence >= config.confidence_threshold
    )


def _silence_confirmed(
    audio: Optional[np.ndarray],
    sample_rate: int,
    time_point: float,
    *,
    distance: float,
) -> bool:
    """Confirm a gap is silent, with a conservative no-audio fallback."""
    if audio is None:
        # Without samples, only a tiny structural correction is safe.
        return distance <= 0.03
    return not _rms_energy_check(
        audio, sample_rate, time_point,
        window_ms=50, threshold_ratio=2.0,
    )


def _record_boundary_diagnostic(
    report: Dict,
    event: object,
    boundary_type: str,
    action: str,
    *,
    reason: str,
    original_time: float,
    candidate_time: Optional[float],
    distance: Optional[float],
) -> None:
    """Append a compact, auditable endpoint decision."""
    report.setdefault("boundary_diagnostics", []).append({
        "stage": "acoustic_boundary",
        "event_id": f"event:index:{int(getattr(event, 'index', 0) or 0):06d}",
        "boundary": boundary_type,
        "action": action,
        "reason": reason,
        "physical_region_id": getattr(event, "physical_region_id", None),
        "physical_bin_id": getattr(event, "physical_bin_id", None),
        "speaker_id": getattr(event, "speaker_id", None),
        "original_time": round(float(original_time), 6),
        "candidate_time": (
            round(float(candidate_time), 6)
            if candidate_time is not None else None
        ),
        "distance_ms": (
            round(float(distance) * 1000, 3)
            if distance is not None else None
        ),
    })
def _is_time_in_speech(
    t: float, skeleton: List[Tuple[float, float]],
) -> bool:
    """判断时间点是否在语音段内"""
    for s_start, s_end in skeleton:
        if s_start <= t <= s_end:
            return True
    return False


def _has_speech_in_range(
    t1: float, t2: float, skeleton: List[Tuple[float, float]],
) -> bool:
    """判断 [t1, t2] 区间内是否有语音"""
    for s_start, s_end in skeleton:
        if s_start < t2 and s_end > t1:
            return True
    return False


def _rms_energy_check(
    audio: np.ndarray,
    sample_rate: int,
    time_point: float,
    window_ms: int = 50,
    threshold_ratio: float = 2.0,
) -> bool:
    """在时间点附近做 RMS 能量确认

    Returns:
        True 如果检测到语音能量
    """
    from ..utils.audio_utils import AudioUtils

    silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)
    half_window = window_ms / 2000.0  # 转秒再折半

    t1 = max(0, time_point - half_window)
    t2 = min(len(audio) / sample_rate, time_point + half_window)

    gap_rms = AudioUtils.get_segment_rms(audio, t1, t2, sample_rate)
    return gap_rms > silence_rms * threshold_ratio


# ------------------------------------------------------------------
# 5.12.3 非人声高能事件仲裁
# ------------------------------------------------------------------


def _compute_vad_overlap(
    start: float,
    end: float,
    vad_segments: List,
) -> float:
    """计算区间与 VAD 检测结果的重叠比例

    Returns:
        0.0 ~ 1.0，重叠比例
    """
    duration = end - start
    if duration <= 0:
        return 0.0

    overlap_total = 0.0
    for seg in vad_segments:
        seg_start = seg.start if hasattr(seg, "start") else seg[0]
        seg_end = seg.end if hasattr(seg, "end") else seg[1]
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)
        if overlap_start < overlap_end:
            overlap_total += overlap_end - overlap_start

    return min(1.0, overlap_total / duration)


def _classify_energy_type(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> str:
    """基于频谱特征区分噪音类型

    使用自相关法检测谐波结构：
    - 有谐波结构 → "music_or_tonal"（音乐、警报等）
    - 无谐波结构 → "transient_noise"（拍桌子、关门等）

    Returns:
        "transient_noise" | "music_or_tonal" | "unknown"
    """
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = audio[start_sample:end_sample]

    if len(segment) < 256:
        return "unknown"

    try:
        # 自相关
        autocorr = np.correlate(segment, segment, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)

        # 找前几个峰值（基频和谐波）
        peaks = []
        for i in range(1, min(len(autocorr) - 1, sample_rate // 50)):  # 50Hz 下限
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > 0.15:  # 显著的峰值
                    peaks.append((i, autocorr[i]))

        if not peaks:
            return "transient_noise"  # 无谐波结构 → 瞬态噪音

        # 谐波比 = 峰值平均
        peak_vals = [p[1] for p in peaks[:10]]
        harmonics_ratio = sum(peak_vals) / len(peak_vals)

        if harmonics_ratio > 0.3:
            return "music_or_tonal"
        else:
            return "transient_noise"
    except Exception:
        return "unknown"


# Route all helper consumers through the isolated policies.  The private names
# remain here as compatibility exports for historical callers and tests.
_find_boundary_in_skeleton = boundary_policy.find_boundary_in_skeleton
_find_directional_boundary = boundary_policy.find_directional_boundary
_boundary_confidence = boundary_policy.boundary_confidence
_preserve_reliable_asr_boundary = boundary_policy.preserve_reliable_asr_boundary
_record_boundary_diagnostic = boundary_policy.record_boundary_diagnostic
_is_time_in_speech = skeleton_queries.is_time_in_speech
_has_speech_in_range = skeleton_queries.has_speech_in_range
_rms_energy_check = skeleton_queries.rms_energy_check
_silence_confirmed = skeleton_queries.silence_confirmed
_compute_vad_overlap = skeleton_queries.compute_vad_overlap
_classify_energy_type = event_checks.classify_energy_type
classify_acoustic_events = event_checks.classify_acoustic_events
