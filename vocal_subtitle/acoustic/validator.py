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
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..merging.merge_engine import _physical_owner_compatible_for_events
from .export import export_skeleton_segments
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
    ) -> Tuple[List, Dict]:
        """校验并修正字幕时间轴

        Args:
            events: SubtitleEvent 列表
            audio_path: 音频文件路径（用于 ffmpeg 调用）
            audio: 音频数组（用于 RMS 双确认）
            sample_rate: 采样率
            ffmpeg_unified_result: 复用的统一 ffmpeg 调用结果

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
        )

        if not speech_skeleton:
            logger.warning("Failed to build acoustic skeleton, skipping validation")
            return events, {"skipped": True, "reason": "skeleton build failed"}

        # Step 2: 吸附修正
        validated, report = self._physical_snap_validation(
            events, speech_skeleton, audio, sample_rate,
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
    # 骨架构建
    # ------------------------------------------------------------------

    def _get_skeleton(
        self,
        audio_path: Optional[Path],
        ffmpeg_unified_result: Optional[Dict],
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

        cfg = self.config
        silence_intervals = FFmpegSilenceVAD._detect_silence(
            audio_path,
            noise_db=cfg.skeleton_noise_db,
            min_silence_duration=cfg.skeleton_min_silence,
        )
        total_duration = FFmpegSilenceVAD._get_duration(audio_path)
        return FFmpegSilenceVAD._invert_intervals(
            silence_intervals, total_duration,
            min_speech_duration=cfg.skeleton_min_speech,
        )

    # ------------------------------------------------------------------
    # 物理吸附
    # ------------------------------------------------------------------

    def _physical_snap_validation(
        self,
        events: List,
        speech_skeleton: List[Tuple[float, float]],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> Tuple[List, Dict]:
        """字幕时间轴向物理声学骨架吸附"""
        cfg = self.config
        report = {
            "snapped_starts": 0,
            "snapped_ends": 0,
            "rms_overrides": 0,
            "skipped_low_confidence": 0,
            "skipped_high_confidence": 0,
            "events_flagged": [],
            "boundary_diagnostics": [],
        }

        for event in events:
            # ---- Start 校验（只向后寻找下一个语音起点） ----
            is_start_in_speech, next_start, _ = _find_directional_boundary(
                event.start, speech_skeleton, "start",
            )
            if not is_start_in_speech and next_start is not None:
                distance = abs(next_start - event.start)
                if _preserve_reliable_asr_boundary(
                    event, "start", self.config,
                ):
                    report["skipped_high_confidence"] += 1
                    _record_boundary_diagnostic(
                        report, event, "start", "skipped",
                        reason="reliable_word_boundary",
                        original_time=event.start,
                        candidate_time=next_start,
                        distance=distance,
                    )
                elif distance <= cfg.max_snap_distance:
                    candidate_start = next_start + cfg.snap_start_margin
                    if candidate_start < event.end:
                        rms_confirmed = True
                        if audio is None:
                            rms_confirmed = distance <= 0.03
                        elif distance > 0.03:
                            rms_confirmed = _rms_energy_check(
                                audio, sample_rate, next_start,
                                window_ms=50, threshold_ratio=2.0,
                            )
                        if rms_confirmed:
                            original_time = event.start
                            event.start = candidate_start
                            report["snapped_starts"] += 1
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
                elif distance <= 0.5:
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
                # End 在连续语音内部时，物理层只报告疑似截尾，不自动延长。
                if containing_speech is not None:
                    speech_end = containing_speech[1]
                    remaining = speech_end - event.end
                    if 0.02 < remaining <= cfg.max_snap_distance:
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
