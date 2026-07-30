"""Acoustic validator — thin public entry point that delegates to domain modules.

Original class from acoustic_validator.py; now leverages:
- acoustic/skeleton.py
- acoustic/boundary.py
- acoustic/event_checks.py
- acoustic/diagnostics.py
- acoustic/export.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.models import AcousticValidationConfig
from ..merging.llm_merge_engine import _physical_owner_compatible_for_events
from .boundary import (
    _find_directional_boundary,
    _preserve_reliable_asr_boundary,
    _record_boundary_diagnostic,
    _silence_confirmed,
)
from .diagnostics import generate_diagnostic_report
from .event_checks import _rms_energy_check

logger = logging.getLogger(__name__)


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
        """校验并修正字幕时间轴"""
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

        # Step 2.5: 微间隙合并
        validated, gap_merged = self._merge_micro_gaps(validated, max_gap=0.05)
        if gap_merged > 0:
            report["gap_merged"] = gap_merged

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
        """生成物理校验诊断报告"""
        return generate_diagnostic_report(events, speech_skeleton, self.config.flag_threshold_ms)
