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

        # Step 3: 生成诊断报告
        if cfg.generate_report:
            diagnostic = self.generate_diagnostic_report(validated, speech_skeleton)
            report.update(diagnostic)

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

        from .vad.ffmpeg_vad import FFmpegSilenceVAD

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
            "events_flagged": [],
        }

        for event in events:
            # ---- Start 校验（双向吸附） ----
            is_start_in_speech, nearest_start = _find_boundary_in_skeleton(
                event.start, speech_skeleton,
            )
            if not is_start_in_speech:
                distance = abs(nearest_start - event.start)
                if distance <= cfg.max_snap_distance:
                    # 当前逻辑：snap to nearest_start + margin（start 延后）
                    if nearest_start > event.start:
                        rms_confirmed = True
                        if audio is not None and distance > 0.03:
                            rms_confirmed = _rms_energy_check(
                                audio, sample_rate, nearest_start,
                                window_ms=50, threshold_ratio=2.0,
                            )
                        if rms_confirmed:
                            event.start = nearest_start + cfg.snap_start_margin
                            report["snapped_starts"] += 1
                        else:
                            report["rms_overrides"] += 1
                    # ★ 新增：向前吸附。start 在语音爆发内部但更接近爆发起点时拉回
                    elif cfg.allow_start_pull_earlier and nearest_start < event.start - 0.01:
                        candidate_start = nearest_start + cfg.snap_start_margin
                        if candidate_start < event.start:
                            rms_confirmed = True
                            if audio is not None and distance > 0.03:
                                rms_confirmed = _rms_energy_check(
                                    audio, sample_rate, nearest_start,
                                    window_ms=50, threshold_ratio=2.0,
                                )
                            if rms_confirmed:
                                event.start = candidate_start
                                report["snapped_starts"] += 1
                elif distance <= 0.5:
                    report["events_flagged"].append({
                        "id": getattr(event, "index", 0),
                        "issue": "start_deviation",
                        "deviation_ms": round(distance * 1000),
                    })

            # ---- End 校验（双向：可延长也可缩短） ----
            is_end_in_speech, nearest_end = _find_boundary_in_skeleton(
                event.end, speech_skeleton,
            )
            if not is_end_in_speech:
                distance = abs(nearest_end - event.end)
                candidate_end = nearest_end - cfg.snap_end_margin

                if distance <= cfg.max_snap_distance:
                    if candidate_end > event.end:
                        # 延长：字幕结束在声学骨架边界之前（防切尾，现有行为）
                        if nearest_end > event.end:
                            event.end = candidate_end
                            report["snapped_ends"] += 1
                        elif distance < 0.05:
                            # 微小偏差，微调
                            event.end = candidate_end
                            report["snapped_ends"] += 1
                    # ★ 新增：缩短。字幕延伸到静音区时回缩到骨架边界
                    elif cfg.allow_end_shorten and candidate_end < event.end - 0.03:
                        # RMS 确认：被切区域确实为静音（防止截断真实语音）
                        rms_confirmed = True
                        if audio is not None:
                            rms_confirmed = not _rms_energy_check(
                                audio, sample_rate,
                                (candidate_end + event.end) / 2,
                                window_ms=50, threshold_ratio=2.0,
                            )
                        if rms_confirmed:
                            event.end = candidate_end
                            report["snapped_ends"] += 1
                            report["ends_shortened"] = (
                                report.get("ends_shortened", 0) + 1
                            )
                elif distance <= 0.5 and nearest_end > event.end:
                    report["events_flagged"].append({
                        "id": getattr(event, "index", 0),
                        "issue": "possible_truncation",
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
                # 合并：延长 prev 覆盖当前事件
                prev.end = event.end
                prev.text = f"{prev.text} {event.text}".strip()
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
        report = {
            "total_events": len(events),
            "start_in_silence": 0,
            "end_in_silence": 0,
            "end_truncated": 0,
            "start_out_of_range": 0,
            "end_out_of_range": 0,
            "events_flagged": [],
        }

        for event in events:
            # 检查 start
            is_start_ok = _is_time_in_speech(event.start, speech_skeleton)
            if not is_start_ok:
                report["start_in_silence"] += 1
                _, nearest = _find_boundary_in_skeleton(
                    event.start, speech_skeleton,
                )
                if abs(nearest - event.start) > 0.2:
                    report["start_out_of_range"] += 1

            # 检查 end
            is_end_ok = _is_time_in_speech(event.end, speech_skeleton)
            if not is_end_ok:
                report["end_in_silence"] += 1
                # 检查是否切尾
                if _has_speech_in_range(
                    event.end, event.end + 0.2, speech_skeleton,
                ):
                    report["end_truncated"] += 1
                    report["events_flagged"].append({
                        "id": getattr(event, "index", 0),
                        "issue": "end_truncated",
                        "current_end": event.end,
                        "text_preview": (
                            getattr(event, "text", "")[:50]
                            if hasattr(event, "text") else ""
                        ),
                    })

                _, nearest = _find_boundary_in_skeleton(
                    event.end, speech_skeleton,
                )
                if abs(nearest - event.end) > 0.2:
                    report["end_out_of_range"] += 1

        # 计算健康度评分
        total_checks = len(events) * 2
        issues = report["start_in_silence"] + report["end_in_silence"]
        report["health_score"] = round(
            (1 - issues / max(total_checks, 1)) * 100, 1,
        )

        return report


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
    from .utils.audio_utils import AudioUtils

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


def classify_acoustic_events(
    skeleton: List[Tuple[float, float]],
    silero_segments: List,
    audio: np.ndarray,
    sample_rate: int,
) -> List[Dict]:
    """将声学骨架中的事件分为"人声"和"非人声高能事件"（文档 5.12.3）

    判定逻辑:
    - ffmpeg 标记为语音 + Silero 也标记为语音 → 人声（可信）
    - ffmpeg 标记为语音 + Silero 未标记 → 非人声高能事件（跳过吸附）
    - ffmpeg 标记为静音 + Silero 标记为语音 → 低音量人声（Silero优先）

    Args:
        skeleton: ffmpeg 声学骨架 [(start, end), ...]
        silero_segments: Silero VAD 检测结果
        audio: 音频数组
        sample_rate: 采样率

    Returns:
        分类后的事件列表 [{"start", "end", "type", "confidence", ...}, ...]
    """
    classified = []

    for sk_start, sk_end in skeleton:
        # 检查该区间是否被 Silero VAD 确认
        silero_overlap = _compute_vad_overlap(
            sk_start, sk_end, silero_segments,
        )

        if silero_overlap > 0.5:
            event_type = "human_speech"
            confidence = "high"
        elif silero_overlap > 0.1:
            event_type = "human_speech"
            confidence = "low"  # 边缘情况，可能是语尾渐弱
        else:
            # ffmpeg 检测到能量但 Silero 不认为是人声
            event_type = "non_human_energy"
            confidence = "high"

            # 进一步分类：音乐 vs 瞬态噪音
            energy_subtype = _classify_energy_type(
                audio, sample_rate, sk_start, sk_end,
            )

        entry = {
            "start": sk_start,
            "end": sk_end,
            "type": event_type,
            "confidence": confidence,
            "silero_overlap_ratio": round(silero_overlap, 2),
        }

        if event_type == "non_human_energy":
            entry["energy_subtype"] = energy_subtype

        classified.append(entry)

    non_human_count = sum(1 for c in classified if c["type"] == "non_human_energy")
    if non_human_count > 0:
        logger.info(
            "Acoustic event classification: %d total, %d non-human energy "
            "(%.0f%%), %d human speech",
            len(classified),
            non_human_count,
            non_human_count / max(len(classified), 1) * 100,
            sum(1 for c in classified if c["type"] == "human_speech"),
        )
    return classified


# ------------------------------------------------------------------
# 骨架段导出（供人工验证）
# ------------------------------------------------------------------


def export_skeleton_segments(
    audio_path: Path,
    output_dir: Path,
    noise_db: float = -40.0,
    min_silence_duration: float = 0.1,
    min_speech_duration: float = 0.05,
    include_silence: bool = True,
    include_mixed: bool = False,
) -> Dict:
    """将声学骨架的语音段和静音段导出为独立音频文件。

    用途：人工验证 ffmpeg silencedetect 的静音/人声划分是否准确。

    对每个骨架段（语音或静音），提取音频并保存为 WAV 文件，
    同时生成一个 metadata.json 描述所有段的时间轴和类型。

    Args:
        audio_path: 原始（人声）音频路径
        output_dir: 导出目录（将创建骨架段子目录）
        noise_db: 静音检测阈值 (dB)
        min_silence_duration: 最小静音段时长 (s)
        min_speech_duration: 最小语音段时长 (s)
        include_silence: 是否同时导出静音段（用于对比）
        include_mixed: 是否导出混合音频（每段前后扩展 200ms 上下文）

    Returns:
        {
            "output_dir": str,
            "total_segments": int,
            "speech_segments": int,
            "silence_segments": int,
            "metadata_path": str,
        }
    """
    import json
    import wave

    from .utils.audio_utils import AudioUtils
    from .vad.ffmpeg_vad import FFmpegSilenceVAD

    output_dir = Path(output_dir)
    segments_dir = output_dir / "skeleton_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 获取声学骨架
    silence_intervals = FFmpegSilenceVAD._detect_silence(
        audio_path, noise_db=noise_db, min_silence_duration=min_silence_duration,
    )
    total_duration = FFmpegSilenceVAD._get_duration(audio_path)
    speech_skeleton = FFmpegSilenceVAD._invert_intervals(
        silence_intervals, total_duration, min_speech_duration=min_speech_duration,
    )

    # Step 2: 加载音频
    audio, sr = AudioUtils.load_audio(audio_path)

    # Step 3: 构建完整的段列表（交替：静音/语音）
    all_segments = []  # [(start, end, type), ...]

    # 开头可能的静音
    cursor = 0.0
    for s_start, s_end in silence_intervals:
        # 静音前的语音
        if cursor < s_start:
            speech_dur = s_start - cursor
            if speech_dur >= min_speech_duration:
                all_segments.append((cursor, s_start, "speech"))
            elif speech_dur > 0:
                all_segments.append((cursor, s_start, "speech_short"))
        # 静音段
        if include_silence and (s_end - s_start) >= min_silence_duration:
            all_segments.append((s_start, s_end, "silence"))
        elif not include_silence:
            all_segments.append((s_start, s_end, "silence"))
        cursor = s_end

    # 最后一段语音
    if cursor < total_duration:
        remaining = total_duration - cursor
        if remaining >= min_speech_duration:
            all_segments.append((cursor, total_duration, "speech"))
        elif remaining > 0:
            all_segments.append((cursor, total_duration, "speech_short"))

    # 如果所有段都是语音（无静音检测到），使用骨架
    if not all_segments and speech_skeleton:
        for s_start, s_end in speech_skeleton:
            all_segments.append((s_start, s_end, "speech"))

    # Step 4: 导出每段
    metadata_segments = []
    speech_count = 0
    silence_count = 0

    for idx, (seg_start, seg_end, seg_type) in enumerate(all_segments):
        start_sample = int(seg_start * sr)
        end_sample = int(seg_end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if end_sample <= start_sample:
            continue

        seg_audio = audio[start_sample:end_sample].copy()

        # 文件名
        type_prefix = {"speech": "S", "speech_short": "SS", "silence": "M"}.get(seg_type, "X")
        time_label = f"{seg_start:.2f}s-{seg_end:.2f}s"
        filename = f"{idx:04d}_{type_prefix}_{time_label}.wav"
        filepath = segments_dir / filename

        AudioUtils.save_audio(seg_audio, filepath, sr)

        meta = {
            "index": idx,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(seg_end - seg_start, 3),
            "type": seg_type,
            "filename": filename,
        }
        metadata_segments.append(meta)

        if "speech" in seg_type:
            speech_count += 1
        else:
            silence_count += 1

    # Step 5: 可选 — 导出带上下文的混合音频（每语音段前后 200ms）
    if include_mixed:
        mixed_dir = output_dir / "skeleton_segments_mixed"
        mixed_dir.mkdir(parents=True, exist_ok=True)
        context_ms = 200

        for meta in metadata_segments:
            if "speech" not in meta["type"]:
                continue

            seg_start = meta["start"]
            seg_end = meta["end"]

            ctx_start = max(0.0, seg_start - context_ms / 1000.0)
            ctx_end = min(total_duration, seg_end + context_ms / 1000.0)

            start_sample = int(ctx_start * sr)
            end_sample = int(ctx_end * sr)
            seg_audio = audio[start_sample:end_sample].copy()

            filename = f"{meta['index']:04d}_CTX_{ctx_start:.2f}s-{ctx_end:.2f}s.wav"
            AudioUtils.save_audio(seg_audio, mixed_dir / filename, sr)

    # Step 6: 写 metadata.json
    metadata = {
        "source_audio": str(audio_path),
        "total_duration": round(total_duration, 3),
        "noise_db": noise_db,
        "min_silence_duration": min_silence_duration,
        "min_speech_duration": min_speech_duration,
        "total_segments": len(metadata_segments),
        "speech_segments": speech_count,
        "silence_segments": silence_count,
        "speech_skeleton": [
            {"start": round(s, 3), "end": round(e, 3)}
            for s, e in speech_skeleton
        ],
        "silence_intervals": [
            {"start": round(s, 3), "end": round(e, 3)}
            for s, e in silence_intervals
        ],
        "segments": metadata_segments,
    }

    metadata_path = segments_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(
        "Exported %d skeleton segments → %s (speech=%d, silence=%d)",
        len(metadata_segments), segments_dir, speech_count, silence_count,
    )

    return {
        "output_dir": str(segments_dir),
        "total_segments": len(metadata_segments),
        "speech_segments": speech_count,
        "silence_segments": silence_count,
        "metadata_path": str(metadata_path),
        "skeleton": speech_skeleton,
        "silence_intervals": silence_intervals,
    }
