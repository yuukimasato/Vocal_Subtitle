"""Export acoustic skeleton intervals for manual inspection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

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

    from ..utils.audio_utils import AudioUtils
    from ..vad.ffmpeg_vad import FFmpegSilenceVAD

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

