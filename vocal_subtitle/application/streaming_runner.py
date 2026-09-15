"""Batch and streaming lifecycle adapters for the subtitle pipeline."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from ..application.pipeline_result import PipelineStats
from ..mapping.time_mapper import SubtitleEvent
from ..pipeline_context import ASRFragment
from ..utils.audio_utils import AudioUtils
from ..utils.progress import ProgressManager

logger = logging.getLogger(__name__)


class PipelineStreamingMixin:
    def run_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        output_format: str = "srt",
        glob_pattern: str = "*.mp3",
        **overrides,
    ) -> list[dict]:
        """批量处理音频文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            output_format: 输出格式
            glob_pattern: 文件匹配模式

        Returns:
            每个文件的结果列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob(glob_pattern))
        logger.info("Batch processing %d files from %s", len(files), input_dir)

        results = []
        for i, file_path in enumerate(files):
            logger.info("[%d/%d] Processing: %s", i + 1, len(files), file_path.name)
            out_path = output_dir / file_path.with_suffix(f".{output_format}").name
            try:
                result = self.run(
                    input_path=file_path,
                    output_path=out_path,
                    output_format=output_format,
                    **overrides,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to process %s: %s", file_path, e)
                results.append({"input_path": file_path, "error": str(e)})

        success = sum(1 for r in results if "error" not in r)
        logger.info("Batch complete: %d/%d succeeded", success, len(results))
        return results

    def run_streaming(
        self,
        input_path: Path,
        output_path: Path,
        output_format: str = "srt",
        progress_callback: callable | None = None,
        skip_separation: bool = False,
        task_id: str | None = None,
        **overrides,
    ) -> dict:
        """流式模式入口（文档 5.12.5）

        使用滑动窗口逐段处理音频，适用于直播/实时字幕场景。
        自动降级依赖全局视角的模块（方案〇/二/七）。

        与离线模式的关键差异：
        - 方案〇 宏观切块 → 禁用（无全局视角）
        - 方案二 三方法融合 → 禁用（简化，仅 Silero VAD）
        - 方案五 LLM 合并 → 降级为本地 NLP + 规则
        - 方案七 声学标尺 → 禁用（无全局标尺）

        Args:
            input_path: 输入音频文件路径
            output_path: 字幕输出路径
            output_format: 输出格式
            progress_callback: 进度回调
            skip_separation: 跳过分离
            task_id: 任务 ID

        Returns:
            dict: 同 run() 的返回格式
        """
        from .streaming import (
            PipelineMode,
            StreamingBuffer,
            StreamingMergeEngine,
        )

        streaming_cfg = self.config.streaming
        pipeline_mode = PipelineMode(
            mode="streaming",
            streaming_chunk_duration=streaming_cfg.chunk_duration,
            streaming_overlap_duration=streaming_cfg.overlap_duration,
            streaming_max_latency=streaming_cfg.max_latency,
        )

        logger.info(
            "Streaming mode: chunk=%.1fs, overlap=%.1fs, max_latency=%.1fs",
            pipeline_mode.streaming_chunk_duration,
            pipeline_mode.streaming_overlap_duration,
            pipeline_mode.streaming_max_latency,
        )

        start_time = time.time()
        stats = PipelineStats(input_path=input_path, duration_seconds=0)

        # ---- Stage 1: 人声分离（如果未跳过） ----
        separation_result = None
        if not skip_separation:
            self._progress = ProgressManager(
                total_stages=1,
                callback=progress_callback,
            )
            self._progress.start_stage("separation", description="人声分离")
            separation_result = self._run_separation(input_path)
            stats.stage_timings["separation"] = separation_result.processing_time
            vocals_path = separation_result.vocals_path
            self._progress.finish_stage()
        else:
            vocals_path = input_path

        # ---- 加载完整音频 ----
        audio, sample_rate = AudioUtils.load_audio(vocals_path)
        stats.duration_seconds = len(audio) / sample_rate

        # ---- 确定活跃模块（流式降级） ----
        active = self._resolve_active_modules()
        logger.info(
            "Streaming active modules: %s",
            {k: v for k, v in active.items() if v},
        )

        # ---- 初始化流式组件 ----
        buffer = StreamingBuffer(
            chunk_duration=pipeline_mode.streaming_chunk_duration,
            overlap_duration=pipeline_mode.streaming_overlap_duration,
        )
        merge_engine = StreamingMergeEngine() if active.get("llm_merge") else None

        total_stages = 4  # vad + merging + asr + mapping
        if active.get("pre_split"):
            total_stages += 1
        if active.get("asr_refine"):
            total_stages += 1
        self._progress = ProgressManager(
            total_stages=total_stages,
            callback=progress_callback,
        )

        # ---- 流式处理循环 ----
        # 将整段音频拆分为滑动窗口
        # 注意：当前实现将完整文件加载后模拟流式处理，
        # 真实流式场景中 audio_stream 来自麦克风或网络
        all_events: list[SubtitleEvent] = []
        total_segments = 0
        window_index = 0

        total_samples = len(audio)
        hop_samples = int(pipeline_mode.streaming_chunk_duration * sample_rate)
        overlap_samples = int(pipeline_mode.streaming_overlap_duration * sample_rate)
        stride = hop_samples - overlap_samples
        if stride <= 0:
            stride = hop_samples // 2

        # 模拟流式音频迭代器
        def _audio_stream():
            pos = 0
            while pos < total_samples:
                chunk_end = min(pos + hop_samples, total_samples)
                yield audio[pos:chunk_end]
                pos += stride

        for audio_chunk in _audio_stream():
            window_index += 1
            buffer.append(audio_chunk)

            if not buffer.ready():
                continue

            window_audio = buffer.get_window()
            if len(window_audio) == 0:
                buffer.advance()
                continue

            window_start_time = (window_index - 1) * stride / sample_rate

            logger.debug(
                "Streaming window %d: %.1fs → %.1fs (size=%d samples)",
                window_index,
                window_start_time,
                window_start_time + len(window_audio) / sample_rate,
                len(window_audio),
            )

            # 为窗口创建临时 WAV
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as tmp_f:
                tmp_path = Path(tmp_f.name)
            try:
                AudioUtils.save_audio(window_audio, tmp_path, sample_rate)

                # 在滑动窗口内执行精简版 Pipeline
                chunk_events, chunk_seg_count, _win_ctx = self._process_chunk_pipeline(
                    audio=window_audio,
                    sample_rate=sample_rate,
                    vocals_path=tmp_path,
                    chunk_label=f"Win {window_index}",
                    parallel_vad=False,  # 流式窗口嵌套线程，避免 PyTorch 死锁
                )
            finally:
                tmp_path.unlink(missing_ok=True)

            # 偏移到全局时间轴
            for evt in chunk_events:
                evt.start += window_start_time
                evt.end += window_start_time

            # 流式合并决策（本地 NLP + 规则）
            if merge_engine and len(all_events) > 0:
                # 用本地模型决定当前窗口首事件是否与上一窗口末事件合并
                prev = all_events[-1]
                for i, curr in enumerate(chunk_events):
                    if curr.start >= prev.end - 0.01:
                        should_merge = merge_engine.decide_merge_streaming(
                            ASRFragment(
                                index=0,
                                start=curr.start,
                                end=curr.end,
                                text=curr.text,
                            ),
                            ASRFragment(
                                index=0,
                                start=prev.start,
                                end=prev.end,
                                text=prev.text,
                            ),
                        )
                        if should_merge:
                            prev.end = curr.end
                            prev.text = f"{prev.text} {curr.text}".strip()
                            chunk_events = chunk_events[i + 1 :]
                        break

            total_segments += chunk_seg_count
            all_events.extend(chunk_events)

            buffer.advance()

        # ---- 后处理 ----
        # 去重（重叠窗口可能产生重复事件）
        if len(all_events) > 1:
            deduped: list[SubtitleEvent] = [all_events[0]]
            for evt in all_events[1:]:
                if abs(evt.start - deduped[-1].start) < 0.05:
                    # 保留文本更长的
                    if len(evt.text) > len(deduped[-1].text):
                        deduped[-1] = evt
                    continue
                deduped.append(evt)
            all_events = deduped

        # 重新编号
        for i, evt in enumerate(all_events):
            evt.index = i + 1

        # 帧级无缝衔接
        if active.get("frame_seamless", True):
            try:
                from ..merging.llm_merge_engine import apply_frame_seamless_stitching

                stitch_gap = self.config.subtitle.max_stitch_gap
                all_events = apply_frame_seamless_stitching(
                    all_events,
                    max_stitch_gap=stitch_gap,
                )
            except Exception as e:
                logger.warning("Frame seamless stitching failed: %s", e)

        all_events = self._finalize_events(all_events, stats, stats.duration_seconds)
        stats.segment_count = total_segments
        stats.total_time = time.time() - start_time

        # 输出字幕
        builder = self._get_subtitle_builder()
        builder.build(all_events, output_path, fmt=output_format)

        logger.info(
            "Streaming pipeline complete: %.1fs total, %d windows, %d events",
            stats.total_time,
            window_index,
            stats.subtitle_count,
        )

        return {
            "subtitle_path": output_path,
            "stats": stats,
            "events": all_events,
            "from_cache": False,
            "vocals_path": str(separation_result.vocals_path)
            if separation_result
            else None,
            "accompaniment_path": str(separation_result.accompaniment_path)
            if separation_result
            else None,
        }

    # ------------------------------------------------------------------
    # Stage 实现方法
    # ------------------------------------------------------------------
