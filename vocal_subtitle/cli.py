"""CLI 命令行入口模块

基于 Click 框架的命令行工具。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .application.pipeline_result import PipelineStats
from .cli_commands.administration import register as register_administration
from .cli_commands.common import (
    device_option as _device_option,
    language_option as _language_option,
    output_format_option as _output_format_option,
    profile_option as _profile_option,
    verbose_option as _verbose_option,
)
from .cli_commands.feedback_commands import register as register_feedback
from .config import ConfigLoader
from .pipeline import Pipeline


# ---------------------------------------------------------------------------
# CLI 组
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.2.0", prog_name="vocal-subtitle")
def main():
    """人声分离 + 字幕生成全链路工具

    从原始音频/视频文件中提取人声并生成精准字幕。
    """
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="字幕输出路径")
@_profile_option
@_output_format_option
@_device_option
@_language_option
@click.option("--separator", default=None, help="分离引擎 (uvr / openunmix / spleeter)")
@click.option("--uvr-model", default=None, help="UVR 模型文件名")
@click.option("--vad-threshold", type=float, default=None, help="VAD 阈值 (0.0–1.0)")
@click.option(
    "--asr-engine",
    type=click.Choice(["auto", "funasr", "qwen", "faster-whisper", "whisper-cpp"]),
    default=None,
    help="主 ASR 引擎；auto 按语言选择主副引擎组合",
)
@click.option("--asr-model", default=None, help="Whisper 模型 (large-v3 / medium / small / tiny)")
@click.option(
    "--primary-engine",
    type=click.Choice(["auto", "funasr", "qwen", "faster-whisper", "whisper-cpp"]),
    default=None,
    help="主引擎配对覆盖；auto 使用语言策略",
)
@click.option(
    "--secondary-engine",
    type=click.Choice(["auto", "funasr", "qwen", "faster-whisper", "whisper-cpp"]),
    default=None,
    help="副引擎配对覆盖；auto 使用语言策略",
)
@click.option(
    "--engine-pair-policy",
    type=click.Choice(["risk_only", "full_quality"]),
    default=None,
    help="主副引擎策略",
)
@click.option("--asr-path", default=None, type=click.Choice(["auto", "global", "segmented"]),
              help="ASR 路径: global (全音频一次识别) 或 segmented (VAD 分段识别，默认)")
@click.option("--llm-optimize", is_flag=True, default=None, help="启用 LLM 字幕优化")
@click.option("--diarization/--no-diarization", default=None, help="启用/禁用说话人分离")
@click.option("--expected-speakers", type=click.IntRange(min=1), default=None,
              help="已知说话人数；省略则自动估计")
@click.option("--speaker-fusion", type=click.Choice(["auto", "embedding", "dual"]),
              default=None, help="说话人融合模式")
@click.option("--global-diarization-model",
              type=click.Choice(["auto", "none", "community-1", "diarization-3.1"]),
              default=None, help="全局 diarization 模型")
@click.option("--speaker-diarization-scope",
              type=click.Choice(["global", "hierarchical"]), default=None,
              help="全局 turns 或全局加逐字幕局部精修")
@click.option("--local-speaker-refinement",
              type=click.Choice(["off", "embedding", "full"]), default=None,
              help="逐字幕局部换人检测级别")
@click.option("--speaker-role/--no-speaker-role", default=None, help="启用/禁用 LLM 角色标注")
@click.option("--skip-separation", is_flag=True, help="跳过分离阶段（输入已是人声）")
@click.option("--skeleton-mode", is_flag=True, default=None,
              help="骨架分段模式：按 ffmpeg 声学骨架逐段独立处理，然后拼接时间轴")
@click.option("--export-skeleton-segments", is_flag=True, default=None,
              help="导出声学骨架的语音/静音段为独立音频文件，供人工验证")
@click.option("--export-skeleton-dir", default=None,
              help="骨架段导出目录（默认：输出目录下的 skeleton_export/）")
@_verbose_option
def run(
    input_path: str,
    output: Optional[str],
    profile: str,
    output_format: str,
    device: Optional[str],
    language: Optional[str],
    separator: Optional[str],
    uvr_model: Optional[str],
    vad_threshold: Optional[float],
    asr_engine: Optional[str],
    asr_model: Optional[str],
    primary_engine: Optional[str],
    secondary_engine: Optional[str],
    engine_pair_policy: Optional[str],
    asr_path: Optional[str],
    llm_optimize: Optional[bool],
    diarization: Optional[bool],
    expected_speakers: Optional[int],
    speaker_fusion: Optional[str],
    global_diarization_model: Optional[str],
    speaker_diarization_scope: Optional[str],
    local_speaker_refinement: Optional[str],
    speaker_role: Optional[bool],
    skip_separation: bool,
    skeleton_mode: Optional[bool],
    export_skeleton_segments: Optional[bool],
    export_skeleton_dir: Optional[str],
    verbose: bool,
):
    """处理单个音频文件，生成字幕

    \b
    Examples:
        vocal-subtitle run input.mp3 -o output.srt
        vocal-subtitle run input.mp3 --profile podcast --language zh
        vocal-subtitle run input.mp3 --separator uvr --uvr-model model_bs_roformer.ckpt
        vocal-subtitle run input.wav --skip-separation --llm-optimize
        vocal-subtitle run input.wav --skip-separation --skeleton-mode
        vocal-subtitle run input.wav --skip-separation --export-skeleton-segments
    """
    # 加载配置
    loader = ConfigLoader()
    config = loader.load_profile(profile)

    # 处理命令行覆盖参数
    overrides = {}
    if separator:
        overrides["separator"] = separator
    if uvr_model:
        overrides["uvr_model"] = uvr_model
    if vad_threshold is not None:
        overrides["vad_threshold"] = vad_threshold
    if asr_model:
        overrides["asr_model"] = asr_model
    if asr_engine:
        overrides["asr_engine"] = asr_engine
    if primary_engine:
        overrides["primary_engine"] = primary_engine
    if secondary_engine:
        overrides["secondary_engine"] = secondary_engine
    if engine_pair_policy:
        overrides["engine_pair_policy"] = engine_pair_policy
    if asr_path:
        overrides["asr_path"] = asr_path
    if device:
        overrides["device"] = device
    if language:
        overrides["language"] = language
    if llm_optimize is not None:
        overrides["llm_optimize"] = llm_optimize
    if diarization is not None:
        overrides["diarization"] = diarization
    if expected_speakers is not None:
        overrides["expected_speakers"] = expected_speakers
    if speaker_fusion:
        overrides["speaker_fusion"] = speaker_fusion
    if global_diarization_model:
        overrides["global_diarization_model"] = global_diarization_model
    if speaker_diarization_scope:
        overrides["speaker_diarization_scope"] = speaker_diarization_scope
    if local_speaker_refinement:
        overrides["local_speaker_refinement"] = local_speaker_refinement
    if speaker_role is not None:
        overrides["speaker_role"] = speaker_role

    config = loader.merge_with_overrides(config, **overrides)

    # 骨架模式覆盖（直接设置，不走 overrides 字典）
    if skeleton_mode is not None:
        config.acoustic_validation.skeleton_mode = skeleton_mode
    if export_skeleton_segments is not None:
        config.acoustic_validation.export_skeleton_segments = export_skeleton_segments
    if export_skeleton_dir is not None:
        config.acoustic_validation.export_skeleton_dir = export_skeleton_dir

    if verbose:
        config.logging.level = "DEBUG"

    # 构建输出路径
    input_file = Path(input_path)
    if output:
        output_path = Path(output)
    else:
        output_path = input_file.with_suffix(f".{output_format}")

    # 执行管道
    click.echo(f"处理: {input_file} → {output_path}")
    click.echo(f"配置: profile={profile}, format={output_format}")

    pipeline = Pipeline(config)

    try:
        result = pipeline.run(
            input_path=input_file,
            output_path=output_path,
            output_format=output_format,
            skip_separation=skip_separation,
        )

        stats: PipelineStats = result["stats"]
        click.echo(f"\n✓ 处理结束: status={stats.status}, quality={stats.quality_status} ({stats.total_time:.1f}s)")
        if stats.error_category:
            click.echo(f"  错误/降级类别: {stats.error_category}")
        if stats.fallback_reason:
            click.echo(f"  回退原因: {stats.fallback_reason}")
        click.echo(f"  片段数: {stats.segment_count}")
        click.echo(f"  字幕条数: {stats.subtitle_count}")
        click.echo(
            f"  说话人: {stats.speaker_count or 0}"
            f" (backend={stats.diarization_backend or 'unknown'},"
            f" status={stats.diarization_status or 'unknown'})"
        )
        click.echo(
            f"  局部换人切分: {stats.local_speaker_split_count}"
            f" (冲突={stats.speaker_conflict_count},"
            f" unknown={stats.unknown_speaker_count})"
        )
        click.echo(f"  输出: {result['subtitle_path']}")

        if verbose:
            click.echo(f"\n各阶段耗时:")
            for stage, elapsed in stats.stage_timings.items():
                click.echo(f"  {stage}: {elapsed:.1f}s")

    except Exception as e:
        click.echo(f"\n✗ 处理失败: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="输出目录")
@_profile_option
@_output_format_option
@_device_option
@_language_option
@click.option("--pattern", default="*.mp3", help="文件匹配模式 (默认: *.mp3)")
@click.option("--separator", default=None, help="分离引擎 (uvr / openunmix / spleeter)")
@click.option("--uvr-model", default=None, help="UVR 模型文件名")
@click.option("--skip-separation", is_flag=True, help="跳过分离阶段")
@_verbose_option
def batch(
    input_dir: str,
    output: Optional[str],
    profile: str,
    output_format: str,
    device: Optional[str],
    language: Optional[str],
    pattern: str,
    separator: Optional[str],
    uvr_model: Optional[str],
    skip_separation: bool,
    verbose: bool,
):
    """批量处理目录中的音频文件

    \b
    Examples:
        vocal-subtitle batch ./inputs/ -o ./outputs/
        vocal-subtitle batch ./videos/ --pattern "*.mp4" --profile education
    """
    input_path = Path(input_dir)
    output_dir = Path(output) if output else input_path / "subtitles"

    # 加载配置
    loader = ConfigLoader()
    config = loader.load_profile(profile)

    overrides = {}
    if device:
        overrides["device"] = device
    if language:
        overrides["language"] = language
    if separator:
        overrides["separator"] = separator
    if uvr_model:
        overrides["uvr_model"] = uvr_model
    config = loader.merge_with_overrides(config, **overrides)

    if verbose:
        config.logging.level = "DEBUG"

    files = sorted(input_path.glob(pattern))
    if not files:
        click.echo(f"无匹配文件: {input_dir}/{pattern}")
        return

    click.echo(f"批量处理: {len(files)} 个文件")
    click.echo(f"输出目录: {output_dir}")

    pipeline = Pipeline(config)
    results = pipeline.run_batch(
        input_dir=input_path,
        output_dir=output_dir,
        output_format=output_format,
        glob_pattern=pattern,
        skip_separation=skip_separation,
    )

    # 统计
    success = sum(1 for r in results if "error" not in r)
    failed = len(results) - success
    click.echo(f"\n✓ 成功: {success}, ✗ 失败: {failed}")

    if verbose and failed > 0:
        for r in results:
            if "error" in r:
                click.echo(f"  ✗ {r['input_path']}: {r['error']}")



# Register command modules after the root group and built-in run/batch commands exist.
register_administration(main)
register_feedback(main)


if __name__ == "__main__":
    main()
