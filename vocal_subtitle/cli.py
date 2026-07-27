"""CLI 命令行入口模块

基于 Click 框架的命令行工具。
"""

import sys
from pathlib import Path
from typing import Optional

import click

from .config import ConfigLoader
from .pipeline import Pipeline, PipelineStats


# ---------------------------------------------------------------------------
# 公共参数
# ---------------------------------------------------------------------------

def _profile_option(f):
    """场景模板选项装饰器"""
    return click.option(
        "--profile", "-p",
        default="default",
        help="场景模板 (default / podcast / education / variety_show / music_live)",
    )(f)


def _output_format_option(f):
    """输出格式选项装饰器"""
    return click.option(
        "--format", "-f", "output_format",
        default="srt",
        type=click.Choice(["srt", "vtt", "ass"]),
        help="输出字幕格式",
    )(f)


def _device_option(f):
    """设备选项装饰器"""
    return click.option(
        "--device", "-d",
        default=None,
        type=click.Choice(["cuda", "cpu"]),
        help="推理设备（默认自动检测）",
    )(f)


def _language_option(f):
    """语言选项装饰器"""
    return click.option(
        "--language", "-l",
        default=None,
        help="语言代码 (zh/en/ja/...)",
    )(f)


def _verbose_option(f):
    """详细输出选项装饰器"""
    return click.option(
        "--verbose", "-v",
        is_flag=True,
        help="详细输出",
    )(f)


# ---------------------------------------------------------------------------
# CLI 组
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="vocal-subtitle")
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
@click.option("--asr-model", default=None, help="ASR 模型 (large-v3 / medium / small)")
@click.option("--asr-path", default=None, type=click.Choice(["auto", "global", "segmented"]),
              help="ASR 路径: global (全音频一次识别) 或 segmented (VAD 分段识别，默认)")
@click.option("--llm-optimize", is_flag=True, default=None, help="启用 LLM 字幕优化")
@click.option("--diarization/--no-diarization", default=None, help="启用/禁用说话人分离")
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
    asr_model: Optional[str],
    asr_path: Optional[str],
    llm_optimize: Optional[bool],
    diarization: Optional[bool],
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
        click.echo(f"\n✓ 完成! ({stats.total_time:.1f}s)")
        click.echo(f"  片段数: {stats.segment_count}")
        click.echo(f"  字幕条数: {stats.subtitle_count}")
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


@main.command()
@click.option("--all", "download_all", is_flag=True, help="下载所有模型")
@click.option("--asr-model", default=None, help="ASR 模型 (large-v3 / medium / small)")
@click.option("--separator", default=None, help="分离引擎 (uvr / spleeter)")
def download_models(
    download_all: bool,
    asr_model: Optional[str],
    separator: Optional[str],
):
    """预下载模型文件

    \b
    Examples:
        vocal-subtitle download-models --all
        vocal-subtitle download-models --asr-model large-v3
    """
    click.echo("模型下载功能将在后续版本实现")
    click.echo("当前请手动放置模型文件到 ~/.cache/vocal-subtitle/")


@main.command()
def profiles():
    """列出可用的场景模板"""
    loader = ConfigLoader()
    click.echo("可用场景模板:")
    for name in loader.list_profiles():
        click.echo(f"  - {name}")


@main.command()
def info():
    """显示系统和设备信息"""
    from .utils.gpu_detector import GPUDetector

    click.echo("=== 系统信息 ===")
    import platform

    click.echo(f"  操作系统: {platform.system()} {platform.release()}")
    click.echo(f"  Python: {platform.python_version()}")

    click.echo("\n=== GPU 信息 ===")
    device_info = GPUDetector.get_device_info()
    click.echo(f"  最佳设备: {device_info['device_type']}")
    click.echo(f"  设备数量: {device_info['device_count']}")
    if device_info["device_names"]:
        click.echo(f"  设备名称: {', '.join(device_info['device_names'])}")
    if device_info["memory_mb"]:
        click.echo(f"  显存 (MB): {device_info['memory_mb']}")
    click.echo(f"  推荐计算精度: {device_info['recommended_compute_type']}")
    click.echo(f"  推荐模型: {GPUDetector.select_whisper_model(GPUDetector.get_best_device())}")

    gpu_mem = GPUDetector.get_gpu_memory_used_mb()
    if gpu_mem is not None:
        click.echo(f"  当前 GPU 显存使用: {gpu_mem:.0f} MB")


# ---------------------------------------------------------------------------
# feedback 子命令组 (Phase 5: 自适应反馈学习)
# ---------------------------------------------------------------------------


@main.group()
def feedback():
    """基于用户修订字幕的自适应参数学习

    上传你对自动生成字幕的修订版（.srt/.ass）+ 原始音频，
    系统自动分析差异并调整管道参数，逐步学习你的偏好。
    """
    pass


@feedback.command()
@click.option("--audio", "-a", required=True, type=click.Path(exists=True),
              help="原始音频文件路径")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True),
              help="用户修订的字幕文件 (.srt / .ass)")
@click.option("--profile", "-p", default="default",
              help="场景模板 (default / podcast / education / variety_show / music_live)")
@click.option("--feedback-profile", default="user_default",
              help="用户配置名称 (默认: user_default)")
@click.option("--dry-run", is_flag=True,
              help="仅预览差异，不实际更新配置")
@_verbose_option
def learn(
    audio: str,
    reference: str,
    profile: str,
    feedback_profile: str,
    dry_run: bool,
    verbose: bool,
):
    """上传修订字幕 + 音频，自动学习用户偏好

    \b
    Examples:
        vocal-subtitle feedback learn -a audio.wav -r revised.srt
        vocal-subtitle feedback learn -a audio.wav -r revised.ass --dry-run
        vocal-subtitle feedback learn -a input.mp3 -r fixed.srt --profile podcast
    """
    from pathlib import Path

    from .config import ConfigLoader
    from .feedback import (
        AudioFingerprinter,
        DiffAnalyzer,
        FewShotBuilder,
        ImpactEstimator,
        ParamLearner,
        SubtitleAligner,
        UserProfileManager,
    )
    from .feedback.aligner import parse_subtitle_file
    from .feedback.conflict_detector import ConflictDetector
    from .feedback.health_scorer import compute_health_score_from_pairs, should_auto_rollback
    from .pipeline import Pipeline

    audio_path = Path(audio)
    reference_path = Path(reference)

    if reference_path.suffix.lower() not in (".srt", ".ass"):
        click.echo(f"✗ 不支持的字幕格式: {reference_path.suffix}，仅支持 .srt / .ass", err=True)
        raise SystemExit(1)

    # 加载配置
    loader = ConfigLoader()
    config = loader.load_profile(profile)

    click.echo(f"📖 解析修订字幕: {reference_path}")
    manual_events = parse_subtitle_file(reference_path)
    if not manual_events:
        click.echo("✗ 未从修订文件中解析到字幕事件", err=True)
        raise SystemExit(1)
    click.echo(f"   解析到 {len(manual_events)} 条字幕")

    # 生成自动版字幕
    click.echo(f"🎤 运行管道生成自动字幕: {audio_path}")
    pipeline = Pipeline(config)
    try:
        result = pipeline.run(
            input_path=audio_path,
            skip_separation=True,  # 反馈模式默认跳过分离
        )
        auto_events = result.get("events", [])
        click.echo(f"   生成 {len(auto_events)} 条自动字幕")
    except Exception as e:
        click.echo(f"✗ 管道运行失败: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)

    # 对齐
    click.echo("🔗 对齐自动版与修订版...")
    feedback_cfg = config.feedback
    aligner = SubtitleAligner(
        min_iou=feedback_cfg.alignment_min_iou,
        min_coverage=feedback_cfg.alignment_min_coverage,
        text_weight=feedback_cfg.alignment_text_weight,
        semantic_weight=feedback_cfg.alignment_semantic_weight,
        semantic_enabled=feedback_cfg.alignment_semantic_enabled,
    )
    try:
        pairs = aligner.align(auto_events, manual_events)
    except Exception as e:
        click.echo(f"✗ 对齐失败: {e}", err=True)
        raise SystemExit(1)

    matched = [p for p in pairs if p.is_matched]
    click.echo(f"   对齐覆盖率: {len(matched)}/{max(len(auto_events), len(manual_events))} "
               f"({len(matched)/max(len(auto_events), len(manual_events))*100:.1f}%)")

    # 差异分析
    click.echo("🔍 分析差异...")
    analyzer = DiffAnalyzer(
        param_isolation_enabled=feedback_cfg.param_isolation_enabled,
    )
    diff_report = analyzer.analyze(pairs)

    # 健康度评分
    health_before, health_detail = compute_health_score_from_pairs(pairs)
    click.echo(f"\n💊 健康度评分: {health_before:.1f}/100")
    if verbose:
        for dim, score in health_detail.items():
            click.echo(f"   {dim}: {score:.1f}")

    # 展示报告
    click.echo(f"\n📊 差异分析报告:")
    click.echo(f"   时间偏移: {len(diff_report.time_shifts)} 处")
    click.echo(f"   合并/拆分: {len(diff_report.merge_actions)} 处")
    click.echo(f"   文本修改: {len(diff_report.text_edits)} 处")

    if diff_report.attribution:
        click.echo(f"\n📈 参数调整建议:")
        for param_path, adj in diff_report.attribution.items():
            direction_icon = "↑" if adj.direction == "increase" else "↓"
            click.echo(f"   {direction_icon} {param_path}: {adj.reason}")
            click.echo(f"     置信度: {adj.confidence:.2f}, 学习权重: {adj.learn_weight:.2f}, 分级: {adj.param_tier}")

        # 影响预估
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            diff_report.attribution,
            {},  # current_overrides (not needed for estimate direction)
        )
        if impacts:
            click.echo(f"\n📊 变更影响预估:")
            for impact in impacts:
                click.echo(f"   {impact.summary}")
    else:
        click.echo("\n✅ 无需调整参数（已匹配用户偏好）")

    if diff_report.structural_revision:
        click.echo("\n⚠️  检测到结构性修订（大幅增删/重排序），学习权重已降低。")

    # 震荡检测
    profile_mgr = UserProfileManager(feedback_cfg)
    profile = profile_mgr.load(feedback_profile)
    history = profile.get("history", [])
    detector = ConflictDetector(window=feedback_cfg.oscillation_detection_window)
    conflicts = detector.detect_all_oscillations(history)
    if conflicts:
        click.echo(f"\n⚠️  参数震荡检测: {len(conflicts)} 个参数出现震荡")
        for cr in conflicts:
            click.echo(f"   {cr.param_path}: {cr.oscillation_count} 次翻转 (建议: {cr.recommended_action})")

    # 应用学习（非 dry-run 模式）
    if dry_run:
        click.echo("\n🔍 [dry-run 模式] 未实际更新配置。")
    elif diff_report.attribution:
        click.echo(f"\n📚 更新用户配置: {feedback_profile}...")

        current_overrides = profile.get("overrides", {})

        learner = ParamLearner(profile_mgr)
        updated = learner.learn_from_diff(
            diff_report=diff_report,
            current_config_overrides=current_overrides,
            profile_name=feedback_profile,
        )

        click.echo(f"   已学习 {len(updated)} 个参数覆盖")
        for k, v in sorted(updated.items()):
            click.echo(f"   {k}: {v}")

        # Few-shot 缓存
        if feedback_cfg.few_shot_enabled:
            few_shot = FewShotBuilder(max_examples=feedback_cfg.few_shot_max_examples)
            few_shot.load_cache(feedback_profile)
            few_shot.build_merge_examples(diff_report.merge_actions)
            if diff_report.text_edits:
                few_shot.build_format_examples(diff_report.text_edits)
            few_shot.save_cache(feedback_profile)

        # 音频指纹提取与存储
        if feedback_cfg.fingerprint_enabled:
            try:
                fingerprinter = AudioFingerprinter(
                    distance_method=feedback_cfg.fingerprint_distance_method,
                    knn_k=feedback_cfg.fingerprint_knn_k,
                    min_absolute_similarity=feedback_cfg.fingerprint_min_absolute_similarity,
                    relative_margin=feedback_cfg.fingerprint_relative_margin,
                )
                fp = fingerprinter.extract(audio_path)
                if fp is not None:
                    audio_hash = AudioFingerprinter.compute_audio_hash(audio_path)
                    fingerprinter.store(
                        profile_id=feedback_profile,
                        fingerprint=fp,
                        audio_hash=audio_hash,
                        config_snapshot=updated,
                    )
                    fingerprinter.record_feedback(
                        profile_id=feedback_profile,
                        audio_hash=audio_hash,
                        alignment_coverage=diff_report.alignment_coverage,
                        diff_summary="; ".join(
                            adj.reason for adj in diff_report.attribution.values()
                        ),
                        adjustments={
                            k: [adj.observed_value, adj.confidence]
                            for k, adj in diff_report.attribution.items()
                        },
                        health_before=health_before,
                        health_after=health_before,  # same pairs, health is relative to this session
                        health_detail=health_detail,
                    )
                    click.echo(f"\n🔊 音频指纹已存储: {fp.audio_signature}")
            except Exception as e:
                if verbose:
                    click.echo(f"   ⚠️ 指纹提取失败: {e}")

        click.echo(f"\n✓ 学习完成！下次运行将自动应用学习到的参数偏好。")
        click.echo(f"  使用 --feedback-profile {feedback_profile} 指定此配置")
    else:
        click.echo("\n✓ 无参数变更。")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default",
              help="用户配置名称 (默认: user_default)")
@_verbose_option
def show(feedback_profile: str, verbose: bool):
    """查看当前用户配置（含学习到的参数偏离量）

    \b
    Examples:
        vocal-subtitle feedback show
        vocal-subtitle feedback show --profile user_default
    """
    from .config import ConfigLoader, FeedbackConfig
    from .feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(feedback_profile)

    click.echo(f"📋 用户配置: {feedback_profile}")
    click.echo(f"   基础模板: {profile.get('base_profile', 'default')}")
    click.echo(f"   反馈次数: {profile.get('feedback_count', 0)}")
    click.echo(f"   创建时间: {profile.get('created_at', 'N/A')}")
    click.echo(f"   更新时间: {profile.get('updated_at', 'N/A')}")

    overrides = profile.get("overrides", {})
    if overrides:
        click.echo(f"\n📈 参数覆盖 ({len(overrides)} 项):")
        _print_nested_dict(overrides)
    else:
        click.echo("\n   无参数覆盖（使用系统默认值）")

    history = profile.get("history", [])
    if history and verbose:
        click.echo(f"\n📝 最近学习记录 ({min(5, len(history))}/{len(history)}):")
        for entry in history[-5:]:
            click.echo(f"   [{entry.get('timestamp', '?')[:19]}] {entry.get('diff_report_summary', 'N/A')}")
            adjustments = entry.get("adjustments", {})
            if adjustments:
                for param, (old, new) in adjustments.items():
                    click.echo(f"      {param}: {old} → {new}")

    few_shot = profile.get("few_shot_examples", [])
    if few_shot:
        click.echo(f"\n🎯 Few-shot 示例: {len(few_shot)} 条")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default",
              help="用户配置名称 (默认: user_default)")
def rollback(feedback_profile: str):
    """回滚用户配置到上一个备份版本

    \b
    Examples:
        vocal-subtitle feedback rollback
    """
    from .config import FeedbackConfig
    from .feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    try:
        profile = mgr.rollback(feedback_profile)
        click.echo(f"✓ 已回滚配置 '{feedback_profile}' 到上一版本")
        click.echo(f"   更新时间: {profile.get('updated_at', 'N/A')}")
        click.echo(f"   反馈次数: {profile.get('feedback_count', 0)}")
    except FileNotFoundError:
        click.echo(f"✗ 无可用备份: {feedback_profile}", err=True)
        raise SystemExit(1)


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default",
              help="用户配置名称 (默认: user_default)")
@click.confirmation_option(prompt="确认重置？此操作不可撤销")
def reset(feedback_profile: str):
    """重置用户配置为系统默认（清除所有学习记录）

    \b
    Examples:
        vocal-subtitle feedback reset
    """
    from .config import FeedbackConfig
    from .feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    mgr.reset(feedback_profile)
    click.echo(f"✓ 已重置配置 '{feedback_profile}' 为系统默认")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default",
              help="用户配置名称 (默认: user_default)")
def fingerprints(feedback_profile: str):
    """列出所有音频指纹→参数映射

    \b
    Examples:
        vocal-subtitle feedback fingerprints
    """
    from .config import FeedbackConfig
    from .feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )

    fps = fingerprinter.list_all()
    if not fps:
        click.echo("📭 指纹库为空。提交一次反馈学习后自动生成指纹。")
        return

    click.echo(f"🔊 音频指纹库 ({len(fps)} 条):\n")
    for fp in fps:
        click.echo(f"  [{fp['id']}] {fp.get('audio_signature', 'N/A')}")
        click.echo(f"       Profile: {fp['profile_id']}")
        click.echo(f"       Audio Hash: {fp['audio_hash'][:16]}...")
        click.echo(f"       Feedback Count: {fp['feedback_count']}")
        click.echo(f"       Created: {fp.get('created_at', 'N/A')[:19]}")
        click.echo()

    click.echo(f"数据库路径: {fingerprinter._db_path}")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default",
              help="要导出的用户配置名称")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="输出 YAML 文件路径")
def export(feedback_profile: str, output: str):
    """导出用户配置（用于分享或备份）

    \b
    Examples:
        vocal-subtitle feedback export -o my_profile.yaml
        vocal-subtitle feedback export --profile user_default -o ~/backup.yaml
    """
    import shutil

    from .config import FeedbackConfig
    from .feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(feedback_profile)

    output_path = Path(output)
    import yaml

    # 清理内部字段后导出
    export_data = {
        "profile_id": profile.get("profile_id"),
        "base_profile": profile.get("base_profile", "default"),
        "description": profile.get("description", ""),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "feedback_count": profile.get("feedback_count", 0),
        "overrides": profile.get("overrides", {}),
        "fingerprint": profile.get("fingerprint", {}),
        "few_shot_examples": profile.get("few_shot_examples", []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(export_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    click.echo(f"✓ 已导出配置 '{feedback_profile}' → {output_path}")
    click.echo(f"   参数覆盖: {len(export_data['overrides'])} 项")
    click.echo(f"   Few-shot 示例: {len(export_data['few_shot_examples'])} 条")


@feedback.command()
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True),
              help="要导入的 YAML 配置文件")
@click.option("--as", "profile_name", default=None,
              help="导入后的配置名称（默认使用文件中的 profile_id）")
def import_(input_file: str, profile_name: str):
    """导入他人分享的用户配置

    \b
    Examples:
        vocal-subtitle feedback import -i friend_profile.yaml
        vocal-subtitle feedback import -i custom.yaml --as user_custom
    """
    import yaml

    from .config import FeedbackConfig
    from .feedback import UserProfileManager

    input_path = Path(input_file)
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not data.get("profile_id") and not profile_name:
        click.echo("✗ 导入文件中缺少 profile_id，请使用 --as 指定配置名称", err=True)
        raise SystemExit(1)

    target_name = profile_name or data["profile_id"]
    mgr = UserProfileManager(FeedbackConfig())

    # 构建新 profile
    profile = mgr.load(target_name)  # 作为基础模板
    profile["profile_id"] = target_name
    profile["base_profile"] = data.get("base_profile", "default")
    profile["description"] = data.get("description", f"Imported from {input_path.name}")
    profile["overrides"] = data.get("overrides", {})
    profile["few_shot_examples"] = data.get("few_shot_examples", [])

    if data.get("fingerprint"):
        profile["fingerprint"] = data["fingerprint"]

    mgr.save(profile)
    click.echo(f"✓ 已导入配置 → '{target_name}'")
    click.echo(f"   参数覆盖: {len(profile['overrides'])} 项")
    click.echo(f"   Few-shot 示例: {len(profile['few_shot_examples'])} 条")


def _print_nested_dict(d: dict, indent: int = 4) -> None:
    """递归打印嵌套字典"""
    for key, value in sorted(d.items()):
        if isinstance(value, dict):
            click.echo(f"{' ' * indent}{key}:")
            _print_nested_dict(value, indent + 2)
        else:
            click.echo(f"{' ' * indent}{key}: {value}")


if __name__ == "__main__":
    main()
