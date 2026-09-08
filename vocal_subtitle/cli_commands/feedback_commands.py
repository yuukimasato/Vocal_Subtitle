"""Feedback-learning and sample-management commands."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .common import verbose_option

logger = logging.getLogger(__name__)


@click.group()
def feedback():
    """基于用户修订字幕的自适应参数学习。"""


@feedback.command()
@click.option("--audio", "-a", required=True, type=click.Path(exists=True), help="原始音频文件路径")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True), help="用户修订的字幕文件 (.srt / .ass)")
@click.option("--profile", "-p", default="default", help="场景模板 (default / podcast / education / variety_show / music_live)")
@click.option("--feedback-profile", default="user_default", help="用户配置名称 (默认: user_default)")
@click.option("--consent", "-c", default="anonymous", type=click.Choice(["local", "anonymous", "full"]), help="反馈数据用途同意级别")
@click.option("--dry-run", is_flag=True, help="仅预览差异，不实际更新配置")
@verbose_option
def learn(audio: str, reference: str, profile: str, feedback_profile: str,
          consent: str, dry_run: bool, verbose: bool):
    """上传修订字幕和音频，自动学习用户偏好。"""
    from ..config import ConfigLoader
    from ..feedback import (AudioFingerprinter, DiffAnalyzer, FewShotBuilder,
                            ImpactEstimator, ParamLearner, SubtitleAligner,
                            UserProfileManager)
    from ..feedback.aligner import parse_subtitle_file
    from ..feedback.conflict_detector import ConflictDetector
    from ..feedback.health_scorer import compute_health_score_from_pairs
    from ..pipeline import Pipeline

    audio_path = Path(audio)
    reference_path = Path(reference)
    if reference_path.suffix.lower() not in (".srt", ".ass"):
        click.echo(f"✗ 不支持的字幕格式: {reference_path.suffix}，仅支持 .srt / .ass", err=True)
        raise SystemExit(1)

    config = ConfigLoader().load_profile(profile)
    click.echo(f"📖 解析修订字幕: {reference_path}")
    manual_events = parse_subtitle_file(reference_path)
    if not manual_events:
        click.echo("✗ 未从修订文件中解析到字幕事件", err=True)
        raise SystemExit(1)
    click.echo(f"   解析到 {len(manual_events)} 条字幕")

    click.echo(f"🎤 运行管道生成自动字幕: {audio_path}")
    try:
        auto_events = Pipeline(config).run(input_path=audio_path, skip_separation=True).get("events", [])
        click.echo(f"   生成 {len(auto_events)} 条自动字幕")
    except Exception as exc:
        click.echo(f"✗ 管道运行失败: {exc}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)

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
    except Exception as exc:
        click.echo(f"✗ 对齐失败: {exc}", err=True)
        raise SystemExit(1)
    matched = [pair for pair in pairs if pair.is_matched]
    denominator = max(len(auto_events), len(manual_events))
    click.echo(f"   对齐覆盖率: {len(matched)}/{denominator} ({len(matched) / denominator * 100:.1f}%)")

    click.echo("🔍 分析差异...")
    diff_report = DiffAnalyzer(param_isolation_enabled=feedback_cfg.param_isolation_enabled).analyze(pairs)
    health_before, health_detail = _health_report(diff_report, pairs, verbose)
    click.echo(f"\n💊 健康度评分: {health_before:.1f}/100")
    if verbose:
        for dimension, score in health_detail.items():
            click.echo(f"   {dimension}: {score:.1f}")
    click.echo("\n📊 差异分析报告:")
    click.echo(f"   时间偏移: {len(diff_report.time_shifts)} 处")
    click.echo(f"   合并/拆分: {len(diff_report.merge_actions)} 处")
    click.echo(f"   文本修改: {len(diff_report.text_edits)} 处")

    if diff_report.attribution:
        click.echo("\n📈 参数调整建议:")
        for param_path, adjustment in diff_report.attribution.items():
            icon = "↑" if adjustment.direction == "increase" else "↓"
            click.echo(f"   {icon} {param_path}: {adjustment.reason}")
            click.echo(f"     置信度: {adjustment.confidence:.2f}, 学习权重: {adjustment.learn_weight:.2f}, 分级: {adjustment.param_tier}")
        impacts = ImpactEstimator().estimate(diff_report.attribution, {})
        if impacts:
            click.echo("\n📊 变更影响预估:")
            for impact in impacts:
                click.echo(f"   {impact.summary}")
    else:
        click.echo("\n✅ 无需调整参数（已匹配用户偏好）")
    if diff_report.structural_revision:
        click.echo("\n⚠️  检测到结构性修订（大幅增删/重排序），学习权重已降低。")

    profile_mgr = UserProfileManager(feedback_cfg)
    stored_profile = profile_mgr.load(feedback_profile)
    conflicts = ConflictDetector(window=feedback_cfg.oscillation_detection_window).detect_all_oscillations(stored_profile.get("history", []))
    if conflicts:
        click.echo(f"\n⚠️  参数震荡检测: {len(conflicts)} 个参数出现震荡")
        for conflict in conflicts:
            click.echo(f"   {conflict.param_path}: {conflict.oscillation_count} 次翻转 (建议: {conflict.recommended_action})")

    if dry_run:
        click.echo("\n🔍 [dry-run 模式] 未实际更新配置。")
    elif diff_report.attribution:
        click.echo(f"\n📚 更新用户配置: {feedback_profile}...")
        updated = ParamLearner(profile_mgr).learn_from_diff(
            diff_report=diff_report,
            current_config_overrides=stored_profile.get("overrides", {}),
            profile_name=feedback_profile,
        )
        click.echo(f"   已学习 {len(updated)} 个参数覆盖")
        for key, value in sorted(updated.items()):
            click.echo(f"   {key}: {value}")
        if feedback_cfg.few_shot_enabled:
            few_shot = FewShotBuilder(max_examples=feedback_cfg.few_shot_max_examples)
            few_shot.load_cache(feedback_profile)
            few_shot.build_merge_examples(diff_report.merge_actions)
            if diff_report.text_edits:
                few_shot.build_format_examples(diff_report.text_edits)
            few_shot.save_cache(feedback_profile)
        if feedback_cfg.fingerprint_enabled:
            try:
                fingerprinter = AudioFingerprinter(
                    distance_method=feedback_cfg.fingerprint_distance_method,
                    knn_k=feedback_cfg.fingerprint_knn_k,
                    min_absolute_similarity=feedback_cfg.fingerprint_min_absolute_similarity,
                    relative_margin=feedback_cfg.fingerprint_relative_margin,
                )
                fingerprint = fingerprinter.extract(audio_path)
                if fingerprint is not None:
                    audio_hash = AudioFingerprinter.compute_audio_hash(audio_path)
                    fingerprinter.store(profile_id=feedback_profile, fingerprint=fingerprint,
                                        audio_hash=audio_hash, config_snapshot=updated)
                    fingerprinter.record_feedback(
                        profile_id=feedback_profile, audio_hash=audio_hash,
                        alignment_coverage=diff_report.alignment_coverage,
                        diff_summary="; ".join(adj.reason for adj in diff_report.attribution.values()),
                        adjustments={key: [adj.observed_value, adj.confidence] for key, adj in diff_report.attribution.items()},
                        health_before=health_before, health_after=health_before,
                        health_detail=health_detail,
                    )
                    click.echo(f"\n🔊 音频指纹已存储: {fingerprint.audio_signature}")
            except Exception as exc:
                if verbose:
                    click.echo(f"   ⚠️ 指纹提取失败: {exc}")
        _ingest_d2_sample(auto_events, manual_events, diff_report.alignment_coverage,
                          consent, diff_report, feedback_cfg, audio_path)
        click.echo("\n✓ 学习完成！下次运行将自动应用学习到的参数偏好。")
        click.echo(f"  使用 --feedback-profile {feedback_profile} 指定此配置")
    else:
        _ingest_d2_sample(auto_events, manual_events, diff_report.alignment_coverage,
                          consent, diff_report, feedback_cfg, audio_path)
        click.echo("\n✓ 无参数变更。")


def _health_report(diff_report, pairs, verbose: bool):
    from ..feedback.health_scorer import compute_health_score_from_pairs

    return compute_health_score_from_pairs(pairs)


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default", help="用户配置名称 (默认: user_default)")
@verbose_option
def show(feedback_profile: str, verbose: bool):
    """查看当前用户配置。"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    profile = UserProfileManager(FeedbackConfig()).load(feedback_profile)
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
    if profile.get("few_shot_examples"):
        click.echo(f"\n🎯 Few-shot 示例: {len(profile['few_shot_examples'])} 条")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default", help="用户配置名称 (默认: user_default)")
def rollback(feedback_profile: str):
    """回滚用户配置到上一个备份版本。"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    try:
        profile = UserProfileManager(FeedbackConfig()).rollback(feedback_profile)
        click.echo(f"✓ 已回滚配置 '{feedback_profile}' 到上一版本")
        click.echo(f"   更新时间: {profile.get('updated_at', 'N/A')}")
        click.echo(f"   反馈次数: {profile.get('feedback_count', 0)}")
    except FileNotFoundError:
        click.echo(f"✗ 无可用备份: {feedback_profile}", err=True)
        raise SystemExit(1)


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default", help="用户配置名称 (默认: user_default)")
@click.confirmation_option(prompt="确认重置？此操作不可撤销")
def reset(feedback_profile: str):
    """重置用户配置为系统默认。"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    UserProfileManager(FeedbackConfig()).reset(feedback_profile)
    click.echo(f"✓ 已重置配置 '{feedback_profile}' 为系统默认")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default", help="用户配置名称 (默认: user_default)")
def fingerprints(feedback_profile: str):
    """列出所有音频指纹到参数的映射。"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    fingerprinter = AudioFingerprinter(distance_method=FeedbackConfig().fingerprint_distance_method)
    fingerprints = fingerprinter.list_all()
    if not fingerprints:
        click.echo("📭 指纹库为空。提交一次反馈学习后自动生成指纹。")
        return
    click.echo(f"🔊 音频指纹库 ({len(fingerprints)} 条):\n")
    for fingerprint in fingerprints:
        click.echo(f"  [{fingerprint['id']}] {fingerprint.get('audio_signature', 'N/A')}")
        click.echo(f"       Profile: {fingerprint['profile_id']}")
        click.echo(f"       Audio Hash: {fingerprint['audio_hash'][:16]}...")
        click.echo(f"       Feedback Count: {fingerprint['feedback_count']}")
        click.echo(f"       Created: {fingerprint.get('created_at', 'N/A')[:19]}\n")
    click.echo(f"数据库路径: {fingerprinter._db_path}")


@feedback.command()
@click.option("--profile", "-p", "feedback_profile", default="user_default", help="要导出的用户配置名称")
@click.option("--output", "-o", required=True, type=click.Path(), help="输出 YAML 文件路径")
def export(feedback_profile: str, output: str):
    """导出用户配置。"""
    import yaml
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    profile = UserProfileManager(FeedbackConfig()).load(feedback_profile)
    export_data = {key: profile.get(key, default) for key, default in {
        "profile_id": None, "base_profile": "default", "description": "",
        "created_at": None, "updated_at": None, "feedback_count": 0,
        "overrides": {}, "fingerprint": {}, "few_shot_examples": [],
    }.items()}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(export_data, allow_unicode=True,
                                          default_flow_style=False, sort_keys=False), encoding="utf-8")
    click.echo(f"✓ 已导出配置 '{feedback_profile}' → {output_path}")
    click.echo(f"   参数覆盖: {len(export_data['overrides'])} 项")
    click.echo(f"   Few-shot 示例: {len(export_data['few_shot_examples'])} 条")


@feedback.command("import")
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True), help="要导入的 YAML 配置文件")
@click.option("--as", "profile_name", default=None, help="导入后的配置名称")
def import_profile(input_file: str, profile_name: str | None):
    """导入他人分享的用户配置。"""
    import yaml
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    input_path = Path(input_file)
    data = yaml.safe_load(input_path.read_text(encoding="utf-8")) or {}
    if not data.get("profile_id") and not profile_name:
        click.echo("✗ 导入文件中缺少 profile_id，请使用 --as 指定配置名称", err=True)
        raise SystemExit(1)
    target_name = profile_name or data["profile_id"]
    manager = UserProfileManager(FeedbackConfig())
    profile = manager.load(target_name)
    profile.update({"profile_id": target_name, "base_profile": data.get("base_profile", "default"),
                    "description": data.get("description", f"Imported from {input_path.name}"),
                    "overrides": data.get("overrides", {}), "few_shot_examples": data.get("few_shot_examples", [])})
    if data.get("fingerprint"):
        profile["fingerprint"] = data["fingerprint"]
    manager.save(profile)
    click.echo(f"✓ 已导入配置 → '{target_name}'")
    click.echo(f"   参数覆盖: {len(profile['overrides'])} 项")
    click.echo(f"   Few-shot 示例: {len(profile['few_shot_examples'])} 条")


@feedback.group("sample")
def feedback_sample_group():
    """D3 分层抽样：从 D2 候选反馈集抽样生成 D3 回归集。"""


@feedback_sample_group.command("plan")
@click.option("--strata", "-s", default="language,scene,speaker_count", help="分层维度（逗号分隔）")
@click.option("--min-per-stratum", default=1, type=int, help="每层最少样本数")
def sample_plan(strata: str, min_per_stratum: int):
    """预览 D3 抽样计划。"""
    from collections import Counter
    from ..feedback.sample_manager import FeedbackSampleManager
    from ..feedback.stratified_sampler import StratifiedSampler

    dimensions = [item.strip() for item in strata.split(",")]
    sampler = StratifiedSampler()
    pool = sampler.pool_from_manager(FeedbackSampleManager())
    if not pool:
        click.echo("⚠️  当前 D2 候选池中没有符合条件的样本")
        click.echo("   需要: review=accepted, anonymization=deidentified, consent≠local")
        return
    click.echo(f"📊 D2 候选池: {len(pool)} 个合格样本")
    for dimension in dimensions:
        values = Counter(sampler._extract_tags(sample).get(dimension, "unknown") for sample in pool)
        click.echo(f"\n  {dimension} 分布:")
        for value, count in values.most_common():
            click.echo(f"    {value}: {count} ({count / len(pool) * 100:.0f}%)")
    plan = sampler.build_plan(pool, strata=dimensions, min_per_stratum=min_per_stratum)
    click.echo(f"\n📋 抽样计划: {plan.version_id}\n   描述: {plan.description}\n   分层: {', '.join(plan.strata)}\n   每层最少: {plan.min_per_stratum}")
    if plan.quotas:
        click.echo("   配额概要:")
        for dimension, quotas in plan.quotas.items():
            click.echo(f"     {dimension}: {sum(quotas.values())} 个")


@feedback_sample_group.command("freeze")
@click.option("--strata", "-s", default="language,scene,speaker_count", help="分层维度（逗号分隔）")
@click.option("--description", "-d", default="", help="D3 版本描述")
def sample_freeze(strata: str, description: str):
    """执行 D3 分层抽样并冻结版本。"""
    from ..feedback.sample_manager import FeedbackSampleManager
    from ..feedback.stratified_sampler import StratifiedSampler

    dimensions = [item.strip() for item in strata.split(",")]
    sampler = StratifiedSampler()
    pool = sampler.pool_from_manager(FeedbackSampleManager())
    if not pool:
        click.echo("✗ D2 候选池中没有符合条件的样本", err=True)
        raise SystemExit(1)
    plan = sampler.build_plan(pool, strata=dimensions, description=description or "D3 分层抽样")
    result = sampler.sample(pool, plan)
    click.echo(f"📊 D2 候选池: {len(pool)} 个合格样本\n📋 抽样计划: {plan.version_id}\n📥 选中: {result.sample_count} 个样本")
    if result.balance.get("warnings"):
        click.echo("⚠️  平衡检查:")
        for warning in result.balance["warnings"]:
            click.echo(f"    {warning['message']}")
    version = sampler.freeze(result, description=description)
    click.echo(f"\n✅ D3 版本已冻结: {version.version_id}\n   样本数: {version.sample_count}\n   冻结日期: {version.freeze_date}")


def _ingest_d2_sample(auto_events, manual_events, alignment_coverage, consent,
                      diff_report, feedback_cfg, audio_path) -> None:
    """将反馈学习结果自动入库到 D2；失败不影响主流程。"""
    try:
        from ..feedback.sample_manager import FeedbackSampleManager

        def events_to_text(events) -> str:
            lines = []
            for index, event in enumerate(events, 1):
                lines.append(f"{index}\n{getattr(event, 'start', 0):.3f} --> {getattr(event, 'end', 0):.3f}\n{getattr(event, 'text', '')}\n")
            return "\n".join(lines)

        edit_types = {}
        if diff_report:
            if diff_report.text_edits:
                edit_types["text_correction"] = len(diff_report.text_edits)
            if diff_report.time_shifts:
                edit_types["time_adjustment"] = len(diff_report.time_shifts)
            if diff_report.merge_actions:
                edit_types["format_preference"] = len(diff_report.merge_actions)
            if diff_report.structural_revision:
                edit_types["structural_rewrite"] = 1
        sample = FeedbackSampleManager().ingest(
            auto_subtitle=events_to_text(auto_events), human_revision=events_to_text(manual_events),
            alignment={"method": "dtw", "coverage_ratio": alignment_coverage,
                       "confidence": getattr(diff_report, "confidence", 0.8) if diff_report else 0.8},
            consent_level=consent, language="unknown", scene="", audio_duration=0.0,
            audio_condition="", speaker_count=0, original_config={}, edit_types=edit_types,
        )
        if sample:
            click.echo(f"📥 D2 反馈样本已入库: {sample.sample_id} (consent={consent})")
    except Exception as exc:
        logger.warning("D2 sample ingestion failed (non-fatal): %s", exc)


def _print_nested_dict(values: dict, indent: int = 4) -> None:
    for key, value in sorted(values.items()):
        if isinstance(value, dict):
            click.echo(f"{' ' * indent}{key}:")
            _print_nested_dict(value, indent + 2)
        else:
            click.echo(f"{' ' * indent}{key}: {value}")


def register(main) -> None:
    """Register the feedback command tree on the public root group."""
    main.add_command(feedback)
