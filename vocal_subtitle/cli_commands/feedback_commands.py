"""Feedback-learning and sample-management commands."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .common import verbose_option

logger = logging.getLogger(__name__)


def _scenarios():
    """D27 四场景标签（延迟导入避免加载 feedback 全家桶）。"""
    from ..feedback.dataset_export import SCENARIOS

    return SCENARIOS


@click.group()
def feedback():
    """基于用户修订字幕的自适应参数学习。"""


@feedback.command()
@click.option(
    "--audio",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="原始音频文件路径",
)
@click.option(
    "--reference",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="用户修订的字幕文件 (.srt / .ass)",
)
@click.option(
    "--profile",
    "-p",
    default="default",
    help="场景模板 (default / podcast / education / variety_show / music_live)",
)
@click.option(
    "--feedback-profile",
    default="user_default",
    help="用户配置名称 (默认: user_default)",
)
@click.option(
    "--consent",
    "-c",
    default="anonymous",
    type=click.Choice(["local", "anonymous", "full"]),
    help="反馈数据用途同意级别",
)
@click.option(
    "--scenario",
    "-s",
    default="external-correction",
    help="D27 场景标签 (inline-review / external-correction / "
    "existing-subtitle / from-scratch-timing)；"
    "CLI 上传外部修订字幕默认 external-correction",
)
@click.option("--dry-run", is_flag=True, help="仅预览差异，不实际更新配置")
@verbose_option
def learn(
    audio: str,
    reference: str,
    profile: str,
    feedback_profile: str,
    consent: str,
    scenario: str,
    dry_run: bool,
    verbose: bool,
):
    """上传修订字幕和音频，自动学习用户偏好。"""
    from ..config import ConfigLoader
    from ..feedback import (
        AudioFingerprinter,
        DiffAnalyzer,
        FewShotBuilder,
        ImpactEstimator,
        ParamLearner,
        SubtitleAligner,
        UserProfileManager,
    )
    from ..feedback.aligner import parse_subtitle_file
    from ..feedback.conflict_detector import ConflictDetector
    from ..pipeline import Pipeline

    if scenario not in _scenarios():
        click.echo(
            f"✗ 无效场景标签: {scenario}，合法值: {' / '.join(_scenarios())}",
            err=True,
        )
        raise SystemExit(1)

    audio_path = Path(audio)
    reference_path = Path(reference)
    if reference_path.suffix.lower() not in (".srt", ".ass"):
        click.echo(
            f"✗ 不支持的字幕格式: {reference_path.suffix}，仅支持 .srt / .ass", err=True
        )
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
        auto_events = (
            Pipeline(config)
            .run(input_path=audio_path, skip_separation=True)
            .get("events", [])
        )
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
    click.echo(
        f"   对齐覆盖率: {len(matched)}/{denominator} ({len(matched) / denominator * 100:.1f}%)"
    )

    click.echo("🔍 分析差异...")
    diff_report = DiffAnalyzer(
        param_isolation_enabled=feedback_cfg.param_isolation_enabled
    ).analyze(pairs)
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
            click.echo(
                f"     置信度: {adjustment.confidence:.2f}, 学习权重: {adjustment.learn_weight:.2f}, 分级: {adjustment.param_tier}"
            )
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
    conflicts = ConflictDetector(
        window=feedback_cfg.oscillation_detection_window
    ).detect_all_oscillations(stored_profile.get("history", []))
    if conflicts:
        click.echo(f"\n⚠️  参数震荡检测: {len(conflicts)} 个参数出现震荡")
        for conflict in conflicts:
            click.echo(
                f"   {conflict.param_path}: {conflict.oscillation_count} 次翻转 (建议: {conflict.recommended_action})"
            )

    if dry_run:
        click.echo("\n🔍 [dry-run 模式] 未实际更新配置。")
    elif diff_report.attribution:
        click.echo(f"\n📚 更新用户配置: {feedback_profile}...")
        updated = ParamLearner(profile_mgr).learn_from_diff(
            diff_report=diff_report,
            current_config_overrides=stored_profile.get("overrides", {}),
            profile_name=feedback_profile,
        )
        if updated:
            click.echo(f"   已学习 {len(updated)} 个参数覆盖")
            for key, value in sorted(updated.items()):
                click.echo(f"   {key}: {value}")
        else:
            learned_count = profile_mgr.load(feedback_profile).get("feedback_count", 0)
            remaining = max(3 - learned_count, 0)
            click.echo(
                f"   已记录第 {learned_count} 次反馈观测（学习率预热期，暂不修改参数）"
            )
            if remaining:
                click.echo(f"   再积累 {remaining} 次反馈后开始应用参数覆盖")
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
                    fingerprinter.store(
                        profile_id=feedback_profile,
                        fingerprint=fingerprint,
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
                            key: [adj.observed_value, adj.confidence]
                            for key, adj in diff_report.attribution.items()
                        },
                        health_before=health_before,
                        health_after=health_before,
                        health_detail=health_detail,
                    )
                    click.echo(f"\n🔊 音频指纹已存储: {fingerprint.audio_signature}")
            except Exception as exc:
                if verbose:
                    click.echo(f"   ⚠️ 指纹提取失败: {exc}")
        _ingest_d2_sample(
            auto_events,
            manual_events,
            diff_report.alignment_coverage,
            consent,
            diff_report,
            feedback_cfg,
            audio_path,
            scenario=scenario,
        )
        try:
            from ..feedback.user_profile import load_apply_overrides_on_run

            apply_enabled = (
                load_apply_overrides_on_run(feedback_cfg) if updated else True
            )
        except Exception:
            apply_enabled = True
        if updated and apply_enabled:
            click.echo("\n✓ 学习完成！下次运行将自动应用学习到的参数偏好。")
        elif updated:
            click.echo("\n✓ 学习完成（参数覆盖已保存）。")
            click.echo(
                "  ⚠️ 运行时开关 apply_overrides_on_run 当前为关："
                "学习到的参数不会在管线运行时生效，可在 8613 处理台「反馈档案」区开启"
            )
        else:
            click.echo("\n✓ 学习完成（观测已记录）。")
        click.echo(f"  使用 --feedback-profile {feedback_profile} 指定此配置")
    else:
        _ingest_d2_sample(
            auto_events,
            manual_events,
            diff_report.alignment_coverage,
            consent,
            diff_report,
            feedback_cfg,
            audio_path,
            scenario=scenario,
        )
        click.echo("\n✓ 无参数变更。")


def _default_sink_dir() -> Path:
    """journal sink 目录（与 webui routes_journal 的推导同源：cache/journal_sink）。"""
    try:
        from ..webui.runtime_state import state

        return state.upload_dir.parent / "journal_sink"
    except Exception:
        return Path(__file__).resolve().parent.parent.parent / "cache" / "journal_sink"


def _load_feedback_config():
    """从场景模板 YAML 读 feedback 配置（D30 保留策略/TTL 的权威来源）；载入失败回退默认值。"""
    try:
        from ..config import ConfigLoader

        return ConfigLoader().load_profile("default").feedback
    except Exception:
        from ..config import FeedbackConfig

        return FeedbackConfig()


def _apply_sink_retention(journal_paths, feedback_cfg) -> None:
    """ingest 成功后对 sink 目录内的源文件执行保留策略（D30）；失败不影响摄取主流程。"""
    try:
        from ..feedback.journal_retention import apply_retention

        policy = feedback_cfg.journal_sink_retention
        sink_dir = _default_sink_dir()
        for path in journal_paths:
            outcome = apply_retention(path, policy=policy, sink_dir=sink_dir)
            if outcome.action == "archived":
                click.echo(
                    f"🗄️  已归档已消费日志: {outcome.path.name} → {outcome.detail}"
                )
            elif outcome.action == "deleted":
                click.echo(f"🗑️  已删除已消费日志: {outcome.path.name}")
            elif outcome.reason == "outside-sink":
                click.echo(f"ℹ️  {outcome.path.name} 不在 journal sink 目录，保留原文件")
    except Exception as exc:
        logger.warning("Journal sink retention failed (non-fatal): %s", exc)


def _health_report(diff_report, pairs, verbose: bool):
    from ..feedback.health_scorer import compute_health_score_from_pairs

    return compute_health_score_from_pairs(pairs)


@feedback.command()
@click.option(
    "--profile",
    "-p",
    "feedback_profile",
    default="user_default",
    help="用户配置名称 (默认: user_default)",
)
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
            click.echo(
                f"   [{entry.get('timestamp', '?')[:19]}] {entry.get('diff_report_summary', 'N/A')}"
            )
    if profile.get("few_shot_examples"):
        click.echo(f"\n🎯 Few-shot 示例: {len(profile['few_shot_examples'])} 条")


@feedback.command()
@click.option(
    "--profile",
    "-p",
    "feedback_profile",
    default="user_default",
    help="用户配置名称 (默认: user_default)",
)
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
@click.option(
    "--profile",
    "-p",
    "feedback_profile",
    default="user_default",
    help="用户配置名称 (默认: user_default)",
)
@click.confirmation_option(prompt="确认重置？此操作不可撤销")
def reset(feedback_profile: str):
    """重置用户配置为系统默认。"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    UserProfileManager(FeedbackConfig()).reset(feedback_profile)
    click.echo(f"✓ 已重置配置 '{feedback_profile}' 为系统默认")


@feedback.command()
@click.option(
    "--profile",
    "-p",
    "feedback_profile",
    default="user_default",
    help="用户配置名称 (默认: user_default)",
)
def fingerprints(feedback_profile: str):
    """列出所有音频指纹到参数的映射。"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    fingerprinter = AudioFingerprinter(
        distance_method=FeedbackConfig().fingerprint_distance_method
    )
    fingerprints = fingerprinter.list_all()
    if not fingerprints:
        click.echo("📭 指纹库为空。提交一次反馈学习后自动生成指纹。")
        return
    click.echo(f"🔊 音频指纹库 ({len(fingerprints)} 条):\n")
    for fingerprint in fingerprints:
        click.echo(
            f"  [{fingerprint['id']}] {fingerprint.get('audio_signature', 'N/A')}"
        )
        click.echo(f"       Profile: {fingerprint['profile_id']}")
        click.echo(f"       Audio Hash: {fingerprint['audio_hash'][:16]}...")
        click.echo(f"       Feedback Count: {fingerprint['feedback_count']}")
        click.echo(f"       Created: {fingerprint.get('created_at', 'N/A')[:19]}\n")
    click.echo(f"数据库路径: {fingerprinter._db_path}")


@feedback.command()
@click.option(
    "--profile",
    "-p",
    "feedback_profile",
    default="user_default",
    help="要导出的用户配置名称",
)
@click.option(
    "--output", "-o", required=True, type=click.Path(), help="输出 YAML 文件路径"
)
def export(feedback_profile: str, output: str):
    """导出用户配置。"""
    import yaml

    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    profile = UserProfileManager(FeedbackConfig()).load(feedback_profile)
    export_data = {
        key: profile.get(key, default)
        for key, default in {
            "profile_id": None,
            "base_profile": "default",
            "description": "",
            "created_at": None,
            "updated_at": None,
            "feedback_count": 0,
            "overrides": {},
            "fingerprint": {},
            "few_shot_examples": [],
        }.items()
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            export_data, allow_unicode=True, default_flow_style=False, sort_keys=False
        ),
        encoding="utf-8",
    )
    click.echo(f"✓ 已导出配置 '{feedback_profile}' → {output_path}")
    click.echo(f"   参数覆盖: {len(export_data['overrides'])} 项")
    click.echo(f"   Few-shot 示例: {len(export_data['few_shot_examples'])} 条")


@feedback.command("import")
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="要导入的 YAML 配置文件",
)
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
    profile.update(
        {
            "profile_id": target_name,
            "base_profile": data.get("base_profile", "default"),
            "description": data.get("description", f"Imported from {input_path.name}"),
            "overrides": data.get("overrides", {}),
            "few_shot_examples": data.get("few_shot_examples", []),
        }
    )
    if data.get("fingerprint"):
        profile["fingerprint"] = data["fingerprint"]
    manager.save(profile)
    click.echo(f"✓ 已导入配置 → '{target_name}'")
    click.echo(f"   参数覆盖: {len(profile['overrides'])} 项")
    click.echo(f"   Few-shot 示例: {len(profile['few_shot_examples'])} 条")


@feedback.command("ingest-journal")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--original",
    "-o",
    type=click.Path(exists=True),
    default=None,
    help="原始字幕文件（缺省在每份日志同目录自动发现 <同名>.srt/.ass/.vtt）",
)
@click.option(
    "--feedback-profile",
    default="user_default",
    help="用户配置名称 (默认: user_default)",
)
@click.option(
    "--consent",
    "-c",
    default="anonymous",
    type=click.Choice(["local", "anonymous", "full"]),
    help="D2 入库的同意级别",
)
@click.option(
    "--apply/--no-apply",
    "apply_prefs",
    default=True,
    help="是否把编辑器维度偏好写入 user_profile",
)
@click.option("--dry-run", is_flag=True, help="仅输出统计报告与归因，不入库不写配置")
@verbose_option
def ingest_journal(
    files: tuple,
    original: str | None,
    feedback_profile: str,
    consent: str,
    apply_prefs: bool,
    dry_run: bool,
    verbose: bool,
):
    """摄取编辑器日志（edit-journal-v1）→ 重放校验 → 统计报告 → D2 入库 → 偏好学习。

    FILES: 一个或多个 .journal.jsonl 文件（编辑器导出搭车产物）。
    """
    from ..config import FeedbackConfig
    from ..feedback import (
        UserProfileManager,
        check_v3_trigger,
        derive_editor_preferences,
        journal_scenario,
        journal_statistics,
        load_journal_files,
        replay_journal,
    )
    from ..feedback.diff_analyzer import analyze_journal_events
    from ..feedback.journal_ingest import (
        find_original_subtitle,
        journal_to_text_sample,
        load_original_events,
    )
    from ..feedback.sample_manager import FeedbackSampleManager

    journal_paths = [Path(f) for f in files]
    try:
        journal_files = load_journal_files(journal_paths)
    except Exception as exc:
        click.echo(f"✗ 日志解析失败: {exc}", err=True)
        raise SystemExit(1)

    all_events = [event for journal in journal_files for event in journal.events]
    if not all_events:
        click.echo("✗ 日志中没有有效事件（schema=edit-journal-v1）", err=True)
        raise SystemExit(1)
    click.echo(
        f"📖 已读取 {len(journal_files)} 份日志，共 {len(all_events)} 个事件"
        f"（跳过 {sum(j.skipped for j in journal_files)} 个无效行）"
    )

    # ---- 重放校验（原始字幕 + 日志 = 最终字幕）----
    replay_summary = []
    for journal in journal_files:
        original_path = (
            Path(original) if original else find_original_subtitle(journal.path)
        )
        if original_path is None:
            click.echo(
                f"⚠️  {journal.path.name}: 同目录未找到原始字幕，跳过重放（可用 --original 指定）"
            )
            continue
        try:
            original_events = load_original_events(original_path)
        except Exception as exc:
            click.echo(f"⚠️  {journal.path.name}: 原始字幕解析失败（{exc}），跳过重放")
            continue
        result = replay_journal(original_events, journal.events)
        replay_summary.append((journal.path.name, original_path, result))
        icon = "✓" if result.exact else "⚠️"
        click.echo(
            f"   {icon} {journal.path.name}: 重放还原 {len(result.final_events)} 条"
            f"（对应原行 {result.matched_ids}，新增 {result.unmatched_ids}）"
        )

    # ---- V1 统计报告 ----
    stats = journal_statistics(journal_files)
    click.echo("\n📊 V1 统计报告:")
    click.echo(f"   事件数: {stats.event_count}（{stats.session_count} 个会话）")
    click.echo(f"   actor 分布: {stats.actors or '{}'}")
    top_commands = sorted(stats.commands.items(), key=lambda kv: kv[1], reverse=True)[
        :8
    ]
    click.echo(f"   高频命令: {top_commands or '无'}")
    if stats.structural:
        click.echo(f"   结构操作: {stats.structural}（删行/拆分/合并是取舍思维信号）")
    if stats.start_delta_median is not None:
        click.echo(
            f"   开始时间偏移: 中位 {stats.start_delta_median * 1000:+.0f}ms / 均值 {stats.start_delta_mean * 1000:+.0f}ms"
        )
    if stats.end_delta_median is not None:
        click.echo(
            f"   结束时间偏移: 中位 {stats.end_delta_median * 1000:+.0f}ms / 均值 {stats.end_delta_mean * 1000:+.0f}ms"
        )
    if stats.gap_after_median is not None:
        click.echo(f"   留白偏好: 中位 {stats.gap_after_median * 1000:.0f}ms")
    if stats.cps_p90 is not None:
        click.echo(f"   CPS P90: {stats.cps_p90:.2f}")
    if stats.boundary_snap_rate is not None:
        click.echo(f"   终点贴合语音边界率: {stats.boundary_snap_rate:.0%}")
    click.echo(f"   出处覆盖（manifest）: {stats.provenance_coverage:.0%}")

    # ---- 事件级归因 ----
    attributions = analyze_journal_events(all_events)
    if attributions:
        click.echo("\n📈 事件级归因 → 管线参数建议:")
        for adjustment in attributions.values():
            icon = "↑" if adjustment.direction == "increase" else "↓"
            click.echo(f"   {icon} {adjustment.param_path}: {adjustment.reason}")
            click.echo(
                f"     置信度: {adjustment.confidence:.2f}, 分级: {adjustment.param_tier}"
            )
    else:
        click.echo("\n✅ 编辑行为不足以归因到管线参数（样本量或偏移幅度不足）")

    preferences = derive_editor_preferences(stats)
    if preferences:
        click.echo("\n🧭 编辑器维度偏好（写入 user_profile）:")
        for key, value in sorted(preferences.items()):
            click.echo(f"   {key}: {value}")

    if dry_run:
        click.echo("\n🔍 [dry-run 模式] 未入库、未更新配置。")
        return

    # ---- D2 候选样本入库（每个日志文件一份，重放终态 = 人工终审）----
    sample_manager = FeedbackSampleManager()
    journal_by_name = {journal.path.name: journal for journal in journal_files}
    consumed_journal_paths: list[Path] = []
    ingested = 0
    for journal_name, original_path, result in replay_summary:
        if not result.final_events:
            continue
        # 重放成功产出终态 = 已消费（样本库去重跳过也算已消费），纳入保留策略
        journal = journal_by_name.get(journal_name)
        if journal is not None:
            consumed_journal_paths.append(journal.path)
        original_events = load_original_events(original_path)
        auto_text, final_text = journal_to_text_sample(
            original_events, result.final_events
        )
        edit_types = {}
        if stats.structural.get("remove"):
            edit_types["structural_rewrite"] = 1
        if stats.start_delta_median is not None or stats.end_delta_median is not None:
            edit_types["time_adjustment"] = len(all_events)
        journal = journal_by_name.get(journal_name)
        header = (journal.header if journal else None) or {}
        sample = sample_manager.ingest(
            auto_subtitle=auto_text,
            human_revision=final_text,
            alignment={
                "method": "journal-replay",
                "coverage_ratio": 1.0 if result.exact else 0.8,
                "confidence": 0.9 if result.exact else 0.7,
            },
            consent_level=consent,
            scene=journal_scenario(journal) if journal else "",
            edit_types=edit_types,
            task_id=str(header.get("task_id") or ""),
            run_id=str(header.get("run_id") or ""),
        )
        if sample:
            ingested += 1
            click.echo(
                f"📥 D2 反馈样本已入库: {sample.sample_id}（来源 {journal_name}）"
            )

    # ---- 编辑器维度偏好 → user_profile ----
    if apply_prefs and preferences:
        from datetime import datetime

        profile_mgr = UserProfileManager(FeedbackConfig())
        profile = profile_mgr.load(feedback_profile)
        merged = dict(profile.get("editor_preferences") or {})
        merged.update(preferences)
        profile["editor_preferences"] = merged
        profile.setdefault("history", []).append(
            {
                "timestamp": datetime.now().isoformat(),
                "source": "journal",
                "event_count": stats.event_count,
                "diff_report_summary": f"journal ingest: {len(preferences)} 个编辑器偏好, {len(attributions)} 条归因",
            }
        )
        profile_mgr.save(profile)
        click.echo(
            f"\n📚 已更新用户配置 {feedback_profile} 的 editor_preferences（{len(merged)} 项）"
        )

    # ---- V3 触发检查（D16）----
    triggered, message = check_v3_trigger(
        sample_manager.count_by_status() + ingested,
        stats.provenance_coverage,
        0.0,
        min_samples=FeedbackConfig().v3_trigger_min_samples,
        min_coverage=FeedbackConfig().v3_trigger_min_coverage,
        max_conflict_rate=FeedbackConfig().v3_trigger_max_conflict_rate,
    )
    if triggered:
        click.echo(f"\n🚀 V3 触发提示: {message}")
    else:
        click.echo(f"\nℹ️  V3: {message}")

    # ---- sink 保留策略（D30）：仅对成功消费（已入库）的日志执行；重放跳过/入库失败的文件不动 ----
    _apply_sink_retention(consumed_journal_paths, _load_feedback_config())
    if len(consumed_journal_paths) < len(journal_paths):
        click.echo(
            f"ℹ️  {len(journal_paths) - len(consumed_journal_paths)} 个日志未成功消费，保留策略不作用于它们"
        )

    click.echo("\n✓ 日志摄取完成。")


@feedback.command("cleanup-journal-sink")
@click.option(
    "--sink-dir",
    "sink_dir_opt",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="journal sink 目录（缺省自动推导 cache/journal_sink）",
)
@click.option(
    "--ttl-days",
    type=int,
    default=None,
    help="覆盖配置的 TTL 天数（缺省读 feedback.journal_sink_ttl_days，<=0 禁用清理）",
)
def cleanup_journal_sink(sink_dir_opt: Path | None, ttl_days: int | None):
    """TTL 兜底清理：删除从未被消费且超过 TTL 的 sink 文件（D30）。

    只清 sink 顶层 *.jsonl；已归档到 consumed/ 的文件不受影响；重复运行幂等。
    """
    from ..feedback.journal_retention import cleanup_expired, consumed_dir

    directory = sink_dir_opt or _default_sink_dir()
    days = (
        ttl_days
        if ttl_days is not None
        else _load_feedback_config().journal_sink_ttl_days
    )
    if days <= 0:
        click.echo(f"⏭️  journal_sink_ttl_days={days}，TTL 清理已禁用（{directory}）")
        return
    removed = cleanup_expired(directory, ttl_days=days)
    if removed:
        click.echo(
            f"🧹 已清理 {len(removed)} 个超期未消费的 sink 文件（TTL {days} 天）:"
        )
        for path in removed:
            click.echo(f"   - {path.name}")
    else:
        click.echo(f"✓ 无超期未消费的 sink 文件（TTL {days} 天）")
    archived_dir = consumed_dir(directory)
    if archived_dir.is_dir():
        archived = len(list(archived_dir.glob("*.jsonl")))
        click.echo(
            f"ℹ️  已消费归档保留于 {archived_dir}（{archived} 份，不参与 TTL 清理）"
        )


@feedback.command("export-dataset")
@click.option(
    "--out",
    "-o",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="数据集输出目录（重复导出到同一目录即追加式新增分片，不改写已存在分片）",
)
@click.option(
    "--scenarios",
    "-s",
    default="",
    help="场景过滤，逗号分隔（inline-review / external-correction / existing-subtitle / from-scratch-timing；缺省导出全部）",
)
@click.option(
    "--license",
    "license_name",
    default=None,
    help="数据集许可（如 CC-BY-4.0 / CC0-1.0）；未指定且为交互终端时会询问，非交互环境拒绝执行",
)
@click.option(
    "--bundle-audio",
    is_flag=True,
    default=False,
    help="把音频实体复制进数据集 audio/ 目录（仅本地使用，不建议推 git；默认不打包）",
)
@click.option(
    "--shard-size",
    default=500,
    show_default=True,
    type=int,
    help="单个 jsonl 分片的最大样本数",
)
@click.option(
    "--name",
    "dataset_name",
    default="subtitle-feedback",
    show_default=True,
    help="数据集名称（写入卡片）",
)
def export_dataset(
    out_dir: Path,
    scenarios: str,
    license_name: str | None,
    bundle_audio: bool,
    shard_size: int,
    dataset_name: str,
):
    """把已接受（accepted）的 D2 样本物化为 dataset-v1 数据集（D31/D32/D33）。

    只导审核队列接受的样本；pending/rejected 不出门。产物为 git-ready 目录
    （README 卡片 + data/*.jsonl 分片 + manifest 清单），推送由用户手动完成。
    """
    import sys

    from ..feedback.dataset_export import (
        SCENARIOS,
        LicenseRequiredError,
        export_dataset,
    )
    from ..feedback.sample_manager import FeedbackSampleManager

    wanted = [item.strip() for item in (scenarios or "").split(",") if item.strip()]
    invalid = [item for item in wanted if item not in SCENARIOS]
    if invalid:
        click.echo(
            f"✗ 未知场景标签: {', '.join(invalid)}；可选值: {', '.join(SCENARIOS)}",
            err=True,
        )
        raise SystemExit(1)

    # 许可门禁（D33）：显式参数优先；交互终端才询问；非交互且未传参必须拒绝，不能卡住等输入
    if not license_name:
        if sys.stdin.isatty():
            license_name = click.prompt(
                "请输入数据集许可（例如 CC-BY-4.0 / CC0-1.0）", type=str
            )
        else:
            click.echo(
                "✗ 未指定许可且当前为非交互环境：dataset-v1 导出强制显式选择许可，请用 --license 指定",
                err=True,
            )
            raise SystemExit(1)

    try:
        outcome = export_dataset(
            FeedbackSampleManager(),
            out_dir,
            license=license_name,
            scenarios=wanted,
            bundle_audio=bundle_audio,
            shard_size=shard_size,
            dataset_name=dataset_name,
        )
    except LicenseRequiredError as exc:
        click.echo(f"✗ {exc}", err=True)
        raise SystemExit(1)
    except ValueError as exc:
        click.echo(f"✗ {exc}", err=True)
        raise SystemExit(1)

    click.echo(f"📦 dataset-v1 导出完成 → {outcome.out_dir}")
    click.echo(
        f"   accepted 样本（过滤后）: {outcome.accepted_total}，本次新增: {outcome.exported_count}"
    )
    if outcome.skipped_missing_text:
        click.echo(
            f"   ⚠️  {len(outcome.skipped_missing_text)} 个样本缺少字幕全文（旧版样本库入库），已跳过: "
            f"{', '.join(outcome.skipped_missing_text)}"
        )
    for shard in outcome.shards:
        click.echo(f"   📄 {shard}")
    if not outcome.shards:
        click.echo("   ℹ️  无新增样本，未写入新分片")
    if bundle_audio:
        if outcome.bundled_audio:
            click.echo(
                f"   🔊 音频实体已打包 {len(outcome.bundled_audio)} 个会话 → audio/"
            )
        if outcome.missing_audio:
            click.echo(
                f"   ⚠️  {len(outcome.missing_audio)} 个音频引用找不到实体文件（引用照写）: "
                f"{', '.join(outcome.missing_audio)}"
            )
        click.echo("   ⚠️  不建议把 audio/ 目录推送到 git 远程（体积大且含原始音频）")
    click.echo(f"   许可: {outcome.license}（已记入 README 数据集卡片）")
    click.echo("   提示: 目录为 git-ready，推送请手动执行；建议以 git tag 标记发行版本")


@feedback.group("sample")
def feedback_sample_group():
    """D3 分层抽样：从 D2 候选反馈集抽样生成 D3 回归集。"""


@feedback_sample_group.command("plan")
@click.option(
    "--strata",
    "-s",
    default="language,scene,speaker_count",
    help="分层维度（逗号分隔）",
)
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
        click.echo(
            "   需要: review=accepted, anonymization=deidentified, consent≠local"
        )
        return
    click.echo(f"📊 D2 候选池: {len(pool)} 个合格样本")
    for dimension in dimensions:
        values = Counter(
            sampler._extract_tags(sample).get(dimension, "unknown") for sample in pool
        )
        click.echo(f"\n  {dimension} 分布:")
        for value, count in values.most_common():
            click.echo(f"    {value}: {count} ({count / len(pool) * 100:.0f}%)")
    plan = sampler.build_plan(pool, strata=dimensions, min_per_stratum=min_per_stratum)
    click.echo(
        f"\n📋 抽样计划: {plan.version_id}\n   描述: {plan.description}\n   分层: {', '.join(plan.strata)}\n   每层最少: {plan.min_per_stratum}"
    )
    if plan.quotas:
        click.echo("   配额概要:")
        for dimension, quotas in plan.quotas.items():
            click.echo(f"     {dimension}: {sum(quotas.values())} 个")


@feedback_sample_group.command("freeze")
@click.option(
    "--strata",
    "-s",
    default="language,scene,speaker_count",
    help="分层维度（逗号分隔）",
)
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
    plan = sampler.build_plan(
        pool, strata=dimensions, description=description or "D3 分层抽样"
    )
    result = sampler.sample(pool, plan)
    click.echo(
        f"📊 D2 候选池: {len(pool)} 个合格样本\n📋 抽样计划: {plan.version_id}\n📥 选中: {result.sample_count} 个样本"
    )
    if result.balance.get("warnings"):
        click.echo("⚠️  平衡检查:")
        for warning in result.balance["warnings"]:
            click.echo(f"    {warning['message']}")
    version = sampler.freeze(result, description=description)
    click.echo(
        f"\n✅ D3 版本已冻结: {version.version_id}\n   样本数: {version.sample_count}\n   冻结日期: {version.freeze_date}"
    )


def _ingest_d2_sample(
    auto_events,
    manual_events,
    alignment_coverage,
    consent,
    diff_report,
    feedback_cfg,
    audio_path,
    *,
    scenario: str = "",
) -> None:
    """将反馈学习结果自动入库到 D2；失败不影响主流程。"""
    try:
        from ..feedback.sample_manager import FeedbackSampleManager

        def events_to_text(events) -> str:
            lines = []
            for index, event in enumerate(events, 1):
                lines.append(
                    f"{index}\n{getattr(event, 'start', 0):.3f} --> {getattr(event, 'end', 0):.3f}\n{getattr(event, 'text', '')}\n"
                )
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
            auto_subtitle=events_to_text(auto_events),
            human_revision=events_to_text(manual_events),
            alignment={
                "method": "dtw",
                "coverage_ratio": alignment_coverage,
                "confidence": getattr(diff_report, "confidence", 0.8)
                if diff_report
                else 0.8,
            },
            consent_level=consent,
            language="unknown",
            scene=scenario,
            audio_duration=0.0,
            audio_condition="",
            speaker_count=0,
            original_config={},
            edit_types=edit_types,
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
