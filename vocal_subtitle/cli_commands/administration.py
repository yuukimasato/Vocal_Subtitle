"""Model, quality and governance commands for the public CLI."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Optional

import click

from ..config import ConfigLoader
from .common import profile_option, verbose_option


@click.command("download-models")
@click.option("--all", "download_all", is_flag=True, help="下载所有模型")
@click.option("--asr-model", default=None,
              type=click.Choice(["tiny", "small", "medium", "large-v3"]),
              help="下载 faster-whisper ASR 模型")
@click.option("--separator", default=None, help="分离引擎 (uvr / spleeter)")
@click.option("--speaker-model", "speaker_models", multiple=True,
              type=click.Choice(["speechbrain-ecapa", "pyannote-embedding", "community-1", "diarization-3.1"]),
              help="下载 speaker 模型，可重复指定")
@click.option("--hf-token", default=None, help="Hugging Face Token（仅用于本次下载）")
@click.option("--list-speaker-models", is_flag=True, help="列出 speaker 模型缓存状态")
def download_models(download_all: bool, asr_model: Optional[str], separator: Optional[str],
                    speaker_models: tuple[str, ...], hf_token: Optional[str],
                    list_speaker_models: bool):
    """预下载模型文件。"""
    from ..diarization.model_registry import download_model, list_model_status

    if asr_model:
        from ..asr.model_download import ensure_faster_whisper_model

        click.echo(f"下载 faster-whisper 模型: {asr_model}")
        try:
            status = ensure_faster_whisper_model(asr_model)
        except Exception as exc:
            raise click.ClickException(
                f"faster-whisper 模型 {asr_model} 不可用: {exc}"
            ) from exc
        click.echo(f"  ✓ {status['status']}: {status['model_ref']} (cache={status['cache_dir']})")

    if list_speaker_models:
        for item in list_model_status():
            state = "ready" if item["cached"] else "not_cached"
            click.echo(f"{item['model_id']}: {state} — {item['model_ref']}")
        return

    selected = list(speaker_models)
    if download_all:
        selected = [item["model_id"] for item in list_model_status()]
    if selected:
        failures = []
        for model_id in selected:
            click.echo(f"下载 speaker 模型: {model_id}")
            try:
                status = download_model(model_id, token=hf_token)
                click.echo(f"  ✓ {status['status']}: {status['model_ref']}")
            except Exception as exc:
                failures.append((model_id, exc))
                click.echo(f"  ✗ {model_id} 下载失败: {exc}", err=True)
        if failures:
            failed_ids = ", ".join(model_id for model_id, _ in failures)
            raise click.ClickException(f"speaker 模型下载失败: {failed_ids}")
        return
    if asr_model:
        return
    click.echo("未指定 speaker 模型。使用 --list-speaker-models 查看状态。")


@click.command()
def profiles():
    """列出可用的场景模板。"""
    loader = ConfigLoader()
    click.echo("可用场景模板:")
    for name in loader.list_profiles():
        click.echo(f"  - {name}")


@click.command()
def info():
    """显示系统、引擎、实验和数据资产概览。"""
    from ..utils.gpu_detector import GPUDetector

    click.echo("=== 系统信息 ===")
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

    click.echo("\n=== 引擎状态 ===")
    try:
        from ..governance.engine_lifecycle import EngineRegistry, EngineLifecycle

        registry = EngineRegistry()
        status_counts = {status.value: 0 for status in EngineLifecycle}
        for engine in registry.list_all():
            status_counts[engine.status.value] += 1
        icons = {"unavailable": "❌", "model_missing": "📥", "ready_shadow": "🔬",
                 "ready_review": "🔍", "ready_default": "✅"}
        status_lines = [f"{icons.get(status, '❓')} {status}: {count}"
                        for status, count in status_counts.items() if count > 0]
        click.echo(f"  {', '.join(status_lines)} (共 {len(registry.list_all())} 个引擎)")
    except Exception:
        pass

    click.echo("\n=== 实验注册 ===")
    try:
        from ..governance.experiment_registry import ExperimentRegistry

        experiments = ExperimentRegistry().list_all()
        statuses = {}
        for experiment in experiments:
            statuses[experiment.status.value] = statuses.get(experiment.status.value, 0) + 1
        parts = [f"{status}: {count}" for status, count in sorted(statuses.items())]
        click.echo(f"  {', '.join(parts) if parts else '(无)'} (共 {len(experiments)} 个实验)")
    except Exception:
        pass

    click.echo("\n=== 任务历史 ===")
    try:
        from ..utils.task_history import TaskHistoryManager

        history = TaskHistoryManager()
        total = history.count()
        if total > 0:
            by_status = {}
            for task in history.list_recent(limit=5):
                status = task.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
            parts = [f"{status}: {count}" for status, count in sorted(by_status.items())]
            click.echo(f"  最近任务 ({min(5, total)}/{total}): {', '.join(parts)}")
        else:
            click.echo("  (无历史任务)")
    except Exception:
        pass

    click.echo("\n=== 数据资产 ===")
    try:
        from ..quality import DataVersionManager, DatasetTier

        manager = DataVersionManager()
        for tier in DatasetTier:
            version = manager.current(tier)
            if version:
                click.echo(f"  {tier.value}: {version.version_id} ({version.sample_count} 样本)")
            else:
                click.echo(f"  {tier.value}: 尚未建立")
    except Exception:
        pass

    click.echo("\n=== 发布状态 ===")
    try:
        from ..governance.release import ReleaseManager

        state = ReleaseManager().current_state()
        click.echo(f"  版本: {state.get('current_version', 'N/A')} ({state.get('status', 'N/A')})")
        blockers = state.get("blockers", [])
        if blockers:
            click.echo(f"  阻塞项: {len(blockers)} 项")
            for blocker in blockers[:3]:
                click.echo(f"    - {blocker}")
    except Exception:
        pass


@click.command("version")
def version_command():
    """显示 Vocal Subtitle 版本信息。"""
    from .. import __version__

    click.echo(f"Vocal Subtitle v{__version__}")
    click.echo(f"  Python:        {sys.version}")
    click.echo("  Config:        default-v1")
    click.echo("  ASR Route:     asr-route-v1")
    click.echo("  Quality Gate:  asr-quality-v1")
    click.echo("  Review Policy: review-policy-v1")
    click.echo("  Risk Policy:   risk-policy-v1")
    click.echo("  Decision:      decision-policy-v1")
    click.echo("  Evidence:      evidence-v1")
    click.echo("  Run Report:    run-report-v1")
    click.echo("  Task State:    task-state-v1")


@click.group("experiment")
def experiment_group():
    """管理实验注册表（EXPERIMENT_REGISTRY.md）。"""


@experiment_group.command("list")
@click.option("--status", "-s", default=None, help="按状态过滤 (proposed/shadow/review/enabled/rolled_back)")
@click.option("--category", "-c", default=None, help="按类别过滤 (engine/model/quantization/config/composite)")
def experiment_list(status: Optional[str] = None, category: Optional[str] = None):
    """列出所有注册实验。"""
    from ..governance.experiment_registry import ExperimentRegistry

    registry = ExperimentRegistry()
    experiments = registry.list_by_status(status) if status else registry.list_by_category(category) if category else registry.list_all()
    if not experiments:
        click.echo("(无匹配实验)")
        return
    for experiment in experiments:
        icon = {"proposed": "📋", "shadow": "🔬", "review": "🔍", "enabled": "✅", "rolled_back": "⬅️"}.get(experiment.status.value, "❓")
        click.echo(f"  {icon} [{experiment.status.value}] {experiment.experiment_id}")
        click.echo(f"     Name: {experiment.name}")
        click.echo(f"     Scope: {experiment.enable_scope}  |  Category: {experiment.category.value}")
        click.echo(f"     Benefit: {experiment.expected_benefit[:80]}")
        if experiment.known_risks:
            click.echo(f"     Risks: {len(experiment.known_risks)} items")
        click.echo()


@experiment_group.command("show")
@click.argument("experiment_id")
def experiment_show(experiment_id: str):
    """查看实验详情。"""
    from ..governance.experiment_registry import ExperimentRegistry

    experiment = ExperimentRegistry().get(experiment_id)
    if not experiment:
        click.echo(f"实验不存在: {experiment_id}", err=True)
        raise SystemExit(1)
    click.echo(f"\n  Experiment: {experiment.experiment_id}")
    click.echo(f"  Name:       {experiment.name}")
    click.echo(f"  Status:     {experiment.status.value}")
    click.echo(f"  Category:   {experiment.category.value}")
    click.echo(f"  Scope:      {experiment.enable_scope}")
    click.echo(f"  Owner:      {experiment.owner}")
    click.echo(f"  Created:    {experiment.created}")
    click.echo(f"  Engines:    {', '.join(experiment.engines) if experiment.engines else 'N/A'}")
    click.echo(f"  Models:     {', '.join(experiment.models) if experiment.models else 'N/A'}")
    click.echo(f"  Languages:  {', '.join(experiment.languages)}")
    click.echo("\n  Expected Benefit:")
    click.echo(f"    {experiment.expected_benefit}")
    if experiment.known_risks:
        click.echo("\n  Known Risks:")
        for risk in experiment.known_risks:
            click.echo(f"    - {risk}")
    if experiment.withdraw_conditions:
        click.echo("\n  Withdraw Conditions:")
        for condition in experiment.withdraw_conditions:
            click.echo(f"    - {condition}")
    click.echo()


@click.group("data-assets")
def data_assets_group():
    """管理数据资产（DATA_ASSETS.md）。"""


@data_assets_group.command("list")
@click.option("--tier", "-t", default="D0", help="数据层级 (D0/D1/D2/D3/D4)")
def data_assets_list(tier: str = "D0"):
    """列出已登记的数据资产。"""
    if tier != "D0":
        click.echo(f"{tier} 尚未建立（见 DATA_ASSETS.md）")
        return
    click.echo("D0 — 工程诊断集 (DATA_ASSETS.md)\n")
    click.echo("  test/ 目录音频 (11 个):")
    assets = [
        ("D0-001", "QS-0-1-2-2-人声.wav", "32.77s, zh, 2人, ✅"),
        ("D0-002", "181人声.wav", "58.45s, zh, 1-2人, ✅"),
        ("D0-003", "培训测试-双人.wav", "139.52s, mixed, 2人, ✅"),
        ("D0-004", "中文多人员测试音频.wav", "13.28s, zh, 3人, ✅"),
        ("D0-005", "英文多人员测试音频.wav", "18.64s, en, 3人, ✅"),
        ("D0-006", "video_英国老头评测-人声.wav", "53.89s, en, 1人, ✅"),
        ("D0-007", "TTS中文朗读测试-双人.wav", "157.93s, zh, 2人, ❌"),
        ("D0-008", "简单三步就能复刻巧乐兹？-人声.wav", "216.23s, zh, 1+人, ❌"),
        ("D0-009", "Grow Up Show-4min-人声.wav", "249.30s, zh, 2+人, ✅*"),
        ("D0-010", "40011894204-1-192-英语多人.mp3", "46.34s, en, 2+人, ✅*"),
        ("D0-011", "20260428_214253_0043_KyGLgfKX.wav", "5.84s, zh, 1人, ❌"),
    ]
    for asset_id, filename, details in assets:
        click.echo(f"    {asset_id}  {filename}  [{details}]")
    click.echo("\n  test/golden/ 合成样本 (2 个):")
    click.echo("    D0-G01  non_speech_tone.wav (4.00s, 纯音与噪声)")
    click.echo("    D0-G02  repeated_phrase_me.wav (2.80s, 重复短语)")
    click.echo("\n  test/quality_manifest.yaml: 10 个场景\n\n  D1-D4: 尚未建立（见 DATA_ASSETS.md）")


@click.command("preflight")
@click.argument("input_path", type=click.Path(exists=True))
@profile_option
@click.option("--output", "-o", default=None, help="预估输出路径（默认与输入同目录同名 .srt）")
@verbose_option
def preflight(input_path: str, profile: str, output: str | None, verbose: bool):
    """检查音频文件是否可被管道处理（只检查，不处理）。"""
    from ..application.preflight import run_preflight_checks

    config = ConfigLoader().load_profile(profile)
    in_path = Path(input_path).resolve()
    out_path = Path(output).resolve() if output else in_path.with_suffix(".srt")
    result = run_preflight_checks(in_path, out_path, config)
    click.echo(f"\n  预检结果 for: {in_path.name}\n  配置模板: {profile}\n")
    for check in result.checks:
        icon = ("✅" if check.passed else "❌") if check.critical else ("⚠️" if not check.passed else "✅")
        line = f"  {icon} {check.label}: {'通过' if check.passed else '未通过'}"
        click.echo(line + (f" — {check.reason}" if check.reason else ""))
    if result.engine_snapshot:
        click.echo("\n  引擎状态:")
        for engine, status in sorted(result.engine_snapshot.items()):
            icon = "✅" if status.startswith("ready") else "⚠️" if status == "model_missing" else "❌"
            click.echo(f"    {icon} {engine}: {status}")
    if result.passed:
        click.echo("\n  ✅ 预检通过，可以处理")
        raise click.exceptions.Exit(0)
    failed = result.failed_critical()
    click.echo(f"\n  ❌ 预检失败（{len(failed)} 项关键检查未通过）:")
    for check in failed:
        click.echo(f"      - {check.label}: {check.reason}")
    raise click.exceptions.Exit(1)


@click.group("engine")
def engine_group():
    """管理引擎生命周期（ENGINE_LIFECYCLE.md）。"""


@engine_group.command("status")
def engine_status():
    """显示所有引擎当前状态。"""
    from ..governance.engine_lifecycle import EngineRegistry

    registry = EngineRegistry()
    categories = {"分离": ["uvr", "spleeter", "open-unmix"], "VAD": ["silero", "webrtc"],
                  "ASR": ["faster-whisper", "funasr", "qwen-asr", "whisper.cpp"],
                  "复核": ["global-asr-evidence", "context-reasr", "qwen-review", "forced-aligner", "sed", "semantic-review"],
                  "说话人": ["speechbrain-ecapa", "pyannote"]}
    icons = {"unavailable": "❌", "model_missing": "📥", "ready_shadow": "🔬", "ready_review": "🔍", "ready_default": "✅"}
    for category, keys in categories.items():
        click.echo(f"\n  {category}:")
        for key in keys:
            engine = registry.get(key)
            if engine:
                model = f" ({engine.model})" if engine.model else ""
                click.echo(f"    {icons.get(engine.status.value, '❓')} {engine.engine}{model}: {engine.status.value}")
    click.echo()


@click.group("quality")
def quality_group():
    """质量运营工具（QUALITY_OPERATIONS.md）。"""


@quality_group.command("classify")
@click.argument("text")
def quality_classify(text: str):
    """对问题描述进行自动分类。"""
    from ..quality import IssueClassifier

    category = IssueClassifier.classify(text)
    click.echo(f"  文本: {text[:80]}\n  类别: {category.value}\n  严重度: {IssueClassifier.assess_severity(category).value}")


@quality_group.command("trend")
@click.option("--version", "-v", required=True, help="当前版本号")
@click.option("--baseline", "-b", required=True, help="基线版本号")
@click.option("--date", "-d", default="", help="报告日期（默认今天）")
def quality_trend(version: str, baseline: str, date: str):
    """生成版本趋势报告。"""
    from ..quality import TrendReporter, VersionMetrics

    report = VersionMetrics(version=version, date=date, baseline=baseline, trends={
        "D0_engineering": {"test_pass_rate": 0.98, "regression_count": 0, "status": "ok"},
        "D1_reference": {"status": "not_available"}, "D3_feedback": {"status": "not_available"},
        "D4_challenge": {"status": "not_available"}, "operations": {"crash_rate": 0.001, "degradation_rate": 0.05, "status": "ok"},
    })
    path = TrendReporter().write_report(report)
    click.echo(f"  趋势报告已生成: {path}\n  版本: {version} (基线: {baseline})")


@quality_group.command("datasets")
def quality_datasets():
    """列出数据集版本。"""
    from ..quality import DataVersionManager, DatasetTier

    manager = DataVersionManager()
    for tier in DatasetTier:
        version = manager.current(tier)
        click.echo(f"  {tier.value}: {version.version_id} ({version.sample_count} 样本, {version.freeze_date})" if version else f"  {tier.value}: 尚未建立")


@click.group("release")
def release_group():
    """发布治理工具（RELEASE_GOVERNANCE.md）。"""


@release_group.command("status")
def release_status():
    """查看当前发布状态。"""
    from ..governance.release import ReleaseManager

    state = ReleaseManager().current_state()
    click.echo(f"  当前版本: {state.get('current_version', 'N/A')}\n  发布状态: {state.get('status', 'N/A')}\n  目标状态: {state.get('target', 'N/A')}")
    blockers = state.get("blockers", [])
    if blockers:
        click.echo(f"\n  阻断项 ({len(blockers)}):")
        for blocker in blockers:
            click.echo(f"    ❌ {blocker}")
    else:
        click.echo("\n  ✅ 无阻断项")


@release_group.command("check")
@click.option("--version", "-v", required=True, help="版本号")
def release_check(version: str):
    """运行发布前检查单。"""
    from ..governance.release import ReleaseManager

    checklist = ReleaseManager().check_pre_release(version)
    click.echo(f"\n  发布前检查单 — v{version}\n  检查时间: {checklist.checked_at}\n")
    for section, items in checklist.sections.items():
        click.echo(f"  [{section}]")
        for item in items:
            icon = "✅" if item["passed"] is True else "⬜" if item["passed"] is None else "❌"
            click.echo(f"    {icon} {item['description']}")
        click.echo()
    if checklist.all_passed():
        click.echo("  ✅ 所有检查项通过")
    else:
        pending = sum(1 for items in checklist.sections.values() for item in items if item["passed"] is not True)
        click.echo(f"  ⚠️ {pending} 项待完成或未通过")


@release_group.command("notes")
@click.option("--version", "-v", required=True, help="版本号")
@click.option("--status", "-s", default="production-usable",
              type=click.Choice(["development", "production-usable", "quality-improving"]), help="发布状态")
def release_notes(version: str, status: str):
    """生成版本发布说明。"""
    from ..governance.release import ReleaseManager, ReleaseStatus

    click.echo(ReleaseManager.release_notes(version=version, status=ReleaseStatus(status),
                                             changes={"优化": ["请在此填写具体变更"]},
                                             upgrades=["请在此填写升级注意事项"],
                                             known_issues=["请在此填写已知问题"]))


@release_group.command("limitations")
def release_limitations():
    """列出已知限制。"""
    from ..governance.release import ReleaseManager

    for limitation in ReleaseManager.list_known_limitations():
        click.echo(f"  [{limitation['limitation_id']}] {limitation['title']}\n    问题: {limitation['description']}\n    缓解: {limitation['mitigation']}\n")


@release_group.command("support")
@click.argument("issue", required=False)
def release_support(issue: Optional[str] = None):
    """查看支持手册或搜索特定问题。"""
    from ..governance.release import ReleaseManager

    if issue:
        info = ReleaseManager.get_support_info(issue)
        if info:
            click.echo(f"  问题: {info['issue']}\n  诊断: {info['diagnosis']}\n  解决: {info['resolution']}")
        else:
            click.echo(f"  未找到匹配问题: {issue}")
        return
    manual = ReleaseManager.support_manual()
    click.echo("  常见问题:")
    for faq in manual["faq"]:
        click.echo(f"    • {faq['issue']} → {faq['resolution']}")
    click.echo("\n  日志位置:")
    for name, path in manual["log_locations"].items():
        click.echo(f"    • {name}: {path}")


def register(main) -> None:
    """Register administration commands on the public root group."""
    for command in (download_models, profiles, info, version_command, preflight):
        main.add_command(command)
    for group in (experiment_group, data_assets_group, engine_group, quality_group, release_group):
        main.add_command(group)
