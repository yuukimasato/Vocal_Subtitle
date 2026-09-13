"""数据集导出 — dataset-v1（定案 D31/D32/D33）

把审核队列中**已接受**的 D2 样本物化为可分享数据集：

  - 只导 accepted（pending/rejected/disputed 不出门；审核队列即发布门禁）
  - 条目 = 文本对（auto vs human）+ 场景/语言/说话人数/时长/出处元数据
    + 音频引用（task_id + sha256 + duration，不搬音频字节）
  - 产物为 git-ready 目录：README 数据集卡片 + data/*.jsonl 分片 + manifest 清单
  - 许可强制显式选择并记入卡片（D33：绕过许可等于白建）
  - 追加式导出：重复导出新增分片，不改写已存在分片（历史分片 checksum 可复现）
  - --bundle-audio 仅把音频实体复制进本地目录，并明确警示不建议推 git

本模块是治理操作：只读样本库，只写用户指定的输出目录；不持有、不调用任何 git 命令。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# schema 字符串固定（定案 §4 dataset-v1）
DATASET_SCHEMA = "dataset-v1"

# D27 四场景标签（样本 scene 字段的合法取值）
SCENARIOS = (
    "inline-review",
    "external-correction",
    "existing-subtitle",
    "from-scratch-timing",
)

# --bundle-audio 从任务会话目录复制的音频实体后缀
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".webm"}

# 数据集卡片头部警示（--bundle-audio 时写入 README 与 stdout）
BUNDLE_AUDIO_WARNING = "audio/ 目录含原始音频实体，体积大且涉及版权，不建议推送到 git 远程"


class LicenseRequiredError(ValueError):
    """许可未显式指定（D33：导出时强制选择，参数或交互皆可）。"""


@dataclass
class ExportOutcome:
    """一次导出的结果摘要"""

    out_dir: Path
    generated_at: str
    license: str = ""
    scenarios: list[str] = field(default_factory=list)
    accepted_total: int = 0            # 过滤后样本库中的 accepted 总数
    exported_ids: list[str] = field(default_factory=list)   # 本次新写入的样本
    skipped_missing_text: list[str] = field(default_factory=list)  # 缺字幕全文无法物化的旧样本
    shards: list[str] = field(default_factory=list)         # 本次新增分片（相对路径）
    bundled_audio: list[str] = field(default_factory=list)  # 已打包的会话目录（sha256 前 16 位）
    missing_audio: list[str] = field(default_factory=list)  # 找不到音频实体的引用

    @property
    def exported_count(self) -> int:
        return len(self.exported_ids)


# ---- 样本收集（只读）----


def collect_accepted_samples(
    manager,
    scenarios: Iterable[str] = (),
) -> list[dict]:
    """按审核状态与可选场景过滤样本库，返回样本详情列表（只读，不改样本库）。

    只取 review 状态为 accepted 的样本（D31：审核队列 = 发布门禁）。
    """
    wanted = {str(item).strip() for item in scenarios if str(item).strip()}
    samples: list[dict] = []
    for sample_id in manager.list_sample_ids(status="accepted"):
        data = manager.get(sample_id)
        if data is None:
            continue
        if wanted and data.get("scene") not in wanted:
            continue
        samples.append(data)
    return sorted(samples, key=lambda item: item.get("sample_id", ""))


def build_entry(sample: dict) -> Optional[dict]:
    """D2 样本 → dataset-v1 条目（字段以定案 §4 条目示例为准）。

    旧版样本只落了字幕哈希没存全文，无法物化文本对 → 返回 None（导出时跳过并记录）。
    """
    auto = str((sample.get("automatic_subtitle") or {}).get("text") or "")
    human = str((sample.get("human_revision") or {}).get("text") or "")
    if not auto or not human:
        return None
    revision = sample.get("human_revision") or {}
    task_id = str(sample.get("task_id") or "")
    duration = float(sample.get("audio_duration_seconds") or 0.0)
    return {
        "schema": DATASET_SCHEMA,
        "sample_id": str(sample.get("sample_id") or ""),
        "scenario": str(sample.get("scene") or "unknown"),
        "language": str(sample.get("language") or "unknown"),
        "speaker_count": int(sample.get("speaker_count") or 0),
        "audio_duration": duration,
        "run_id": str(sample.get("run_id") or ""),
        "task_id": task_id,
        "auto_subtitle": auto,
        "human_revision": human,
        "edit_types": dict(revision.get("edit_types") or {}),
        "audio_ref": {
            "task_id": task_id,
            "sha256": str(sample.get("audio_sha256") or ""),
            "duration": duration,
        },
    }


# ---- 产物目录 ----


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(out_dir: Path) -> dict:
    """读取已有 manifest（追加式导出依据）；不存在则给空骨架。"""
    path = out_dir / "manifest.json"
    if not path.exists():
        return {"schema": DATASET_SCHEMA, "shards": [], "exports": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"输出目录已有 manifest.json 无法解析（{exc}），请人工检查后处理") from exc
    if not isinstance(manifest, dict):
        raise ValueError("输出目录已有 manifest.json 格式异常（顶层不是对象）")
    manifest.setdefault("schema", DATASET_SCHEMA)
    manifest.setdefault("shards", [])
    manifest.setdefault("exports", [])
    return manifest


def _next_shard_index(out_dir: Path) -> int:
    """下一个分片序号 = 已存在分片最大序号 + 1（以文件系统为准，追加不改写）。"""
    data_dir = out_dir / "data"
    highest = 0
    if data_dir.is_dir():
        for path in data_dir.glob("shard-*.jsonl"):
            try:
                highest = max(highest, int(path.stem.split("-")[-1]))
            except ValueError:
                continue
    return highest + 1


def _default_session_root(manager) -> Path:
    """任务会话目录根（cache/uploads），音频实体定位 cache/uploads/{sha256[:16]}/。"""
    return Path(manager.storage_dir).parent / "uploads"


def _bundle_audio(samples: list[dict], out_dir: Path, session_root: Path) -> tuple[list[str], list[str]]:
    """把音频引用对应的会话音频实体复制进 out_dir/audio/{sha256[:16]}/。

    找不到会话目录或音频文件的引用照写，只记录缺失并告警（不阻塞导出）。
    """
    audio_dir = out_dir / "audio"
    bundled: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        sha = str(sample.get("audio_sha256") or "").strip()
        if not sha or sha in seen:
            continue
        seen.add(sha)
        prefix = sha[:16]
        source_dir = session_root / prefix
        audio_files = sorted(
            path for path in source_dir.glob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        ) if source_dir.is_dir() else []
        if not audio_files:
            missing.append(prefix)
            continue
        target_dir = audio_dir / prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in audio_files:
            shutil.copy2(source, target_dir / source.name)
        bundled.append(prefix)
    return bundled, missing


def _aggregate_stats(out_dir: Path, manifest: dict) -> dict:
    """读回全部分片，累计统计（场景/语言分布、总时长、样本数）。"""
    by_scenario: Counter = Counter()
    by_language: Counter = Counter()
    total_duration = 0.0
    total_samples = 0
    for shard in manifest["shards"]:
        path = out_dir / shard["file"]
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            total_samples += 1
            by_scenario[str(entry.get("scenario") or "unknown")] += 1
            by_language[str(entry.get("language") or "unknown")] += 1
            total_duration += float(entry.get("audio_duration") or 0.0)
    return {
        "total_samples": total_samples,
        "by_scenario": dict(by_scenario),
        "by_language": dict(by_language),
        "total_duration_seconds": round(total_duration, 2),
    }


def _counter_block(title: str, values: dict) -> str:
    if not values:
        return f"- {title}: 无"
    pairs = "，".join(f"{key} {value} 条" for key, value in sorted(values.items()))
    return f"- {title}: {pairs}"


def render_readme(
    *,
    dataset_name: str,
    manifest: dict,
    stats: dict,
    latest_license: str,
    latest_scenarios: list[str],
    generated_at: str,
    bundle_audio: bool,
) -> str:
    """渲染 README 数据集卡片（schema/统计/来源/许可 + 发行建议，D33）。"""
    exports = manifest.get("exports", [])
    history_rows = []
    for position, export in enumerate(exports, 1):
        scenarios = export.get("scenarios") or []
        filter_text = "，".join(scenarios) if scenarios else "全部场景"
        shards = export.get("shards") or []
        shard_text = "、".join(shards) if shards else "无（无新增样本）"
        history_rows.append(
            f"| {position} | {export.get('generated_at', '?')} | {filter_text} "
            f"| {export.get('license', '?')} | {export.get('new_samples', 0)} | {shard_text} |"
        )
    licenses = []
    for export in reversed(exports):
        license_name = str(export.get("license") or "")
        if license_name and license_name not in licenses:
            licenses.append(license_name)
    license_block = latest_license
    if len(licenses) > 1:
        license_block += f"（历史导出许可：{'、'.join(licenses[1:])}）"
    audio_line = "- 音频：条目仅含音频引用（task_id + sha256 + duration），不含音频字节"
    if bundle_audio:
        audio_line += f"；本次导出另将音频实体复制到了 audio/ 目录（⚠️ {BUNDLE_AUDIO_WARNING}）"
    lines = [
        f"# {dataset_name} 数据集卡片",
        "",
        f"- schema: `{DATASET_SCHEMA}`",
        f"- 生成时间: {generated_at}",
        f"- 样本总数: {stats['total_samples']}",
        f"- 许可: {license_block}",
        "",
        "## 统计",
        "",
        _counter_block("按场景", stats["by_scenario"]),
        _counter_block("按语言", stats["by_language"]),
        f"- 字幕总时长: {stats['total_duration_seconds']} 秒",
        "",
        "## 来源",
        "",
        "- 来源: 本地反馈样本库（D2 候选反馈集）中审核状态为 accepted 的样本（审核队列即发布门禁）",
        "- 内容: 管线自动字幕 vs 人工修订字幕的文本对，附场景/语言/说话人数/时长/出处元数据",
        "- 场景: " + "、".join(SCENARIOS),
        audio_line,
        "",
        "## 导出历史（追加式分片）",
        "",
        "| # | 时间(UTC) | 过滤器 | 许可 | 新增样本 | 分片 |",
        "| --- | --- | --- | --- | --- | --- |",
        *history_rows,
        "",
        "## 许可",
        "",
        f"本数据集按导出时的显式选择以 {latest_license} 许可发布；分发前请自行确认字幕内容版权。",
        "",
        "## 发行建议",
        "",
        "- 本目录为 git-ready：推送由用户手动完成，工具不持有 git 凭据",
        "- 建议以 git tag 标记发行版本（追加式导出会新增分片，tag 保证已发布版本可复现）",
        "- 已存在分片不会被改写，各分片 checksum 见 manifest.json",
    ]
    if bundle_audio:
        lines.append(f"- ⚠️ {BUNDLE_AUDIO_WARNING}")
    return "\n".join(lines) + "\n"


# ---- 导出主流程 ----


def export_dataset(
    manager,
    out_dir: Path | str,
    *,
    license: str,
    scenarios: Iterable[str] = (),
    bundle_audio: bool = False,
    shard_size: int = 500,
    dataset_name: str = "subtitle-feedback",
    session_root: Optional[Path] = None,
    now: Optional[str] = None,
) -> ExportOutcome:
    """把 accepted 样本追加导出为 dataset-v1 数据集目录。

    Args:
        manager: FeedbackSampleManager（只读使用）
        out_dir: 数据集输出目录（重复导出到同一目录即追加式新增分片）
        license: 数据集许可，必填（D33 强制显式选择）
        scenarios: 场景过滤（D27 标签，缺省全部）
        bundle_audio: 是否把音频实体打包进 audio/（默认不打包）
        shard_size: 单个 jsonl 分片的最大样本数
        dataset_name: 数据集名称（写入卡片）
        session_root: 任务会话目录根（缺省 cache/uploads）
        now: 覆盖生成时间（测试用）

    Raises:
        LicenseRequiredError: 许可为空
        ValueError: 首次导出没有可导出样本，或已有 manifest 损坏
    """
    if not str(license or "").strip():
        raise LicenseRequiredError("数据集许可未指定：dataset-v1 导出强制显式选择许可（--license 或交互确认）")
    license_name = str(license).strip()
    out_dir = Path(out_dir)
    generated_at = now or _utc_now_iso()
    wanted = sorted({str(item).strip() for item in scenarios if str(item).strip()})

    selected = collect_accepted_samples(manager, wanted)
    manifest = _load_manifest(out_dir)
    first_export = not manifest["shards"]
    if not selected and first_export:
        hint = f"（场景过滤: {'，'.join(wanted)}）" if wanted else ""
        raise ValueError(
            f"样本库中没有审核状态为 accepted 的样本{hint}；"
            "请先在审核队列接受样本再导出（pending/rejected 不出门）"
        )

    # 追加式：已导出的样本不重复，只为新增样本写新分片
    already = {
        sample_id
        for shard in manifest["shards"]
        for sample_id in shard.get("sample_ids", [])
    }
    entries: list[dict] = []
    skipped_ids: list[str] = []
    for sample in selected:
        if sample.get("sample_id") in already:
            continue
        entry = build_entry(sample)
        if entry is None:
            skipped_ids.append(str(sample.get("sample_id") or "?"))
            continue
        entries.append(entry)

    shards_written: list[str] = []
    if not entries and not first_export and not bundle_audio:
        # 无新增样本：幂等返回，不写任何文件（已存在分片/manifest/README 保持原样）
        return ExportOutcome(
            out_dir=out_dir,
            generated_at=generated_at,
            license=license_name,
            scenarios=wanted,
            accepted_total=len(selected),
            exported_ids=[],
            skipped_missing_text=skipped_ids,
        )
    if entries:
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        index = _next_shard_index(out_dir)
        for start in range(0, len(entries), max(1, shard_size)):
            chunk = entries[start:start + max(1, shard_size)]
            shard_rel = f"data/shard-{index:04d}.jsonl"
            payload = "".join(
                json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                for entry in chunk
            )
            (out_dir / shard_rel).write_text(payload, encoding="utf-8")
            manifest["shards"].append({
                "file": shard_rel,
                "sample_count": len(chunk),
                "sha256": _sha256_text(payload),
                "sample_ids": [entry["sample_id"] for entry in chunk],
                "created_at": generated_at,
            })
            shards_written.append(shard_rel)
            index += 1

    bundled: list[str] = []
    missing_audio: list[str] = []
    if bundle_audio:
        root = Path(session_root) if session_root else _default_session_root(manager)
        bundled, missing_audio = _bundle_audio(selected, out_dir, root)

    manifest.update({
        "schema": DATASET_SCHEMA,
        "generated_at": manifest.get("generated_at") or generated_at,
        "updated_at": generated_at,
        "license": license_name,
        "total_samples": sum(shard.get("sample_count", 0) for shard in manifest["shards"]),
        "scenarios_filter": wanted,
    })
    manifest["exports"].append({
        "generated_at": generated_at,
        "license": license_name,
        "scenarios": wanted,
        "bundle_audio": bundle_audio,
        "new_samples": len(entries),
        "skipped_missing_text": skipped_ids,
        "shards": shards_written,
    })

    stats = _aggregate_stats(out_dir, manifest)
    readme = render_readme(
        dataset_name=dataset_name,
        manifest=manifest,
        stats=stats,
        latest_license=license_name,
        latest_scenarios=wanted,
        generated_at=generated_at,
        bundle_audio=bundle_audio,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    return ExportOutcome(
        out_dir=out_dir,
        generated_at=generated_at,
        license=license_name,
        scenarios=wanted,
        accepted_total=len(selected),
        exported_ids=[entry["sample_id"] for entry in entries],
        skipped_missing_text=skipped_ids,
        shards=shards_written,
        bundled_audio=bundled,
        missing_audio=missing_audio,
    )


# ---- 导出预览（只读统计；8613 数据集工作区使用，CLI 不受影响）----


def summarize_export(
    manager,
    scenarios: Iterable[str] = (),
    out_dir: Path | str | None = None,
) -> dict:
    """导出前统计预览（只读，不写任何文件）：与 export_dataset 完全同口径。

    样本收集走同一个 collect_accepted_samples（accepted-only + 场景过滤），
    条目物化走同一个 build_entry；给定 out_dir 时还按导出的追加式语义读取
    已有 manifest、剔除已导出样本——保证"预览数字 = 实际导出条目数"。

    与 export_dataset 的差异仅在边界行为：无 accepted 样本 / 目录尚不存在时
    不抛错，返回零值统计（预览应友好提示"暂无可导样本"而非失败）。

    Raises:
        ValueError: out_dir 已有 manifest.json 但无法解析（与导出同一报错）
    """
    wanted = sorted({str(item).strip() for item in scenarios if str(item).strip()})
    selected = collect_accepted_samples(manager, wanted)

    already: set[str] = set()
    existing_samples = 0
    if out_dir is not None:
        directory = Path(out_dir)
        if (directory / "manifest.json").exists():
            manifest = _load_manifest(directory)
            already = {
                sample_id
                for shard in manifest["shards"]
                for sample_id in shard.get("sample_ids", [])
            }
            existing_samples = int(manifest.get("total_samples") or 0)

    by_scenario: Counter = Counter()
    by_language: Counter = Counter()
    total_duration = 0.0
    exportable_ids: list[str] = []
    skipped_ids: list[str] = []
    already_count = 0
    for sample in selected:
        sample_id = str(sample.get("sample_id") or "")
        if sample_id in already:
            already_count += 1
            continue
        entry = build_entry(sample)
        if entry is None:
            skipped_ids.append(sample_id or "?")
            continue
        exportable_ids.append(sample_id)
        by_scenario[str(entry["scenario"])] += 1
        by_language[str(entry["language"])] += 1
        total_duration += float(entry["audio_duration"] or 0.0)

    return {
        "schema": DATASET_SCHEMA,
        "scenarios_filter": wanted,
        "accepted_total": len(selected),       # 过滤后样本库中的 accepted 总数
        "already_exported": already_count,     # 其中已在此输出目录导出过的
        "exportable_count": len(exportable_ids),  # 本次将新增导出的条目数 = 导出 exported_count
        "skipped_missing_text": skipped_ids,   # 缺字幕全文无法物化的旧样本
        "by_scenario": dict(by_scenario),
        "by_language": dict(by_language),
        "total_duration_seconds": round(total_duration, 2),
        "existing_dataset_samples": existing_samples,  # 输出目录已有数据集的累计样本数
    }
