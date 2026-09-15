"""数据集导出端点（定案 D35：CLI export-dataset 为核，8613 数据集工作区套壳）。

导出与统计全部复用 feedback/dataset_export 服务层（与 CLI `feedback
export-dataset` 共用同一实现，本模块不持有第二份导出逻辑），只做 HTTP 适配：

  - GET  /api/feedback/dataset/preview  导出前统计预览（accepted-only + 场景过滤，
          与导出走同一个收集函数，预览数字 = 导出条目数）
  - POST /api/feedback/dataset/export   执行导出（许可必选，D33 门禁语义由服务层把守）

执行方式：当前样本量小（单条样本为文本对，导出与 checksum 毫秒级），同步执行
即可；若未来样本量增长到万级（分片写入 + bundle-audio 复制音频实体会明显
耗时，HTTP 同步可能超时），应改为后台任务立即返回任务标识（参考 D28 学习
任务的异步化做法），届时再扩展本端点契约。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_DATASET_NAME = "subtitle-feedback"


class DatasetExportRequest(BaseModel):
    """数据集导出请求体（D33：许可强制显式选择，缺失由服务层门禁拒绝）"""

    license: str = Field(
        default="", description="数据集许可（如 CC-BY-4.0 / CC0-1.0），必选"
    )
    scenarios: list[str] = Field(
        default_factory=list, description="场景过滤（D27 标签；空 = 全部场景）"
    )
    out_dir: str | None = Field(
        default=None, description="输出目录覆盖；缺省放用户数据目录 datasets/<name>"
    )
    bundle_audio: bool = Field(
        default=False, description="把音频实体复制进 audio/（仅本地使用，不建议推 git）"
    )
    dataset_name: str = Field(
        default=DEFAULT_DATASET_NAME, description="数据集名称（写入 README 卡片）"
    )


def _default_dataset_dir(dataset_name: str) -> Path:
    """默认输出目录：用户数据目录下的 datasets/<name>（与 journal_sink 推导同源）"""
    safe = str(dataset_name or DEFAULT_DATASET_NAME).strip() or DEFAULT_DATASET_NAME
    return state.upload_dir.parent / "datasets" / safe


def _sample_manager():
    """样本库（导出只读使用）；函数内导入，测试可整体替换隔离"""
    from ..feedback.sample_manager import FeedbackSampleManager

    return FeedbackSampleManager()


def _clean_scenarios(raw) -> list[str]:
    """清洗并校验场景标签（与 CLI export-dataset 同一取值集）"""
    from ..feedback.dataset_export import SCENARIOS

    wanted = [str(item).strip() for item in (raw or []) if str(item).strip()]
    invalid = [item for item in wanted if item not in SCENARIOS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"未知场景标签: {', '.join(invalid)}；可选值: {', '.join(SCENARIOS)}",
        )
    return wanted


@router.get("/feedback/dataset/preview")
async def dataset_preview(
    scenarios: str = Query(default="", description="场景过滤，逗号分隔；缺省全部场景"),
    dataset_name: str = Query(
        default=DEFAULT_DATASET_NAME, description="数据集名称（决定默认输出目录）"
    ),
    out_dir: str | None = Query(
        default=None, description="输出目录覆盖；缺省 datasets/<name>"
    ),
):
    """导出前统计预览：可导样本总数、按场景/按语言分布（与导出同一过滤器）"""
    from ..feedback.dataset_export import summarize_export

    wanted = _clean_scenarios([item.strip() for item in (scenarios or "").split(",")])
    directory = Path(out_dir) if out_dir else _default_dataset_dir(dataset_name)
    try:
        stats = summarize_export(_sample_manager(), wanted, out_dir=directory)
    except ValueError as exc:
        # 已有 manifest 损坏：预览与导出报同一错误，提示人工检查目录
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **stats,
        "out_dir": str(directory),
        "default_out_dir": str(_default_dataset_dir(dataset_name)),
    }


@router.post("/feedback/dataset/export")
async def dataset_export(body: DatasetExportRequest):
    """执行导出：产出与 CLI `feedback export-dataset` 相同的 git-ready 数据集目录"""
    from ..feedback.dataset_export import LicenseRequiredError, export_dataset

    wanted = _clean_scenarios(body.scenarios)
    name = (
        str(body.dataset_name or DEFAULT_DATASET_NAME).strip() or DEFAULT_DATASET_NAME
    )
    directory = Path(body.out_dir) if body.out_dir else _default_dataset_dir(name)
    try:
        outcome = export_dataset(
            _sample_manager(),
            directory,
            license=body.license,
            scenarios=wanted,
            bundle_audio=body.bundle_audio,
            dataset_name=name,
        )
    except LicenseRequiredError as exc:
        # D33 许可门禁：语义来自服务层，HTTP 侧仅转译为 400
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # 空库（无 accepted 样本）/ 已有 manifest 损坏：与 CLI 同样拒绝
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Dataset export failed")
        raise HTTPException(
            status_code=500, detail=f"数据集目录写入失败: {exc}"
        ) from exc

    total_samples = 0
    manifest_path = Path(outcome.out_dir) / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            total_samples = int(manifest.get("total_samples") or 0)
        except (ValueError, OSError):
            total_samples = 0

    return {
        "status": "ok",
        "out_dir": str(outcome.out_dir),
        "dataset_name": name,
        "license": outcome.license,
        "scenarios": outcome.scenarios,
        "accepted_total": outcome.accepted_total,
        "exported_count": outcome.exported_count,
        "total_samples": total_samples,
        "shards": outcome.shards,
        "skipped_missing_text": outcome.skipped_missing_text,
        "bundled_audio": outcome.bundled_audio,
        "missing_audio": outcome.missing_audio,
        "generated_at": outcome.generated_at,
        "message": f"已导出 {outcome.exported_count} 条样本（累计 {total_samples} 条）到 {outcome.out_dir}",
    }


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DatasetExportRequest",
    "dataset_export",
    "dataset_preview",
    "router",
]
