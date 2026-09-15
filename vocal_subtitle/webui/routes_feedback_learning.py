"""HTTP adapters for feedback learning and preview."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .feedback_services import FeedbackLearningService, TaskBaselineError
from .pipeline_tasks import (
    ASYNC_LEARN_SCENARIOS,
    PipelineSubmissionError,
    submit_learn_task,
)
from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()
_service = FeedbackLearningService()

# D27 场景标签：V1 工作台内修正 / V2 外部工具修正上传 / V3 存量字幕学习 / V4 从零打轴
VALID_SCENARIOS = (
    "inline-review",
    "external-correction",
    "existing-subtitle",
    "from-scratch-timing",
)


async def _run_feedback(
    audio: UploadFile | None,
    reference: UploadFile,
    *,
    task_id: str = "",
    scenario: str = "",
    profile: str,
    feedback_profile: str,
    run_pipeline_first: bool,
    dry_run: bool,
):
    ref_suffix = Path(reference.filename or "").suffix.lower()
    if ref_suffix not in (".srt", ".ass"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的字幕格式: {ref_suffix}，仅支持 .srt / .ass",
        )
    if scenario and scenario not in VALID_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"无效的场景标签: {scenario}，允许取值: {'/'.join(VALID_SCENARIOS)}",
        )
    # 校验顺序（定案 §4）：task_id → audio → 400
    if not task_id and audio is None:
        raise HTTPException(
            status_code=400,
            detail="task_id 与音频文件至少需要一个：带 task_id 时复用任务历史基线，"
            "否则请上传音频重跑学习",
        )
    # D28 冷重跑异步化：V3/V4 场景缺 task_id 且需要跑管线时，转为内部学习
    # 任务提交进现有任务系统，立即返回任务标识，不阻塞 HTTP 等待管线；
    # 带 task_id 的同步引用路径与旧客户端音频路径行为不变，dry_run 预览不变。
    if (
        not task_id
        and run_pipeline_first
        and not dry_run
        and scenario in ASYNC_LEARN_SCENARIOS
    ):
        try:
            audio_contents = await audio.read()
            reference_contents = await reference.read()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"文件读取失败: {exc}") from exc
        try:
            return await submit_learn_task(
                audio_contents,
                audio.filename or "audio.wav",
                reference_contents,
                reference.filename or "reference.srt",
                scenario=scenario,
                profile=profile,
                feedback_profile=feedback_profile,
            )
        except PipelineSubmissionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    audio_dir = state.upload_dir / f"feedback_{uuid.uuid4().hex[:8]}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path: Path | None = None
    reference_path = audio_dir / (reference.filename or "reference.srt")
    try:
        try:
            if audio is not None:
                audio_path = audio_dir / (audio.filename or "audio.wav")
                audio_path.write_bytes(await audio.read())
            reference_path.write_bytes(await reference.read())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc
        return _service.learn(
            audio_path,
            reference_path,
            task_id=task_id,
            scenario=scenario,
            profile=profile,
            feedback_profile=feedback_profile,
            run_pipeline_first=run_pipeline_first,
            dry_run=dry_run,
        )
    except TaskBaselineError as exc:
        # 任务记录不存在（404）或结果/会话产物已清理（410）：提示改走音频上传
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feedback learn failed")
        return {"status": "error", "message": str(exc)}
    finally:
        shutil.rmtree(audio_dir, ignore_errors=True)


@router.post("/feedback/learn")
async def feedback_learn(
    audio: UploadFile | None = File(None),
    reference: UploadFile = File(...),
    task_id: str = Form(default=""),
    scenario: str = Form(default=""),
    profile: str = Form(default="default"),
    feedback_profile: str = Form(default="user_default"),
    run_pipeline_first: bool = Form(default=True),
    dry_run: bool = Form(default=False),
):
    return await _run_feedback(
        audio,
        reference,
        task_id=task_id,
        scenario=scenario,
        profile=profile,
        feedback_profile=feedback_profile,
        run_pipeline_first=run_pipeline_first,
        dry_run=dry_run,
    )


@router.post("/feedback/preview")
async def feedback_preview(
    audio: UploadFile = File(...),
    reference: UploadFile = File(...),
    profile: str = Form(default="default"),
):
    return await _run_feedback(
        audio,
        reference,
        profile=profile,
        feedback_profile="user_default",
        run_pipeline_first=True,
        dry_run=True,
    )


__all__ = ["feedback_learn", "feedback_preview", "router"]
