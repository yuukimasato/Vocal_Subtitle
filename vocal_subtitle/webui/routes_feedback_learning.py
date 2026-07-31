"""HTTP adapters for feedback learning and preview."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .feedback_services import FeedbackLearningService
from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()
_service = FeedbackLearningService()


async def _run_feedback(
    audio: UploadFile,
    reference: UploadFile,
    *,
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
    audio_dir = state.upload_dir / f"feedback_{uuid.uuid4().hex[:8]}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / (audio.filename or "audio.wav")
    reference_path = audio_dir / (reference.filename or "reference.srt")
    try:
        try:
            audio_path.write_bytes(await audio.read())
            reference_path.write_bytes(await reference.read())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc
        return _service.learn(
            audio_path,
            reference_path,
            profile=profile,
            feedback_profile=feedback_profile,
            run_pipeline_first=run_pipeline_first,
            dry_run=dry_run,
        )
    except Exception as exc:
        logger.exception("Feedback learn failed")
        return {"status": "error", "message": str(exc)}
    finally:
        shutil.rmtree(audio_dir, ignore_errors=True)


@router.post("/feedback/learn")
async def feedback_learn(
    audio: UploadFile = File(...),
    reference: UploadFile = File(...),
    profile: str = Form(default="default"),
    feedback_profile: str = Form(default="user_default"),
    run_pipeline_first: bool = Form(default=True),
    dry_run: bool = Form(default=False),
):
    return await _run_feedback(
        audio,
        reference,
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
