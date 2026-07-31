"""Configuration, device and model-management API routes."""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Form, HTTPException, Query

from ..config import ConfigLoader
from ..utils.gpu_detector import GPUDetector
from . import api_services as services
from .models import *

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/profiles", response_model=List[ProfileInfo])
async def list_profiles():
    """获取所有可用场景模板"""
    loader = ConfigLoader()
    profiles = []
    for name in loader.list_profiles():
        try:
            config = loader.load_profile(name)
            profiles.append(
                ProfileInfo(
                    name=name,
                    description=services.profile_description(name),
                    config_summary=services.config_summary(config),
                )
            )
        except Exception as e:
            logger.warning("Failed to load profile '%s': %s", name, e)
    return profiles


@router.get("/profiles/{name}")
async def get_profile_config(name: str):
    """获取指定场景模板的完整配置"""
    loader = ConfigLoader()
    try:
        config = loader.load_profile(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

    return {
        "name": name,
        "description": services.profile_description(name),
        "config": services.config_to_overrides(config),
    }


# ---------------------------------------------------------------------------
# 设备信息
# ---------------------------------------------------------------------------


@router.get("/device", response_model=DeviceInfoResponse)
async def get_device_info():
    """获取系统/GPU 设备信息"""
    info = GPUDetector.get_device_info()
    device_type = GPUDetector.get_best_device()
    return DeviceInfoResponse(
        device_type=info["device_type"],
        device_count=info["device_count"],
        device_names=info["device_names"],
        memory_mb=info["memory_mb"],
        recommended_compute_type=info["recommended_compute_type"],
        gpu_memory_used_mb=GPUDetector.get_gpu_memory_used_mb(),
        recommended_model=GPUDetector.select_whisper_model(device_type),
    )


@router.get("/speaker-embedding/license")
async def get_speaker_embedding_license():
    """获取说话人嵌入模型的协议信息

    前端展示 pyannote 模型的协议要求，
    用户需在 huggingface.co 上接受协议后才能使用。
    """
    try:
        from vocal_subtitle.diarization.speaker_embedding import (
            PyannoteEmbeddingEngine,
        )

        info = PyannoteEmbeddingEngine.license_info()
        return {
            "engine": info["engine"],
            "code_license": info["code_license"],
            "model_license": info["model_license"],
            "license_url": info["license_url"],
            "preset_models": [
                {
                    "ref": k,
                    "name": v["name"],
                    "description": v["description"],
                    "embedding_dim": v["embedding_dim"],
                    "license_url": v["license_url"],
                    "model_license_type": v["model_license_type"],
                    "size_mb": v["size_mb"],
                    "requires_token": v["requires_token"],
                }
                for k, v in info["preset_models"].items()
            ],
        }
    except ImportError:
        return {
            "engine": "unavailable",
            "note": "pyannote.audio 未安装。安装: pip install pyannote.audio",
        }


@router.get("/speaker-models")
async def list_speaker_models():
    """列出 embedding/global speaker 模型及本地缓存状态。"""
    from ..diarization.model_registry import list_model_status

    return {"models": list_model_status()}


@router.get("/speaker-models/{model_id}/status")
async def get_speaker_model_status(model_id: str):
    """查询单个 speaker 模型状态。"""
    from ..diarization.model_registry import model_status

    try:
        return model_status(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/speaker-models/{model_id}/download")
async def download_speaker_model(model_id: str, token: str = Form(default="")):
    """下载 speaker 模型，并将用户输入的 Token 加密保存到本机。"""
    from ..diarization.model_registry import download_model

    submitted_token = (token or "").strip()
    if submitted_token == "***":
        submitted_token = ""
    if submitted_token:
        try:
            from ..utils.hf_token_store import store_hf_token

            store_hf_token(submitted_token)
        except ImportError:
            logger.warning("HF Token storage is unavailable; using token for this request only")
        except (OSError, ValueError) as exc:
            logger.warning("Could not persist HF Token securely: %s", type(exc).__name__)

    try:
        result = await asyncio.to_thread(download_model, model_id, token=submitted_token or None)
        result.pop("cache_dir", None)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Speaker model download failed for %s: %s", model_id, exc)
        detail = services.speaker_model_download_detail(exc)
        status_code = 503 if isinstance(exc, ImportError) else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc


@router.get("/asr/funasr/status")
async def get_funasr_status(model: str = Query(default="")):
    """Check FunASR package/model readiness without network access."""
    return await asyncio.to_thread(services.funasr_status, model)


@router.post("/asr/funasr/prepare")
async def prepare_funasr(body: FunASRPrepareRequest):
    """Install FunASR if needed and download only a missing local model."""
    try:
        return await asyncio.to_thread(services.ensure_funasr_ready, body.model)
    except FunASRPrepareError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected FunASR preparation failure")
        raise HTTPException(
            status_code=500,
            detail="FunASR 准备失败，请查看服务日志",
        ) from exc


# ---------------------------------------------------------------------------
# Pipeline 执行
# ---------------------------------------------------------------------------
