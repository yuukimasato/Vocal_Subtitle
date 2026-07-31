"""HTTP adapters for LLM provider and model discovery."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .llm_services import LLMModelLookupError, LLMModelService

router = APIRouter()
_service = LLMModelService()


@router.get("/llm/providers")
async def list_llm_providers():
    return _service.list_providers()


@router.post("/llm/models")
async def fetch_llm_models(body: dict):
    try:
        return _service.fetch_models(body)
    except LLMModelLookupError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


__all__ = ["fetch_llm_models", "list_llm_providers", "router"]
