"""LLM provider/model discovery service used by the WebUI."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List

from .api_services import LLM_PROVIDERS


class LLMModelLookupError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _normalise_url(base_url: str) -> str:
    try:
        from llm_subtitle_optimizer.llm_client import normalize_base_url

        return normalize_base_url(base_url)
    except ImportError:
        return base_url if base_url.endswith("/v1") else base_url.rstrip("/") + "/v1"


def _models_from_payload(data: Any) -> List[Dict[str, str]]:
    items = data.get("data", []) if isinstance(data, dict) else data
    if isinstance(data, dict) and "data" not in data:
        items = data.get("models", data.get("result", data.get("items", [])))
    if not isinstance(items, list):
        return []
    models = []
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model", "")
            owner = item.get("owned_by", item.get("provider", ""))
        else:
            model_id, owner = str(item), ""
        if model_id:
            models.append({"id": model_id, "owned_by": owner})
    return models


class LLMModelService:
    def list_providers(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": provider_id,
                "name": info["name"],
                "base_url": info["base_url"],
                "default_model": info["default_model"],
                "default_models": info.get("default_models", []),
            }
            for provider_id, info in LLM_PROVIDERS.items()
        ]

    def fetch_models(self, body: Dict[str, Any]) -> Dict[str, Any]:
        base_url = body.get("base_url", "").strip()
        api_key = body.get("api_key", "").strip()
        if not base_url:
            raise LLMModelLookupError(400, "base_url is required")
        base_url = _normalise_url(base_url)

        if api_key:
            try:
                request = urllib.request.Request(
                    f"{base_url}/models",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    models = _models_from_payload(json.loads(response.read().decode()))
                if models:
                    models.sort(key=lambda model: (
                        not any(keyword in model["id"].lower() for keyword in ("chat", "gpt", "claude", "deepseek", "qwen", "glm", "llama", "command")),
                        model["id"],
                    ))
                    return {"models": models, "total": len(models), "source": "api"}
            except Exception:
                pass

        clean_url = base_url.rstrip("/").removesuffix("/v1").rstrip("/")
        for provider in LLM_PROVIDERS.values():
            if provider["base_url"].rstrip("/") == clean_url and provider.get("default_models"):
                models = [{"id": model, "owned_by": provider["name"]} for model in provider["default_models"]]
                return {"models": models, "total": len(models), "source": "preset"}

        try:
            request = urllib.request.Request(f"{base_url}/models", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                models = _models_from_payload(json.loads(response.read().decode()))
            if models:
                models.sort(key=lambda model: model["id"])
                return {"models": models, "total": len(models), "source": "api"}
        except Exception:
            pass

        if not api_key:
            raise LLMModelLookupError(400, "未提供 API 密钥且 API 地址需要认证，请先输入密钥后重试")
        raise LLMModelLookupError(502, "获取模型列表失败：API 不可达或密钥无效。将使用预设默认模型。")


__all__ = ["LLMModelLookupError", "LLMModelService"]
