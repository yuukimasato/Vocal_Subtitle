"""OpenAI 兼容协议 LLM 客户端。

支持所有 OpenAI 兼容 API（DeepSeek / OpenAI / 硅基流动 / Ollama 等）。
默认使用 DeepSeek，无需额外配置即可使用。

注意：openai 和 tenacity 为可选依赖，仅在调用 LLM 功能时才需要。
     基础安装 (--cpu) 不包含这些依赖，使用时请先安装:
         pip install -e ".[llm]"
"""

import os
from typing import Any, List, Optional

# 延迟导入可选依赖，仅在实际使用时才要求安装
_openai_module = None
_OpenAI = None
_tenacity_retry = None
_tenacity_retry_if_exc = None
_tenacity_stop = None
_tenacity_wait = None


def _ensure_llm_deps():
    """按需加载 LLM 依赖 (openai, tenacity)，未安装时给出明确提示。"""
    global _openai_module, _OpenAI, _tenacity_retry, _tenacity_retry_if_exc
    global _tenacity_stop, _tenacity_wait

    if _openai_module is not None:
        return  # 已加载

    try:
        import openai as _openai_mod
        from openai import OpenAI as _OpenAI_cls

        _openai_module = _openai_mod
        _OpenAI = _OpenAI_cls
    except ImportError:
        raise ImportError(
            "LLM 功能需要 openai 库，请安装: pip install -e '.[llm]'\n"
            "或直接: pip install openai"
        )

    try:
        from tenacity import (
            RetryCallState,
            retry as _retry,
            retry_if_exception_type as _retry_if_exc,
            stop_after_attempt as _stop,
            wait_random_exponential as _wait,
        )

        _tenacity_retry = _retry
        _tenacity_retry_if_exc = _retry_if_exc
        _tenacity_stop = _stop
        _tenacity_wait = _wait
    except ImportError:
        raise ImportError(
            "LLM 重试机制需要 tenacity 库，请安装: pip install -e '.[llm]'\n"
            "或直接: pip install tenacity"
        )

# ---- DeepSeek 默认配置 ----
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-pro"

# 兼容多种环境变量名
_API_KEY_ENV_NAMES = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
_BASE_URL_ENV_NAMES = ["DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"]


def _read_env(names: List[str]) -> str:
    """从多个环境变量名中读取第一个非空值"""
    for name in names:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def normalize_base_url(base_url: str) -> str:
    """规范化 API base URL，确保 /v1 后缀"""
    from urllib.parse import urlparse, urlunparse

    url = base_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if not path:
        path = "/v1"

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return normalized


def get_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """获取 LLM 客户端实例。

    优先级：参数 > 环境变量 > DeepSeek 默认值。
    Ollama 本地服务无需 API 密钥。

    Args:
        base_url: API 基础 URL（默认读取 DEEPSEEK_BASE_URL / OPENAI_BASE_URL，
                  都未设置则使用 https://api.deepseek.com/v1）
        api_key: API 密钥（默认读取 DEEPSEEK_API_KEY / OPENAI_API_KEY，
                 Ollama 本地服务可留空）

    Returns:
        OpenAI 兼容客户端实例
    """
    _ensure_llm_deps()
    base_url = base_url or _read_env(_BASE_URL_ENV_NAMES) or DEFAULT_BASE_URL
    api_key = api_key or _read_env(_API_KEY_ENV_NAMES) or "ollama"  # Ollama doesn't require a real key

    base_url = normalize_base_url(base_url)

    return _OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def _call_llm_api(
    client,
    messages: List[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """实际调用 LLM API（带速率限制重试）"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        **kwargs,
    )
    return response


def call_llm(
    messages: List[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 1,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client=None,
    **kwargs: Any,
) -> Any:
    """调用 LLM API（OpenAI 兼容协议）。

    Args:
        messages: 对话消息列表 [{"role": "system", "content": "..."}, ...]
        model: 模型名称（默认 "deepseek-v4-pro"）
        temperature: 温度参数（默认 1）
        base_url: API 基础 URL（可选，默认 DeepSeek）
        api_key: API 密钥（可选，默认读环境变量）
        client: 预初始化的 OpenAI 客户端（可选，优先级最高）
        **kwargs: 传递给 API 的其他参数

    Returns:
        API 响应对象

    Raises:
        ValueError: API 返回空响应
    """
    _ensure_llm_deps()

    # 运行时应用 tenacity 重试装饰器
    _retrying_call = _tenacity_retry(
        stop=_tenacity_stop(10),
        wait=_tenacity_wait(multiplier=1, min=5, max=60),
        retry=_tenacity_retry_if_exc(_openai_module.RateLimitError),
    )(_call_llm_api)

    if client is None:
        client = get_llm_client(base_url=base_url, api_key=api_key)

    response = _retrying_call(client, messages, model, temperature, **kwargs)

    if not (
        response
        and hasattr(response, "choices")
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
        and response.choices[0].message.content
    ):
        raise ValueError("Invalid API response: empty choices or content")

    return response
