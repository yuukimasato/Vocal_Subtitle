"""离线优先的模型加载器

统一管理 sentence-transformers 模型的加载策略，确保：
1. 优先检测本地缓存（零网络，即时返回）
2. 本地缺失时默认返回 None（规则逻辑优雅降级，不阻塞 Pipeline）
3. 仅在显式允许时尝试网络下载，并回退到 HF 镜像站

策略优先级:
  第一路径 — local_files_only=True（已缓存 → 即时返回）
  第二路径 — 离线降级（返回 None，不影响 Pipeline 主流程）
  第三路径 — 显式允许下载时的网络下载（10s 超时）
  第四路径 — HF 镜像站（国内网络优化）

关键设计:
  - 模块级 setdefault 设置 HF_HUB_OFFLINE=1，优先影响后续导入
  - 网络下载路径直接操作 huggingface_hub.constants.HF_HUB_OFFLINE
    （因为该库在首次导入时将 os.environ 缓存为模块级常量，
    后续修改 os.environ 不会生效）
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 模块级别：确保默认离线
# ------------------------------------------------------------------
# setdefault 只在未设置时生效，用户显式设置优先
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 模型名称常量（使用完整的 org/model 路径，确保缓存目录匹配）
MINILM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ALLOW_DOWNLOAD_ENV = "VOCAL_SUBTITLE_ALLOW_MODEL_DOWNLOAD"

# 缓存标记：记录已确认存在于本地的模型，避免重复尝试
_cached_models: set = set()


def is_model_cached(model_name: str) -> bool:
    """检查模型是否在本地缓存中（不触发网络请求）

    通过检查 HuggingFace 缓存目录中是否存在对应的 snapshot。
    支持短名称（不含 org 前缀）和完整名称。

    缓存目录命名规则：models--{org}--{model_name}
    """
    if model_name in _cached_models:
        return True

    cache_dir = os.path.expanduser(
        os.environ.get("HF_HOME", "~/.cache/huggingface/hub")
    )

    # 尝试的模型名称列表：完整名 + 短名（自动补充 sentence-transformers 前缀）
    candidates = [model_name]
    if "/" not in model_name:
        # 短名称可能对应 sentence-transformers 或 transformers 等 org
        # 逐个检查常见 org 前缀
        for prefix in ["sentence-transformers", "transformers", "huggingface"]:
            candidates.append(f"{prefix}/{model_name}")

    for name in candidates:
        model_dir = "models--" + name.replace("/", "--")
        model_path = os.path.join(cache_dir, model_dir, "snapshots")

        if os.path.isdir(model_path):
            try:
                for entry in os.listdir(model_path):
                    snap_path = os.path.join(model_path, entry)
                    if os.path.isdir(snap_path):
                        if any(
                            f.endswith(
                                (".safetensors", ".bin", ".pt", ".h5", ".msgpack")
                            )
                            or f == "config.json"
                            for f in os.listdir(snap_path)
                        ):
                            _cached_models.add(model_name)
                            return True
            except OSError:
                pass

    return False


def load_sentence_transformer(
    model_name: str = MINILM_MODEL_NAME,
    *,
    device: str = "cpu",
    allow_download: bool | None = None,
) -> Optional["SentenceTransformer"]:
    """离线优先加载 sentence-transformers 模型

    加载策略（按优先级）：
    1. 本地缓存 → local_files_only=True（零网络，即时返回）
    2. 默认离线降级 → 返回 None（调用方使用规则逻辑）
    3. 显式允许下载时 → 网络下载并回退到 HF 镜像站

    Args:
        model_name: HuggingFace 模型名称
        device: 推理设备 ("cpu" | "cuda")
        allow_download: 是否允许本地缓存缺失时访问网络。未传入时仅当
            ``VOCAL_SUBTITLE_ALLOW_MODEL_DOWNLOAD=1`` 才允许。

    Returns:
        SentenceTransformer 实例，加载失败时返回 None
    """
    # ── 第一路径：本地缓存（零网络，立即可用）──────────────────
    if is_model_cached(model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic NLP features will use rule-only fallback. "
                "Install with: pip install sentence-transformers"
            )
            return None
        try:
            model = _load_cached_model(
                SentenceTransformer,
                model_name,
                device,
            )
            logger.info("Model '%s' loaded from local cache", model_name)
            return model
        except Exception as e:
            logger.warning(
                "Failed to load cached model '%s': %s.",
                model_name,
                e,
            )

    if allow_download is None:
        allow_download = os.environ.get(ALLOW_DOWNLOAD_ENV, "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not allow_download:
        logger.info(
            "Model '%s' is not available locally; offline semantic fallback is "
            "active. Set %s=1 to allow download.",
            model_name,
            ALLOW_DOWNLOAD_ENV,
        )
        return None

    # ── 第三路径：显式允许时的限时网络下载 ────────────────────────
    try:
        return _download_model(model_name, device)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; cannot download semantic model"
        )
        return None


def _load_cached_model(
    sentence_transformer: type,
    model_name: str,
    device: str,
):
    """Load a cached model while isolating mutable Hugging Face offline state."""
    saved_env = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    saved_hf_offline = None
    saved_transformers_offline = None
    try:
        for key in saved_env:
            os.environ[key] = "1"
        try:
            import huggingface_hub.constants as hf_constants

            saved_hf_offline = hf_constants.HF_HUB_OFFLINE
            hf_constants.HF_HUB_OFFLINE = True
        except (ImportError, AttributeError):
            pass
        try:
            import transformers.utils.hub as transformers_hub

            saved_transformers_offline = transformers_hub._is_offline_mode
            transformers_hub._is_offline_mode = True
        except (ImportError, AttributeError):
            pass
        return sentence_transformer(
            model_name,
            local_files_only=True,
            device=device,
        )
    finally:
        if saved_hf_offline is not None:
            try:
                import huggingface_hub.constants as hf_constants

                hf_constants.HF_HUB_OFFLINE = saved_hf_offline
            except (ImportError, AttributeError):
                pass
        if saved_transformers_offline is not None:
            try:
                import transformers.utils.hub as transformers_hub

                transformers_hub._is_offline_mode = saved_transformers_offline
            except (ImportError, AttributeError):
                pass
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _download_model(
    model_name: str,
    device: str,
) -> Optional["SentenceTransformer"]:
    """尝试通过网络下载模型（含镜像站回退）

    需要操作 huggingface_hub 的模块常量来解除离线限制。
    """
    from sentence_transformers import SentenceTransformer

    # 保存 + 关闭离线模式常量
    # huggingface_hub 在首次导入时将 os.environ 缓存为模块级常量，
    # 后续修改 os.environ 不会生效 → 必须直接操作常量
    try:
        import huggingface_hub.constants as hf_constants

        _saved_hf_offline = hf_constants.HF_HUB_OFFLINE
        hf_constants.HF_HUB_OFFLINE = False
    except ImportError:
        _saved_hf_offline = None

    try:
        import transformers.utils.hub as tf_hub

        _saved_tf_offline = tf_hub._is_offline_mode
        tf_hub._is_offline_mode = False
    except ImportError:
        _saved_tf_offline = None

    def _restore_offline():
        """恢复离线模式常量"""
        if _saved_hf_offline is not None:
            try:
                import huggingface_hub.constants as hf_c

                hf_c.HF_HUB_OFFLINE = _saved_hf_offline
            except ImportError:
                pass
        if _saved_tf_offline is not None:
            try:
                import transformers.utils.hub as tf_h

                tf_h._is_offline_mode = _saved_tf_offline
            except ImportError:
                pass

    # 设置下载超时（10s，防止启动/测试时无限阻塞）
    saved_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "10"

    try:
        logger.info("Attempting to download model '%s'...", model_name)
        model = SentenceTransformer(model_name, device=device)
        _cached_models.add(model_name)
        logger.info("Model '%s' downloaded successfully", model_name)
        return model
    except Exception as e:
        logger.warning(
            "Network download failed for '%s': %s. Trying HF mirror...",
            model_name,
            e,
        )

        # ── 第三路径：HF 镜像站 ──────────────────────────────
        saved_endpoint = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            model = SentenceTransformer(model_name, device=device)
            _cached_models.add(model_name)
            logger.info("Model '%s' loaded via HF mirror", model_name)
            return model
        except Exception as e2:
            logger.warning(
                "All loading strategies failed for '%s': %s. "
                "Semantic NLP features will use rule-only fallback.",
                model_name,
                e2,
            )
            return None
        finally:
            os.environ.pop("HF_ENDPOINT", None)
            if saved_endpoint is not None:
                os.environ["HF_ENDPOINT"] = saved_endpoint
    finally:
        _restore_offline()
        if saved_timeout is None:
            os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
        else:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = saved_timeout
