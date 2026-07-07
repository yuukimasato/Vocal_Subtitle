"""说话人嵌入引擎 — 从音频片段中提取说话人特征向量

提供可替换的引擎架构，默认实现为 pyannote.audio。
用户需先接受 Hugging Face 模型协议，输入模型路径后自动下载。

引擎接口:
- SpeakerEmbeddingEngine: 抽象基类
- PyannoteEmbeddingEngine: pyannote.audio ECAPA-TDNN 实现 (~100MB)
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 模型缓存目录
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "speaker_models"


class SpeakerEmbeddingEngine(ABC):
    """说话人嵌入引擎抽象基类

    从音频片段中提取固定维度的说话人特征向量（embedding），
    用于后续的说话人聚类。

    使用示例:
        engine = PyannoteEmbeddingEngine()
        engine.load_model("pyannote/embedding", token="hf_xxx")
        emb = engine.extract_embedding(audio_snippet, sample_rate=16000)
        # emb.shape → (512,)
    """

    @abstractmethod
    def load_model(
        self,
        model_ref: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs,
    ) -> None:
        """加载说话人嵌入模型

        Args:
            model_ref: 模型引用（HuggingFace repo ID 或本地路径）
            token: HuggingFace API token（用于下载需授权模型）
        """
        ...

    @abstractmethod
    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """提取单个音频片段的说话人嵌入向量

        Args:
            audio: 单声道音频 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            1D float64 嵌入向量
        """
        ...

    @abstractmethod
    def extract_embeddings_batch(
        self, audios: List[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        """批量提取嵌入向量

        Args:
            audios: 音频片段列表
            sample_rate: 采样率

        Returns:
            (n_audios × embedding_dim) float64 矩阵
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """嵌入向量维度"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        ...

    @property
    @abstractmethod
    def model_loaded(self) -> bool:
        """模型是否已加载"""
        ...

    @staticmethod
    @abstractmethod
    def license_info() -> dict:
        """返回协议信息，供前端展示"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class PyannoteEmbeddingEngine(SpeakerEmbeddingEngine):
    """pyannote.audio 说话人嵌入引擎

    使用 pyannote.audio 的 ECAPA-TDNN 模型（speaker-diarization-3.1 的嵌入部分）
    或独立的 pyannote/embedding 模型。

    协议:
    - 代码: MIT
    - 模型: pyannote.audio models 需接受 Hugging Face 协议
      (https://huggingface.co/pyannote/embedding)
    - 用户需在 huggingface.co 上签署模型使用协议，
      然后用 HF token 下载模型

    使用示例:
        engine = PyannoteEmbeddingEngine()
        engine.load_model(
            model_ref="pyannote/embedding",
            token="hf_your_token_here",
        )
        emb = engine.extract_embedding(audio, sr)
    """

    # ------------------------------------------------------------------
    # 预置模型信息
    # ------------------------------------------------------------------

    PRESET_MODELS = {
        "pyannote/embedding": {
            "name": "pyannote/embedding",
            "description": "ECAPA-TDNN speaker embedding (pyannote.audio 3.1)",
            "embedding_dim": 512,
            "hf_url": "https://huggingface.co/pyannote/embedding",
            "license_url": "https://huggingface.co/pyannote/embedding",
            "license_type": "mit",  # 代码协议
            "model_license_type": "pyannote-eula",  # 模型协议
            "size_mb": 100,
            "requires_token": True,
        },
        "speechbrain/spkrec-ecapa-voxceleb": {
            "name": "speechbrain/spkrec-ecapa-voxceleb",
            "description": "ECAPA-TDNN on VoxCeleb (SpeechBrain, Apache 2.0)",
            "embedding_dim": 192,
            "hf_url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
            "license_url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
            "license_type": "apache-2.0",
            "model_license_type": "apache-2.0",
            "size_mb": 80,
            "requires_token": False,
        },
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None or (isinstance(cache_dir, Path) and str(cache_dir) in ("", ".")):
            self._cache_dir = DEFAULT_CACHE_DIR
        else:
            self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None          # speechbrain EncoderClassifier 或通用
        self._inference = None      # pyannote.audio Inference
        self._model_loaded = False
        self._model_ref: Optional[str] = None
        self._model_type: str = ""  # "pyannote" | "speechbrain"
        self._embedding_dim: int = 512

    # ------------------------------------------------------------------
    # SpeakerEmbeddingEngine 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._model_type or "pyannote"

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @staticmethod
    def license_info() -> dict:
        return {
            "engine": "pyannote.audio",
            "code_license": "MIT",
            "model_license": "用户需在 huggingface.co 接受模型使用协议",
            "license_url": "https://huggingface.co/pyannote/embedding",
            "preset_models": PyannoteEmbeddingEngine.PRESET_MODELS,
        }

    def load_model(
        self,
        model_ref: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs,
    ) -> None:
        """加载说话人嵌入模型

        自动检测模型类型并使用对应的加载方式：
        - pyannote/embedding → pyannote.audio.Inference (需 token)
        - speechbrain/*      → speechbrain.pretrained.EncoderClassifier
        """
        model_ref = model_ref or "speechbrain/spkrec-ecapa-voxceleb"
        self._model_ref = model_ref

        preset = self.PRESET_MODELS.get(model_ref, {})
        self._embedding_dim = preset.get("embedding_dim", 512)

        logger.info("Loading speaker embedding model: %s", model_ref)

        # 解析 token
        use_auth_token = token or kwargs.get("use_auth_token")
        if use_auth_token is None:
            import os
            use_auth_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        import torch
        _torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # speechbrain @ PyTorch 2.x 兼容：run_opts["device"] 需要字符串
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if model_ref.startswith("speechbrain/"):
            # ---- SpeechBrain 模型 ----
            # 预注册 dummy k2 模块，防止其懒加载失败阻塞整个 speechbrain import
            import sys as _sys
            _k2_mod = type(_sys)("speechbrain.integrations.k2_fsa")
            _k2_mod.__file__ = "k2_not_installed"
            _k2_mod.__package__ = "speechbrain.integrations.k2_fsa"
            _sys.modules["speechbrain.integrations.k2_fsa"] = _k2_mod

            try:
                from speechbrain.pretrained import EncoderClassifier
            except ImportError:
                raise ImportError(
                    "speechbrain is required for speechbrain models. "
                    "Install with: pip install speechbrain"
                )

            logger.info("SpeechBrain embedding using device: %s", device_str)
            try:
                _savedir = str((self._cache_dir / model_ref.replace("/", "_")).resolve())

                # Patch fetch 确保所有文件从本地 savedir 加载
                import speechbrain.utils.fetching as _sb_fetch
                _orig_fetch = _sb_fetch.fetch
                _patched_modules = []
                import sys

                def _offline_fetch(filename, source, savedir=None, save_filename=None, **kw):
                    import pathlib as _pl
                    fname = save_filename or filename
                    if savedir is not None:
                        local = _pl.Path(savedir) / fname
                        if local.exists() and local.stat().st_size > 200:
                            return local
                    # 从 _savedir 检查（捕获多模块调用时 savedir 不一致的情况）
                    local2 = _pl.Path(_savedir) / fname
                    if local2.exists() and local2.stat().st_size > 200:
                        return local2
                    if fname.endswith(".ckpt"):
                        txt_name = fname.replace(".ckpt", ".txt")
                        for _sd in (savedir, _savedir):
                            if _sd is not None:
                                local_txt = _pl.Path(_sd) / txt_name
                                if local_txt.exists() and local_txt.stat().st_size > 200:
                                    return local_txt
                    return _orig_fetch(filename, source, savedir=savedir, save_filename=save_filename, **kw)

                _sb_fetch.fetch = _offline_fetch
                for _mn, _m in list(sys.modules.items()):
                    if _mn.startswith("speechbrain."):
                        try:
                            if hasattr(_m, "fetch"):
                                _patched_modules.append((_mn, _m.fetch))
                                _m.fetch = _offline_fetch
                        except Exception:
                            pass

                try:
                    # 传入本地 hyperparams.yaml 的绝对路径
                    self._model = EncoderClassifier.from_hparams(
                        source=model_ref,
                        hparams_file=f"{_savedir}/hyperparams.yaml",
                        savedir=_savedir,
                        run_opts={"device": device_str},
                    )
                finally:
                    _sb_fetch.fetch = _orig_fetch
                    for _mn, _prev_fetch in _patched_modules:
                        _m = sys.modules.get(_mn)
                        if _m is not None:
                            try:
                                _m.fetch = _prev_fetch
                            except Exception:
                                pass
                self._model_loaded = True
                self._model_type = "speechbrain"
                # speechbrain ECAPA 实际嵌入维度
                if hasattr(self._model, 'hparams'):
                    self._embedding_dim = getattr(
                        self._model.hparams, 'emb_dim', self._embedding_dim,
                    )
                logger.info(
                    "SpeechBrain model loaded: %s (dim=%d)",
                    model_ref, self._embedding_dim,
                )
            except Exception as e:
                logger.error(
                    "Failed to load speechbrain model '%s': %s",
                    model_ref, e,
                )
                raise

        else:
            # ---- pyannote.audio 模型 ----
            try:
                from pyannote.audio import Inference
            except ImportError:
                raise ImportError(
                    "pyannote.audio is required for pyannote models. "
                    "Install with: pip install pyannote.audio"
                )

            logger.info("Pyannote embedding using device: %s", device)
            try:
                self._inference = Inference(
                    model=model_ref,
                    device=device,
                    use_auth_token=use_auth_token,
                )
                self._model_loaded = True
                self._model_type = "pyannote"
                logger.info(
                    "Pyannote model loaded: %s (dim=%d)",
                    model_ref, self._embedding_dim,
                )
            except Exception as e:
                logger.error(
                    "Failed to load pyannote model '%s': %s\n"
                    "Make sure you have:\n"
                    "  1. Accepted the license at https://huggingface.co/%s\n"
                    "  2. Set HF_TOKEN or passed token parameter\n"
                    "  3. Installed pyannote.audio: pip install pyannote.audio",
                    model_ref, e, model_ref,
                )
                raise

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """提取单个音频片段的嵌入向量"""
        self._ensure_loaded()
        return self._extract_single(audio, sample_rate)

    def extract_embeddings_batch(
        self, audios: List[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        """批量提取嵌入向量"""
        self._ensure_loaded()
        embeddings = []
        for audio in audios:
            emb = self._extract_single(audio, sample_rate)
            embeddings.append(emb)
        return np.vstack(embeddings)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._model_loaded:
            raise RuntimeError(
                "Speaker embedding model not loaded. Call load_model() first."
            )

    def _extract_single(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """提取单个音频的嵌入（自动适配 pyannote / speechbrain）"""
        import torch

        # 预处理
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 最小长度保护
        min_samples = int(0.6 * sample_rate)
        if len(audio) < min_samples:
            padded = np.zeros(min_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            audio = padded

        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

        try:
            with torch.no_grad():
                if self._model_type == "speechbrain":
                    # speechbrain: EncoderClassifier.encode_batch
                    # 需要相对长度归一化
                    emb = self._model.encode_batch(waveform)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    embedding = np.squeeze(emb)
                else:
                    # pyannote.audio: Inference
                    embedding = self._inference(waveform)
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    embedding = np.squeeze(embedding)
        except Exception as e:
            logger.warning("Embedding extraction failed: %s, returning zeros", e)
            embedding = np.zeros(self._embedding_dim, dtype=np.float64)

        result = np.array(embedding, dtype=np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result


class DummyEmbeddingEngine(SpeakerEmbeddingEngine):
    """空嵌入引擎 — 用户未配置 pyannote 模型时的占位实现

    返回零向量，确保 pipeline 不会因缺少嵌入引擎而崩溃。
    实际聚类会降级到间隙交替方案。
    """

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def model_loaded(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return 8

    @staticmethod
    def license_info() -> dict:
        return {"engine": "none", "note": "请配置 pyannote 模型以获得最佳说话人分离效果"}

    def load_model(self, **kwargs) -> None:
        pass

    def extract_embedding(self, audio, sample_rate):
        return np.zeros(8, dtype=np.float64)

    def extract_embeddings_batch(self, audios, sample_rate):
        return np.zeros((len(audios), 8), dtype=np.float64)


def create_embedding_engine(config) -> SpeakerEmbeddingEngine:
    """根据配置创建说话人嵌入引擎

    Args:
        config: SpeakerEmbeddingConfig 实例

    Returns:
        SpeakerEmbeddingEngine 子类实例
    """
    if not config.enabled:
        return DummyEmbeddingEngine()

    engine_name = config.engine
    model_ref = config.model_ref
    token = config.hf_token

    if engine_name == "pyannote":
        engine = PyannoteEmbeddingEngine(cache_dir=Path(config.cache_dir))
        try:
            # 缩短 HuggingFace Hub 下载超时，避免离线环境下长时间阻塞
            import os as _os
            _prev_timeout = _os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
            _prev_requests_timeout = _os.environ.get("REQUESTS_TIMEOUT_SECONDS")
            _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3"
            _os.environ["REQUESTS_TIMEOUT_SECONDS"] = "3"
            try:
                token_to_use = token if token else None
                engine.load_model(model_ref=model_ref, token=token_to_use)
            finally:
                # 恢复原环境变量
                if _prev_timeout is not None:
                    _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = _prev_timeout
                else:
                    _os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
                if _prev_requests_timeout is not None:
                    _os.environ["REQUESTS_TIMEOUT_SECONDS"] = _prev_requests_timeout
                else:
                    _os.environ.pop("REQUESTS_TIMEOUT_SECONDS", None)
            return engine
        except ImportError as e:
            # 缺少依赖包
            pkg = "speechbrain" if model_ref.startswith("speechbrain/") else "pyannote.audio"
            logger.warning(
                "Speaker embedding package '%s' not installed. "
                "Install with: pip install %s. "
                "Falling back to gap-based alternation.",
                pkg, pkg,
            )
            return DummyEmbeddingEngine()
        except Exception as e:
            logger.warning(
                "Speaker embedding engine failed to load '%s': %s. "
                "Falling back to gap-based alternation.",
                model_ref, e,
            )
            return DummyEmbeddingEngine()

    logger.warning("Unknown embedding engine: %s, using dummy", engine_name)
    return DummyEmbeddingEngine()
