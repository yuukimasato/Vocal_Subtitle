"""UVR (Ultimate Vocal Remover) 人声分离引擎

使用 audio-separator 库调用 UVR 模型进行高质量人声分离。
代码协议: MIT | 模型权重协议: MIT (UVR 自训模型)

要求: pip install audio-separator
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from .base import LicenseInfo, SeparationEngine, SeparationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# tqdm 进度钩子 — 将 audio-separator 内部的 tqdm 迭代进度导向外部队列
# ---------------------------------------------------------------------------

_ORIGINAL_TQDM_INIT = None
"""保存 tqdm.tqdm.__init__ 的原始引用，用于卸载钩子"""

_progress_callback_ref: Optional[Callable] = None
"""当前活跃的进度回调 (current: int, total: int) -> None"""


def _install_tqdm_hook(callback: Callable) -> None:
    """安装全局 tqdm 进度钩子

    替换 tqdm.tqdm.__init__，使得后续所有新创建的 tqdm 实例
    在 update() 时自动调用 callback(current, total)。

    通过修改类方法（而非替换类引用）确保无论调用方使用
    ``from tqdm import tqdm`` 还是 ``import tqdm`` 都能生效。
    """
    global _ORIGINAL_TQDM_INIT, _progress_callback_ref
    import tqdm

    if _ORIGINAL_TQDM_INIT is not None:
        return  # 避免重复安装

    _ORIGINAL_TQDM_INIT = tqdm.tqdm.__init__
    _progress_callback_ref = callback

    def _patched_init(self, *args, **kwargs):
        _ORIGINAL_TQDM_INIT(self, *args, **kwargs)
        # 保存原始 update 并用带回调的版本替换
        original_update = self.update
        cb = _progress_callback_ref  # 闭包捕获当前回调

        def _hooked_update(n: int = 1):
            original_update(n)
            if cb and self.total and self.n > 0:
                cb(self.n, self.total)

        self.update = _hooked_update

    tqdm.tqdm.__init__ = _patched_init


def _uninstall_tqdm_hook() -> None:
    """卸载全局 tqdm 进度钩子，恢复原始 __init__"""
    global _ORIGINAL_TQDM_INIT, _progress_callback_ref
    import tqdm

    if _ORIGINAL_TQDM_INIT is not None:
        tqdm.tqdm.__init__ = _ORIGINAL_TQDM_INIT
        _ORIGINAL_TQDM_INIT = None
    _progress_callback_ref = None


class UVREngine(SeparationEngine):
    """UVR 分离引擎

    基于 audio-separator 的高品质人声分离。
    品质: ★★★★★ | 速度: ★★★★☆ | 适用: 品质天花板 + 商用安全

    使用示例:
        engine = UVREngine()
        engine.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        result = engine.separate(Path("input.mp3"), Path("output/"))
    """

    DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None

    @property
    def name(self) -> str:
        return "uvr"

    @property
    def license_info(self) -> LicenseInfo:
        return LicenseInfo(
            code_license="MIT",
            model_license="MIT",
            source_url="https://github.com/nomadkaraoke/python-audio-separator",
        )

    # 模型缓存目录（持久化在项目内，避免 /tmp 重启丢失）
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    MODEL_DIR = os.path.join(_PROJECT_ROOT, "cache", "models") + os.sep
    OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "cache", "models", "tmp") + os.sep
    # 本地捆绑的 download_checks.json（避免每次启动都从 GitHub 拉取）
    _BUNDLED_CHECKSUM = Path(__file__).parent / "data" / "download_checks.json"

    def load_model(self, model_name: Optional[str] = None) -> None:
        """加载 UVR 模型

        遵循 ML 模型加载模式：优先使用本地缓存避免联网。
        仅在模型文件不存在时才尝试从 GitHub 下载（此时会临时清除
        SOCKS 代理以规避 httpx/requests 对 socks:// scheme 的不兼容）。

        Args:
            model_name: 模型文件名，默认使用 BS-Roformer 模型
        """
        model_name = model_name or self.DEFAULT_MODEL

        if self._model is not None and self._model_name == model_name:
            return

        logger.info("Loading UVR model: %s", model_name)

        try:
            from audio_separator.separator import Separator
        except ImportError:
            raise ImportError(
                "audio-separator is required. Install it with: "
                "pip install audio-separator"
            )

        # -----------------------------------------------------------------
        # 确保本地存在有效的 download_checks.json（离线优先）
        # audio-separator 每次 load_model 都会调用 list_supported_model_files，
        # 如果 download_checks.json 不存在就会尝试从 GitHub 下载。
        # 我们预先放置一份捆绑的 JSON，完全跳过这次网络请求。
        # -----------------------------------------------------------------
        checks_path = os.path.join(self.MODEL_DIR, "download_checks.json")
        self._ensure_valid_checks_file(checks_path)

        self._model = Separator(
            model_file_dir=self.MODEL_DIR,
            output_dir=self.OUTPUT_DIR,
        )

        self._model.load_model(model_filename=model_name)
        self._model_name = model_name

    @staticmethod
    def _ensure_valid_checks_file(checks_path: str) -> None:
        """确保 download_checks.json 有效，否则从捆绑副本恢复

        audio-separator 只检查 os.path.isfile()，不验证 JSON 完整性。
        前次网络中断留下的截断文件会同时（a）阻止重新下载，
        （b）导致 JSON 解析崩溃。这里做两层防护：
        1. 文件不存在 → 从捆绑副本复制（零网络）
        2. 文件存在但 JSON 损坏 → 删除后从捆绑副本恢复
        """
        os.makedirs(os.path.dirname(checks_path), exist_ok=True)

        need_copy = False
        if os.path.isfile(checks_path):
            try:
                with open(checks_path, encoding="utf-8") as f:
                    json.load(f)
                # 文件有效，无需操作
                return
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(
                    "Removing corrupted download_checks.json, "
                    "restoring from bundled copy"
                )
                os.remove(checks_path)
                need_copy = True
        else:
            need_copy = True

        if need_copy:
            bundled = UVREngine._BUNDLED_CHECKSUM
            if bundled.is_file():
                import shutil
                shutil.copy2(str(bundled), checks_path)
                logger.info(
                    "Placed bundled download_checks.json → %s", checks_path
                )
            else:
                logger.warning(
                    "Bundled download_checks.json not found at %s; "
                    "will trigger network download", bundled
                )

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> SeparationResult:
        """执行人声分离

        Args:
            input_path: 输入音频文件路径
            output_dir: 输出目录
            progress_callback: 进度回调 (current: int, total: int) -> None，
                              用于将 audio-separator 内部 tqdm 进度透传出去

        Returns:
            SeparationResult
        """
        if self._model is None:
            self.load_model()

        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        logger.info("Separating: %s → %s", input_path, output_dir)

        # 预定义输出路径
        vocals_path = output_dir / "vocals.wav"
        accompaniment_path = output_dir / "accompaniment.wav"

        # 安装 tqdm 进度钩子（将 audio-separator 内部迭代进度导向外部回调）
        if progress_callback is not None:
            _install_tqdm_hook(progress_callback)

        try:
            # audio-separator 的输出文件名由库内部生成
            output_files = self._model.separate(str(input_path))

            # audio-separator 可能返回相对路径，需要拼接 output_dir
            sep_output_dir = Path(getattr(self._model, "output_dir", "/tmp"))

            def _resolve_path(p: str) -> Path:
                f_path = Path(p)
                if not f_path.is_absolute():
                    f_path = sep_output_dir / f_path
                return f_path

            if isinstance(output_files, list) and len(output_files) >= 2:
                # 将输出文件复制到指定目录
                import shutil

                vocals_src = None
                accomp_src = None

                for f in output_files:
                    f_path = _resolve_path(f)
                    f_lower = f_path.name.lower()
                    if "vocals" in f_lower:
                        vocals_src = f_path
                    elif "instrumental" in f_lower:
                        accomp_src = f_path

                if vocals_src and vocals_src.exists():
                    shutil.copy2(vocals_src, vocals_path)
                if accomp_src and accomp_src.exists():
                    shutil.copy2(accomp_src, accompaniment_path)
            else:
                # 单个输出文件（可能是人声），复制到 vocals_path
                # 处理空列表等异常情况
                import shutil

                if isinstance(output_files, list):
                    if len(output_files) == 0:
                        raise RuntimeError(
                            "UVR separation produced no output files — "
                            "the input file may have been deleted or the "
                            "separation process encountered an internal error"
                        )
                    src = output_files[0]
                else:
                    src = output_files
                src = _resolve_path(str(src)) if src else None
                if src and src.exists():
                    shutil.copy2(src, vocals_path)
                else:
                    logger.warning(
                        "UVR output file not found at %s", src
                    )

        except Exception as e:
            logger.error("UVR separation failed: %s", e)
            raise
        finally:
            # 确保钩子在分离完成后卸载（无论成功或失败）
            if progress_callback is not None:
                _uninstall_tqdm_hook()

        elapsed = time.time() - start_time

        return SeparationResult(
            vocals_path=vocals_path,
            accompaniment_path=accompaniment_path,
            engine_name=self.name,
            processing_time=elapsed,
        )
