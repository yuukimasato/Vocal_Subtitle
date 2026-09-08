"""工具层模块

提供音频处理、缓存管理、GPU 检测、进度管理、日志、模型加载、持久化和会话管理。
"""

from .audio_utils import AudioUtils
from .cache_manager import CacheManager
from .gpu_detector import GPUDetector
from .logger import setup_logging
from .progress import ProgressManager
from .persistence_manager import PersistenceManager
from .session_manager import SessionManager, create_task_id, create_run_id, create_config_snapshot
from .task_history import ErrorCategory, PreflightChecklist, TaskHistoryManager
from .model_loader import is_model_cached, load_sentence_transformer
from .model_identity import ModelIdentity, parse_model_id
from .file_hasher import compute_config_hash, compute_file_hash

__all__ = [
    "AudioUtils",
    "CacheManager",
    "GPUDetector",
    "ProgressManager",
    "setup_logging",
    "PersistenceManager",
    "SessionManager",
    "create_task_id",
    "create_run_id",
    "create_config_snapshot",
    "ErrorCategory",
    "PreflightChecklist",
    "TaskHistoryManager",
    "is_model_cached",
    "load_sentence_transformer",
    "ModelIdentity",
    "parse_model_id",
    "compute_config_hash",
    "compute_file_hash",
]
