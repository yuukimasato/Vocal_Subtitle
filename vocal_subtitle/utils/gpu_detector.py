"""GPU 检测与设备选择模块

自动检测可用的 GPU 设备，支持 CUDA / Apple Silicon / CPU 降级。
"""

import logging
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU = "cpu"


class GPUDetector:
    """GPU 检测器

    自动检测可用的计算设备并提供推荐配置。

    使用示例:
        detector = GPUDetector()
        device = detector.get_best_device()
        # 'cuda' | 'mps' | 'cpu'

        info = detector.get_device_info()
        # {'device_type': 'cuda', 'device_count': 1, 'memory_mb': 8192, ...}
    """

    @staticmethod
    def detect_cuda() -> Tuple[bool, Optional[str]]:
        """检测 CUDA GPU 可用性

        Returns:
            (available, reason_if_not)
        """
        try:
            import torch

            if torch.cuda.is_available():
                return True, None
            return False, "CUDA not available in PyTorch"
        except ImportError:
            return False, "PyTorch not installed"

    @staticmethod
    def detect_mps() -> Tuple[bool, Optional[str]]:
        """检测 Apple Silicon (MPS) 可用性

        Returns:
            (available, reason_if_not)
        """
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True, None
            return False, "MPS not available"
        except ImportError:
            return False, "PyTorch not installed"

    @classmethod
    def get_best_device(cls) -> DeviceType:
        """获取最佳可用设备

        优先级: CUDA > MPS > CPU

        Returns:
            DeviceType 枚举值
        """
        cuda_ok, _ = cls.detect_cuda()
        if cuda_ok:
            return DeviceType.CUDA

        mps_ok, _ = cls.detect_mps()
        if mps_ok:
            return DeviceType.MPS

        return DeviceType.CPU

    @classmethod
    def get_device_info(cls) -> dict:
        """获取设备详细信息

        Returns:
            dict: {
                'device_type': str,
                'device_count': int,
                'device_names': list[str],
                'memory_mb': list[int],
                'recommended_compute_type': str,
            }
        """
        device_type = cls.get_best_device()

        info = {
            "device_type": device_type.value,
            "device_count": 0,
            "device_names": [],
            "memory_mb": [],
            "recommended_compute_type": "int8",  # CPU 默认
        }

        if device_type == DeviceType.CUDA:
            try:
                import torch

                info["device_count"] = torch.cuda.device_count()
                for i in range(info["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    info["device_names"].append(props.name)
                    info["memory_mb"].append(props.total_memory // (1024 * 1024))

                # 根据显存推荐计算类型
                if info["memory_mb"]:
                    max_mem = max(info["memory_mb"])
                    if max_mem >= 8000:
                        info["recommended_compute_type"] = "float16"
                    elif max_mem >= 4000:
                        info["recommended_compute_type"] = "int8_float16"
                    else:
                        info["recommended_compute_type"] = "int8"
            except Exception as e:
                logger.warning(f"Failed to get CUDA device info: {e}")

        elif device_type == DeviceType.MPS:
            info["device_count"] = 1
            info["device_names"] = ["Apple Silicon (MPS)"]
            info["recommended_compute_type"] = "float16"

        return info

    @classmethod
    def get_gpu_memory_used_mb(cls) -> Optional[float]:
        """获取当前 GPU 显存使用量 (MB)

        Returns:
            显存使用量，如果无法获取返回 None
        """
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass

        return None

    @classmethod
    def select_whisper_model(cls, device_type: DeviceType) -> str:
        """根据设备选择最佳的 Whisper 模型

        Args:
            device_type: 设备类型

        Returns:
            推荐的模型名称
        """
        if device_type == DeviceType.CUDA:
            info = cls.get_device_info()
            max_mem = max(info["memory_mb"]) if info["memory_mb"] else 0

            if max_mem >= 8000:
                return "large-v3"
            elif max_mem >= 4000:
                return "medium"
            else:
                return "small"
        elif device_type == DeviceType.MPS:
            return "medium"
        else:
            return "small"
