"""测试 GPUDetector GPU 检测模块"""

import pytest

from vocal_subtitle.utils.gpu_detector import DeviceType, GPUDetector


class TestGPUDetector:
    """GPU 检测器测试"""

    def test_get_best_device_returns_valid_type(self):
        """获取最佳设备返回有效类型"""
        device = GPUDetector.get_best_device()
        assert device in (DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU)

    def test_detect_cuda_returns_tuple(self):
        """CUDA 检测返回 (bool, reason)"""
        available, reason = GPUDetector.detect_cuda()
        assert isinstance(available, bool)
        if not available:
            assert reason is not None

    def test_detect_mps_returns_tuple(self):
        """MPS 检测返回 (bool, reason)"""
        available, reason = GPUDetector.detect_mps()
        assert isinstance(available, bool)

    def test_get_device_info_has_required_keys(self):
        """设备信息包含所有必要字段"""
        info = GPUDetector.get_device_info()
        assert "device_type" in info
        assert "device_count" in info
        assert "device_names" in info
        assert "memory_mb" in info
        assert "recommended_compute_type" in info

    def test_get_device_info_type_is_str(self):
        """device_type 是字符串"""
        info = GPUDetector.get_device_info()
        assert isinstance(info["device_type"], str)
        assert info["device_type"] in ("cuda", "mps", "cpu")

    def test_select_whisper_model_cpu(self):
        """CPU 设备推荐 small 模型"""
        model = GPUDetector.select_whisper_model(DeviceType.CPU)
        assert model in ("small", "medium", "large-v3")
        # CPU 默认推荐 small
        assert model == "small"

    def test_select_whisper_model_mps(self):
        """MPS 设备推荐 medium 模型"""
        model = GPUDetector.select_whisper_model(DeviceType.MPS)
        assert model == "medium"

    def test_select_whisper_model_cuda(self):
        """CUDA 设备推荐根据显存"""
        model = GPUDetector.select_whisper_model(DeviceType.CUDA)
        assert model in ("small", "medium", "large-v3")

    def test_get_gpu_memory_used_returns_number_or_none(self):
        """GPU 显存使用返回数字或 None"""
        mem = GPUDetector.get_gpu_memory_used_mb()
        if mem is not None:
            assert isinstance(mem, (int, float))
            assert mem >= 0

    def test_device_type_enum_values(self):
        """DeviceType 枚举值"""
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"
        assert DeviceType.CPU.value == "cpu"
