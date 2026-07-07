"""进度管理模块

提供管道各阶段的进度跟踪和展示，基于 tqdm。
"""

import time
from typing import Any, Callable, Dict, Optional


class ProgressManager:
    """进度管理器

    跟踪管道各阶段的执行进度，支持回调通知。

    使用示例:
        pm = ProgressManager(total_stages=5)
        pm.start_stage("separation", total_items=10)
        for i in range(10):
            process_item(i)
            pm.update_stage(1)
        pm.finish_stage()

        stats = pm.get_stats()
    """

    def __init__(
        self,
        total_stages: int = 5,
        callback: Optional[Callable] = None,
        use_tqdm: bool = True,
    ):
        self._total_stages = total_stages
        self._callback = callback
        self._use_tqdm = use_tqdm

        self._current_stage: str = ""
        self._stage_progress: Dict[str, dict] = {}
        self._start_time = time.time()
        self._stage_times: Dict[str, float] = {}

        # tqdm 实例
        self._pbar = None

    @property
    def current_stage(self) -> str:
        return self._current_stage

    def start_stage(
        self,
        stage_name: str,
        total_items: int = 1,
        description: str = "",
    ) -> None:
        """开始一个处理阶段

        Args:
            stage_name: 阶段名称
            total_items: 总处理项数
            description: 阶段描述
        """
        self._current_stage = stage_name
        self._stage_progress[stage_name] = {
            "total": total_items,
            "current": 0,
            "description": description,
            "start_time": time.time(),
        }

        if self._use_tqdm:
            from tqdm import tqdm

            self._pbar = tqdm(
                total=total_items,
                desc=description or stage_name,
                unit="item",
            )

        if self._callback:
            self._callback(
                {
                    "type": "stage_start",
                    "stage": stage_name,
                    "total": total_items,
                    "description": description,
                }
            )

    def report_progress(
        self, current: int, total: int, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """报告绝对进度（不依赖内部计数器，直接推送到回调）

        用于子引擎（如 UVR）将自身的迭代进度透传到 WebSocket，
        绕开 ProgressManager 内部的 tqdm 计数器。

        Args:
            current: 当前进度值
            total: 总进度值
            extra: 额外信息（如阶段描述文本）
        """
        if self._callback:
            self._callback(
                {
                    "type": "progress",
                    "stage": self._current_stage,
                    "current": current,
                    "total": total,
                    "extra": extra or {},
                }
            )

    def update_stage(self, n: int = 1, extra: Optional[Dict[str, Any]] = None) -> None:
        """更新当前阶段的进度

        Args:
            n: 进度增量
            extra: 额外信息
        """
        if self._current_stage not in self._stage_progress:
            return

        self._stage_progress[self._current_stage]["current"] += n

        if self._pbar:
            self._pbar.update(n)
            if extra:
                self._pbar.set_postfix(extra)

        if self._callback:
            self._callback(
                {
                    "type": "progress",
                    "stage": self._current_stage,
                    "current": self._stage_progress[self._current_stage]["current"],
                    "total": self._stage_progress[self._current_stage]["total"],
                    "extra": extra or {},
                }
            )

    def finish_stage(self) -> float:
        """完成当前阶段

        多次调用同一阶段（如多块处理）时，耗时自动累加，
        前端可通过 stage_timings 获取累计耗时。

        Returns:
            本次阶段耗时（秒），非累计值（前端自行累加）
        """
        if self._current_stage not in self._stage_progress:
            return 0.0

        elapsed = (
            time.time() - self._stage_progress[self._current_stage]["start_time"]
        )
        # 累加耗时：同一阶段多次调用时自动累加
        prev = self._stage_times.get(self._current_stage, 0.0)
        self._stage_times[self._current_stage] = prev + elapsed

        if self._pbar:
            self._pbar.close()
            self._pbar = None

        if self._callback:
            self._callback(
                {
                    "type": "stage_finish",
                    "stage": self._current_stage,
                    "elapsed_seconds": elapsed,        # 本次耗时（前端累加用）
                    "accumulated_seconds": prev + elapsed,  # 累计耗时（最终值）
                }
            )

        return elapsed

    def get_stats(self) -> dict:
        """获取统计信息

        Returns:
            dict: 包含各阶段耗时和总时间
        """
        total_time = time.time() - self._start_time
        return {
            "total_time_seconds": total_time,
            "stage_timings": dict(self._stage_times),
            "stages_completed": len(self._stage_times),
        }
