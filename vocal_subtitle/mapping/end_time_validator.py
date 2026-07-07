"""结束时间后校验器

在所有 Pipeline 后处理完成后、字幕输出前运行，对结束时间做全局一致性校验。

校验规则:
1. 相邻字幕不应有 < 30ms 的微小间隙（视频播放器一帧 = 33ms@30fps，不可见）
2. 同说话人的字幕间距不应 > 1.5s（可能是漏检）
3. 结束时间不应早于开始时间（兜底异常检测）
4. 单条字幕时长不应 < 200ms（不可读）
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class EndTimePostValidator:
    """结束时间后校验器

    在 LLM 优化之后、字幕输出之前运行，对所有字幕事件做
    最后一轮结束时间校验和自动修正。

    使用示例:
        validator = EndTimePostValidator()
        events = validator.validate(events)
    """

    def __init__(
        self,
        min_gap_ms: float = 30,
        min_duration_ms: float = 200,
        max_speaker_gap_sec: float = 1.5,
    ):
        self.min_gap_ms = min_gap_ms
        self.min_duration_ms = min_duration_ms
        self.max_speaker_gap_sec = max_speaker_gap_sec

    def validate(self, events: List) -> List:
        """对字幕事件列表做结束时间校验和修正

        返回修正后的事件列表（可能修改 end 时间）。

        Args:
            events: SubtitleEvent 列表（按 start 排序）

        Returns:
            修正后的事件列表
        """
        if not events:
            return events

        fixed_count = 0

        for i, evt in enumerate(events):
            # 规则 1: 结束时间 > 开始时间（兜底）
            if evt.end <= evt.start:
                logger.warning(
                    "EndTimeValidator: event #%d end(%.3f) <= start(%.3f), fixing",
                    getattr(evt, "index", i), evt.end, evt.start,
                )
                evt.end = evt.start + 0.5  # 默认 500ms
                fixed_count += 1

            # 规则 2: 最小时长保护
            duration_ms = (evt.end - evt.start) * 1000
            if duration_ms < self.min_duration_ms:
                evt.end = evt.start + self.min_duration_ms / 1000
                fixed_count += 1

            # 规则 3: 与前一条的微小间隙合并
            if i > 0:
                prev = events[i - 1]
                gap_ms = (evt.start - prev.end) * 1000
                if 0 < gap_ms < self.min_gap_ms:
                    # 微小间隙：延长前一条至当前开始（消除不可见帧间隙）
                    prev.end = evt.start
                    fixed_count += 1

            # 规则 4: 同说话人最大间隙告警（可能漏检语音）
            if i > 0:
                prev = events[i - 1]
                gap_sec = evt.start - prev.end
                same_speaker = (
                    getattr(prev, "speaker_id", None) is not None
                    and getattr(evt, "speaker_id", None) is not None
                    and prev.speaker_id == evt.speaker_id
                )
                if same_speaker and gap_sec > self.max_speaker_gap_sec:
                    logger.warning(
                        "EndTimeValidator: large gap (%.1fs) between same-speaker "
                        "events #%d and #%d — possible missed speech",
                        gap_sec,
                        getattr(prev, "index", i - 1),
                        getattr(evt, "index", i),
                    )

        if fixed_count > 0:
            logger.info(
                "EndTimePostValidator: fixed %d/%d events",
                fixed_count, len(events),
            )

        return events
