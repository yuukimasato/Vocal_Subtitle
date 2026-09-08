"""反馈样本脱敏器

D2 自动入库前对反馈样本进行脱敏处理。
对应 FEEDBACK_LOOP.md 的 D2 自动入库流程：
  - 脱敏：移除敏感字段，设置 anonymization 标记
  - 质量门控：验证样本满足 D2 准入条件

三种同意级别:
  - local: 仅本地分析，不入库
  - anonymous: 匿名化后入库
  - full: 完全授权，脱敏后入库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 敏感字段列表（入库前移除）
SENSITIVE_FIELDS = (
    "audio_path", "task_id", "user_name", "email",
    "reviewer", "uploader_ip", "machine_id", "original_filename",
    "input_path", "output_path",
)


@dataclass
class AnonymizationResult:
    """脱敏结果"""

    sample: dict              # 脱敏后的样本
    removed_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "anonymized": True,
            "removed_fields": self.removed_fields,
        }


class FeedbackAnonymizer:
    """反馈样本脱敏器。

    使用示例:
        anon = FeedbackAnonymizer()
        result = anon.anonymize(sample_dict, consent_level="anonymous")
        # result.sample["anonymization"] = "deidentified"
        # result.removed_fields = ["audio_path", "task_name", ...]
    """

    @staticmethod
    def anonymize(
        sample: dict[str, Any],
        *,
        consent_level: str = "anonymous",
    ) -> AnonymizationResult:
        """对反馈样本进行脱敏处理。

        Args:
            sample: 原始样本 dict
            consent_level: local | anonymous | full

        Returns:
            AnonymizationResult 包含脱敏后的样本和已移除字段列表
        """
        sanitized = dict(sample)
        removed: list[str] = []

        # 移除敏感字段
        for field in SENSITIVE_FIELDS:
            if field in sanitized:
                sanitized.pop(field, None)
                removed.append(field)

        # 按同意级别设置 anonymization 标记
        if consent_level == "local":
            sanitized["anonymization"] = "local_only"
        elif consent_level in ("anonymous", "full"):
            sanitized["anonymization"] = "deidentified"
        else:
            sanitized["anonymization"] = "unknown"

        # 对嵌套 original 字段也做清理
        original = sanitized.get("original", {})
        if isinstance(original, dict):
            cleaned_original = dict(original)
            for field in SENSITIVE_FIELDS:
                cleaned_original.pop(field, None)
            sanitized["original"] = cleaned_original

        return AnonymizationResult(
            sample=sanitized,
            removed_fields=removed,
        )

    @classmethod
    def is_eligible_for_d2(cls, sample: dict[str, Any]) -> bool:
        """检查样本是否满足 D2 入库条件。

        条件:
          1. consent_level 不能为 "local"（仅本地分析不入库）
          2. alignment.coverage_ratio >= 0.70
          3. automatic_subtitle 和 human_revision 均非空

        Args:
            sample: 样本 dict

        Returns:
            是否满足 D2 条件
        """
        consent = sample.get("consent_level", "")
        if consent == "local":
            return False

        alignment = sample.get("alignment", {})
        coverage = alignment.get("coverage_ratio", 0.0)
        if coverage < 0.70:
            return False

        auto = sample.get("automatic_subtitle", {})
        human = sample.get("human_revision", {})
        if not auto or not human:
            return False

        return True


__all__ = ["AnonymizationResult", "FeedbackAnonymizer", "SENSITIVE_FIELDS"]
