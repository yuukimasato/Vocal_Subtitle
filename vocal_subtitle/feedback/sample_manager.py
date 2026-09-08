"""反馈样本管理器 — D2 候选反馈集

管理用户反馈样本的生命周期：
  - 入库前：脱敏、去重、质量门控
  - 差异分类统计
  - 不可逆样本 ID 生成
  - 审核状态管理

对应 FEEDBACK_LOOP.md §2 (反馈数据 Schema) 和 §7 (审核队列)。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 反馈样本最小对齐覆盖率（低于此值拒绝入库）
MIN_ALIGNMENT_COVERAGE = 0.70

# 差异分类
DIFF_CATEGORIES = {
    "text_correction": "内容更正",
    "time_adjustment": "时间调整",
    "format_preference": "格式偏好",
    "speaker_label": "角色标注",
    "structural_rewrite": "结构改写",
}

# 审核结果
REVIEW_RESULTS = {"accepted", "rejected", "disputed"}


@dataclass
class FeedbackSample:
    """D2 候选反馈样本记录"""

    sample_id: str = ""
    data_source: str = "user_revision"
    consent_level: str = "anonymous"  # local | anonymous | full
    language: str = "unknown"
    scene: str = ""
    audio_duration_seconds: float = 0.0
    audio_condition: str = ""
    speaker_count: int = 0

    original: dict = field(default_factory=dict)
    automatic_subtitle: dict = field(default_factory=dict)
    human_revision: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)

    retention_days: int = 365
    anonymization: str = "deidentified"
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "data_source": self.data_source,
            "consent_level": self.consent_level,
            "language": self.language,
            "scene": self.scene,
            "audio_duration_seconds": self.audio_duration_seconds,
            "audio_condition": self.audio_condition,
            "speaker_count": self.speaker_count,
            "original": self.original,
            "automatic_subtitle": self.automatic_subtitle,
            "human_revision": self.human_revision,
            "alignment": self.alignment,
            "review": self.review,
            "retention_days": self.retention_days,
            "anonymization": self.anonymization,
            "created_at": self.created_at,
        }


class FeedbackSampleManager:
    """管理 D2 候选反馈样本的收集、质量门控和审核。

    使用示例:
        mgr = FeedbackSampleManager(storage_dir)
        mgr.ingest(auto_subtitle, human_revision, alignment, consent="anonymous")
        pending = mgr.list_pending_review()
        mgr.review("sample-id", "accepted", reviewer="admin")
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._storage_dir = Path(storage_dir or (
            Path(__file__).parent.parent.parent / "cache" / "feedback_samples"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._storage_dir / "sample_index.json"
        self._index: dict[str, dict] = self._load_index()

    # ---- 索引持久化 ----

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- 样本 ID 生成 ----

    @staticmethod
    def generate_sample_id(
        auto_text: str,
        human_text: str,
        timestamp: str = "",
    ) -> str:
        """生成不可逆样本 ID。

        SHA256(auto_text + human_text + timestamp) 前 16 位。
        """
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()
        raw = f"{auto_text}|{human_text}|{timestamp}"
        return f"D2-{datetime.now().strftime('%Y%m%d')}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

    # ---- 入库 ----

    def ingest(
        self,
        auto_subtitle: str,
        human_revision: str,
        alignment: dict,
        *,
        consent_level: str = "anonymous",
        language: str = "unknown",
        scene: str = "",
        audio_duration: float = 0.0,
        audio_condition: str = "",
        speaker_count: int = 0,
        original_config: dict | None = None,
        edit_types: dict | None = None,
    ) -> Optional[FeedbackSample]:
        """将反馈提交入库。

        质量门控：
          - alignment.coverage_ratio >= 0.70
          - 自动字幕和人工修订均非空
          - consent_level 必须非空

        Returns:
            入库成功返回 FeedbackSample，被门控拒绝返回 None
        """
        # 质量门控
        if not auto_subtitle or not human_revision:
            logger.warning("Feedback rejected: empty subtitle")
            return None

        coverage = alignment.get("coverage_ratio", 0.0)
        if coverage < MIN_ALIGNMENT_COVERAGE:
            logger.warning(
                "Feedback rejected: coverage %.2f < %.2f",
                coverage, MIN_ALIGNMENT_COVERAGE,
            )
            return None

        if not consent_level:
            logger.warning("Feedback rejected: no consent level")
            return None

        now = datetime.now(timezone.utc).isoformat()
        sample_id = self.generate_sample_id(auto_subtitle, human_revision)

        # 去重
        if sample_id in self._index:
            logger.info("Feedback already exists: %s", sample_id)
            return None

        sample = FeedbackSample(
            sample_id=sample_id,
            data_source="user_revision",
            consent_level=consent_level,
            language=language,
            scene=scene or "unknown",
            audio_duration_seconds=round(audio_duration, 2),
            audio_condition=audio_condition or "unknown",
            speaker_count=speaker_count,
            original=original_config or {},
            automatic_subtitle={
                "version": "auto-v1",
                "subtitle_hash": hashlib.sha256(auto_subtitle.encode()).hexdigest()[:16],
                "event_count": auto_subtitle.count("\n\n") + 1,
            },
            human_revision={
                "version": "human-v1",
                "subtitle_hash": hashlib.sha256(human_revision.encode()).hexdigest()[:16],
                "event_count": human_revision.count("\n\n") + 1,
                "edit_types": edit_types or {},
            },
            alignment=alignment,
            review={"reviewer": "", "result": "pending", "timestamp": ""},
            created_at=now,
        )

        # 持久化
        sample_path = self._storage_dir / f"{sample_id}.json"
        sample_path.write_text(
            json.dumps(sample.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._index[sample_id] = {
            "sample_id": sample_id,
            "status": "pending",
            "language": language,
            "scene": scene,
            "coverage": coverage,
            "created_at": now,
        }
        self._save_index()

        logger.info("Feedback ingested: %s (coverage=%.2f)", sample_id, coverage)
        return sample

    # ---- 审核队列 ----

    def list_pending_review(self) -> list[dict]:
        """列出所有待审核的反馈样本。"""
        pending = []
        for sample_id, entry in self._index.items():
            if entry.get("status") == "pending":
                sample_path = self._storage_dir / f"{sample_id}.json"
                if sample_path.exists():
                    try:
                        data = json.loads(sample_path.read_text(encoding="utf-8"))
                        pending.append({
                            "sample_id": sample_id,
                            "diff_summary": data.get("human_revision", {}).get("edit_types", {}),
                            "confidence": data.get("alignment", {}).get("confidence", 0.0),
                            "submitted_at": entry.get("created_at", ""),
                            "language": entry.get("language", ""),
                            "scene": entry.get("scene", ""),
                        })
                    except (json.JSONDecodeError, IOError):
                        pass
        return sorted(pending, key=lambda x: x.get("submitted_at", ""), reverse=True)

    def review(
        self,
        sample_id: str,
        result: str,
        *,
        reviewer: str = "",
        notes: str = "",
    ) -> bool:
        """审核一个反馈样本。

        Args:
            sample_id: 样本 ID
            result: accepted | rejected | disputed
            reviewer: 审核人
            notes: 审核备注

        Returns:
            是否成功
        """
        if result not in REVIEW_RESULTS:
            raise ValueError(f"Invalid review result: {result!r}. Must be one of {REVIEW_RESULTS}")

        if sample_id not in self._index:
            logger.warning("Sample not found: %s", sample_id)
            return False

        now = datetime.now(timezone.utc).isoformat()

        # 更新样本文件
        sample_path = self._storage_dir / f"{sample_id}.json"
        if sample_path.exists():
            try:
                data = json.loads(sample_path.read_text(encoding="utf-8"))
                data["review"] = {
                    "reviewer": reviewer,
                    "result": result,
                    "notes": notes,
                    "timestamp": now,
                }
                sample_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, IOError):
                pass

        # 更新索引
        self._index[sample_id]["status"] = result
        self._index[sample_id]["reviewed_at"] = now
        self._save_index()

        logger.info("Feedback reviewed: %s → %s (by %s)", sample_id, result, reviewer)
        return True

    def get(self, sample_id: str) -> Optional[dict]:
        """获取单个样本详情。"""
        if sample_id not in self._index:
            return None
        sample_path = self._storage_dir / f"{sample_id}.json"
        if sample_path.exists():
            try:
                return json.loads(sample_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def count_by_status(self, status: str | None = None) -> int:
        if status:
            return sum(1 for e in self._index.values() if e.get("status") == status)
        return len(self._index)
