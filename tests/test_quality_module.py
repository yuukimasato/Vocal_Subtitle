"""Tests for the vocal_subtitle.quality module.

Covers: quality_classifier, scene_slicer, priority_calculator,
        trend_reporter, controlled_runner, data_version_manager
"""

import logging
import tempfile
from pathlib import Path

import pytest

from vocal_subtitle.quality.controlled_runner import (
    ControlledRunManager,
    RunKind,
)
from vocal_subtitle.quality.data_version_manager import (
    DatasetTier,
    DatasetVersionStatus,
    DataVersionManager,
)
from vocal_subtitle.quality.priority_calculator import (
    IssuePriorityCalculator,
    PriorityFactors,
    PriorityLevel,
)
from vocal_subtitle.quality.quality_classifier import (
    IssueCategory,
    IssueClassifier,
    IssueSeverity,
)
from vocal_subtitle.quality.scene_slicer import (
    MAX_SINGLE_DIMENSION_RATIO,
    SceneSlicer,
    SceneTag,
)
from vocal_subtitle.quality.trend_reporter import (
    TrendReporter,
    VersionMetrics,
)

# ---------------------------------------------------------------------------
# IssueClassifier
# ---------------------------------------------------------------------------


class TestIssueClassifier:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("程序崩溃了", IssueCategory.USABILITY),
            ("无法导出字幕", IssueCategory.USABILITY),
            ("app crashes on startup", IssueCategory.USABILITY),
            ("长音频尾部漏句", IssueCategory.COMPLETENESS),
            ("missing sentences", IssueCategory.COMPLETENESS),
            ("识别出大量错字", IssueCategory.TEXT_ACCURACY),
            ("wrong word recognized", IssueCategory.TEXT_ACCURACY),
            ("字幕时间轴偏移", IssueCategory.TIME_ACCURACY),
            ("audio out of sync", IssueCategory.TIME_ACCURACY),
            ("说话人标注错误", IssueCategory.SPEAKER),
            ("speaker mislabel", IssueCategory.SPEAKER),
            ("字幕过长且断句不当", IssueCategory.READABILITY),
            ("formatting is too long", IssueCategory.READABILITY),
            ("处理超时内存不足", IssueCategory.PERFORMANCE),
            ("processing is too slow", IssueCategory.PERFORMANCE),
            ("API 费用太高", IssueCategory.COST),
            ("high api cost", IssueCategory.COST),
            ("模型是黑盒来源不明", IssueCategory.EXPLAINABILITY),
            ("untraceable black box", IssueCategory.EXPLAINABILITY),
        ],
    )
    def test_classify_keywords(self, text, expected):
        assert IssueClassifier.classify(text) == expected

    def test_classify_unknown(self):
        assert IssueClassifier.classify("普通的描述文本") == IssueCategory.UNKNOWN

    def test_assess_severity_defaults(self):
        assert (
            IssueClassifier.assess_severity(IssueCategory.USABILITY)
            == IssueSeverity.CRITICAL
        )
        assert (
            IssueClassifier.assess_severity(IssueCategory.COMPLETENESS)
            == IssueSeverity.HIGH
        )
        assert (
            IssueClassifier.assess_severity(IssueCategory.READABILITY)
            == IssueSeverity.LOW
        )
        assert (
            IssueClassifier.assess_severity(IssueCategory.UNKNOWN)
            == IssueSeverity.MEDIUM
        )

    def test_assess_severity_workaround_downgrades(self):
        assert (
            IssueClassifier.assess_severity(
                IssueCategory.COMPLETENESS, {"workaround_available": True}
            )
            == IssueSeverity.MEDIUM
        )

    def test_assess_severity_user_blocked_upgrades(self):
        assert (
            IssueClassifier.assess_severity(
                IssueCategory.TEXT_ACCURACY, {"user_blocked": True}
            )
            == IssueSeverity.CRITICAL
        )

    def test_analyze_combines_fields(self):
        issue = IssueClassifier.analyze(
            {
                "title": "长音频尾部漏句",
                "description": "&gt;30min 音频最后一段缺失",
            }
        )
        assert issue.category == IssueCategory.COMPLETENESS
        assert issue.severity == IssueSeverity.HIGH

    def test_batch_analyze(self):
        results = IssueClassifier.batch_analyze(
            [
                {"title": "崩溃"},
                {"title": "字幕偏移"},
            ],
            detected_by="report",
        )
        assert len(results) == 2
        assert results[0].category == IssueCategory.USABILITY


# ---------------------------------------------------------------------------
# SceneSlicer
# ---------------------------------------------------------------------------


class TestSceneSlicer:
    def test_tag_full(self):
        tag = SceneSlicer.tag(
            {
                "language": "zh",
                "speaker_count": 2,
                "background_noise": "clean",
                "words_per_second": 4.2,
                "duration_seconds": 600,
                "device_name": "cuda",
                "vram_gb": 16,
                "scene_type": "podcast",
            }
        )
        assert tag.language == "zh"
        assert tag.speaker_count == "dual"
        assert tag.background_noise == "clean"
        assert tag.speech_rate == "normal"
        assert tag.audio_length == "medium"
        assert tag.device == "gpu_12gb_plus"
        assert tag.scene_type == "podcast"

    def test_tag_id_stable(self):
        t1 = SceneTag(language="zh", speaker_count="dual")
        assert t1.tag_id() == SceneTag(language="zh", speaker_count="dual").tag_id()
        assert len(t1.tag_id()) == 12

    def test_as_key_all_dimensions(self):
        tag = SceneTag(
            language="zh",
            speaker_count="dual",
            background_noise="clean",
            speech_rate="normal",
            audio_length="medium",
            device="gpu_8gb",
            scene_type="podcast",
        )
        assert tag.as_key() == "zh|dual|clean|normal|medium|gpu_8gb|podcast"

    def test_balance_report_warns_above_60(self):
        samples = [{"language": "zh"}] * 4 + [{"language": "en"}]
        report = SceneSlicer.balance_report(samples)
        assert report["total"] == 5
        lang_warn = [w for w in report["warnings"] if w["dimension"] == "language"]
        assert lang_warn
        assert lang_warn[0]["ratio"] == 0.8

    def test_max_ratio_constant(self):
        assert MAX_SINGLE_DIMENSION_RATIO == 0.60


# ---------------------------------------------------------------------------
# IssuePriorityCalculator
# ---------------------------------------------------------------------------


class TestPriorityCalculator:
    def test_calculate_multiplies(self):
        factors = PriorityFactors(
            impact_range=3, user_severity=4, reproducibility=2, fix_confidence=3
        )
        result = IssuePriorityCalculator.calculate(factors)
        assert result.score == 72
        assert result.level == PriorityLevel.IMMEDIATE

    @pytest.mark.parametrize(
        ("factors", "level"),
        [
            (PriorityFactors(5, 3, 2, 1), PriorityLevel.IMMEDIATE),
            (PriorityFactors(5, 2, 2, 1), PriorityLevel.THIS_VERSION),
            (PriorityFactors(1, 1, 1, 1), PriorityLevel.SCHEDULED),
        ],
    )
    def test_level_thresholds(self, factors, level):
        assert IssuePriorityCalculator.calculate(factors).level == level

    def test_validate_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            IssuePriorityCalculator.calculate(
                PriorityFactors(
                    impact_range=0, user_severity=1, reproducibility=1, fix_confidence=1
                )
            )

    def test_batch_sorts_descending(self):
        issues = [
            {"impact_range": 2, "user_severity": 2},  # default: repr=1, conf=1 → 4
            {
                "impact_range": 5,
                "user_severity": 3,
                "reproducibility": 2,
                "fix_confidence": 2,
            },  # 60
        ]
        results = IssuePriorityCalculator.batch_calculate(issues)
        assert results[0].score == 60
        assert results[1].score == 4


# ---------------------------------------------------------------------------
# TrendReporter
# ---------------------------------------------------------------------------


class TestTrendReporter:
    def test_build_report_fills_defaults(self):
        report = TrendReporter.build_report("0.2.0", "0.1.0", trends={})
        assert report.version == "0.2.0"
        assert report.baseline == "0.1.0"
        assert report.trends["D1_reference"] == {
            "scenarios": 0,
            "status": "not_available",
        }
        assert report.trends["D3_feedback"] == {"status": "not_available"}

    def test_write_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            reporter = TrendReporter(storage_dir=Path(tmp))
            report = reporter.build_report(
                "0.2.0", "0.1.0", trends={"operations": {"crash_rate": 0.01}}
            )
            path = reporter.write_report(report)
            assert path.exists()
            loaded = reporter.load_report(path)
            assert loaded.to_dict() == report.to_dict()

    def test_compare_detects_improvements(self):
        prev = VersionMetrics(
            version="0.1.0",
            date="2026-07-01",
            trends={
                "D0_engineering": {"test_pass_rate": 0.95, "regression_count": 3},
            },
        )
        curr = VersionMetrics(
            version="0.2.0",
            date="2026-08-01",
            trends={
                "D0_engineering": {"test_pass_rate": 0.98, "regression_count": 1},
            },
        )
        diff = TrendReporter.compare(prev, curr)
        assert diff.improvement_count == 2
        assert diff.regression_count == 0
        assert len(diff.changes) == 2

    def test_render_monthly_report(self):
        report = TrendReporter.build_report(
            "0.2.0", "0.1.0", date="2026-08-15", trends={"operations": {}}
        )
        prev = VersionMetrics(version="0.1.0", date="2026-07-01", trends={})
        curr = VersionMetrics(version="0.2.0", date="2026-08-01", trends={})
        diff = TrendReporter.compare(prev, curr)
        md = TrendReporter.render_monthly_report(report, diff)
        assert "# 月度质量报告" in md
        assert "0.2.0" in md
        assert "## 用户反馈统计" in md


# ---------------------------------------------------------------------------
# ControlledRunManager
# ---------------------------------------------------------------------------


class TestControlledRunner:
    def test_run_kind_enum(self):
        assert RunKind.BASELINE.value == "baseline"
        assert RunKind.CANDIDATE.value == "candidate"
        assert RunKind.SHADOW.value == "shadow"

    def test_create_and_load_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "runs"
            mgr = ControlledRunManager(storage_dir=storage)
            audio = Path(tmp) / "input.wav"
            config = Path(tmp) / "cfg.yaml"
            spec = mgr.create_run(
                RunKind.CANDIDATE, audio, config, overrides={"asr.model": "turbo"}
            )
            assert spec.kind == RunKind.CANDIDATE
            assert spec.overrides == {"asr.model": "turbo"}
            loaded = mgr.load_spec(spec.run_id)
            assert loaded.to_dict() == spec.to_dict()

    def test_recommend_enable(self):
        rec = ControlledRunManager.recommend(
            {
                "benefited_scenes": [{"scene": "podcast"}],
                "degraded_scenes": [],
                "uncertain_areas": [],
            }
        )
        assert rec.recommend_enable is True

    def test_recommend_disable_with_degradation(self):
        rec = ControlledRunManager.recommend(
            {
                "benefited_scenes": [{"scene": "a"}],
                "degraded_scenes": [{"scene": "b"}],
                "uncertain_areas": [],
            }
        )
        assert rec.recommend_enable is False


# ---------------------------------------------------------------------------
# DataVersionManager
# ---------------------------------------------------------------------------


def _balanced_entries():
    return [
        {
            "sample_id": f"b{i}",
            "metadata": {"language": "zh", "speaker_count": 1, "duration_seconds": 60},
        }
        for i in range(3)
    ] + [
        {
            "sample_id": f"b{i}",
            "metadata": {"language": "en", "speaker_count": 1, "duration_seconds": 60},
        }
        for i in range(3, 5)
    ]


class TestDataVersionManager:
    def test_d0_auto_seeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DataVersionManager(storage_dir=Path(tmp) / "datasets")
            d0 = mgr.current(DatasetTier.D0)
            assert d0 is not None
            assert d0.tier == DatasetTier.D0
            assert d0.version_id == "D0-20260802-001"
            assert d0.sample_count == 13
            assert d0.status == DatasetVersionStatus.ACTIVE

    def test_register_persists_to_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "datasets"
            mgr = DataVersionManager(storage_dir=store)
            mgr.register(
                DatasetTier.D1,
                "D1-20260803-001",
                [{"sample_id": "s1"}],
                description="回归集",
            )
            mgr2 = DataVersionManager(storage_dir=store)
            assert len(mgr2.list_versions(DatasetTier.D1)) == 1

    def test_freeze_d3_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DataVersionManager(storage_dir=Path(tmp) / "datasets")
            v = mgr.freeze(
                DatasetTier.D3,
                "D3-20260803-001",
                _balanced_entries(),
                description="首次回归集",
            )
            assert v.version_id == "D3-20260803-001"
            assert v.sample_count == 5

    def test_freeze_unbalanced_logs_warning(self, caplog):
        entries = [
            {
                "sample_id": f"u{i}",
                "metadata": {
                    "language": "zh",
                    "speaker_count": 1,
                    "duration_seconds": 60,
                },
            }
            for i in range(4)
        ]
        entries.append(
            {
                "sample_id": "u4",
                "metadata": {
                    "language": "en",
                    "speaker_count": 1,
                    "duration_seconds": 60,
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DataVersionManager(storage_dir=Path(tmp) / "datasets")
            with caplog.at_level(
                logging.WARNING, logger="vocal_subtitle.quality.data_version_manager"
            ):
                mgr.freeze(DatasetTier.D3, "D3-20260803-002", entries)
            assert any("Balance warnings" in r.message for r in caplog.records)

    def test_supersede(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DataVersionManager(storage_dir=Path(tmp) / "datasets")
            mgr.freeze(DatasetTier.D3, "D3-v1", _balanced_entries())
            mgr.freeze(DatasetTier.D3, "D3-v2", _balanced_entries())
            assert mgr.supersede(DatasetTier.D3, "D3-v1", reason="被取代") is True
            assert mgr.current(DatasetTier.D3).version_id == "D3-v2"

    def test_cleanup_expired(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DataVersionManager(storage_dir=Path(tmp) / "datasets")
            mgr.register(DatasetTier.D2, "D2-20240101-001", [{"sample_id": "old"}])
            mgr._registry["D2"]["D2-20240101-001"]["freeze_date"] = "2024-01-01"
            archived = mgr.cleanup_expired(now="2025-01-01")
            assert archived == ["D2-20240101-001"]
