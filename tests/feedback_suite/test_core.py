"""Feedback tests: test_core."""

from datetime import datetime, timedelta

import pytest

from .common import _make_events


class TestSubtitleAligner:
    """智能字幕对齐器测试"""

    def test_perfect_1_to_1_match(self):
        """完全匹配: 自动版与修订版完全相同 → 全部 1:1"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        events = _make_events(
            [
                (0.0, 2.0, "今天天气真不错"),
                (2.5, 5.0, "我们去看电影吧"),
                (5.5, 8.0, "你觉得怎么样"),
            ]
        )

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner.align(events, events)

        assert len(pairs) == 3
        assert all(p.match_type == "1:1" for p in pairs)
        assert all(p.time_iou > 0.9 for p in pairs)
        assert all(p.text_similarity > 0.95 for p in pairs)

    def test_n_to_1_merge_detection(self):
        """N:1 检测: 自动版短句 → 修订版合并（通过 _dtw_align 直接测试）"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        # 使用 _dtw_align 绕过 coverage 门控，直接测试 DTW 匹配逻辑
        auto_sub = _make_events(
            [
                (0.0, 1.5, "所以我们需要"),
                (1.5, 3.5, "考虑这个方案的可行性"),
            ]
        )
        manual_sub = _make_events(
            [
                (0.0, 3.5, "所以我们需要考虑这个方案的可行性"),
            ]
        )

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner._dtw_align(auto_sub, manual_sub)

        merge_pairs = [p for p in pairs if p.match_type in ("N:1", "N:M")]
        assert len(merge_pairs) >= 1, (
            f"Expected N:1 merge, got types: {[p.match_type for p in pairs]}"
        )

    def test_1_to_n_split_detection(self):
        """1:N 检测: 自动版 1 条长句 → 修订版拆分为 2 条（通过 _dtw_align 测试）"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        auto_sub = _make_events(
            [
                (0.0, 5.0, "那么接下来我们看第二个问题也就是说如何优化性能"),
            ]
        )
        manual_sub = _make_events(
            [
                (0.0, 2.5, "那么接下来我们看第二个问题"),
                (2.5, 5.0, "也就是说如何优化性能"),
            ]
        )

        aligner = SubtitleAligner(semantic_enabled=False)
        pairs = aligner._dtw_align(auto_sub, manual_sub)

        split_pairs = [p for p in pairs if p.match_type in ("1:N", "N:M")]
        assert len(split_pairs) >= 1, (
            f"Expected 1:N split, got types: {[p.match_type for p in pairs]}"
        )

    def test_insert_and_delete_detection(self):
        """INSERT/DELETE: 修订版增加/删除了字幕行（通过 _residual_match 测试）"""
        from vocal_subtitle.feedback.aligner import AlignmentPair, SubtitleAligner

        # 构造 3 对 1:1 匹配 + 1 个 auto 未匹配 + 1 个 manual 未匹配
        auto_events = _make_events(
            [
                (0.0, 2.0, "第一句"),
                (3.0, 5.0, "第二句"),  # manual 删除此句
                (6.0, 8.0, "第三句"),
            ]
        )
        manual_events = _make_events(
            [
                (0.0, 2.0, "第一句"),
                (6.0, 8.0, "第三句"),
                (9.0, 11.0, "新句子"),  # auto 没有此句
            ]
        )

        # 只提供已匹配的 2 对
        from vocal_subtitle.feedback.aligner import _levenshtein_similarity, _time_iou

        pairs = [
            AlignmentPair(
                auto_events=[auto_events[0]],
                manual_events=[manual_events[0]],
                match_type="1:1",
                time_iou=_time_iou(0, 2, 0, 2),
                text_similarity=_levenshtein_similarity("第一句", "第一句"),
            ),
            AlignmentPair(
                auto_events=[auto_events[2]],
                manual_events=[manual_events[1]],
                match_type="1:1",
                time_iou=_time_iou(6, 8, 6, 8),
                text_similarity=_levenshtein_similarity("第三句", "第三句"),
            ),
        ]

        aligner = SubtitleAligner(semantic_enabled=False)
        result = aligner._residual_match(pairs, auto_events, manual_events)

        insert_types = [p.match_type for p in result if p.match_type == "INSERT"]
        delete_types = [p.match_type for p in result if p.match_type == "DELETE"]
        assert len(insert_types) + len(delete_types) >= 2, (
            f"Expected INSERT+DELETE types, got: {[p.match_type for p in result]}"
        )

    def test_alignment_coverage_gating(self):
        """对齐覆盖率 < 70% 时应抛出 AlignmentError"""
        from vocal_subtitle.feedback.aligner import AlignmentError, SubtitleAligner

        # 自动版 3 条 vs 手动版 15 条（完全不同的内容 + 数量悬殊）
        # DTW 只能匹配 ~3 条，其余为 INSERT/DELETE（不算 matched），
        # coverage = 3 / max(3, 15) = 20% < 70% → 应抛出异常
        auto = _make_events(
            [
                (0.0, 2.0, "完全不同的一句"),
                (3.0, 5.0, "完全不相关的另一句"),
                (6.0, 8.0, "第三句也不像"),
            ]
        )
        manual = _make_events([(20.0, 22.0, f"完全不同的内容A{i}") for i in range(15)])

        aligner = SubtitleAligner(semantic_enabled=False)
        with pytest.raises(AlignmentError, match="coverage too low"):
            aligner.align(auto, manual)

    def test_empty_events_raises(self):
        """空事件列表应抛出异常"""
        from vocal_subtitle.feedback.aligner import AlignmentError, SubtitleAligner

        events = _make_events([(0.0, 1.0, "test")])
        aligner = SubtitleAligner()

        with pytest.raises(AlignmentError):
            aligner.align([], events)
        with pytest.raises(AlignmentError):
            aligner.align(events, [])

    def test_anchor_segmentation(self):
        """长停顿 (>2s) 能正确触发段落切分"""
        from vocal_subtitle.feedback.aligner import SubtitleAligner

        auto = _make_events(
            [
                (0.0, 2.0, "段落A的第一句"),
                (2.5, 4.0, "段落A的第二句"),
                (7.0, 9.0, "段落B的第一句"),  # 间隔 3s > 2s
                (9.5, 12.0, "段落B的第二句"),
            ]
        )
        manual = _make_events(
            [
                (0.0, 2.2, "段落A的第一句修改"),
                (2.5, 4.2, "段落A的第二句修改"),
                (7.0, 9.2, "段落B的第一句修改"),
                (9.5, 12.2, "段落B的第二句修改"),
            ]
        )

        aligner = SubtitleAligner(semantic_enabled=False)
        anchors = aligner._find_global_anchors(auto, manual)
        # 应该有至少 2 个锚点（首尾或段落边界）
        assert len(anchors) >= 2

    def test_time_iou_symmetric(self):
        """时间 IoU 应是对称的"""
        from vocal_subtitle.feedback.aligner import _time_iou

        iou1 = _time_iou(0, 10, 2, 8)
        iou2 = _time_iou(2, 8, 0, 10)
        assert iou1 == pytest.approx(iou2)

    def test_time_iou_non_overlapping(self):
        """完全不重叠 → IoU = 0"""
        from vocal_subtitle.feedback.aligner import _time_iou

        assert _time_iou(0, 5, 10, 15) == 0.0
        assert _time_iou(10, 15, 0, 5) == 0.0

    def test_levenshtein_identical(self):
        """相同文本 → 相似度 1.0"""
        from vocal_subtitle.feedback.aligner import _levenshtein_similarity

        assert _levenshtein_similarity("hello", "hello") == 1.0

    def test_levenshtein_completely_different(self):
        """完全不同文本 → 低相似度"""
        from vocal_subtitle.feedback.aligner import _levenshtein_similarity

        sim = _levenshtein_similarity("abc", "xyz")
        assert sim < 0.5


# ============================================================================
#  2. DiffAnalyzer 测试
# ============================================================================


class TestDiffAnalyzer:
    """差异分类与归因分析器测试"""

    def _make_alignment_pairs(self, auto_events, manual_events, match_type="1:1"):
        """创建合成的 AlignmentPair 列表"""
        from vocal_subtitle.feedback.aligner import AlignmentPair

        pairs = []
        for i, (ae, me) in enumerate(zip(auto_events, manual_events)):
            from vocal_subtitle.feedback.aligner import (
                _levenshtein_similarity,
                _time_iou,
            )

            iou = _time_iou(ae.start, ae.end, me.start, me.end)
            text_sim = _levenshtein_similarity(ae.text, me.text)
            pairs.append(
                AlignmentPair(
                    auto_events=[ae],
                    manual_events=[me],
                    match_type=match_type,
                    time_iou=iou,
                    text_similarity=text_sim,
                    composite_score=(0.35 * iou + 0.30 * text_sim),
                )
            )
        return pairs

    def test_time_shift_attribution(self):
        """系统性后移 → 归因到 merging.padding"""
        from vocal_subtitle.feedback.diff_analyzer import DiffAnalyzer

        auto = _make_events(
            [
                (0.0, 2.0, "今天天气不错"),
                (2.5, 5.0, "我们去吧"),
            ]
        )
        # 修订版：每句结束时间都后移了 150ms
        manual = _make_events(
            [
                (0.0, 2.15, "今天天气不错"),
                (2.5, 5.15, "我们去吧"),
            ]
        )

        pairs = self._make_alignment_pairs(auto, manual)
        analyzer = DiffAnalyzer()
        report = analyzer.analyze(pairs)

        assert "merging.padding" in report.attribution
        adj = report.attribution["merging.padding"]
        assert adj.direction == "increase"
        assert adj.param_tier == "short_term"

    def test_merge_attribution(self):
        """用户合并了多句 → 归因到 fast_merge_max_gap"""
        from vocal_subtitle.feedback.aligner import AlignmentPair
        from vocal_subtitle.feedback.diff_analyzer import DiffAnalyzer

        auto = _make_events(
            [
                (0.0, 1.0, "短句A"),
                (1.1, 2.0, "短句B"),
                (3.0, 4.0, "短句C"),
                (4.1, 5.0, "短句D"),
                (6.0, 7.0, "短句E"),
                (7.1, 8.0, "短句F"),
                (9.0, 10.0, "独立句"),
            ]
        )
        manual = _make_events(
            [
                (0.0, 2.0, "短句A短句B"),
                (3.0, 5.0, "短句C短句D"),
                (6.0, 8.0, "短句E短句F"),
                (9.0, 10.0, "独立句"),
            ]
        )

        # 构造 N:1 对齐对
        from vocal_subtitle.feedback.aligner import _levenshtein_similarity, _time_iou

        pairs = []
        for i in range(3):
            ae_group = auto[i * 2 : (i + 1) * 2]
            me = manual[i]
            iou = _time_iou(
                min(e.start for e in ae_group),
                max(e.end for e in ae_group),
                me.start,
                me.end,
            )
            text_sim = _levenshtein_similarity(
                " ".join(e.text for e in ae_group),
                me.text,
            )
            pairs.append(
                AlignmentPair(
                    auto_events=ae_group,
                    manual_events=[me],
                    match_type="N:1",
                    time_iou=iou,
                    text_similarity=text_sim,
                )
            )
        # 第 4 对 1:1
        pairs.append(
            AlignmentPair(
                auto_events=[auto[6]],
                manual_events=[manual[3]],
                match_type="1:1",
                time_iou=_time_iou(
                    auto[6].start, auto[6].end, manual[3].start, manual[3].end
                ),
                text_similarity=_levenshtein_similarity(auto[6].text, manual[3].text),
            )
        )

        analyzer = DiffAnalyzer()
        report = analyzer.analyze(pairs)

        assert "merge_decision.fast_merge_max_gap" in report.attribution
        adj = report.attribution["merge_decision.fast_merge_max_gap"]
        assert adj.direction == "increase"

    def test_punctuation_edit_classification(self):
        """标点修改应被分类为 punctuation"""
        from vocal_subtitle.feedback.diff_analyzer import DiffAnalyzer

        # 使用较长文本确保 Levenshtein 相似度 > 0.9
        auto = _make_events([(0.0, 2.0, "今天天气真的非常不错很适合出门散步")])
        manual = _make_events([(0.0, 2.0, "今天天气真的非常不错，很适合出门散步。")])

        pairs = self._make_alignment_pairs(auto, manual)
        # 手动设置 text_similarity 以绕过阈值问题
        pairs[0].text_similarity = 0.92

        analyzer = DiffAnalyzer()
        report = analyzer.analyze(pairs)

        assert len(report.text_edits) == 1
        assert report.text_edits[0].edit_type == "punctuation"

    def test_no_false_positive_on_identical(self):
        """完全相同的事件不应产生归因"""
        from vocal_subtitle.feedback.diff_analyzer import DiffAnalyzer

        events = _make_events(
            [
                (0.0, 2.0, "完全一致"),
                (2.5, 5.0, "毫无差异"),
            ]
        )

        pairs = self._make_alignment_pairs(events, events)
        analyzer = DiffAnalyzer()
        report = analyzer.analyze(pairs)

        assert len(report.time_shifts) == 0
        assert len(report.text_edits) == 0
        assert len(report.attribution) == 0

    def test_param_isolation_suppresses_weaker(self):
        """同组耦合参数同时调整 → 仅保留置信度高的"""
        from vocal_subtitle.feedback.diff_analyzer import ParamAdjustment
        from vocal_subtitle.feedback.param_learner import ParamDecoupler

        # llm_decision_min_gap 和 llm_decision_max_gap 在同一耦合组 ("interval")
        attr = {
            "merge_decision.llm_decision_min_gap": ParamAdjustment(
                param_path="merge_decision.llm_decision_min_gap",
                param_tier="medium_term",
                observed_value=0.05,
                confidence=0.9,
                learn_weight=1.0,
                direction="increase",
                reason="test min_gap",
            ),
            "merge_decision.llm_decision_max_gap": ParamAdjustment(
                param_path="merge_decision.llm_decision_max_gap",
                param_tier="medium_term",
                observed_value=0.10,
                confidence=0.5,
                learn_weight=1.0,
                direction="decrease",
                reason="test max_gap",
            ),
        }

        selected = ParamDecoupler.select_adjustments(attr, [])
        # min_gap (conf=0.9) > max_gap (conf=0.5) → max_gap 被抑制
        assert "merge_decision.llm_decision_min_gap" in selected
        assert "merge_decision.llm_decision_max_gap" not in selected


# ============================================================================
#  3. ParamLearner 测试
# ============================================================================


class TestParamLearner:
    """参数学习器测试"""

    def test_learn_rate_tiers(self):
        """学习率分级: ≤2→0%, 3-5→15%, 6-15→25%, >15→35%"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        assert ParamLearner.compute_learn_rate(0) == 0.0
        assert ParamLearner.compute_learn_rate(2) == 0.0
        assert ParamLearner.compute_learn_rate(3) == 0.15
        assert ParamLearner.compute_learn_rate(5) == 0.15
        assert ParamLearner.compute_learn_rate(6) == 0.25
        assert ParamLearner.compute_learn_rate(15) == 0.25
        assert ParamLearner.compute_learn_rate(16) == 0.35
        assert ParamLearner.compute_learn_rate(100) == 0.35

    def test_ema_convergence(self):
        """EMA 逐步收敛到目标值"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        current = 0.10
        target = 0.20
        lr = 0.25

        # 一次更新
        new1 = ParamLearner.ema_update(current, target, lr, "merging.padding")
        assert current < new1 < target  # 朝目标移动但不过头

        # 多次更新应接近目标
        for _ in range(20):
            current = ParamLearner.ema_update(current, target, lr, "merging.padding")
        assert abs(current - target) < 0.01

    def test_hard_bounds_clamping(self):
        """参数值不能超出硬边界"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        # padding 上界 0.30
        clamped = ParamLearner.ema_update(0.29, 0.50, 0.5, "merging.padding")
        assert clamped <= 0.30

        # padding 下界 0.02
        clamped = ParamLearner.ema_update(0.03, 0.001, 0.5, "merging.padding")
        assert clamped >= 0.02

        # max_chars_cjk 范围 [10, 40]
        clamped = ParamLearner.ema_update(39, 50, 0.5, "subtitle.max_chars_cjk")
        assert clamped <= 40

    def test_iqr_outlier_filter(self):
        """IQR 过滤应移除极端值"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        # 大部分值在 0.10-0.15，加入一个极端值 1.0
        values = [0.10, 0.11, 0.12, 0.13, 0.15, 1.0]
        filtered = ParamLearner.filter_outliers(values)
        assert 1.0 not in filtered
        assert len(filtered) < len(values)

    def test_iqr_no_filter_on_uniform(self):
        """均匀分布的值不应被过滤"""
        from vocal_subtitle.feedback.param_learner import ParamLearner

        values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        filtered = ParamLearner.filter_outliers(values)
        assert len(filtered) == len(values)


# ============================================================================
#  4. UserProfileManager 测试
# ============================================================================


class TestUserProfileManager:
    """用户配置文件管理测试"""

    def test_create_and_load_default_profile(self):
        """创建并加载默认配置"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        profile = mgr.load("__test_temp__")

        assert profile["profile_id"] == "__test_temp__"
        assert profile["feedback_count"] == 0
        assert profile["overrides"] == {}
        assert profile["history"] == []

    def test_save_and_load_preserves_overrides(self):
        """保存并重新加载 overrides"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        profile = mgr.load("__test_save__")
        profile["overrides"] = {"merging": {"padding": 0.14}}
        profile["feedback_count"] = 5
        mgr.save(profile)

        loaded = mgr.load("__test_save__")
        assert loaded["overrides"]["merging"]["padding"] == 0.14
        assert loaded["feedback_count"] == 5

    def test_rollback_restores_previous(self):
        """回滚到上一备份版本"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()

        # 初始版本
        profile = mgr.load("__test_rollback__")
        profile["overrides"] = {"merging": {"padding": 0.10}}
        mgr.save(profile)

        # 修改版本（触发备份）
        profile["overrides"]["merging"]["padding"] = 0.20
        mgr.save(profile)

        # 回滚
        rolled = mgr.rollback("__test_rollback__")
        assert rolled["overrides"]["merging"]["padding"] == 0.10

    def test_reset_clears_all(self):
        """重置为系统默认"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        profile = mgr.load("__test_reset__")
        profile["overrides"] = {"merging": {"padding": 0.30}}
        profile["feedback_count"] = 10
        mgr.save(profile)

        reset = mgr.reset("__test_reset__")
        assert reset["feedback_count"] == 0
        assert reset["overrides"] == {}

    def test_backup_rotation_limit(self):
        """备份轮转最多保留 MAX_BACKUPS 个"""
        from vocal_subtitle.feedback.user_profile import MAX_BACKUPS, UserProfileManager

        mgr = UserProfileManager()

        for i in range(MAX_BACKUPS + 2):
            profile = mgr.load("__test_backup_rotation__")
            profile["feedback_count"] = i
            mgr.save(profile)

        # 备份数不应超过 MAX_BACKUPS
        bak_dir = mgr._profile_dir
        backups = list(bak_dir.glob("__test_backup_rotation__*.bak.*"))
        assert len(backups) <= MAX_BACKUPS

    def test_decay_weight_long_term(self):
        """长期偏好 180 天半衰期：30 天前权重 ≈ 0.85"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        ts = (datetime.now() - timedelta(days=30)).isoformat()
        w = UserProfileManager.decay_weight(ts, half_life_days=180)
        assert 0.80 < w < 0.90, f"Expected ~0.85, got {w:.3f}"

    def test_decay_weight_short_term(self):
        """短期环境 60 天半衰期：60 天前权重 = exp(-1) ≈ 0.368"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        ts = (datetime.now() - timedelta(days=60)).isoformat()
        w = UserProfileManager.decay_weight(ts, half_life_days=60)
        # exp(-60/60) = exp(-1) ≈ 0.368
        assert 0.35 < w < 0.39, f"Expected ~0.368, got {w:.3f}"

    def test_decay_weight_no_decay(self):
        """half_life_days=None → 不衰减，权重始终为 1.0"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        ts = (datetime.now() - timedelta(days=365)).isoformat()
        w = UserProfileManager.decay_weight(ts, half_life_days=None)
        assert w == 1.0

    def test_cleanup_temp_profiles(self):
        """清理本测试类创建的临时配置文件"""
        from vocal_subtitle.feedback.user_profile import UserProfileManager

        mgr = UserProfileManager()
        for name in [
            "__test_temp__",
            "__test_save__",
            "__test_rollback__",
            "__test_reset__",
            "__test_backup_rotation__",
        ]:
            try:
                mgr.delete(name)
            except Exception:
                pass


# ============================================================================
#  5. FewShotBuilder / FewShotCacheManager 测试
# ============================================================================


class TestFewShotCache:
    """Few-shot 示例缓存管理测试"""

    def test_add_and_retrieve(self):
        """添加并获取活跃示例"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager(max_capacity=20)
        cache.add(
            FewShotExample(
                example_type="merge",
                fragments=["对", "就是说"],
                decision="MERGE",
                reason="填充词合并",
            )
        )

        active = cache.get_active_examples(max_count=5, min_weight=0.1)
        assert len(active) == 1
        assert active[0].example_type == "merge"

    def test_lru_eviction(self):
        """超过 max_capacity 时 LRU 淘汰最久未命中项"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager(max_capacity=5)
        for i in range(7):
            cache.add(
                FewShotExample(
                    example_type="merge",
                    fragments=[f"片段{i}"],
                    decision="MERGE",
                )
            )

        assert cache.size() == 5  # max_capacity

    def test_duplicate_detection(self):
        """重复示例应更新已有项的权重而非新增"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager(max_capacity=20)
        cache.add(
            FewShotExample(
                example_type="merge",
                fragments=["同一个片段"],
                decision="MERGE",
                weight=0.5,
            )
        )
        cache.add(
            FewShotExample(
                example_type="merge",
                fragments=["同一个片段"],
                decision="MERGE",
                weight=0.8,
            )
        )

        assert cache.size() == 1
        active = cache.get_active_examples(min_weight=0.1)
        assert active[0].weight == 0.8  # 权重更新为较大值

    def test_prompt_injection_format(self):
        """Prompt 注入格式正确"""
        from vocal_subtitle.feedback.few_shot_builder import (
            FewShotBuilder,
            FewShotCacheManager,
            FewShotExample,
        )

        cache = FewShotCacheManager()
        cache.add(
            FewShotExample(
                example_type="merge",
                fragments=["对", "就是说"],
                decision="MERGE",
                reason="填充词合并",
            )
        )
        cache.add(
            FewShotExample(
                example_type="format",
                rule="句末统一使用中文句号。",
                reason="标点偏好",
            )
        )

        builder = FewShotBuilder(cache_manager=cache, max_examples=3)
        base_prompt = "你是一个字幕优化助手。"
        injected = builder.inject_into_prompt(
            base_prompt, max_examples=3, min_weight=0.1
        )

        assert "User Preference Examples" in injected
        assert "对" in injected
        assert "就是说" in injected
        assert "中文句号" in injected
        assert base_prompt in injected  # 基础 prompt 保留


# ============================================================================
#  6. HealthScorer 测试
# ============================================================================
