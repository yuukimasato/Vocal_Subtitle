"""用户反馈自适应学习模块 (Phase 5)

基于用户修订字幕的自适应参数学习引擎。

核心流程:
    1. 对齐 (Aligner): 自动版字幕 vs 用户修订版逐行对齐
    2. 差异分析 (DiffAnalyzer): 分类修改行为 → 映射到参数调整
    3. 参数学习 (ParamLearner): EMA 增量更新用户配置
    4. 指纹提取 (Fingerprinter): 音频声学特征 → 相似音频匹配
    5. 健康度评分 (HealthScorer): 字幕质量综合评分
    6. 影子模式 (ShadowMode): 新参数零风险后台验证
    7. 冲突检测 (ConflictDetector): 参数震荡检测与交互引导
    8. 影响预估 (ImpactEstimator): 参数变更效果预估

使用示例:
    from vocal_subtitle.feedback import (
        SubtitleAligner, DiffAnalyzer, ParamLearner,
        UserProfileManager, FewShotBuilder,
        AudioFingerprinter, MahalanobisMatcher,
        compute_health_score, ShadowModeEvaluator,
        ConflictDetector, ImpactEstimator,
        AlignmentError,
    )

    # 对齐
    aligner = SubtitleAligner()
    pairs = aligner.align(auto_events, manual_events)

    # 分析差异
    analyzer = DiffAnalyzer()
    report = analyzer.analyze(pairs)

    # 学习参数
    profile_mgr = UserProfileManager()
    learner = ParamLearner(profile_mgr)
    learner.learn_from_diff(report)

    # 音频指纹
    fingerprinter = AudioFingerprinter()
    fp = fingerprinter.extract(audio_path)
    match = fingerprinter.find_similar(fp)
"""

from .aligner import AlignmentError, AlignmentPair, SemanticScorer, SubtitleAligner
from .audio_fingerprint import AudioFingerprint, AudioFingerprinter, MahalanobisMatcher
from .conflict_detector import ConflictDetector, ConflictReport, OscillationEntry
from .diff_analyzer import DiffAnalyzer, DiffReport, ParamAdjustment
from .few_shot_builder import FewShotBuilder, FewShotCacheManager
from .health_scorer import (
    HealthScoreResult,
    compute_health_score,
    compute_health_score_from_pairs,
    health_score_result,
    should_auto_rollback,
)
from .impact_estimator import ImpactEstimator, ImpactPrediction
from .param_learner import ParamDecoupler, ParamLearner
from .shadow_mode import ShadowEvaluation, ShadowModeEvaluator, ShadowRunResult
from .user_profile import UserProfileManager

__all__ = [
    # Aligner
    "SubtitleAligner",
    "AlignmentError",
    "AlignmentPair",
    "SemanticScorer",
    # Audio Fingerprint
    "AudioFingerprint",
    "AudioFingerprinter",
    "MahalanobisMatcher",
    # Conflict Detector
    "ConflictDetector",
    "ConflictReport",
    "OscillationEntry",
    # DiffAnalyzer
    "DiffAnalyzer",
    "DiffReport",
    "ParamAdjustment",
    # Health Scorer
    "HealthScoreResult",
    "compute_health_score",
    "compute_health_score_from_pairs",
    "health_score_result",
    "should_auto_rollback",
    # Impact Estimator
    "ImpactEstimator",
    "ImpactPrediction",
    # ParamLearner
    "ParamLearner",
    "ParamDecoupler",
    # Shadow Mode
    "ShadowEvaluation",
    "ShadowModeEvaluator",
    "ShadowRunResult",
    # UserProfile
    "UserProfileManager",
    # FewShot
    "FewShotBuilder",
    "FewShotCacheManager",
]
