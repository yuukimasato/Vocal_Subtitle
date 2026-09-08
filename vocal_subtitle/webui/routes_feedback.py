"""Feedback and adaptive-learning API routes."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..config import FeedbackConfig
from .models import *
from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/feedback/profiles")
async def list_feedback_profiles():
    """列出所有用户配置及其学习统计"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile_names = mgr.list_profiles()

    profiles = []
    for name in profile_names:
        p = mgr.load(name)
        profiles.append({
            "profile_id": name,
            "base_profile": p.get("base_profile", "default"),
            "feedback_count": p.get("feedback_count", 0),
            "created_at": p.get("created_at", ""),
            "updated_at": p.get("updated_at", ""),
            "is_active": p.get("is_active", True),
            "overrides": p.get("overrides", {}),
            "history_count": len(p.get("history", [])),
        })

    return {"profiles": profiles, "total": len(profiles)}


@router.get("/feedback/profile/{name}")
async def get_feedback_profile(name: str):
    """获取指定用户配置的详细信息"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(name)

    if not profile or profile.get("profile_id") != name:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")

    return {
        "profile_id": profile.get("profile_id"),
        "base_profile": profile.get("base_profile", "default"),
        "description": profile.get("description", ""),
        "feedback_count": profile.get("feedback_count", 0),
        "created_at": profile.get("created_at", ""),
        "updated_at": profile.get("updated_at", ""),
        "is_active": profile.get("is_active", True),
        "overrides": profile.get("overrides", {}),
        "fingerprint": profile.get("fingerprint", {}),
        "history": profile.get("history", [])[-10:],  # 最近 10 条
        "few_shot_examples_count": len(profile.get("few_shot_examples", [])),
    }


@router.post("/feedback/profile/{name}/rollback")
async def rollback_feedback_profile(name: str):
    """回滚指定用户配置"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    try:
        profile = mgr.rollback(name)
        return {
            "status": "ok",
            "profile_id": name,
            "updated_at": profile.get("updated_at", ""),
            "feedback_count": profile.get("feedback_count", 0),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No backup found for profile: {name}")


@router.delete("/feedback/profile/{name}")
async def delete_feedback_profile(name: str):
    """删除指定用户配置"""
    from ..config import FeedbackConfig
    from ..feedback import UserProfileManager

    if name == "user_default":
        raise HTTPException(status_code=400, detail="Cannot delete the default profile")

    mgr = UserProfileManager(FeedbackConfig())
    ok = mgr.delete(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Profile not found: {name}")
    return {"status": "ok", "deleted": name}


# ---------------------------------------------------------------------------
# 指纹管理端点 (Phase 5.3)
# ---------------------------------------------------------------------------


@router.get("/feedback/fingerprints", response_model=FingerprintListResponse)
async def list_fingerprints():
    """列出所有音频指纹"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )

    fps = fingerprinter.list_all()
    return FingerprintListResponse(
        fingerprints=[
            FingerprintInfo(
                id=f["id"],
                profile_id=f["profile_id"],
                audio_hash=f["audio_hash"],
                audio_signature=f.get("audio_signature", ""),
                feedback_count=f.get("feedback_count", 1),
                created_at=f.get("created_at", ""),
            )
            for f in fps
        ],
        total=len(fps),
        db_path=str(fingerprinter._db_path),
    )


@router.post("/feedback/fingerprints/match")
async def match_fingerprint(
    audio: UploadFile = File(...),
    profile: str = Form(default="user_default"),
):
    """上传音频，查找匹配的指纹和配置

    Returns:
        匹配结果（含 profile_id 和 confidence）
    """
    import tempfile

    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()

    # 保存音频到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        fingerprinter = AudioFingerprinter(
            distance_method=cfg.fingerprint_distance_method,
            knn_k=cfg.fingerprint_knn_k,
            min_absolute_similarity=cfg.fingerprint_min_absolute_similarity,
            relative_margin=cfg.fingerprint_relative_margin,
        )

        # 拟合匹配器（从已有指纹库）
        fingerprinter.refit_matcher()

        fp = fingerprinter.extract(tmp_path)
        if fp is None:
            return FingerprintMatchResponse(matched=False)

        result = fingerprinter.find_similar(fp)
        if result:
            matched_profile, confidence = result
            return FingerprintMatchResponse(
                matched=True,
                profile_id=matched_profile,
                confidence=round(confidence, 4),
                audio_signature=fp.audio_signature,
            )

        return FingerprintMatchResponse(
            matched=False,
            audio_signature=fp.audio_signature,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.delete("/feedback/fingerprints/{fp_id}")
async def delete_fingerprint(fp_id: int):
    """删除指定指纹"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )
    ok = fingerprinter.delete_by_id(fp_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Fingerprint not found: {fp_id}")
    return {"status": "ok", "deleted": fp_id}


# ---------------------------------------------------------------------------
# 健康度评分端点 (Phase 5.4)
# ---------------------------------------------------------------------------


@router.get("/feedback/health/{profile_name}")
async def get_health_trend(
    profile_name: str,
    limit: int = Query(default=20, le=100),
):
    """获取健康度趋势数据（用于前端趋势图）"""
    from ..config import FeedbackConfig
    from ..feedback import AudioFingerprinter

    cfg = FeedbackConfig()
    fingerprinter = AudioFingerprinter(
        distance_method=cfg.fingerprint_distance_method,
    )

    trend = fingerprinter.get_health_trend(profile_name, limit=limit)
    return {
        "profile_id": profile_name,
        "data_points": len(trend),
        "trend": [
            HealthTrendEntry(
                timestamp=e["timestamp"],
                health_before=e["health_before"],
                health_after=e["health_after"],
                shadow_mode=e["shadow_mode"],
                summary=e["summary"],
            )
            for e in trend
        ],
    }


@router.post("/feedback/health/compute")
async def compute_health(
    auto_subtitle: UploadFile = File(...),
    reference_subtitle: UploadFile = File(...),
):
    """上传自动版与修订版字幕，计算健康度评分

    Returns:
        HealthScoreDetail 包含综合评分和各子项得分
    """
    import tempfile

    from ..feedback.aligner import SubtitleAligner, parse_subtitle_file
    from ..feedback.health_scorer import health_score_result

    # 保存文件
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        auto_path = tmp_dir / (auto_subtitle.filename or "auto.srt")
        ref_path = tmp_dir / (reference_subtitle.filename or "reference.srt")

        auto_path.write_bytes(await auto_subtitle.read())
        ref_path.write_bytes(await reference_subtitle.read())

        # 解析
        auto_events = parse_subtitle_file(auto_path)
        manual_events = parse_subtitle_file(ref_path)

        if not auto_events or not manual_events:
            raise HTTPException(status_code=400, detail="Empty subtitle files")

        # 计算健康度
        result = health_score_result(auto_events, manual_events)

        return HealthScoreDetail(
            overall=result.overall,
            alignment_coverage=result.alignment_coverage,
            semantic_similarity=result.semantic_similarity,
            time_iou=result.time_iou,
            structure_consistency=result.structure_consistency,
            grade=result.grade,
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 影子模式端点 (Phase 5.4)
# ---------------------------------------------------------------------------



@router.get("/feedback/shadow/{profile_name}")
async def get_shadow_status(profile_name: str):
    """获取影子模式状态"""
    from ..feedback import ShadowModeEvaluator

    evaluator = state.shadow_evaluators.get(profile_name)
    if evaluator is None:
        return ShadowModeStatus(
            enabled=False,
            total_runs=0,
            reason="Shadow mode not active for this profile",
        )

    eval_result = evaluator.should_upgrade()
    return ShadowModeStatus(
        enabled=True,
        total_runs=evaluator.run_count,
        current_mean_health=eval_result.current_mean_health,
        shadow_mean_health=eval_result.shadow_mean_health,
        health_delta=eval_result.health_delta,
        recommendation=eval_result.recommendation,
        reason=eval_result.reason,
        runs=evaluator.to_dict().get("runs", []),
    )


@router.post("/feedback/shadow/{profile_name}/toggle")
async def toggle_shadow_mode(profile_name: str, body: ShadowModeToggleRequest):
    """启用/停用影子模式"""
    from ..config import FeedbackConfig
    from ..feedback import ShadowModeEvaluator

    if body.enabled:
        cfg = FeedbackConfig()
        evaluator = ShadowModeEvaluator(
            min_shadow_runs=cfg.shadow_min_runs,
            upgrade_threshold=cfg.shadow_upgrade_threshold,
            max_shadow_duration_days=cfg.shadow_max_duration_days,
        )
        state.shadow_evaluators[profile_name] = evaluator
        return {
            "status": "ok",
            "enabled": True,
            "message": f"Shadow mode enabled for '{profile_name}'",
        }
    else:
        state.shadow_evaluators.pop(profile_name, None)
        return {
            "status": "ok",
            "enabled": False,
            "message": f"Shadow mode disabled for '{profile_name}'",
        }


@router.post("/feedback/shadow/{profile_name}/record")
async def record_shadow_run(
    profile_name: str,
    health_current: float = Form(...),
    health_shadow: float = Form(...),
):
    """记录一次影子运行结果"""
    from ..feedback import ShadowModeEvaluator, ShadowRunResult

    evaluator = state.shadow_evaluators.get(profile_name)
    if evaluator is None:
        cfg = FeedbackConfig()
        evaluator = ShadowModeEvaluator(
            min_shadow_runs=cfg.shadow_min_runs,
            upgrade_threshold=cfg.shadow_upgrade_threshold,
            max_shadow_duration_days=cfg.shadow_max_duration_days,
        )
        state.shadow_evaluators[profile_name] = evaluator

    evaluator.add_run(ShadowRunResult(
        health_current=health_current,
        health_shadow=health_shadow,
    ))

    eval_result = evaluator.should_upgrade()
    return {
        "status": "ok",
        "total_runs": evaluator.run_count,
        "should_upgrade": eval_result.should_upgrade,
        "recommendation": eval_result.recommendation,
        "reason": eval_result.reason,
    }


# ---------------------------------------------------------------------------
# 冲突检测端点 (Phase 5.4)
# ---------------------------------------------------------------------------


@router.get("/feedback/conflicts/{profile_name}")
async def detect_conflicts(profile_name: str):
    """检测参数冲突（震荡）"""
    from ..config import FeedbackConfig
    from ..feedback import ConflictDetector, UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(profile_name)
    history = profile.get("history", [])

    detector = ConflictDetector(window=5)
    reports = detector.detect_all_oscillations(history)

    return {
        "profile_id": profile_name,
        "conflicts": [
            ConflictInfo(
                param_path=r.param_path,
                is_oscillating=r.is_oscillating,
                oscillation_count=r.oscillation_count,
                severity=r.severity,
                recommended_action=r.recommended_action,
                possible_causes=r.possible_causes,
                suggested_actions=r.suggested_actions,
                entries=[
                    {
                        "timestamp": e.timestamp,
                        "direction": e.direction,
                        "delta": e.delta,
                        "summary": e.summary,
                    }
                    for e in r.entries
                ],
            )
            for r in reports
        ],
        "total": len(reports),
    }


@router.post("/feedback/conflicts/{profile_name}/resolve")
async def resolve_conflict(profile_name: str, body: ConflictResolutionRequest):
    """解决参数冲突"""
    from ..config import FeedbackConfig
    from ..feedback import ConflictDetector, UserProfileManager

    detector = ConflictDetector(window=5)
    mgr = UserProfileManager(FeedbackConfig())

    # 查找冲突
    profile = mgr.load(profile_name)
    history = profile.get("history", [])
    reports = detector.detect_all_oscillations(history)

    target = None
    for r in reports:
        if r.param_path == body.param_path:
            target = r
            break

    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"No oscillation detected for param: {body.param_path}",
        )

    result = detector.resolve(target, body.action)

    # 如果锁定，写入 locked_params 到 profile
    if body.action == "lock":
        locked = profile.get("locked_params", [])
        if body.param_path not in locked:
            locked.append(body.param_path)
        profile["locked_params"] = locked
        mgr.save(profile)

    return result


# ---------------------------------------------------------------------------
# 影响预估端点 (Phase 5.4)
# ---------------------------------------------------------------------------


@router.post("/feedback/impact/preview")
async def preview_impact(profile_name: str = "user_default"):
    """预览当前待调整参数的影响

    基于用户配置中最近的差异分析结果。
    """
    from ..config import FeedbackConfig
    from ..feedback import ImpactEstimator, UserProfileManager

    mgr = UserProfileManager(FeedbackConfig())
    profile = mgr.load(profile_name)
    overrides = profile.get("overrides", {})
    history = profile.get("history", [])

    if not history:
        return {"impacts": [], "message": "尚无学习历史"}

    # 从最近的历史记录中获取归因
    last_entry = history[-1]
    latest_adjustments = last_entry.get("adjustments", {})
    if not latest_adjustments:
        return {"impacts": [], "message": "最近一次反馈无参数调整"}

    # 构建简易 ParamAdjustment 列表
    from ..feedback.diff_analyzer import ParamAdjustment

    adj_map = {}
    for param_path, (old_val, new_val) in latest_adjustments.items():
        delta = new_val - old_val
        adj_map[param_path] = ParamAdjustment(
            param_path=param_path,
            param_tier="medium_term",
            observed_value=abs(delta),
            confidence=0.8,
            learn_weight=1.0,
            direction="increase" if delta > 0 else "decrease",
            reason="Historical adjustment",
        )

    estimator = ImpactEstimator()
    impacts = estimator.estimate(adj_map, overrides)

    return {
        "profile_id": profile_name,
        "impacts": [
            ImpactPredictionInfo(
                param_path=ip.param_path,
                current_value=ip.current_value,
                new_value=ip.new_value,
                delta=ip.delta,
                delta_pct=ip.delta_pct,
                summary=ip.summary,
                avg_duration_change_pct=ip.avg_duration_change_pct,
                merge_frequency_change_pct=ip.merge_frequency_change_pct,
                split_frequency_change_pct=ip.split_frequency_change_pct,
                end_truncation_change_pct=ip.end_truncation_change_pct,
                total_line_count_change_pct=ip.total_line_count_change_pct,
                confidence_low=ip.confidence_low,
                confidence_high=ip.confidence_high,
            )
            for ip in impacts
        ],
        "total": len(impacts),
    }


# ---------------------------------------------------------------------------
# 反馈审核队列端点 (FEEDBACK_LOOP.md §7)
# ---------------------------------------------------------------------------


@router.get("/feedback/review-queue")
async def feedback_review_queue(status: str = Query(default="pending")):
    """列出待审核反馈样本。

    对应 FEEDBACK_LOOP.md §7:
      GET /api/feedback/review-queue?status=pending
      → [{sample_id, diff_summary, confidence, submitted_at}]
    """
    from ..feedback.sample_manager import FeedbackSampleManager

    mgr = FeedbackSampleManager()
    queue = mgr.list_pending_review()

    if status != "pending":
        queue = [
            item for item in queue
            if item.get("status", "pending") == status
        ]

    return {
        "samples": queue,
        "total": len(queue),
        "status": status,
    }


@router.get("/feedback/review/{sample_id}")
async def feedback_review_detail(sample_id: str):
    """获取单个反馈样本的完整审核详情。

    返回原始字幕、人工修订、对齐信息和审核状态。
    """
    from ..feedback.sample_manager import FeedbackSampleManager

    mgr = FeedbackSampleManager()
    sample = mgr.get(sample_id)

    if sample is None:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    return sample


@router.post("/feedback/review/{sample_id}")
async def feedback_review(sample_id: str, result: str = Form(...), notes: str = Form(default="")):
    """审核一个反馈样本。

    对应 FEEDBACK_LOOP.md §7:
      POST /api/feedback/review/{sample_id}
      Body: {result: "accepted"|"rejected"|"disputed", notes: "..."}
    """
    from ..feedback.sample_manager import FeedbackSampleManager

    valid_results = {"accepted", "rejected", "disputed"}
    if result not in valid_results:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid review result: {result!r}. Must be one of {sorted(valid_results)}",
        )

    mgr = FeedbackSampleManager()
    success = mgr.review(sample_id, result, reviewer="webui", notes=notes)

    if not success:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    return {
        "status": "ok",
        "sample_id": sample_id,
        "result": result,
    }


# ---------------------------------------------------------------------------
# D3 分层抽样 (FEEDBACK_LOOP.md §5, StratifiedSampler)
# ---------------------------------------------------------------------------


@router.post("/feedback/d3-sample")
async def feedback_d3_sample(
    strata: str = Form(default="language,scene,speaker_count"),
    description: str = Form(default=""),
):
    """从已审核的 D2 样本中执行分层抽样，冻结为 D3 版本。

    对应 FEEDBACK_LOOP.md §5 影子模式验证流程:
      POST /api/feedback/d3-sample
      Body: {strata: "language,scene,speaker_count", description: "D3 分层抽样"}

    Returns:
      {status, version_id, sample_count, balance_warnings, freeze_date}
    """
    from ..feedback.sample_manager import FeedbackSampleManager
    from ..feedback.stratified_sampler import StratifiedSampler

    strata_list = [s.strip() for s in strata.split(",")]

    mgr = FeedbackSampleManager()
    sampler = StratifiedSampler()

    pool = sampler.pool_from_manager(mgr)
    if not pool:
        raise HTTPException(
            status_code=400,
            detail="D2 候选池中没有符合条件的样本（需要 review=accepted, anonymization=deidentified, consent≠local）",
        )

    plan = sampler.build_plan(
        pool, strata=strata_list,
        description=description or "WebUI D3 分层抽样",
    )

    result = sampler.sample(pool, plan)
    version = sampler.freeze(result, description=description)

    return {
        "status": "ok",
        "version_id": version.version_id,
        "sample_count": version.sample_count,
        "freeze_date": version.freeze_date,
        "pool_size": len(pool),
        "selection_count": result.sample_count,
        "balance_warnings": result.balance.get("warnings", []),
    }
