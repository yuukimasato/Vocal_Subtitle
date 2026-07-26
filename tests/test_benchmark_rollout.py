from pathlib import Path

from scripts.run_benchmarks import (
    BenchmarkSummary,
    SceneResult,
    evaluate_rollout_eligibility,
    summary_to_dict,
)


def test_rollout_is_ineligible_without_real_scenes(tmp_path):
    summary = BenchmarkSummary()

    eligible, reasons = evaluate_rollout_eligibility(
        {}, [], summary, min_scenes=3, target_metrics={"end_mae": 300}
    )

    assert eligible is False
    assert any("至少 3 个场景" in reason for reason in reasons)


def test_rollout_requires_global_path_and_scene_categories(tmp_path):
    scene_dirs = {}
    results = []
    for name in ("studio_monologue", "meeting_3_speakers", "music_voiceover"):
        scene_dir = tmp_path / name
        scene_dir.mkdir()
        (scene_dir / "metadata.yaml").write_text("scene: test\n", encoding="utf-8")
        scene_dirs[name] = scene_dir
        results.append(
            SceneResult(
                scene=name,
                success=True,
                output_dir=scene_dir,
                asr_path="legacy_degraded" if name == "music_voiceover" else "global",
            )
        )

    summary = BenchmarkSummary(
        total_scenes=3,
        success_scenes=3,
        avg_start_mae_ms=100,
        avg_end_mae_ms=200,
    )
    eligible, reasons = evaluate_rollout_eligibility(
        scene_dirs,
        results,
        summary,
        min_scenes=3,
        target_metrics={"end_mae": 300, "start_mae": 200},
    )

    assert eligible is False
    assert any("没有使用 global ASR" in reason for reason in reasons)


def test_summary_serializes_rollout_state(tmp_path):
    summary = BenchmarkSummary(
        rollout_eligible=False,
        eligibility_reasons=["no_valid_scenes"],
        results=[
            SceneResult(
                scene="demo",
                success=False,
                output_dir=Path(tmp_path),
                fallback_category="dependency_unavailable",
            )
        ],
    )

    payload = summary_to_dict(summary)

    assert payload["rollout_eligible"] is False
    assert payload["eligibility_reasons"] == ["no_valid_scenes"]
    assert payload["scenes"][0]["fallback_category"] == "dependency_unavailable"
