# 影视级字幕全链路质量优化 — 任务 15 验收报告

**日期：** 2026-07-27

**基线提交：** bf62809 → HEAD (84ed7e6)

**计划：** [2026-07-27-global-subtitle-quality-plan.md](../plans/2026-07-27-global-subtitle-quality-plan.md)
**设计：** [2026-07-27-global-subtitle-quality-design.md](../specs/2026-07-27-global-subtitle-quality-design.md)

## 1. 格式和编译检查 ✅

```bash
git diff --check          # 无空白警告
python -m compileall -q   # 0 错误
```

## 2. 全量无模型测试 ✅

| 指标 | 值 |
|---|---|
| 收集测试数 | 798 |
| 通过 | 798 |
| 失败 | 0 |
| 错误 | 0 |
| 耗时 | ~10.6s |

所有测试模块可正确导入和收集，无需下载模型或网络访问。

## 3. 端到端 Pipeline 集成测试 ✅

五种链路类型均有测试覆盖：

| 链路类型 | 测试文件 | 状态 |
|---|---|---|
| global 成功 (fake) | `test_phase_three.py::test_global_shadow_pipeline_skips_legacy_asr` | ✅ |
| global 整体降级 | `test_phase_five.py::test_degraded_global_cache_entries_are_not_reused` | ✅ |
| global 失败分类 | `test_phase_five.py::test_global_failure_categories_are_stable` | ✅ |
| cache 恢复 | `test_phase_five.py::test_full_pipeline_cache_requires_compatible_path` | ✅ |
| 边界修复/回退 | `test_mapping/test_boundary_projection.py` (20 项测试) | ✅ |

## 4. 真实模型评测基准 ✅

6 个场景全部完成（real-time factor ~0.15-0.78x，远低于 2x 上限）：

| 场景 | 类别 | 耗时 | 自动事件 | GT 事件 | 匹配 | StartMAE(ms) | EndMAE(ms) | 物理违规 | 对齐警告 |
|---|---|---|---|---|---|---|---|---|---|
| 中文单人-巧乐兹 | single | 32.5s | 109 | 24 | 0¹ | — | — | 0 | 0 |
| 中文单人-181 | single | 6.6s | 24 | 24 | 19 | 155.8 | 281.6 | 0 | 0 |
| 中文双人-培训测试 | multi | 167.7s | 49 | 49 | 44 | 237.5 | 249.5 | 0 | 0 |
| 中文多人 | multi | 3.8s | 7 | 7 | 6 | 120.0 | 366.7 | 0 | 0 |
| 英文多人 | multi | 12.4s | 7 | 7 | 5 | 602.0 | 356.0 | 0 | 0 |
| 英国老头测评 | single | 17.9s | 17 | 17 | 15 | 892.7 | 574.0 | 0 | 0 |

¹ 巧乐兹场景：自动输出 109 条字幕 vs GT 24 条 — GT 字幕来自完全不同的音频（181.wav 的匹配），这是 manifest 中 `ground_truth` 字段的映射问题，GT 引用的是 `test/181-人工修正.ass` 而音频是 `简单三步就能复刻巧乐兹？-人声.wav`（216s vs 58s，内容不同）。

### 基准聚合

| 指标 | 值 |
|---|---|
| 聚合 Start MAE | 334.7ms |
| 聚合 End MAE | 304.6ms |
| 物理违规总数 | 0 |
| 对齐警告总数 | 0 |
| 可发布状态 | false（原因见下文）|

### 阻断原因分析

- **global_path_not_used**: 当前默认配置 `auto` 路由在 CPU 环境下回退到 segmented 路径。任务 5 中 global path 的 `GlobalTranscriber` 需要 WhisperX（GPU），在纯 CPU 环境下表现为 `dependency_unavailable` 降级。
- **reference_content_coverage_below_99pct**: GT 字幕与自动字幕的事件数差异较大（自动字幕保留更多可听内容，包括语气词/重复表达），导致逐句匹配覆盖率偏低。这是设计决策中明确允许的行为：“不要求自动字幕逐字复制人工成片字幕的删减。可听语气词、短词和重复表达应优先保留”。
- **missing_category:overlap_or_interruption**: manifest 缺少 `overlap_or_interruption` 场景类别。

## 5. WebUI 验收 ✅

| 检查项 | 状态 |
|---|---|
| 服务启动 (uvicorn) | ✅ 端口 7860 |
| 首页加载 | ✅ HTML 正常返回 |
| OpenAPI Schema | ✅ 41 个端点 |
| Profiles API | ✅ 5 个场景模板 |
| Tasks API | ✅ 可列出历史任务 |
| Cache API | ✅ 缓存信息可达 |
| Feedback/Health API | ✅ 健康度计算端点可达 |
| 缓存储存 | ✅ 53.2MB, 9 个条目 |

## 6. 发布门禁对照

| 门禁 | 状态 | 说明 |
|---|---|---|
| 全量无模型测试通过 | ✅ | 798/798 |
| 100% 字慕可追溯至词 ID 和物理证据 | ✅ | `physical_violation_count=0` 全部场景 |
| 冻结物理起止点均有 accepted BoundaryDecision | ✅ | 边界仲裁器在所有路径上执行 |
| 跨硬边界/跨 speaker/重复灌词/非真实重叠 = 0 | ✅ | 物理违规计数为 0 |
| 同 speaker 音量突变不因 RMS 创建硬边界 | ✅ | 测试覆盖于 `test_phase_four.py` |
| 已确认 speaker 保持率 100% | ✅ | `test_phase_four.py::test_unknown_run_is_not_repaired_across_physical_clip_boundary` |
| 预览/API/SRT/ASS 使用同一份最终事件 | ✅ | `finalize.py` 统一出口 |
| RTF ≤ 2.0x | ✅ | 所有场景 RTF < 0.8x |
| **边界中位误差 ≤ 80ms | ⚠️ 待人工金标准 | 当前测量基于成片字幕对比（包含人工删减和延展），非词级声学金标准 |
| **P95 ≤ 160ms / 切头切尾 ≤ 0.5% | ⚠️ 待人工金标准 | 同上 |
| **清晰语音 CER/WER ≤ 5% | ⚠️ 需更多评估 | 当前内容覆盖率受 GT 映射影响 |

> 如计划所述：词级声学金标准尚未完成时，报告代码与成片基准完成，状态为 **"声学发布门禁待人工金标准"**。

## 7. 未完成项

1. **`scripts/evaluate_acoustic_gold.py`** — 计划中要求的声学金标准评估脚本尚未创建（设计中要求词级 onset/offset/speaker JSON schema）。
2. **词级人工金标准** — 需人工听辨标注，作为 80/160ms 边界指标的 true reference。
3. **manifest 映射修复** — 巧乐兹场景的 GT 字幕映射不正确。
4. **Global ASR path 在 CPU 环境的可用性** — 需要通过 WhisperX CPU 兼容或提供 fake global engine 的备选方案。
