# 影视级字幕全链路质量优化代码实况审计

**日期：** 2026-07-27
**审计基线：** `84ed7e6` + 当前工作区修复
**依据：** `2026-07-27-global-subtitle-quality-design.md` 与实施计划

## 结论

原有实现的模块和测试覆盖面较完整，但实际生产接线未完全符合计划，旧验收报告有三项结论偏早：

1. `finalize_subtitle_events()` 只有孤立测试，Pipeline 没有调用；Builder 仍会在导出时隐式合并、拆分和去重。
2. global ASR 主路径只执行 `allocate_words()`，随后直接用原始 ASR 时间构造事件，没有执行词级边界仲裁；因此不能声称全部冻结边界都有 accepted `BoundaryDecision`。
3. 计划要求的 `scripts/evaluate_acoustic_gold.py` 缺失，质量 manifest 把巧乐兹音频错误映射到 181 的人工字幕。

本轮已修复上述代码级偏差。自动测试和本地 WebUI 烟测通过，但影视级声学指标仍待人工词级金标准与真实模型全样本复验。

## 已修复

- Pipeline 离线和流式路径均在导出前进入唯一 finalizer；统计、返回事件和主字幕使用同一 finalized list。
- 启用 LLM 优化时，`subtitle_path` 指向最终 LLM 主结果，不再返回 pre-LLM 干净版作为主字幕。
- `SubtitleBuilder` 只做格式渲染和换行，不改变逻辑 cue 数、时间、编号或调用方对象。
- full-pipeline cache 恢复为 `SubtitleEvent` 对象，并执行 ASR path 兼容检查。
- `WordAllocation` 正式定义 aligned time、physical bin、confidence、status、起止决策和 evidence IDs，不再动态注入冻结对象。
- 新增独立 `boundary_arbiter.py`，确保硬约束先于评分且保留结构化拒绝原因；global 路径改为 `allocate -> align -> build_events -> SubtitleEvent`。
- 物理 bin 外包络只保存在 `physical_bin_start/end`，不再伪装成词级物理时间。
- accepted 边界决策写入 `revision_trace`；未对齐的局部恢复明确记录 `timing_degraded`。
- WebUI 编辑重写和导出使用 `SubtitleEvent.from_dict()`，保留 physical/source/speaker/revision provenance。
- 新增声学金标准评估器，支持严格 schema、median/P95、切头/切尾率、speaker error 和 CI 非零门禁。
- manifest 用匹配的 QS 音频/人工 ASS 替换错误巧乐兹映射，并补充语言、speaker 数和场景标签。

## 自动验证

```text
git diff --check                                      PASS
python -m compileall -q vocal_subtitle scripts        PASS
import vocal_subtitle heavy modules                   []
pytest --collect-only                                 805 tests
pytest -q                                             805 passed, 8 warnings
```

8 个 warning 均为既有依赖弃用或空数组/短音频数值 warning，没有测试失败。

本地浏览器烟测：

- 首页加载成功，profiles/cache/device/history 请求均为 HTTP 200。
- 历史任务详情请求 `/api/history/90342781` 为 HTTP 200。
- Swagger 加载成功，无 console/page error。
- OpenAPI 共 41 个路径，包含 `/api/subtitle/{task_id}/export`。

## 仍受阻的发布门禁

1. 尚无人工听辨复核的 `acoustic-gold-v1` 词级 onset/offset/speaker JSON，不能验证 median `<=80ms`、P95 `<=160ms` 或切头切尾率 `<=0.5%`。
2. 本轮没有重新运行 8 个真实媒体的模型推理、LLM 路径和逐任务下载对照；已有 dogfood 产物来自修复前代码。
3. 当前环境的 global WhisperX 路径仍可能因依赖/GPU 条件降级，真实 global rollout 尚未证明。
4. QS 标签覆盖 interruption 类别，但不是经人工标注的同期重叠声学金标准，不能替代真实 overlap 样本。
5. `BoundaryArbiter` 的硬约束核心已接入，但当前词对齐尚未把真实 RMS 梯度、VAD 概率、FFmpeg 边界距离和 `LocalNoiseProfile` 逐候选注入评分；任务 6 目前是基础可审计版本，不是设计中的完整多源仲裁。

因此当前状态是：**代码契约和无模型回归通过；声学发布门禁待人工金标准，真实模型/WebUI 全样本验收待重跑。**
