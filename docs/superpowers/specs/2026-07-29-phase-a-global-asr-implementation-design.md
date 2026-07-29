# 方案 A：Global ASR 主路径接入设计

**日期**：2026-07-29
**状态**：已批准并实施
**范围**：离线 Pipeline 的 global/segmented 路由、物理 shadow 接入、回退状态和回归测试

## 1. 目标

将仓库中已有的 `PhysicalTimeline`、物理 bins、词分配器和 `_run_global_transcription_path()` 接入真实离线主流程，解决复杂音频中因 VAD 切段造成的上下文断裂和局部时间坐标恢复问题。`GlobalTranscriber` 的长音频窗口化不在本阶段实现。

成功标准：

- `auto` 在离线任务中真实尝试 global ASR；
- global 成功时不执行 segmented ASR；
- global 失败时，`auto` 只回退为一套完整的 segmented 结果；
- 显式 `global` 失败时直接报告失败，不静默降级；
- `stats.asr_path`、`global_attempted`、fallback 分类和诊断真实反映运行路径；
- streaming、现有 segmented 路径、LLM、WebUI 和场景参数行为保持兼容。

## 2. 非目标

本阶段不做以下变更：

- 不修改 Silero、ffmpeg、RMS 的阈值；
- 不打开 `fusion.enabled`；
- 不把降噪默认改为开启；
- 不引入新的 ASR 模型或依赖；
- 不改写 LLM 合并和说话人融合算法；
- 不把长音频窗口化策略和自适应噪声策略混入本阶段，长音频仅复用现有安全边界或明确降级。

## 3. 路由设计

离线 `Pipeline.run()` 在加载完整 audio 后先执行 `_resolve_asr_path()`：

```text
streaming                 -> 现有 streaming 路径
explicit segmented        -> 现有 skeleton/segmented 路径
global                    -> 物理 shadow + global ASR，失败即任务失败
auto                      -> 先 global；失败后完整回退 segmented
```

`skeleton_mode` 不再抢占 global 路径。global 分支可以使用 skeleton/ffmpeg 结果构建物理证据，但不再对每个 skeleton 段调用一次完整 VAD → ASR。显式 `segmented` 时保留现有 skeleton 行为，确保本阶段可回滚。

## 4. 物理 shadow 构建

新增一个 Pipeline 内部的早期构建步骤，复用已有数据结构：

1. 创建完整音频对应的 `PipelineContext`；
2. 对完整音频运行一次现有 Silero VAD 和 ffmpeg unified pass；
3. 将 `silero_segments`、`ffmpeg_unified_result` 和可用 fused segments 写入 context；
4. 调用 `build_shadow_artifacts()`，得到 `ShadowBuildResult`；
5. 校验 `physical_timeline.validate()`，并将 context 中的 timeline、speaker timeline 和 ffmpeg 结果挂到 global 调用需要的 shadow 对象上；
6. 将检测结果、证据数量、跳过数量和构建异常写入 global diagnostics。

本阶段不改变原始 detector 输出。物理 evidence 只记录来源和置信度，不能把粗粒度 evidence 伪装成词级时间。

## 5. Global ASR 与事件生成

优先复用现有 `_run_global_transcription_path()` 和物理分配模块：

```text
global ASR transcript
  -> GlobalWord / GlobalTranscript
  -> PhysicalSubtitleBin
  -> allocate_words()
  -> align_words_to_physical()
  -> build_events()
  -> SubtitleEvent
```

需要补齐的行为：

- global ASR 调用传入已解析的语言，而不是丢失语言配置；
- global 事件保留 `source_word_ids`、词时间、physical spans 和 alignment warnings；
- 空 transcript、空事件、非法时间和物理覆盖不足都必须进入可分类的失败/降级判断；
- 不将 global 与 segmented 的部分事件混合输出。

长音频在本阶段使用保守策略：超过现有安全时长上限时，`auto` 直接走 segmented 并记录 `resource_unavailable`/`duration_limit`，显式 `global` 返回明确错误。后续方案可单独接入 `GlobalTranscriber` 的窗口化执行。

## 6. 回退状态机

```text
auto
  -> global attempted
       -> usable events + acceptable coverage: asr_path=global
       -> empty/invalid/exception: discard global result
                                  -> segmented once
                                  -> asr_path=legacy_degraded

global
  -> usable events: asr_path=global
  -> any global failure: raise classified error

segmented
  -> existing behavior: asr_path=legacy
```

回退不得复用 global 已产出的部分事件，也不得把 global 失败原因写成普通 segmented 成功。缓存校验必须根据实际 `asr_path` 区分旧结果。

## 7. 测试设计

新增或扩展无模型测试：

- 路由决策：`global`、`segmented`、`auto` 和 streaming；
- fake global engine 成功时 segmented engine 不被调用；
- fake global engine 返回空/非法/异常时，`auto` 只调用一次 segmented；
- 显式 global 失败不发生 segmented fallback；
- physical shadow 能把 Silero、ffmpeg 和 fused evidence 保留到 timeline；
- global 词时间经过 offset 后不重复加偏移；
- global event 的 word IDs 和 physical provenance 不丢失；
- skeleton 无语音回退的返回值契约一致；
- 旧 physical、mapping、segmented 测试保持通过。

验证命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_physical \
  tests/test_pipeline.py \
  tests/test_phase_five.py \
  tests/test_cli.py
```

完成实现后再运行全量测试，并单独运行真实质量基准；模型不可用时不将端到端质量结果伪装成通过。

## 8. 风险与回滚

主要风险是 global 物理 evidence 不完整、现有后处理不接受 global event 字段、以及进度阶段数量与新路径不一致。缓解方式是优先使用 fake engine 和现有 physical fixtures，global 失败时保持完整回退，且不删除现有 segmented 分支。

回滚方式：将 `global_asr.enabled=false` 或任务级 `asr_path=segmented`，即可恢复现有离线路径。
