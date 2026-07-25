# 阶段五：全局 ASR 默认路径切换设计

日期：2026-07-26  
状态：已确认，待实施

关联文档：

- `docs/Vocal_Subtitle-离线高精度字幕系统设计文档.md` 第 13 节“阶段五：切换默认路径”
- `docs/superpowers/specs/2026-07-26-phase-three-global-transcription-design.md`
- `docs/superpowers/specs/2026-07-26-phase-four-subtitle-segmentation-validation-design.md`
- `docs/ARCHITECTURE.md` 第 11 节“自适应反馈学习”

## 1. 目标与范围

阶段五把阶段二至阶段四已经实现的物理时间线、全局转录、词级物理分配、严格断句和最终校验接入离线默认路径。离线 Pipeline 默认优先尝试全局 ASR；旧分段 ASR 作为完整、可诊断的降级路径保留。流式模式继续使用旧分段 ASR，不加载 WhisperX。

本阶段还统一路径状态、失败原因、缓存身份、CLI/WebUI 结果字段和基准验收入口，使默认路径切换可观察、可回归、可回滚。

本阶段不包含：删除旧 ASR 引擎、修改流式协议、在导入阶段加载 WhisperX/Torch、用不完整的 benchmark 素材伪造验收通过、重做阶段二至阶段四的算法。

## 2. 运行路径契约

离线任务的路径决策如下：

```text
global preferred
  -> physical timeline
  -> global ASR + word allocation
     -> success: asr_path=global
     -> dependency/resource/execution/result failure
        -> fallback_to_segmented=true: asr_path=legacy_degraded
        -> fallback_to_segmented=false: structured task failure

explicit segmented -> asr_path=legacy
streaming         -> asr_path=legacy
```

任务级路径覆盖支持 `auto`、`global`、`segmented`。profile 配置不因任务级覆盖而被持久修改。全局路径成功时必须保留 `source_word_ids`、`physical_spans`、`logical_sentence_id`、speaker 和最终校验结果。全局路径不能把部分结果静默混入 legacy 结果。

全局失败分类固定为：

- `dependency_unavailable`：WhisperX 或 alignment 依赖不可用；
- `resource_unavailable`：设备、显存或模型资源不足；
- `execution_failed`：全局转录或对齐执行异常；
- `invalid_result`：结果为空、降级、时间非法或无法通过物理校验；
- `explicit_legacy`：用户或流式入口明确选择旧路径。

可降级结果至少包含：

```json
{
  "asr_path": "legacy_degraded",
  "global_attempted": true,
  "global_status": "degraded",
  "fallback_category": "dependency_unavailable",
  "fallback_reason": "WhisperX is not installed"
}
```

错误和诊断不得包含 API key、HF token 或完整敏感本地凭据。

## 3. 配置与缓存

`global_asr.enabled=true` 是离线默认的全局优先开关；设为 `false` 表示显式 legacy。`fallback_to_segmented=true` 保持默认安全降级；设为 `false` 时全局失败直接结束任务。全局引擎和 alignment 继续延迟加载。

`PipelineStats` 和任务结果统一提供：

- `asr_path`：`global`、`legacy` 或 `legacy_degraded`；
- `global_attempted`；
- `fallback_category` 与 `fallback_reason`；
- `global_status`、`global_statistics`、`global_diagnostics`。

全流水线缓存恢复必须保留完整统计，而非只恢复字幕数和片段数。缓存身份至少包含路径策略、全局配置和结果 schema 版本。默认全局请求不得恢复没有全局路径标记的旧 legacy 结果。阶段三的 `whisperx_asr` 分阶段缓存继续使用独立 schema 版本。

## 4. 基准验收

`scripts/run_benchmarks.py` 增加全局、legacy 和 auto 路径标识，并为每个场景记录实际路径、降级原因、时间误差、物理越界和健康度。CI 验收至少要求三类真实素材：单人长段、多人交替、重叠或抢话。素材不足、ground truth 缺失或场景数量不足时，报告 `rollout_eligible=false`，不得宣称默认路径已通过真实素材验收。

当前仓库只有 benchmark 目录骨架，没有足够的真实 ground truth；本阶段实现验收机制，但不生成虚假基准数据。

## 5. 测试策略

无模型测试覆盖：配置默认值和任务覆盖、全局成功、依赖缺失、资源异常、执行异常、无效结果、显式 legacy、流式强制 legacy、统计字段和缓存恢复。Pipeline/API 回归测试验证任务历史、WebSocket 完成事件、CLI 输出和 WebUI payload 保留路径诊断。

真实素材测试在安装 WhisperX 和补齐 benchmark 后执行全局/legacy 对照。基础测试不得依赖 GPU、HF token、网络或云端 LLM。

## 6. 回滚策略

将 `global_asr.enabled=false` 或任务级选择 `segmented` 即可恢复 legacy 默认行为；不需要删除缓存或修改用户反馈配置。若全局路径质量异常，保留诊断和 benchmark 报告，修复后重新运行验收再恢复全局优先。
