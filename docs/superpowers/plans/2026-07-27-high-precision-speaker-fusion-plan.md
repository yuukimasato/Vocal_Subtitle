# 高精度说话人融合开发计划

对应设计：[高精度说话人融合设计](../specs/2026-07-27-high-precision-speaker-fusion-design.md)

## 目标

将当前事件级滑窗聚类升级为可诊断的两条主线：

- 线路一：SpeechBrain ECAPA 或 `pyannote/embedding` 声纹身份线。
- 线路四：Community-1 / Diarization 3.1 的全局 turns 加逐字幕局部换人精修。

默认仍可在没有全局模型时使用 SpeechBrain ECAPA；模型和任一线路失败时不使用停顿交替或字幕顺序伪造 speaker。

## 实施任务

### 1. 配置与模型注册

- 扩展 `DiarizationConfig`：`fusion_mode`、`global_model`、`diarization_scope`、局部精修阈值和最小时长。
- 保持旧 `backend=pyannote|legacy` 配置兼容。
- 新增统一 speaker model registry，暴露四个模型的元数据、HF cache 状态、协议和脱敏错误。
- 将新增字段纳入配置 hash。

### 2. 声学融合核心

- 新增纯数据/算法模块，完成 global turn 与 embedding cluster 的 overlap 映射。
- 对完整音频只运行一次全局 diarization，结果跨 chunk/skeleton 复用。
- 使用词时间戳和左右局部声纹变化，在单条字幕内部生成候选换人边界。
- 按词边界切分 `SubtitleEvent`，复制物理跨度、word IDs 和溯源字段。
- 全局/局部结果冲突时保留 `unknown` 或不切分，并记录诊断。
- 移除 `_gap_based_speaker_assignment` 在生产 speaker 赋值中的调用。

### 3. Pipeline 集成

- 在最终事件集合形成后运行统一 speaker analysis stage，覆盖单块、多块和 skeleton 路径。
- 将声纹/全局/局部模型、speaker 数、split 数和降级原因写入 `PipelineStats`。
- 确保 LLM 合并器不能跨最终 speaker 边界合并。
- 旧 legacy/单角色缓存不被当前配置复用。

### 4. WebUI/API

- 新增 `/api/speaker-models` catalog/status/download 接口。
- 恢复独立模型下载面板，展示 SpeechBrain ECAPA、`pyannote/embedding`、Community-1、Diarization 3.1。
- 说话人设置增加 global model、线路范围、融合模式、人数和局部精修级别。
- 提交 override、任务结果和缓存 hash 必须包含这些字段。
- 结果页展示 speaker count、backend、模型、局部切分、冲突、unknown 和降级原因。

### 5. CLI

- 增加 `--expected-speakers`、`--speaker-fusion`、`--global-diarization-model`、`--speaker-diarization-scope` 和 `--local-speaker-refinement`。
- `download-models` 支持列出/下载 speaker embedding 和 global diarization 模型。
- 输出实际 backend、模型、speaker 数和局部切分诊断。

### 6. 测试与验收

- 配置、模型注册表、cache 状态、CLI override、WebUI API。
- fake global turns 与 fake embedding 的身份映射、已知/未知人数、冲突和失败降级。
- 单字幕内部换人切分，保留 word IDs 和物理字段。
- 跨 chunk/skeleton 的 speaker continuity。
- 全量既有测试、`compileall`、`git diff --check`。

## 验收命令

```bash
pytest -q
python -m compileall -q vocal_subtitle tests
git diff --check
```

## 实施状态

已完成：

- 配置、模型注册表、SpeechBrain ECAPA 默认路径和 pyannote 可选路径；
- 线路一声纹身份聚类与线路四全局 turns + 局部换人检测；
- 已知/未知人数约束、冲突保留 `unknown`、相同嵌入的质量门控；
- Pipeline、缓存哈希、WebUI/API、模型下载面板和 CLI 参数；
- 按词边界切分字幕，保留词时间戳、物理字段和来源词 ID；
- LLM 合并的 speaker 边界保护与事件溯源字段保留；
- 818 项全量测试、`compileall` 和 `git diff --check`。

验证环境：`venv/bin/python`（Python 3.12.3，pytest 9.1.1）。
