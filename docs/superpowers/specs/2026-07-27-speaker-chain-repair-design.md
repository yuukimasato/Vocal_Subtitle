# 双人说话人链路修复设计

日期：2026-07-27

## 背景

双人测试音频的字幕文本已经完成标点归属和短碎片修复，但同一条链路仍出现两类说话人问题：CLI 默认没有 speaker 信息，WebUI 在全局 diarization 不可用时使用旧 fallback，产生 A-H 等过多身份。诊断表明默认配置关闭 diarization 和 speaker embedding；本地 SpeechBrain ECAPA 模型可加载并能将测试音频聚为 2 类，而本地 pyannote 全局模型当前不存在。

人工修订 ASS 仅作为结果参考，不作为自动切分或说话人标注的输入。

## 目标

1. CLI 和 WebUI 使用同一套全局说话人配置与降级规则。
2. 双人任务可以显式设置 `expected_speakers=2`，聚类最多且通常固定为两类。
3. pyannote 可用时优先使用；不可用时使用本地 SpeechBrain ECAPA；两者都不可用时保留 unknown，不按字幕顺序或停顿伪造 speaker。
4. 旧的 degraded/A-H 全管道缓存不能污染新配置的结果。
5. 保留上一轮字幕完整性修复：无首部标点、无纯标点事件、词语不被物理边界截断。

## 非目标

- 不把人工 ASS 强制对齐为自动结果。
- 不在本次工作中引入新的说话人角色名称推断或 LLM 角色标注。
- 不将所有 profile 都切换到高成本的在线 pyannote 下载路径。
- 不改变已完成的 global ASR 主路径和字幕文本规则，除非测试暴露直接回归。

## 方案

### 配置与入口

默认运行配置启用可用的全局 diarization 路径，并将本地 SpeechBrain ECAPA 作为无 pyannote 模型时的可用 fallback。CLI 增加 `--expected-speakers`，WebUI 在说话人选项中增加人数约束输入；两者都映射到 `diarization.expected_speakers`。未指定人数时仍允许后端自动估计，但不允许 fallback 生成轮换式伪 speaker。

WebUI 提交的 backend、fallback、embedding 模型和人数约束必须由最终 config hash 覆盖，不能只依赖浏览器 localStorage 的旧设置。

### 全局说话人链路

完整音频只运行一次 speaker diarization。得到的全局 speaker turns 经过 canonicalization 后，作为后续物理分段、ASR 词时间和字幕事件的唯一 speaker 来源。`expected_speakers=2` 同时作为 embedding 聚类的最小/最大人数约束，并作为 canonicalization 的最大人数约束。

fallback 的成功状态必须记录 backend、model、speaker count 和 silhouette。fallback 失败或质量不可信时，输出字幕文本和时间，但 speaker_id 保持 `None`，统计状态为 degraded，并显示可诊断原因。

### 缓存

diarization stage cache 的 key 包含音频哈希、backend、全局模型、fallback backend、embedding 模型、人数约束及相关边界参数。WebUI full-pipeline cache 校验必须区分：

- 可靠的 pyannote 结果；
- 配置允许且模型确实加载成功的 ECAPA fallback 结果；
- 不可复用的旧 legacy/degraded 结果。

如果新配置与历史结果的配置哈希不同，直接重新运行，不恢复旧字幕事件。

## 错误处理

- pyannote 缺失、模型不存在、依赖或 Token 失败：记录脱敏原因，进入 ECAPA fallback。
- ECAPA 加载或提取失败：停止 speaker 赋值，不使用 gap 轮换或字幕序号推断身份。
- 聚类结果超过 `expected_speakers`：确定性合并并标记 degraded；结果仍不得超过约束。
- 缓存结果缺少 speaker provenance、模型信息或状态不符合当前配置：视为 stale。

## 测试与验收

使用 `test/中文朗读测试-双人.wav`，不使用人工 ASS 驱动生成。

1. 单元测试覆盖 CLI/WebUI override、expected speaker 约束、ECAPA fallback、fallback 失败保持 unknown、缓存可用性判断。
2. CLI 测试使用显式 `--expected-speakers 2`，确认输出包含两类 speaker，且没有首部标点、纯标点事件和 `八`/`回`、`二十分`/`钟` 等截断。
3. 浏览器测试通过 WebUI 提交同一音频，确认任务状态、speaker 统计、字幕预览与 CLI 结果使用同一全局 ASR 路径，且不命中旧 A-H 缓存。
4. 全量既有测试必须通过。

