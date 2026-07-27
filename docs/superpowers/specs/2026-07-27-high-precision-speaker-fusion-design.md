# 高精度说话人标注与片段级换人检测设计

**日期：** 2026-07-27  
**状态：** 待用户审阅  
**范围：** 后端 Pipeline、模型下载中心、WebUI、CLI、缓存和测试

## 1. 背景与问题

当前项目已经包含 SpeechBrain ECAPA、`pyannote/embedding`、Community-1 和
Diarization 3.1 的部分模型适配代码，但实际运行链路仍存在三个问题：

1. 事件级滑动窗口聚类可能把多个说话人压成一个角色。
2. 当前片段级处理没有把全局 diarization turns 纳入最终字幕边界。
3. 质量不足时的 gap-based speaker alternation 会制造伪身份，且跨宏观块/骨架段重新编号会破坏同一角色的连续性。

典型错误是将下面一条字幕：

```text
18.55 --> 20.36  说话人A  还好意思说我。 行行行
```

正确拆分为两个说话人事件：

```text
18.53 --> 19.48  甲  还好意思说我。
19.69 --> 20.42  乙  行行行
```

## 2. 目标与非目标

### 目标

- 默认使用 SpeechBrain ECAPA，开箱即用，不要求 Hugging Face 授权。
- 支持用户选择 `pyannote/embedding` 作为更高精度声纹嵌入模型。
- 保留并接入 Community-1、Diarization 3.1 全局 diarization 模型。
- 支持用户已知说话人数时填写人数；未填写时自动估计。
- 支持三层证据：全音频 speaker turns、声纹身份聚类、逐字幕局部换人检测。
- 一条字幕内部发生换人时，按词时间和声学证据切分，并保持稳定 speaker ID。
- WebUI、CLI 和 Python 配置使用同一套模型、人数约束和融合逻辑。
- 任何模型失败时保留字幕文本和时间，但 speaker 结果必须可诊断；不能使用停顿交替、字幕序号或随机方式伪造 speaker。

### 非目标

- 不在本次设计中训练新的声纹或 diarization 模型。
- 不要求自动推断真实姓名；姓名/角色仍由现有可选 LLM role labeler 负责。
- 不把不同模型返回的原始 `SPEAKER_00`、`SPEAKER_01` 当作跨模型稳定身份。

## 3. 用户可见模式

### 3.1 模型组

模型中心展示四个模型，并分别显示下载状态、缓存路径、大小、协议和授权要求：

| 组 | 模型 | 默认 | 作用 |
|---|---|---:|---|
| 声纹嵌入 | `speechbrain/spkrec-ecapa-voxceleb` | 是 | 默认身份聚类，Apache 2.0，无需 HF 授权 |
| 声纹嵌入 | `pyannote/embedding` | 否 | 可选高精度声纹嵌入，需要接受模型协议 |
| 全局 diarization | `pyannote/speaker-diarization-community-1` | 否 | 全音频换人边界与全局 turns |
| 全局 diarization | `pyannote/speaker-diarization-3.1` | 否 | 可选全局 diarization pipeline |

默认配置仍以 SpeechBrain ECAPA 为基础。全局模型模式为 `auto` 时，只有本地全局模型完整可用才启用双路融合，否则自动回到 ECAPA 单路。

### 3.2 说话人数与融合模式

- `expected_speakers` 留空：自动估计。
- `expected_speakers=N`：将 N 同时传给全局模型和嵌入聚类；全局模型使用 `min_speakers=N,max_speakers=N`，嵌入聚类以 N 为目标簇数。
- `fusion_mode=auto`：本地有全局模型时启用双路，否则 ECAPA 单路。
- `fusion_mode=embedding`：只使用声纹嵌入线。
- `fusion_mode=dual`：要求同时运行声纹嵌入和全局 diarization；全局模型不可用时结果降级，不静默伪造。
- `global_model=community-1|diarization-3.1|auto|none`：选择全局模型。
- `local_refinement=off|embedding|full`：控制逐字幕局部换人检测。

高精度模式默认 `local_refinement=embedding`；`full` 会复用已加载的全局模型对带上下文的局部窗口复核，速度更慢。

## 4. 三层处理架构

### 4.1 线路一：全音频声纹身份线

使用配置的 embedding 模型在完整音频上建立稳定的 speaker cluster。短字幕不直接作为唯一 embedding 输入，而是使用 2--3 秒上下文窗口或相邻语音窗口，降低短句声纹不稳定导致的误聚类。

线路一输出：

- ECAPA/pyannote embedding cluster ID；
- 每个原子语音片段的身份分数；
- 聚类簇数、silhouette、模型和 provenance。

### 4.2 线路二：全局 diarization turn 线

Community-1 或 Diarization 3.1 对完整音频运行一次。模型只加载一次，不能为每条字幕重新初始化或独立推理完整上下文。

线路二输出：

- 全局 speaker turns；
- turn 边界、重叠区间和全局模型 speaker 标签；
- 实际 speaker 数、后端状态和模型 provenance。

全局模型返回的 speaker 标签只在本次音频内有效，必须通过重叠时长与线路一的 cluster 做映射，不能直接比较数字编号。

### 4.3 线路三：逐字幕局部换人精修

对每一条最终候选字幕事件执行局部检测。该阶段不重复加载模型，使用已加载的 embedding 或全局模型实例。

处理步骤：

1. 获取事件内的 ASR word timestamps。
2. 枚举词间边界、标点边界、短暂停顿和全局 turn 交叉点。
3. 对候选点左右截取带上下文音频，默认比较约 0.4--1.2 秒有效语音。
4. `embedding` 模式计算左右声纹相似度和局部变化分数。
5. `full` 模式额外使用全局模型对字幕前后 padding 窗口复核。
6. 通过最小片段时长、停顿长度、speaker 分数和证据支持数进行质量门控。
7. 切点优先选择 word boundary；没有词时间戳时，跨 speaker 的事件标记为 `unknown`，不将整句话强行归给一人。

局部检测的职责是发现“全局 turn 没有覆盖或字幕合并隐藏的内部换人”，不是替代全音频全局结果。

## 5. 融合与边界决策

### 5.1 原子片段构建

取以下边界的并集：

- ASR word boundaries；
- 原字幕事件边界；
- 全局 diarization turn boundaries；
- 通过质量门控的局部换人边界。

在这些边界上构建最小 `AtomicSpeechSpan`。同一个 speaker 的相邻原子片段只有在不跨硬边界且满足现有物理约束时才允许合并。

### 5.2 speaker 身份对齐

使用重叠时长矩阵把线路二的全局标签映射到线路一 cluster：

1. 计算每个全局 turn 与每个 embedding cluster 的时间/能量/嵌入重叠得分。
2. 在已知人数约束下进行一一映射；未知人数时允许未匹配的低置信 speaker。
3. 对映射结果做全音频一致性检查，禁止按首次出现顺序重新覆盖已有全局身份。

### 5.3 决策规则

- 两条线路映射后一致：使用该 speaker，状态为 `fused`。
- 全局线检测到边界且局部 embedding 也支持变化：确认切分。
- 只有一条线路支持边界：要求高置信度、合理停顿和最小片段时长；否则保留原事件并记录冲突。
- 两条线路 speaker 身份冲突且无法通过映射/分数解决：speaker 为 `unknown`，不强行选择。
- 局部精修发现边界后，最终事件不得再被 LLM 合并器跨 speaker 合并。

每个输出事件保留：`speaker_id`、`speaker_label`、`speaker_source`、`speaker_confidence`、`speaker_model` 和可选冲突原因。

## 6. 后端实现边界

### 6.1 模型注册与下载

新增统一模型注册表，复用现有 HF cache 检查和 token 存储能力，提供：

- 模型元数据、类别、协议、是否需要 token；
- 本地 snapshot 完整性检查；
- 下载/加载状态和脱敏错误原因；
- 可重复调用的下载操作，已完整缓存时不重复下载。

新增 Web API：

- `GET /speaker-models`：返回四个模型的 catalog 和本地状态；
- `POST /speaker-models/{model_id}/download`：开始或复用下载任务；
- `GET /speaker-models/{model_id}/status`：查询下载/缓存/错误状态；
- 保留现有 license API，并扩展为同时返回 embedding 和 global 模型信息。

下载操作不能把 HF token 写入日志、任务历史或返回值；前端只显示掩码状态。

### 6.2 Pipeline 编排

新增统一 speaker analysis stage，在完整音频可用后执行一次：

1. 解析 fusion/global/embedding 配置。
2. 运行线路一；如果启用双路且全局模型可用，运行线路二。
3. 对线路结果做 canonicalization 和 speaker mapping。
4. 在 ASR word timestamps 可用后运行逐字幕局部精修。
5. 构建 atomic spans，替换事件级滑窗聚类的最终赋值逻辑。
6. 将结果注入事件并运行现有物理边界、字幕构建和导出流程。

长音频和宏观分块必须共享完整音频的 speaker timeline。禁止每个 chunk 或 skeleton segment 重新从 speaker 0 开始编号。

现有 `_gap_based_speaker_assignment` 不再用于生产 speaker 赋值。其质量不足时改为 `unknown`，保留诊断信息。

### 6.3 配置兼容

扩展 `DiarizationConfig`，建议字段如下：

```yaml
diarization:
  enabled: true
  backend: auto
  fusion_mode: auto
  global_model: auto
  expected_speakers: null
  local_refinement: embedding
  local_context_seconds: 0.6
  min_local_segment_seconds: 0.25
  min_change_confidence: 0.70
```

现有 `backend=pyannote|legacy` 配置继续解析，但映射到新模式并在统计中标记兼容路径；旧 legacy 结果不能被新双路配置的缓存复用。

## 7. WebUI 修改

### 模型下载面板

恢复独立的“模型下载”分组，四个模型按 embedding/global 分类展示：选择、协议链接、token 状态、缓存状态、下载按钮和错误信息。下载完成后自动刷新配置可选项，不自动覆盖用户当前 embedding 选择。

### 说话人配置面板

增加：

- embedding 模型选择；
- global 模型选择；
- 融合模式；
- 说话人数输入，空值表示自动；
- 局部换人检测级别；
- 局部检测阈值和最小片段时长。

提交任务时，这些字段必须进入最终 config override 和 config hash，不能只保存到浏览器 localStorage。

### 结果与诊断

任务详情显示：

- 最终 speaker 数；
- embedding/global/fused/unknown 事件数；
- 实际使用的模型和后端；
- 局部切分次数；
- 冲突数、unknown 数和降级原因。

字幕预览、SRT/ASS 导出和 API 响应必须来自同一份最终事件集合。

## 8. CLI 修改

在现有 `run` 命令增加：

```text
--expected-speakers INTEGER
--speaker-fusion [auto|embedding|dual]
--global-diarization-model [auto|none|community-1|diarization-3.1]
--local-speaker-refinement [off|embedding|full]
```

CLI 参数优先级高于 profile 配置，并写入 config hash。命令行输出和最终 stats 显示实际后端、模型、speaker 数、局部切分数与降级原因。

## 9. 错误处理与缓存

- 全局模型缺失、pyannote 依赖缺失、HF 协议/token 失败：记录脱敏原因，`auto` 进入 ECAPA；`dual` 标记 degraded。
- embedding 加载/提取失败：如果全局模型可用则使用全局线，否则 speaker 保持 unknown。
- 两路质量不足：禁止 gap alternation 和字幕序号推断。
- 期望人数为非法值、模型返回超过人数约束或聚类结果不可解释：做确定性约束并标记 degraded；不能静默声称高精度。
- 缓存 key 必须包含 embedding model、global model、fusion mode、local refinement、expected speakers、阈值和相关边界参数。
- 缓存结果必须包含 provenance、speaker source、模型状态和 config hash；缺字段、旧 legacy、单角色伪 fallback 结果视为 stale。

## 10. 测试与验收

### 单元测试

- 四类模型 catalog、缓存检测、授权状态和脱敏错误。
- CLI/WebUI override 正确写入配置和 hash。
- 已知人数同时约束两条线路；未知人数不强行补齐。
- 全局 speaker ID 与 embedding cluster 的一一映射不依赖原始数字编号。
- 局部字幕事件在词边界处分裂，并保留 source word IDs、物理跨度和时间约束。
- 一致、冲突、低置信度三种融合结果。
- 两路失败后 speaker 保持 unknown，确认不会调用 gap alternation。
- 跨 chunk/skeleton 的 speaker ID 保持全局一致。

### 集成测试

使用双人音频构造或真实 fixture，验证：

1. 输入 `还好意思说我。 行行行` 的单事件可拆为两个 speaker 事件。
2. 已知 `expected_speakers=2` 时最终 speaker 不超过且应能识别两类。
3. 未知人数时自动识别，不因字幕数量生成 A-H 等伪身份。
4. 同一 speaker 跨多个字幕保持同一 ID。
5. 全局模型不可用时 ECAPA 路径可用；两者不可用时字幕仍可导出且 speaker 为 unknown。
6. 新配置不会命中旧单角色或 legacy cache。

### CLI/WebUI 验收

- CLI 显式设置 `--expected-speakers 2`，输出包含两类 speaker 和融合诊断。
- WebUI 下载/选择模型、填写人数、选择局部精修级别后提交任务，后端实际使用配置与页面一致。
- WebUI 结果统计、字幕预览、SRT、ASS 使用同一最终事件集合。
- 全量既有测试通过；无模型测试不得访问网络。

## 11. 实施顺序

1. 扩展配置、模型注册表和缓存 provenance。
2. 接入统一全音频 speaker analysis stage，移除生产路径的伪 speaker fallback。
3. 实现 global turn 与 embedding cluster 对齐及融合。
4. 实现逐字幕局部换人检测和原子片段切分。
5. 增加 Web API、模型下载面板、说话人配置和结果诊断。
6. 增加 CLI 参数和 stats 输出。
7. 完成单元、集成、CLI 和 WebUI 验收。
