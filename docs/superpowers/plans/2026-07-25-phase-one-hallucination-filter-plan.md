# 阶段一实施计划：ASR 幻觉过滤与数据契约扩展

关联规格：[2026-07-25-phase-one-hallucination-filter-design.md](../specs/2026-07-25-phase-one-hallucination-filter-design.md)

## 实施约束

- 保留工作区中用户已有的阶段零变更，不回滚、不重排无关修改。
- 过滤层只作用于离线 `_run_asr()`；streaming 路径保持现状。
- 不引入 WhisperX、全局 ASR、PhysicalTimeline 或新的模型依赖。
- 所有新数据字段使用追加式默认值，保持旧 positional dataclass 构造兼容。
- 每项任务先补测试再实现或同步实现，完成后运行目标测试与可用的全量测试。

## 任务 1：配置契约与策略适配

**目标**：把阶段一的阈值、开关和版本纳入 `ASRConfig` 及 YAML 加载/校验。

**文件**：

- `vocal_subtitle/config.py`
- `configs/default.yaml`
- `configs/podcast.yaml`
- `configs/education.yaml`
- `configs/variety_show.yaml`
- `configs/music_live.yaml`
- `tests/test_pipeline.py` 或新增阶段一配置测试

**步骤**：

1. 在 `ASRConfig` 追加三个解码阈值、过滤开关、短语/相邻重复开关和 `v1` 默认版本。
2. 在 `ConfigLoader._parse_config()` 映射这些字段；未配置时使用 dataclass 默认值。
3. 在现有配置校验入口加入有限数值、非负阈值和非空版本校验。
4. 从默认 profile 读取字段；其余 profile 使用相同默认值，避免场景配置行为漂移。
5. 增加配置加载、非法配置和默认值测试。

**验收**：配置对象能生成过滤策略；非法值在任务开始前被报告；旧 YAML 仍可加载。

## 任务 2：纯函数幻觉过滤器

**目标**：实现无副作用、可序列化诊断的统一过滤模块。

**文件**：

- `vocal_subtitle/asr/hallucination.py`
- `tests/test_asr/test_hallucination.py`

**步骤**：

1. 定义 `HallucinationFilterPolicy`、删除记录、警告记录和 `HallucinationFilterResult`。
2. 实现文本规范化：首尾空白、连续空白、大小写和标点边界归一化；不改变返回对象的原始文本。
3. 实现有效词级证据判断：非空文本、有限时间、正向区间和最低置信度 `0.35`。
4. 按规格顺序实现空文本、训练短语、无声概率、低 logprob、压缩比重复和相邻重复规则。
5. 让过滤器不修改输入列表、segment 或 words，并保留输入顺序。
6. 让缺失/异常质量字段按保守规则跳过；规则异常形成可读诊断而不是静默丢失。
7. 编写纯函数测试，覆盖真实短词、短前缀+完整短语、缺失字段、输入不可变和计数稳定性。

**验收**：所有规则均可由测试独立证明；过滤器不依赖 Pipeline、音频或外部模型。

## 任务 3：faster-whisper 显式解码保护

**目标**：将阶段一阈值从 Pipeline 配置传入 faster-whisper 底层调用。

**文件**：

- `vocal_subtitle/asr/faster_whisper_engine.py`
- `vocal_subtitle/pipeline.py`
- `tests/test_asr/test_faster_whisper_engine.py`

**步骤**：

1. 在 `FasterWhisperEngine.__init__()` 追加三个可选阈值参数，默认值与规格一致，保持旧构造调用有效。
2. 在 `_get_asr_engine()` 将 `ASRConfig` 的值传入引擎。
3. 在 `_model.transcribe()` 调用中显式传递三个阈值，不依赖库默认值。
4. 保持现有 beam、词时间戳、previous text 和外部 VAD 参数行为。
5. 用 mock 模型断言底层调用收到精确阈值；继续断言质量字段和词级 speaker 字段透传。

**验收**：默认和自定义阈值均实际传到模型；旧的直接实例化测试不需修改调用签名。

## 任务 4：Pipeline 统一过滤与统计

**目标**：让缓存命中和新识别结果在相同位置经过过滤，并把结果写入任务统计。

**文件**：

- `vocal_subtitle/pipeline.py`
- `tests/test_phase_one.py` 或 `tests/test_pipeline.py`

**步骤**：

1. 在 Pipeline 中增加策略构造和过滤辅助方法，避免缓存分支与新识别分支复制规则。
2. 调整 `_run_asr()`：规范化、段内重叠去重之后调用过滤器，再写缓存和返回结果。
3. 确保缓存命中分支在 `continue` 前也调用同一过滤器。
4. 过滤异常时保留原始结果并增加诊断；过滤关闭时保持旧结果集合。
5. 扩展 `PipelineStats`、`to_dict()` 和当前任务初始化/结束流程，记录版本、开关、删除数、原因和警告数。
6. 验证过滤后的空结果不会导致后续时间映射异常。
7. 使用 fake ASR、fake CacheManager 或临时缓存测试新识别与缓存命中一致。

**验收**：同一输入无论命中缓存与否得到相同过滤结果；统计可序列化；streaming 不调用新过滤路径。

## 任务 5：ASR 缓存键隔离

**目标**：所有会改变过滤结果或引擎解码结果的参数进入 transcription 缓存键。

**文件**：

- `vocal_subtitle/pipeline.py`
- `tests/test_utils/test_cache_manager.py` 或阶段一测试

**步骤**：

1. 在 `_run_asr()` 的缓存键加入过滤器版本、开关、三个阈值和两个规则开关。
2. 保持已有语言策略、模型、beam、词时间戳等键字段不变。
3. 添加参数差异键测试，确保每个策略字段变化都会产生不同 key。
4. 验证阶段零已有 `expected_speakers` 缓存隔离测试不受影响。

**验收**：规则升级或配置变化不会命中旧转录结果；缓存写入和读取使用完全相同参数集合。

## 任务 6：回归验证与文档状态

**目标**：完成阶段一验收并记录环境限制。

**步骤**：

1. 运行阶段一纯函数、配置、引擎、Pipeline 和缓存测试。
2. 运行阶段零相关测试和现有 ASR/映射/Pipeline 回归测试。
3. 尝试运行完整 `pytest`；若依赖缺失，记录具体命令和缺失项，不伪造通过结果。
4. 使用 `git diff --check` 检查本轮改动。
5. 更新阶段一规格状态和必要的项目文档，只修改与本阶段相关内容。

**完成定义**：阶段一规格中的七项完成标准全部满足，且任何未运行测试有明确报告。

## 推荐执行顺序

```text
任务 1 配置
  → 任务 2 纯函数过滤器
  → 任务 3 引擎阈值
  → 任务 4 Pipeline/统计
  → 任务 5 缓存隔离
  → 任务 6 回归验证
```

每个任务完成后先运行其目标测试，再进入下一任务；任务 4 与任务 5 共享 `_run_asr()`，需要在同一轮集成回归中共同验证。
