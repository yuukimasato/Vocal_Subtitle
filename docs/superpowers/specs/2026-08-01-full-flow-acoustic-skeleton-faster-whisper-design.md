# 全流程验收与声学骨架字幕设计

日期：2026-08-01

## 目标

本次工作验证并修复 Vocal Subtitle 的正式生产路径，生产 ASR 使用 `faster-whisper`，模型矩阵覆盖 `tiny`、`small`、`medium` 和 `large-v3`。`whisper.cpp + tiny` 仅作为轻量 fallback 路径验收，不作为黄金集的生产代表。

最终字幕事件必须由声学骨架约束：每条字幕落在一个物理 speech bin 内，不跨越声学静音，不把内部词级时间戳直接暴露为字幕行。词级时间戳仍可保留在内部，用于对齐、证据审计和诊断。

## 现状问题

1. 离线黄金集 runner 默认写死 `whisper-cpp + tiny`，导致质量报告不能代表正式 `faster-whisper` 链路。
2. WebUI 默认的 `global_evidence` 路径会把多个审核 decision 的词级候选直接投影成最终事件。多个候选属于同一物理 bin 时没有统一聚合，最终 WebUI 返回和导出的事件可能退化为单词级字幕，并触发重叠质量失败。
3. CLI 的 `download-models --asr-model` 参数存在但没有执行 ASR 下载，当前会无条件打印成功退出，容易误导用户。

## 设计

### 生产 ASR 与模型矩阵

- 正式生产验收 runner 默认使用 `faster-whisper`。
- 模型通过本地 Hugging Face cache 加载；缺失模型时明确报告 `blocked`，不把未运行计为通过。
- 四个模型分别运行同一批测试场景，报告包含模型、引擎、设备、耗时、事件数、质量状态、物理违规和跨静音指标。
- `whisper.cpp/tiny` 保留独立 smoke，用于验证轻量 fallback，不参与生产模型结论。

### 声学骨架事件投影

数据流保持现有方向：ASR 词级结果 -> evidence review -> physical allocation -> subtitle bin -> final subtitle export。

在物理投影出口增加 bin 级聚合约束：

- 先按 `PhysicalSubtitleBin` 分配并聚合所有已接受词。
- 同一 bin 内可以有多个内部 decision，但最终只生成一个逻辑字幕事件。
- 文本按时间顺序合并并去重；保留来源词 ID、decision trace、speaker 和 alignment warning。
- 事件的展示时间使用 bin 的物理边界或其经过验证的安全边界，不能使用跨 bin 的词级范围扩张事件。
- 跨 bin、跨 speaker 或存在硬边界时必须分开；不得用普通合并规则重新跨越声学静音。
- 没有可分配词的 bin 不生成空字幕，也不能合成虚假文本。

最终化阶段继续负责最大时长、显示时间轴和格式导出，但不得把一个已经按物理 bin 聚合的事件重新拆成词级事件。必要的长字幕拆分必须只发生在同一 bin 内，并在诊断中明确记录。

### CLI 模型下载

`download-models --asr-model` 改为调用正式 ASR 模型加载/下载入口：

- 已缓存：报告 `ready`。
- 下载成功：报告模型名称、缓存位置和状态。
- 依赖缺失、网络失败或模型不可用：返回非零退出码并给出可行动错误。
- 不影响 speaker/separation 模型下载路径。

### WebUI

WebUI 保持 `faster-whisper`、四种模型和骨架模式的可选配置，但最终结果统一消费 pipeline finalizer 的物理 bin 事件。验证范围包括：上传、参数选择、任务提交、WebSocket/轮询进度、完成态、字幕预览、单条编辑、批量编辑和 SRT/VTT/ASS 导出。

浏览器自动化上传若不能可靠构造本地 multipart 文件，会使用 HTTP multipart 验证后端，并单独记录浏览器 DOM 事件状态，不将自动化工具限制误判为产品错误。

## 错误处理与门禁

- ASR 模型未安装：阻断该模型，不回退到 tiny 并伪装成功。
- 生产路径出现 `physical_violation_count > 0`、`cross_silence_count > 0`、`raw_event_bypass_count > 0` 或最终事件跨 bin：门禁失败。
- 最终事件出现单词级碎片时，记录 bin、来源词和 decision trace，回归测试失败。
- Qwen/FunASR/可选复核模型缺失时，保留结构化 `unavailable`/`blocked` 诊断，不降低声学骨架门禁。
- 现有工作区改动不回滚；修复只覆盖本设计涉及的 runner、投影、CLI 和测试文件。

## 验证矩阵

1. 全部现有 Python 自动化测试。
2. `faster-whisper` 四模型短音频加载与识别 smoke。
3. 代表性中文、英文、多人、非语音、重复短语和长音频真实文件的生产路径。
4. CLI 单文件、骨架模式、导出格式和模型参数路径。
5. WebUI REST/API 任务生命周期、字幕编辑和导出。
6. 浏览器端上传、参数选择、运行状态、字幕预览与结果交互。
7. 独立 `whisper.cpp/tiny` fallback smoke。

验收重点顺序为：物理骨架约束 > 不出现词级字幕 > 任务链路无异常 > 模型识别质量 > 性能统计。
