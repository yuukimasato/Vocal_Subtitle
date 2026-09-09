# VAD Provider 契约（Tier 4）

- 版本：`vad-provider-request-1` / `vad-provider-response-1`
- 归属：**本仓库（subtitle-editor）拥有并版本化此契约**。实现方（任何语言的上层流水线、用户脚本）只需满足本文。
- 定位：外部 VAD 是**可选增强**。内置能量 VAD（Tier 3，`engine: "energy-vad"`）是零依赖默认路径；provider 不可用或输出非法时，CLI 自动降级回内置实现并在 JSON 输出中记录 `degraded.reason`。**集成，不是依赖。**

## 调用方式

```
node agent/cli.mjs vad <media> --provider "<命令行>"
```

CLI 经 `sh -c` 启动 provider 进程，向其 **stdin 写入一行 JSON 请求**，从其 **stdout 读取一行 JSON 响应**。

## 请求（stdin，单行 JSON）

```json
{"schema_version":"vad-provider-request-1","audio_path":"...","sample_rate":16000,
 "threshold":0.5,"min_speech_ms":150,"min_silence_ms":400}
```

| 字段 | 说明 |
|---|---|
| `audio_path` | 媒体文件路径。**解码由 provider 自负**（本 CLI 不为其解出 PCM） |
| `sample_rate` | 建议的内部处理采样率 |
| `threshold` / `min_speech_ms` / `min_silence_ms` | 建议参数，provider 可自行映射 |

## 响应（stdout，单行 JSON）

```json
{"schema_version":"vad-provider-response-1","engine":"silero",
 "segments":[{"start":1.02,"end":3.00,"confidence":0.91}]}
```

- `segments` 必须为数组；每段 `start`/`end` 为有限数值（秒），`confidence` 可选（0–1）。
- JSON 行前后的其他输出会被宽容跳过（CLI 取含 `schema_version` 的那一行）。
- `engine` 为实现自报标识（如 `silero`、`ffmpeg-silence`）。

## 失败语义

- 退出码非 0、超时（默认 120s）、或响应不满足上述结构 → 视为 provider 失败；
- CLI **自动退回内置 VAD**，输出 `source:"builtin"` 并带 `degraded:{reason}`；整个命令仍以退出码 0 成功（降级是正常路径，不是错误）。

## 参考适配脚本（放实现方自己的仓库，不属于本组件）

Python/上层流水线只需把已有 VAD 的输出映射成上述响应格式，约 20 行：

```python
import json, sys
from my_pipeline.vad import SileroVAD   # 任意实现

req = json.loads(sys.stdin.readline())
segs = SileroVAD().detect(req["audio_path"])
print(json.dumps({
    "schema_version": "vad-provider-response-1",
    "engine": "silero",
    "segments": [{"start": s.start, "end": s.end, "confidence": s.conf} for s in segs],
}))
```

## 语义边界

- provider 返回的是**语音/静音区间候选**；与 ASR 文本的对齐、最终裁决属于 Agent/流水线侧；
- 本 CLI 的 `check --media` 用语音区间回答**事实性问题**（边界是否深入静音区），不评价"应该吸附到哪里"——那是 provider/流水线的语义（词汇表见《开发文档》§4.1，勿把上层吸附算法的字段名混入编辑器输出）。
