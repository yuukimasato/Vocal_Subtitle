# 任务状态机

**版本**: task-state-v1
**更新日期**: 2026-08-02

## 概述

所有离线任务（CLI 和 WebUI）使用统一的状态语义和转换规则。状态机由 `TaskHistoryManager` (SQLite) 持久化，`PipelineStats` 在内存中携带当前状态。

## 状态定义

```
                  ┌──────────┐
                  │  pending  │  任务已创建，等待调度
                  └────┬─────┘
                       │ dispatch
                       ▼
                  ┌──────────┐
                  │ preflight │  预检阶段（模型可用性检查、资源校验）
                  └────┬─────┘
                       │ preflight_ok
                       ▼
                  ┌──────────┐
                  │  running  │  主链执行中
                  └────┬─────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────┐
   │ completed  │ │ degraded │ │  failed  │
   │            │ │_completed│ │          │
   └────────────┘ └──────────┘ └──────────┘
                       ▲
                       │
                  ┌──────────┐
                  │cancelled │  用户主动取消
                  └──────────┘
```

## 状态详情

| 状态 | 含义 | 进入条件 | 退出条件 |
|------|------|----------|----------|
| `pending` | 已创建，等待调度 | 任务创建时 | 调度器分配资源后进入 `preflight` |
| `preflight` | 预检中 | 调度后、执行前 | 预检通过 → `running`; 预检失败 → `failed` |
| `running` | 执行中 | 预检通过 | 完成 → `completed`/`degraded_completed`; 异常 → `failed`; 用户取消 → `cancelled` |
| `completed` | 全链路正常完成 | 所有阶段正常结束 | 终态 |
| `degraded_completed` | 降级完成 | 部分引擎不可用但主链完成 | 终态 |
| `failed` | 执行失败 | 不可恢复的错误 | 终态（可重试创建新任务） |
| `cancelled` | 用户取消 | 用户主动取消 | 终态 |

## 转换规则

```yaml
transitions:
  pending:
    - preflight
    - cancelled          # 排队中取消
  preflight:
    - running
    - failed             # 模型缺失、资源不足等
    - cancelled
  running:
    - completed
    - degraded_completed
    - failed
    - cancelled
  completed: []           # 终态
  degraded_completed: []  # 终态
  failed: []              # 终态
  cancelled: []           # 终态
```

## 预检清单 (Preflight)

| 检查项 | 失败动作 | 错误分类 |
|--------|----------|----------|
| 输入文件存在且可读 | → `failed` | `input_missing` |
| 音频格式支持 | → `failed` | `format_unsupported` |
| 分离引擎可用 | → `degraded_completed` (若 skip_separation) | `separation_unavailable` |
| VAD 引擎可用 | → `failed` (VAD 不可降级) | `vad_unavailable` |
| 至少一个 ASR 引擎可用 | → `failed` | `asr_unavailable` |
| 磁盘空间充足 (> 输出预估 × 3) | → `failed` | `disk_space` |
| 输出目录可写 | → `failed` | `output_unwritable` |

## 降级完成 vs 失败

### 降级完成 (`degraded_completed`) 的条件

- 主链（分离 → VAD → ASR → 字幕构建 → 导出）产生有效输出
- 可选引擎不可用（如 FunASR 缺失、Qwen 超时）
- `fallback_category` 和 `fallback_reason` 已记录
- 输出字幕文件可正常打开

### 失败 (`failed`) 的条件

- 主链任一步骤不可恢复地失败
- 输出文件为空或损坏
- 任务执行超过资源边界（超时/OOM）
- 用户取消

## CLI/WebUI 统一状态

| CLI 行为 | WebUI 行为 | 状态 |
|----------|-----------|------|
| 显示 "排队中..." | 显示在队列中 | `pending` |
| 显示 "检查模型..." | 显示预检进度 | `preflight` |
| 显示进度条 | WebSocket 推送进度 | `running` |
| 显示完成 + 输出路径 | 显示下载按钮 | `completed` |
| 显示完成 + 降级警告 | 显示警告 + 下载按钮 | `degraded_completed` |
| 显示错误 + 退出码非零 | 显示错误消息 | `failed` |
| 响应 Ctrl+C | 点击取消按钮 | `cancelled` |

## 残留状态修复

`TaskHistoryManager.fixup_stale_running_tasks()` 在服务启动时自动将残留 `running` 状态修复为 `failed`，原因标注为 "Server restarted during task execution"。

## 错误分类

| 分类 | 含义 | 用户提示 |
|------|------|----------|
| `input_missing` | 输入文件不存在 | "找不到输入文件：{path}" |
| `format_unsupported` | 音频格式不支持 | "不支持的音频格式：{format}，支持 MP3/WAV/M4A/FLAC" |
| `model_missing` | 所需模型未下载 | "缺少模型：{model}，运行 `vocal-subtitle download-models`" |
| `dependency_missing` | 系统依赖缺失 | "缺少系统依赖：{dep}，运行 `bash install.sh`" |
| `resource_exhausted` | 内存/显存不足 | "资源不足，尝试使用更小的模型或 --cpu" |
| `engine_timeout` | 引擎执行超时 | "{engine} 执行超时（{timeout}s），已降级" |
| `recoverable_degradation` | 可恢复降级 | "部分可选功能不可用，继续生成基础字幕" |
| `unrecoverable_failure` | 不可恢复失败 | "处理失败：{reason}" |
