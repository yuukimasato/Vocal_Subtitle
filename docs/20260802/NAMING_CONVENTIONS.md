# 命名规范与运行标识约定

**更新日期**: 2026-08-02
**适用范围**: 任务命名、运行 ID、配置快照、模型标识、报告位置、变更日志

## 1. 任务命名

### 格式

```
{task_type}-{date}-{id_digest}
```

- `task_type`: `offline` | `streaming` | `batch`
- `date`: `YYYYMMDD`
- `id_digest`: 输入文件路径的 SHA256 前 8 位

### 示例

```
offline-20260802-a1b2c3d4
```

### 实现位置

- `vocal_subtitle/utils/session_manager.py` — `SessionManager.create_task_id()`

---

## 2. 运行 ID

### 格式

```
run-{task_id}-{timestamp}
```

- `task_id`: 父任务 ID
- `timestamp`: Unix 毫秒时间戳

### 示例

```
run-offline-20260802-a1b2c3d4-1690972800000
```

### 使用场景

- 每次 Pipeline 执行产生唯一 `run_id`
- 写入运行报告、缓存 key 和日志 context
- WebUI 任务历史以 `run_id` 为主键关联

---

## 3. 配置快照

### 格式

配置快照随运行报告持久化，包含：

```yaml
config_snapshot:
  config_profile: "default"          # 使用的场景模板
  config_overrides: {}               # 用户覆盖的参数
  config_version: "default-v1"       # 配置版本标识
  snapshot_digest: "sha256:..."      # 完整配置内容的哈希
```

### 存储位置

- 运行报告: `cache/reports/{run_id}/run_report.json`
- 快照文件: `cache/reports/{run_id}/config_snapshot.yaml`

---

## 4. 模型标识

### 格式

```
{engine}:{model_name}@{quantization}
```

### 示例

```
faster-whisper:large-v3@float16
faster-whisper:tiny@int8
qwen-asr:Qwen3-ASR-1.7B@bf16
funasr:paraformer-zh@fp32
```

### 写入位置

- 运行报告的 `engine_availability` 字段
- 反馈样本的 `model_version` 字段
- 数据集版本的依赖标注

---

## 5. 报告位置

### 目录结构

```
cache/reports/{run_id}/
├── run_report.json             # 统一运行报告（schema: run-report-v1）
├── config_snapshot.yaml        # 配置快照
├── engine_availability.json    # 引擎可用性快照
├── degradation_log.jsonl       # 降级事件日志
├── coverage_audit.json         # 物理覆盖审计
├── decision_trace.jsonl        # 证据决策追踪
├── stage_timings.json          # 各阶段耗时
├── output/                     # 输出文件
│   ├── subtitle.srt
│   ├── subtitle.vtt
│   └── subtitle.ass
└── diagnostics/                # 诊断制品
    ├── skeleton_segments/      # 骨架分段（调试用）
    └── acoustic_report.json    # 声学校验报告
```

### 查找规则

1. 按 `run_id` 直接定位: `cache/reports/{run_id}/`
2. 按 `task_id` 模糊搜索: 列出 `cache/reports/run-{task_id}-*/`
3. WebUI 历史视图通过 SQLite 任务历史索引

---

## 6. 变更日志格式

### Git Commit 规范

```
{type}({scope}): {summary}

{body}

ADR: {adr-ref}        # 可选，关联 ADR
Co-Authored-By: Claude <noreply@anthropic.com>
```

- `type`: `feat` | `fix` | `refactor` | `docs` | `test` | `chore` | `revert`
- `scope`: 受影响的子系统（如 `asr`, `physical`, `mapping`, `webui`, `config`）
- `summary`: 简要描述（中文或英文，当前项目惯例为中文）

### 示例

```
feat(physical): 添加 PhysicalTimeline 硬静音边界门控

在 DecisionEventProjector 中增加硬静音检查，
禁止后处理跨物理 bin 合并字幕事件。

ADR: ADR-002
Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 7. 版本策略标识符

项目中使用的版本标识符及其命名约定：

| 标识符 | 命名格式 | 示例 | 更新触发条件 |
|--------|----------|------|-------------|
| `route_version` | `asr-route-v{N}` | `asr-route-v1` | 路由逻辑变更 |
| `quality_gate_version` | `asr-quality-v{N}` | `asr-quality-v1` | 质量门禁规则变更 |
| `review_policy_version` | `review-policy-v{N}` | `review-policy-v1` | 复核策略变更 |
| `risk_policy_version` | `risk-policy-v{N}` | `risk-policy-v1` | 风险评估逻辑变更 |
| `decision_policy_version` | `decision-policy-v{N}` | `decision-policy-v1` | 决策逻辑变更 |
| `evidence_schema_version` | `evidence-v{N}` | `evidence-v1` | 证据数据结构变更 |
| `golden_quality_gate_version` | `golden-quality-v{N}` | `golden-quality-v1` | 黄金质量门禁变更 |
| `run_report_schema` | `run-report-v{N}` | `run-report-v1` | 运行报告结构变更 |

版本号递增规则：主版本号（v1 → v2）表示不向后兼容的变更。

---

## 8. 数据集版本命名

```
{layer}-{date}-{id}
```

- `layer`: `D0` | `D1` | `D2` | `D3` | `D4`
- `date`: `YYYYMMDD`（冻结日期）
- `id`: 递增序号

### 示例

```
D1-20260802-001
D3-20260901-001
```

---

## 9. Profile 版本命名

```
profile-{scope}-{name}-v{version}
```

- `scope`: `user` | `scene` | `device` | `global`
- `name`: 唯一名称
- `version`: 递增版本号

### 示例

```
profile-user-default-v3
profile-scene-podcast-v1
profile-global-quality-first-v2
```
