# 反馈学习闭环规范

**版本**: feedback-loop-v1
**更新日期**: 2026-08-02
**依据**: [ADR-003](../adr/003-feedback-shadow-consumption.md), [ADR-004](../adr/004-dataset-tiering.md)

## 1. 反馈生命周期

```
用户修订字幕
  │
  ├─ 1. 同意确认
  │     CLI: --consent local|anonymous|full
  │     WebUI: 勾选用途说明复选框
  │
  ├─ 2. 加密存储 + 样本 ID
  │     不可逆样本 ID = SHA256(audio_hash + revision_hash + timestamp)
  │
  ├─ 3. 自动质量检查
  │     对齐覆盖、空字幕、时间异常、文本长度异常
  │
  ├─ 4. 自动字幕 ↔ 人工修订对齐
  │     feedback/aligner.py: 动态时间扭曲 (DTW) 对齐
  │
  ├─ 5. 差异分类
  │     feedback/diff_analyzer.py: 内容更正/时间调整/格式偏好/角色标注/结构改写
  │
  ├─ 6. 人工审核
  │     接受 → 进入 D2
  │     拒绝 → 标记原因，保留作诊断
  │     待补充 → 标记 disputed
  │
  ├─ 7. 候选反馈集 D2
  │     脱敏、去重、质量门控
  │
  ├─ 8. 分层抽样 + 匿名化 + 二次标注
  │
  ├─ 9. 冻结为 D3 反馈回归集版本
  │
  └─ 10. 用于影子评估 + 场景 profile + 趋势分析
```

## 2. 反馈数据 Schema

```json
{
  "sample_id": "D2-20260802-a1b2c3d4e5f6",
  "data_source": "user_revision",
  "consent_level": "anonymous",
  "language": "zh",
  "scene": "podcast",
  "audio_duration_seconds": 1234.5,
  "audio_condition": "clean_indoor",
  "speaker_count": 2,
  "original": {
    "config_version": "default-v1",
    "engine": "faster-whisper",
    "model": "large-v3",
    "route_version": "asr-route-v1"
  },
  "automatic_subtitle": {
    "version": "auto-v1",
    "subtitle_hash": "sha256:...",
    "event_count": 156
  },
  "human_revision": {
    "version": "human-v1",
    "subtitle_hash": "sha256:...",
    "event_count": 148,
    "edit_types": {
      "text_correction": 12,
      "time_adjustment": 5,
      "format_preference": 3,
      "speaker_label": 8,
      "structural_rewrite": 0
    }
  },
  "alignment": {
    "method": "dtw",
    "coverage_ratio": 0.94,
    "confidence": 0.87
  },
  "review": {
    "reviewer": "admin",
    "result": "accepted",
    "timestamp": "2026-08-02T12:00:00Z"
  },
  "retention_days": 365,
  "anonymization": "deidentified"
}
```

## 3. 差异分类与消费策略

| 差异类型 | 示例 | 学习策略 | 生效范围 |
|----------|------|----------|----------|
| **内容更正** | 错字、漏词、错词 | → D3 回归样本 + 场景 profile | 经审核后全局 |
| **时间调整** | 边界偏移、拆分/合并 | → 参数学习 + 影子验证 | 场景级 |
| **格式偏好** | 标点风格、数字格式 | → 用户 profile 快速生效 | 用户级 |
| **角色标注** | SPEAKER_00 → "主持人" | → 用户偏好存储 | 用户级 |
| **结构改写** | 大幅重写、删减段落 | → 仅诊断，不自动学习 | N/A |

## 4. Profile 版本与回滚

### Profile 作用域

| 作用域 | 优先级 | 示例 |
|--------|--------|------|
| `user` | 最高 | 单个用户的偏好 |
| `scene` | 中 | podcast/education 场景 |
| `device` | 低 | GPU/CPU 配置差异 |
| `global` | 最低 | 全局默认 |

### 版本管理

```
profile-user-default-v1  →  profile-user-default-v2  →  profile-user-default-v3
                              差异: {param: old→new}
                              健康度: 0.85
                              可回滚: true
```

### 自动回退触发条件

- 底层模型版本变更（如 faster-whisper 升级）
- VAD 引擎替换
- 主链版本变更（config_version 不匹配）
- Profile 健康度连续 3 次低于 0.6

## 5. 影子模式验证

```
用户修订 → 生成建议 profile (v_next)
  │
  ├─ Shadow Mode: 对已授权历史样本运行 v_current vs v_next
  │    ├─ 受益场景: 覆盖率提高、错误率降低
  │    ├─ 退化场景: 覆盖率下降、新错误引入
  │    └─ 不确定: 样本量不足、场景不匹配
  │
  ├─ 无明显退化 + 样本量足够 + 场景明确 → 建议启用
  └─ 有退化 → 保持 shadow，生成诊断报告
```

## 6. 反馈入口

### CLI

```bash
vocal-subtitle feedback learn -a input.wav -r revised.srt --consent anonymous
vocal-subtitle feedback learn -a input.wav -r fixed.srt --dry-run
vocal-subtitle feedback show
vocal-subtitle feedback rollback
vocal-subtitle feedback export -o my_profile.yaml
```

### WebUI

- 上传修订字幕 → 关联原任务 → 展示用途说明
- 三个同意选项：不保留音频 / 仅本地分析 / 允许匿名化改进
- 预览差异对比 → 确认提交

## 7. 审核队列

```
GET /api/feedback/review-queue?status=pending
  → [{sample_id, diff_summary, confidence, submitted_at}]

POST /api/feedback/review/{sample_id}
  → {result: accepted|rejected|disputed, notes: "..."}
```

## 8. 当前实施状态

| 模块 | 状态 | 说明 |
|------|------|------|
| `feedback/aligner.py` | ✅ 已实施 | DTW 字幕对齐 |
| `feedback/diff_analyzer.py` | ✅ 已实施 | 差异分类与归因 |
| `feedback/param_learner.py` | ✅ 已实施 | 参数学习引擎 |
| `feedback/shadow_mode.py` | ✅ 已实施 | 影子模式验证 |
| `feedback/health_scorer.py` | ✅ 已实施 | 健康度评分 |
| `feedback/audio_fingerprint.py` | ✅ 已实施 | 音频指纹匹配 |
| `feedback/user_profile.py` | ✅ 已实施 | Profile 持久化 |
| `feedback/sample_manager.py` | ✅ 已实施 | D2 样本存储、质量门控、审核队列 API |
| `feedback/stratified_sampler.py` | ✅ 已实施 | D2→D3 分层抽样引擎 |
| `feedback/anonymizer.py` | ✅ 已实施 | 反馈数据脱敏 |
| `feedback/pipeline_stage.py` | ✅ 已实施 | 管道反馈学习 + D2 自动入库 |
| CLI feedback 命令 | ✅ 已实施 | learn/show/rollback/reset/fingerprints/export/import + sample plan/freeze |
| WebUI 反馈入口 | ✅ 已实施 | routes_feedback.py (含审核队列 API + D3 抽样端点) |
| 反馈审核队列 | 🔧 部分实施 | 后端 API 已实施，前端审核界面待开发 |
| D2 自动入库 | 🔧 部分实施 | 核心类已实施（sample_manager.py），pipeline/CLI/WebUI 已接入 |
| D3 分层抽样 | 🔧 部分实施 | 核心类已实施（stratified_sampler.py），CLI + WebUI 端点已接入 |

## 9. 必做检查

1. 反馈对齐质量门控：覆盖率 < 0.7 → 拒绝入库
2. 差异归因置信度 < 0.5 → 标记低可信，降权
3. 结构性修订检测 → 隔离，不自动学习
4. Profile 回滚 → 保留回滚前版本快照
5. 过期的 profile（30 天无活动）→ 标记 inactive
