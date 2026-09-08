# 质量运营规范

**版本**: quality-ops-v1
**更新日期**: 2026-08-02
**依据**: [ADR-004](../adr/004-dataset-tiering.md)

## 1. 质量问题分类

| 类别 | 定义 | 严重度评估 | 示例 |
|------|------|-----------|------|
| **可用性 (Usability)** | 系统能否正常运行和产出 | Critical/High | 崩溃、无法导出、输出损坏 |
| **完整性 (Completeness)** | 是否遗漏语音内容 | High/Medium | 漏句、漏词、跳段 |
| **文本准确性 (Text Accuracy)** | 识别文本是否准确 | Medium/Low | 错字、同音字、多字少字 |
| **时间准确性 (Time Accuracy)** | 时间轴是否精确 | Medium/Low | 偏移、跨静音、过度拉伸 |
| **说话人 (Speaker)** | 说话人标注是否正确 | Medium/Low | 错标、漏标、重复 |
| **格式可读性 (Readability)** | 断句、标点、长度是否合理 | Low | 过长字幕、不当断句 |
| **性能 (Performance)** | 处理速度、资源消耗 | Medium | 超时、OOM、4h+ 处理 |
| **成本 (Cost)** | LLM API 调用、计算资源 | Low | 高 API 费用 |
| **可解释性 (Explainability)** | 输出能否被理解和审计 | Medium | 无 trace、来源不明 |

## 2. 场景切片

| 切片维度 | 取值 |
|----------|------|
| 语言 | zh, en, mixed, other |
| 说话人数 | single, dual, multi (3+) |
| 背景噪声 | clean, light_noise, heavy_noise, music |
| 语速 | slow (<3 wps), normal (3-5), fast (>5) |
| 音频长度 | short (<3min), medium (3-30min), long (30-120min), very_long (>2h) |
| 设备 | cpu, gpu_8gb, gpu_12gb+, mac_mps |
| 场景 | podcast, education, variety_show, music_live, meeting, outdoor |

## 3. 版本趋势报告

每个版本发布时输出趋势报告（不追求单一总分）：

```yaml
version: "0.2.0"
date: "2026-08-15"
baseline: "0.1.0"

trends:
  D0_engineering:
    test_pass_rate: 0.98           # 上一版本 0.97
    regression_count: 0            # 新增回归
    build_time_seconds: 45         # 上一版本 42

  D1_reference:
    scenarios: 6
    coverage_avg: 0.92             # 上一版本 0.90
    text_accuracy_avg: 0.87        # 上一版本 0.85
    time_mae_avg_ms: 320           # 上一版本 350
    failures: []
    regressions:
      - scene: "英文多人"
        metric: "coverage"
        change: "0.95 → 0.91"
        severity: "medium"

  D3_feedback:                     # 尚未建立时为 N/A
    status: "not_available"

  D4_challenge:
    scenarios: 2
    long_audio_pass: true
    heavy_noise_degraded: false
    mixed_language_acceptable: true

  operations:
    crash_rate: 0.001
    degradation_rate: 0.05
    avg_duration_seconds: 120
    p95_duration_seconds: 450
    user_revision_rate: "N/A"      # 依赖反馈数据
```

## 4. 问题优先级公式

```
优先级 = 影响范围 × 用户严重度 × 可复现性 × 修复置信度
```

| 因子 | 权重 | 评分标准 |
|------|------|----------|
| 影响范围 | 1-5 | 1=单一场景, 3=多场景, 5=所有用户 |
| 用户严重度 | 1-5 | 1=格式偏好, 3=漏句, 5=输出损坏 |
| 可复现性 | 1-3 | 1=偶发, 2=条件触发, 3=必然复现 |
| 修复置信度 | 1-3 | 1=不确定, 2=有方案, 3=确定性修复 |

优先级 ≥ 30：立即修复。15-29：本版本修复。≤ 14：排期。

## 5. 对照运行规范

每次质量改动必须对照运行：

```bash
# 1. 基线运行
vocal-subtitle run input.wav -o baseline/ --config configs/default.yaml

# 2. 候选运行
vocal-subtitle run input.wav -o candidate/ --config configs/default.yaml \
    --override asr.model=large-v3-turbo

# 3. Shadow 运行（可选，用于生产数据）
vocal-subtitle run input.wav -o shadow/ --config configs/default.yaml \
    --override evidence_review.shadow_mode=true

# 4. 对照报告
python scripts/compare_timeline.py \
    --auto candidate/subtitle.srt \
    --ground-truth baseline/subtitle.srt \
    --report comparison-report.json
```

对照报告内容：
- 受益场景和量化改善
- 退化场景和量化退化
- 不确定性和需要更多样本的领域
- 是否建议启用

## 6. 数据版本管理

| 数据集 | 版本号 | 冻结日期 | 样本数 | 变更说明 |
|--------|--------|----------|--------|----------|
| D0-v1 | D0-20260802-001 | 2026-08-02 | 13 音频 + 10 manifest 场景 | 初始登记 |
| D1 | 待建立 | - | - | - |
| D2 | 待建立 | - | - | - |
| D3 | 待建立 | - | - | - |
| D4 | 待建立 | - | - | - |

## 7. 月度质量报告模板

```markdown
# 月度质量报告 — 2026年8月

## 总体状态
- 生产可用性: production-usable
- 关键问题: 0
- 高优先级问题: 2
- 本月解决: 5

## 版本对比 (v0.1.0 → v0.2.0)
- D0 回归: 0 新增
- D1 覆盖率: 0.90 → 0.92 (+2%)
- D1 文本准确率: 0.85 → 0.87 (+2%)
- 性能: avg 120s → 110s (-8%)

## 用户反馈统计
- 本月反馈数: N/A
- 修订率: N/A
- 主要投诉: N/A

## 已知问题 Top 5
1. [HIGH] 多人重叠场景说话人错标 — 影响范围 3, 严重度 3
2. [MEDIUM] 长音频 (>30min) 尾部漏句 — 影响范围 2, 严重度 4
3. ...

## 下月计划
- 优化多人重叠场景
- 建立 D1 参考回归集
- Qwen 复核从 shadow 进入 review
```

## 8. 样本归档策略

- D0: 不归档（随功能演进增减）
- D1: 每季度审查，移除不再有代表性的样本
- D2: 超过保留期限（默认 365 天）自动清理
- D3: 版本冻结后只增量，不删除（保留回滚能力）
- D4: 按场景更新，旧样本标记为 superseded

避免反馈集被某类用户、某个语言或某个模型版本过度主导（任何单一维度占比不超过 60%）。
