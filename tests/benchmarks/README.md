# 基准数据集 (Benchmarks)

多场景字幕时间轴精度基准数据集，用于回归测试和参数调优。

> 文档参考：[字幕时间轴精度优化方案](../../docs/字幕时间轴精度优化方案.md) 第 5.8.2 节

## 目录结构

```
tests/benchmarks/
├── hotel_front_desk/       # 酒店前台（当前测试样本）
├── studio_monologue/       # 录音室独白（单人旁白/有声书）
├── meeting_3_speakers/     # 三人会议
├── outdoor_interview/      # 户外采访
├── podcast_conversation/   # 播客对话
└── music_voiceover/        # 带背景音乐
```

## 每个样本应包含

| 文件 | 说明 |
|------|------|
| `audio.wav` | 原始音频（16kHz 单声道） |
| `vocals.wav` | 人声分离后音频（如适用） |
| `ground_truth.ass` | 手动校正的 ASS 字幕（Aegisub 逐条精校） |
| `metadata.yaml` | 场景描述、说话人数、语言、时长、环境类型 |

## metadata.yaml 示例

```yaml
# hotel_front_desk/metadata.yaml
scene: hotel_front_desk
description: 酒店前台电话预订对话
duration_seconds: 139
speakers: 2
languages: ["zh", "en"]
environment: quiet_indoor
noise_level: low
ground_truth_source: Aegisub manual correction
correction_date: 2026-07-04
```

## 使用方法

```bash
# 全场景基准测试
python scripts/run_benchmarks.py \
    --benchmark-dir tests/benchmarks/ \
    --config configs/default.yaml \
    --output-dir test/benchmark_results/ \
    --metrics start_mae,end_mae,end_error_gt_300ms,health_score

# 输出：
# benchmark_results/
# ├── summary.json           # 全场景汇总
# ├── hotel_front_desk/
# │   ├── subtitle.ass
# │   ├── comparison.json
# │   └── diagnostic_report.json
# └── ...
```

## 状态

当前仅有 hotel_front_desk 样本（项目测试音频，139 秒酒店对话）。
其余场景待后续收集补充。
