# 声学发布门禁验收报告

**日期：** 2026-07-27
**状态：** 阻塞，不可宣称影视级发布

## 已完成

- 新增 `acoustic-gold-v1` JSON Schema 和严格 Python 评估器。
- `BoundaryArbiter` 逐候选保存 ASR、RMS、VAD、FFmpeg、局部噪声、来源一致性特征及分项评分。
- 全局路径在没有直接 detector 参数时，会从绝对时间 `PhysicalTimeline` 重建 VAD/FFmpeg 证据，并计算局部噪声档案。
- 8 个真实媒体均使用修复后代码完成一次真实模型回归，输出位于 `test/benchmark_results/real_material-20260727/`。

## 自动验证

- 定向评估/仲裁/对齐/噪声测试：`33 passed`。
- 8 媒体回归：`8/8` Pipeline 成功导出。
- 实际 ASR 路径：全部为 `segmented`；当前 CPU 环境没有证明 global 路径可用。
- 6 个样本有人工成片字幕参考，2 个样本仅完成稳定性运行，没有人工参考。

## 发布阻塞项

1. 当前没有任何人工词级 acoustic gold 文件，故无法计算真实 onset/offset median、P95、切头率和切尾率。
2. 6 个成片字幕对比不是词级声学真值，重跑汇总的 aggregate start MAE 为 `500.1ms`，end MAE 为 `616.0ms`，最小内容覆盖率为 `0.805`；这些结果不能通过 `80/160ms` 声学门禁。
3. 8 个媒体实际均走 segmented，global + 完整 BoundaryArbiter 的真实模型 rollout 仍未被证明。
4. 两个媒体没有人工字幕参考，不能计入发布精度门禁。

因此本报告只证明代码契约、逐候选诊断和 8 媒体重跑已经执行，不能证明影视级发布标准已达成。
