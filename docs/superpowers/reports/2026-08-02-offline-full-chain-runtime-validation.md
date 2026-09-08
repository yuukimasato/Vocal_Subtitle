# 离线字幕全链路运行时验收报告

日期：2026-08-02

结论：**全链路可运行，但当前不可发布**。

## 验收分层

| 层级 | 结果 | 证据 |
|---|---|---|
| runnable | 通过 | CLI、Web 后端上传/轮询/下载、浏览器上传/处理/结果展示均已完成真实运行；当前 Web 服务仍运行在 `127.0.0.1:8765` |
| benchmarked | 通过 | `1005 passed, 1 skipped`，全量 pytest 用时约 6.82 秒 |
| gated | 未通过 | 10 个黄金场景已用 `faster-whisper large-v3 + Qwen risk_only` 重新生成并执行 `--ci`，返回 `exit=3` |
| release-default | 不允许 | 真实语音漏检率超过门槛，不能将当前结果标记为生产质量通过 |

## 全量测试

执行：

```bash
./.venv-production/bin/python -m pytest -q
```

结果：`1005 passed, 1 skipped`。

## 黄金集

本轮使用 `test/quality_manifest.yaml` 的 10 个场景，其中 8 个有人工参考、2 个无人工参考。生产配对为：

- 主引擎：`faster-whisper`
- 主模型：`large-v3`，CPU/int8
- 副引擎：`qwen`
- 策略：`risk_only`
- Context Re-ASR：启用

产物（`docs/superpowers/reports/` 下，input/quality 报告体积较大且含本机路径，仅保留在本地，可用 `scripts/run_offline_golden_production.py` 复现）：

- golden input: `2026-08-02-offline-production-paired-golden-input.json`
- quality report: `2026-08-02-offline-production-paired-golden-quality-report.json`

门禁结果：

| 指标 | 实测 | 阈值 | 结论 |
|---|---:|---:|---|
| real speech drop rate | 0.175000 | 0.05 | 未通过 |
| hallucination retention rate | 0.000000 | 0.05 | 通过 |
| physical violation rate | 0.000000 | 0 | 通过 |
| cross silence rate | 0.000000 | 0 | 通过 |
| unresolved rate | 0.000000 | 0.30 | 通过 |
| split rate | 0.013298 | 0.80 | 通过 |
| drop rate | 0.000000 | 0.50 | 通过 |
| trace missing rate | 0.000000 | 0 | 通过 |
| raw bypass count | 0 | 0 | 通过 |
| diagnostics complete | true | true | 通过 |

本轮在同一 10 场景清单上完成了 `faster-whisper large-v3 + Qwen risk_only` 的 authoritative 运行。物理违规、跨静音、幻觉保留、决策追踪和 raw bypass 均通过；真实语音漏检/不匹配为 `21/120`，集中在 QS、181、培训测试、英文多人、长实录和重复短语场景。归因显示 `drop=0`、`unresolved=0`，问题主要在 ASR 候选漏词以及后处理合并后时间覆盖不足，不能通过放宽黄金集阈值解决。

## 启动器

已更新仓库启动器和桌面实际副本（本机 `.desktop` 文件，不入库）：

- 仓库启动器: `Vocal_Subtitle.desktop`
- 桌面启动器: `~/桌面/Vocal_Subtitle.desktop`

两者都使用 `.venv-production/bin/vocal-subtitle-gui`，绑定 `127.0.0.1:7860`，不再依赖旧 `.venv` 或 shell 激活脚本；`desktop-file-validate` 和 GUI 命令入口检查通过。

## 下一步

保持当前 `faster-whisper large-v3` 为生产默认候选，继续修复人工参考场景的漏检/文本不匹配；修复后用同一 10 场景清单重新生成输入并执行同一 `--ci` gate。黄金集在 gate 通过前不标记为 `gated` 或 `release-default`。
