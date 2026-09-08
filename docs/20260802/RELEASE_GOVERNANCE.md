# 发布、支持与维护治理

**版本**: release-governance-v1
**更新日期**: 2026-08-02

## 1. 发布状态分类

| 状态 | 含义 | 适用条件 |
|------|------|----------|
| `development` | 开发验证中 | 功能开发阶段，通过工程验证 |
| `production-usable` | 正常需求可用 | 基础任务可完成、关键失败可解释、输出不损坏 |
| `quality-improving` | 持续优化中 | 在 production-usable 基础上持续提升质量 |

## 2. production-usable 最低标准

- [x] 安装脚本 `bash install.sh --production` 可成功执行 (✅)
- [x] 依赖锁定 (`pyproject.toml` + `uv.lock`) (✅)
- [x] 默认配置任务可完成并导出字幕 (✅)
- [x] CLI `vocal-subtitle run input.mp3 -o output.srt` 可运行 (✅)
- [x] WebUI `vocal-subtitle-gui` 可启动并提供图形界面 (✅)
- [x] 可选引擎缺失时主链不崩溃，正确降级 (✅)
- [x] 失败任务返回可行动的错误原因（非神秘崩溃）(✅)
- [x] 人工修订入口可用（CLI feedback + WebUI 上传修订）(✅)
- [ ] D0 回归测试全部通过（当前有 pending 修复）
- [ ] 少量 D1 回放验证
- [ ] CLI/Web 冒烟测试通过

## 3. 发布前检查单

### 每次发布前必做

```markdown
## 发布检查单 — v{version}

### 依赖与安装
- [ ] `pyproject.toml` 依赖版本已锁定
- [ ] `uv.lock` 已更新
- [ ] `bash install.sh --production --cpu` 全新安装成功
- [ ] `pip install -e ".[all]"` 无冲突

### 模型可用性
- [ ] faster-whisper large-v3 可正常运行
- [ ] Silero VAD 模型可正常加载
- [ ] FunASR 模型路径配置正确（可选）
- [ ] Qwen3-ASR 模型路径配置正确（可选）

### 自动化测试
- [ ] `pytest tests/ -x --tb=short` 全部通过
- [ ] `pytest tests/test_physical/ -v` 物理层测试通过
- [ ] `pytest tests/test_asr/test_router.py -v` 路由测试通过
- [ ] `pytest tests/test_cli.py -v` CLI 测试通过

### D0 回归
- [ ] `python scripts/run_quality_benchmark.py` 无可检测回归
- [ ] D0 样本无覆盖率下降 > 5%
- [ ] D0 样本无新增幻觉

### 冒烟测试
- [ ] CLI: `vocal-subtitle run test/中文多人员测试音频.wav -o /tmp/test.srt`
- [ ] CLI: `vocal-subtitle info` 无异常
- [ ] WebUI: 启动 → 上传 → 处理 → 下载 完整流程
- [ ] WebUI: 字幕编辑功能可用
- [ ] WebUI: 反馈上传功能可用

### 导出兼容
- [ ] SRT 格式可用标准播放器打开
- [ ] VTT 格式可用浏览器打开
- [ ] ASS 格式可用 Aegisub 打开

### 文档
- [ ] README 示例命令可执行
- [ ] ARCHITECTURE_STATE.md 已更新
- [ ] DOCUMENT_INDEX.md 已更新
- [ ] 版本号 `pyproject.toml version` 已更新
```

## 4. 发布后观测

| 指标 | 数据来源 | 告警阈值 |
|------|----------|----------|
| 崩溃/失败率 | `TaskHistoryManager.status=failed` 计数 | > 10% |
| 任务平均耗时 | `PipelineStats.total_time` | 较上一版本 > +50% |
| 降级率 | `production_path=*fallback` 占比 | > 20% |
| 导出失败 | 导出阶段失败计数 | > 5% |
| 用户修订率 | `feedback learn` 调用频率 | 趋势上升 |
| 高严重度问题 | 人工抽检发现 | > 0 |

## 5. 回滚机制

### 配置回滚（优先）

```bash
# 回滚到上一版本配置
vocal-subtitle feedback rollback

# 或手动指定
vocal-subtitle run input.wav --config configs/default.yaml.backup
```

配置回滚是首选方案：不改变代码，仅恢复配置参数。

### 模型/引擎禁用

```yaml
# configs/default.yaml
evidence_review:
  qwen_enabled: false          # 独立禁用 Qwen
  forced_aligner_enabled: false
  sed_enabled: false
```

每个可选引擎有独立开关，无需修改代码即可禁用。

### 代码回滚

```bash
git revert <commit>
# 或
git checkout <previous-tag>
```

### 数据集结论回滚

如果 D1/D3 评估结论因数据错误而误判：
- 标记该版本数据集为 `superseded`
- 在新数据集版本下重新评估
- 撤回基于错误数据的发布决策

## 6. 已知限制清单

以下限制在发布说明中明确告知用户：

| 限制 | 说明 | 缓解措施 |
|------|------|----------|
| 多人重叠场景 (>3人) | 说话人标注可能不准确 | 启用 pyannote 全局聚类改善 |
| 高背景噪声 | ASR 准确率下降 | 启用噪声抑制预处理 |
| 中英频繁切换 | 语言检测可能出错 | 手动指定 `--language mixed` |
| 长音频 (>2h) | 处理时间线性增长，可能触发超时 | 启用宏观切块分治 |
| 专业词汇 | 可能识别错误 | 考虑启用 LLM 后处理 |
| 远场录音 | VAD 可能漏检远距离语音 | 降低 VAD 阈值 |
| 无 GPU 设备 | CPU 推理速度慢 5-10x | 使用 whisper.cpp 引擎 |
| 1GB 以下显存 | 大模型无法加载 | 使用 tiny/small 模型或 CPU |

## 7. 版本说明模板

```markdown
# Vocal Subtitle v{version} 发布说明

## 发布状态
production-usable / quality-improving

## 主要变更
- **新增**: ...
- **修复**: ...
- **优化**: ...
- **废弃**: ...

## 升级注意事项
- 配置变更: ...
- 模型变更: ...
- API 变更: ...

## 已知问题
1. ...
2. ...

## 安装
```bash
bash install.sh --production
```

## 验证
```bash
pytest tests/ -x
```
```

## 8. 支持手册

### 常见问题

| 问题 | 诊断 | 解决 |
|------|------|------|
| "模型未找到" | 检查 `~/.cache/vocal-subtitle/` | `vocal-subtitle download-models --all` |
| "CUDA out of memory" | 检查 `nvidia-smi` | `--device cpu` 或 `--model small` |
| "ffmpeg not found" | `which ffmpeg` | `sudo apt install ffmpeg` |
| "端口被占用" (WebUI) | `lsof -i :7860` | `vocal-subtitle-gui --port 8080` |
| "处理超时" | 检查音频长度和可用内存 | 启用 `macro_chunking` 分治 |
| "字幕时间偏移" | 对比参考字幕 | 检查 `acoustic_validation.report` |

### 日志位置

| 日志 | 路径 |
|------|------|
| 管道日志 | `logs/pipeline.log` |
| WebUI 日志 | stdout (uvicorn) |
| 任务历史 | `cache/task_history.db` |
| 运行报告 | `cache/reports/{run_id}/run_report.json` |
| 声学校验 | `cache/reports/{run_id}/diagnostics/acoustic_report.json` |

## 9. 当前发布状态

| 项目 | 状态 |
|------|------|
| 当前版本 | 0.1.0 (Alpha) |
| 发布状态 | `development` |
| 目标状态 | `production-usable` |
| 阻塞项 | D0 回归测试未完全通过、D1 回放未建立、冒烟测试未完整覆盖 |
