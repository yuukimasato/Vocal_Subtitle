# 阶段五开发说明：全局 ASR 默认路径

## 目的

离线任务默认优先使用全局 ASR、词级物理分配和严格物理校验。全局路径不可用或失败时，系统使用旧分段 ASR 完整降级，并在结果中明确说明。

## 路径选择

配置入口位于 `pipeline.asr.global`：

```yaml
global:
  enabled: true
  backend: whisperx
  fallback_to_segmented: true
```

任务级路径支持：

- `auto`：离线默认行为，优先全局，失败后 legacy；
- `global`：要求全局路径成功，禁止静默回退；
- `segmented`：显式使用旧分段 ASR。

流式模式始终使用 `segmented`，不会加载 WhisperX。

## 结果诊断

查看 `PipelineStats` 或 WebUI 任务结果中的：

- `asr_path`：`global`、`legacy` 或 `legacy_degraded`；
- `global_attempted`：是否尝试过全局路径；
- `fallback_category`：依赖、资源、执行、结果或显式 legacy；
- `fallback_reason`：可操作的失败摘要；
- `global_diagnostics`：窗口、对齐、分配和缓存信息。

`legacy_degraded` 表示本次任务由于全局路径失败而回退，不表示字幕生成失败。`global` 路径成功时，事件应包含 `source_word_ids` 和 `physical_spans`。

## 常见问题

### WhisperX 未安装

安装可选依赖后重试：

```bash
uv pip install -e '.[whisperx]'
```

未安装时 `auto` 会输出 `dependency_unavailable` 并回退到 legacy。若使用 `global`，任务会失败并保留诊断。

### 需要临时回退旧路径

使用 CLI 的 `--asr-path segmented`，或在自定义配置中关闭 `pipeline.asr.global.enabled`。该设置只影响当前配置/任务，不删除全局缓存。

### 缓存结果看起来仍是旧路径

检查任务结果中的 `asr_path` 和 `global_diagnostics.cache_hit`。全局配置、路径策略或结果 schema 变化会使不兼容的全流水线缓存失效；旧结果不会被默认全局请求直接复用。

## 基准验收

### 真实素材验收

仓库中的真实音频和人工字幕由 `test/quality_manifest.yaml` 明确配对。运行全部
素材的 legacy 基线：

```bash
venv/bin/python scripts/run_quality_benchmark.py \
  --manifest test/quality_manifest.yaml \
  --config configs/default.yaml \
  --asr-path segmented \
  --output-dir test/benchmark_results/real_material
```

运行 global 发布门（需要已安装 WhisperX 和所需模型）：

```bash
venv/bin/python scripts/run_quality_benchmark.py \
  --manifest test/quality_manifest.yaml \
  --config configs/default.yaml \
  --asr-path auto \
  --require-global --ci \
  --output-dir test/benchmark_results/real_material
```

`auto` 的 fallback 结果可以用于回归比较，但不会计入 global 发布通过。每个场景
的字幕、对比报告和诊断报告位于输出目录下，汇总文件为
`summary-auto.json` 或 `summary-segmented.json`。

执行路径对照：

```bash
uv run python scripts/run_benchmarks.py \
  --benchmark-dir tests/benchmarks \
  --config configs/default.yaml \
  --asr-path auto \
  --output-dir test/benchmark_results/auto
```

发布前至少需要单人长段、多人交替、重叠/抢话三类真实素材，并为每个场景提供音频、ground truth 和 metadata。CI 模式在素材不足、ground truth 缺失或指标不达标时返回失败；空 benchmark 目录不会被视为通过。

## 回滚

将全局路径关闭或选择 `segmented` 即可恢复旧默认行为：

```yaml
global:
  enabled: false
```

回滚不需要删除用户反馈学习配置、分阶段缓存或任务历史。修复全局路径后重新运行 benchmark，再恢复 `enabled: true`。
