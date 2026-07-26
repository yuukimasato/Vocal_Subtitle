# Vocal Subtitle 部署指南

## 一键部署

源码部署需要 Python 3.10-3.13 和 ffmpeg；默认安装还会安装 WhisperX 及其
PyTorch CPU/CUDA 运行时：

```bash
bash install.sh
source venv/bin/activate
vocal-subtitle-gui
```

默认安装包含 WhisperX 全局 ASR、faster-whisper legacy fallback、WebRTC VAD、CLI
和 Web GUI。没有 NVIDIA 驱动时使用 CPU；CUDA 运行时安装失败会自动回退 CPU。
如需不安装 WhisperX/PyTorch 的轻量旧路径，使用 `bash install.sh --no-torch`。
模型在第一次使用时下载；`--download-models` 可在安装阶段预下载 faster-whisper 模型。

带背景音乐的原始音频安装 UVR 扩展：

```bash
bash install.sh --profile separation
```

其他扩展：

```bash
bash install.sh --profile gpu
bash install.sh --profile gpu-monitor
bash install.sh --profile silero-vad
bash install.sh --profile diarization
bash install.sh --profile llm
bash install.sh --profile local-nlp
bash install.sh --profile full

# 只安装轻量 legacy 路径
bash install.sh --no-torch
```

## Debian 安装

构建和安装：

```bash
bash build-deb.sh --check
bash build-deb.sh
sudo apt install ./vocal-subtitle_0.1.0_all.deb
```

安装包会创建 `base + whisperx` Python 环境并启动本机 GUI：

```text
http://127.0.0.1:7862
```

扩展或修复（安装脚本会复用同一个 venv）：

```bash
sudo vocal-subtitle-setup --profile separation
sudo vocal-subtitle-setup --profile gpu
sudo vocal-subtitle-setup --profile silero-vad
sudo vocal-subtitle-setup --profile full --download-models

# 新安装时跳过 WhisperX/PyTorch；已有包不会被卸载
sudo vocal-subtitle-setup --profile base --no-torch
```

## 目录

| 路径 | 用途 |
| --- | --- |
| `/usr/share/vocal-subtitle` | 只读项目源码和资源 |
| `/var/lib/vocal-subtitle/venv` | Debian 专用 Python 虚拟环境 |
| `/etc/vocal-subtitle` | 服务环境配置 |
| `/var/cache/vocal-subtitle` | 任务和模型缓存 |
| `/var/log/vocal-subtitle` | 服务日志 |

## 服务管理

```bash
systemctl status vocal-subtitle-gui
systemctl restart vocal-subtitle-gui
journalctl -u vocal-subtitle-gui -f
```

服务默认只监听 `127.0.0.1`。需要远程访问时，应在 systemd override 中显式配置监听地址，并确认防火墙和访问控制策略。

## 升级和卸载

普通升级会保留虚拟环境、配置和模型缓存：

```bash
sudo apt install ./vocal-subtitle_0.1.0_all.deb
```

普通卸载保留生成数据；完全清理使用：

```bash
sudo apt purge vocal-subtitle
```

## 常见问题

- `ffmpeg not found`：安装 `sudo apt install ffmpeg`，或在容器中使用 `--no-system-deps`。
- GUI 启动失败：先查看 `journalctl -u vocal-subtitle-gui -e`，再运行 `sudo vocal-subtitle-setup --profile base` 修复。
- 原始音频无法分离：安装 `sudo vocal-subtitle-setup --profile separation`，然后重新处理。
- CPU 速度较慢：基础配置使用 `small` + `int8`；有兼容 GPU 时安装 `gpu` profile，并在配置或命令行选择 GPU。
- pyannote 模型需要授权：安装 `diarization` 后，根据 Web GUI 提示配置 Hugging Face Token；基础字幕流程不依赖它。

模型不会打包进 `.deb`，以避免包体过大和模型协议混入系统包。模型下载位置由 `VOCAL_SUBTITLE_MODEL_DIR` 或应用默认缓存决定。
# 阶段五离线路径说明

离线模式默认优先尝试全局 ASR + 物理对齐。WhisperX、alignment 模型或运行资源
不可用时，系统会完整回退到旧分段 ASR，并在 `PipelineStats`、CLI、WebUI 和
任务历史中记录 `asr_path`、`fallback_category` 与 `fallback_reason`。

临时使用旧路径：

```bash
vocal-subtitle run input.wav --asr-path segmented
```

强制全局路径并禁止静默降级：

```bash
vocal-subtitle run input.wav --asr-path global
```

安装全局 ASR 可选依赖：

```bash
uv pip install -e '.[whisperx]'
```

发布前运行基准验收：

```bash
uv run python scripts/run_benchmarks.py \
  --benchmark-dir tests/benchmarks \
  --config configs/default.yaml \
  --asr-path auto \
  --ci
```

基准素材不足或未达到指标时，命令会返回非零状态并生成
`rollout_eligible=false` 报告；空目录不会被视为通过。

仓库内真实素材的 legacy 对照入口为：

```bash
venv/bin/python scripts/run_quality_benchmark.py \
  --manifest test/quality_manifest.yaml \
  --config configs/default.yaml \
  --asr-path segmented \
  --output-dir test/benchmark_results/real_material
```

WhisperX 已安装时，使用 global 技术发布门并生成质量报告：

```bash
venv/bin/python scripts/run_quality_benchmark.py \
  --manifest test/quality_manifest.yaml \
  --config configs/default.yaml \
  --asr-path global --require-global --ci \
  --output-dir test/benchmark_results/real_material/global-cpu-20260726-final
```

该命令确认六个场景实际使用 global；命令通过只代表全局技术链路可运行，仍需检查报告中的字幕质量指标。

发布门要求所有场景实际使用 global：

```bash
venv/bin/python scripts/run_quality_benchmark.py \
  --manifest test/quality_manifest.yaml \
  --config configs/default.yaml \
  --asr-path auto --require-global --ci
```
