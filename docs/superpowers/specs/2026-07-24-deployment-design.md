# Vocal Subtitle 一键部署与 Debian 打包设计

## 目标

基于当前项目实际入口和延迟导入结构，收敛安装链路，移除基础部署中的重量级及已停止维护依赖，提供可直接安装的 Debian 软件包，并让源码安装与 Debian 安装使用同一套 profile 逻辑。

完成后的默认部署应满足：

- CPU 环境可直接运行；
- 提供 CLI 和 Web GUI；
- 默认使用 `faster-whisper` ASR 和 WebRTC VAD；
- 不安装 PyTorch、Silero、UVR、Spleeter、Open-Unmix、pyannote、SpeechBrain、本地 NLP、LLM 或开发工具；
- `.deb` 安装后可在本机 `127.0.0.1:7862` 提供 GUI 服务；
- GPU、人声分离、说话人分离、LLM 等能力可通过明确的扩展 profile 后续安装。

## 现状与问题

当前根目录 `install.sh`、`pyproject.toml`、`requirements-all.txt` 和 Debian `postinst` 对生产环境的定义不一致：

- `requirements.txt` 把 ASR、Torch 和 Web GUI 标为可选，但生产安装逻辑会主动安装 CPU/GPU Torch、UVR 和 diarization；
- Debian `postinst` 调用 `install.sh --all`，会在安装包阶段安装全量可选依赖；
- Debian control 声明了基础包不需要的 `libsndfile1`、`python3-tk` 和 CUDA 建议项；
- 默认 YAML 使用 `uvr`、`silero`、pyannote 和较大的 ASR 配置，轻量环境安装成功后首次运行仍可能失败；
- `Vocal_Subtitle.旧项目/` 是历史副本，不应进入当前 Debian 包；
- Spleeter 已停止维护，不应继续出现在受支持的一键安装路径；
- 代码安装目录 `/usr/share` 下现有 venv 会把可变运行时文件写入静态共享目录；
- 文档中 GUI 端口 `7860` 与当前服务端口 `7862` 不一致。

## 方案

采用分层安装方案：默认安装轻量基础 profile，重量级能力按需追加。

### Profile

`pyproject.toml` 作为依赖单一来源，增加以下 profile：

- `base`：核心依赖、`faster-whisper`、`webrtcvad`、`webui`；
- `gpu`：GPU 推理所需的 CTranslate2/PyTorch 相关扩展，按当前代码实际使用情况定义；
- `separation`：UVR 人声分离；
- `diarization`：声学特征、聚类和 pyannote/SpeechBrain 相关依赖；
- `llm`：云端 LLM 优化；
- `local-nlp`：本地 NLP 语义合并；
- `dev`：测试、格式化、Lint 和类型检查工具；
- `full`：组合所有受支持扩展，不包含 Spleeter。

`install.sh` 默认执行 `base`。现有 `--production`、`--base`、`--webui`、`--gui`、`--uvr`、`--gpu`、`--cpu`、`--llm`、`--diarization`、`--local-nlp`、`--dev`、`--all` 等参数保留兼容，并统一转换到 profile 集合。Spleeter 参数不再安装依赖；若仍传入，应给出迁移提示或明确错误。

`requirements.txt` 保持核心运行依赖，`requirements-all.txt` 改为引用完整 extra，避免重复维护完整依赖列表。

### 默认运行配置

基础 profile 的默认配置调整为：

- separation engine 为 `none`，输入被视为已有人声；
- VAD engine 为 `webrtc`；
- ASR device 为 `cpu`；
- ASR model 为 `small`；
- diarization disabled；
- LLM 和本地 NLP disabled。

`Pipeline` 增加 `none` 分离引擎的明确旁路：当配置为 `none` 时跳过分离阶段，直接对输入音频运行 VAD、ASR、时间轴映射和字幕导出。CLI 和 Web GUI 的显式 `--skip-separation` 语义保持不变。其他场景 profile 可继续使用 UVR 等分离引擎。

配置数据类的默认回退值必须与 `configs/default.yaml` 一致，避免缺少字段时恢复到 `spleeter`、`silero` 或 GPU 设备。

## Debian 包结构

Debian 包仍采用当前 `dpkg-deb` 手动构建脚本，以减少构建环境要求。包内容包括当前项目源码、配置、运行所需资源、启动器、systemd unit 和部署文档；明确排除历史项目副本、构建产物、虚拟环境和测试音频。

安装路径：

- `/usr/share/vocal-subtitle`：只读项目源码和资源；
- `/var/lib/vocal-subtitle/venv`：由 `postinst` 创建的可变 Python 虚拟环境；
- `/etc/vocal-subtitle`：环境配置和服务配置；
- `/var/cache/vocal-subtitle`：任务缓存及模型缓存；
- `/var/log/vocal-subtitle`：服务日志；
- `/usr/bin/vocal-subtitle`、`/usr/bin/vocal-subtitle-gui`：CLI/GUI wrapper；
- `/usr/sbin/vocal-subtitle-setup`：扩展安装和修复入口。

基础 Debian `Depends` 只保留 Python 3.10+、`python3-venv` 和 `ffmpeg`。基础包不声明 `libsndfile1`、`python3-tk`、CUDA toolkit 或开发依赖。

### postinst

`postinst configure` 按以下顺序执行：

1. 创建目录和 `vocal-subtitle` 系统用户；
2. 创建或复用 `/var/lib/vocal-subtitle/venv`；
3. 调用项目 `install.sh --profile base --no-system-deps --no-model-dl`；
4. 验证 Python 包、CLI 和 GUI 入口；
5. 写入 `/etc/vocal-subtitle/env.conf`，设置安装、缓存、日志和模型目录；
6. 设置最小必要权限，并在 systemd 可用时 enable/start GUI 服务。

重复安装和升级不得删除现有 venv、模型缓存或配置。Python 依赖安装失败时应以失败状态退出，并保留现场供 `vocal-subtitle-setup` 修复。

### postrm

普通 remove 保留 venv、缓存、配置和日志。purge 删除这些生成目录和系统用户。删除前只操作明确的项目专属路径，不使用宽泛的递归目标。

### systemd

GUI 服务使用专用系统用户，工作目录为缓存目录，代码目录只读，启用基础权限限制。默认监听 `127.0.0.1:7862`，不在安装后直接向局域网暴露服务。用户需要远程访问时，通过服务配置显式修改监听地址并重启服务。

## 安装入口

源码安装：

```bash
bash install.sh
```

扩展安装：

```bash
bash install.sh --profile separation
bash install.sh --profile gpu
bash install.sh --profile diarization
bash install.sh --profile full
```

Debian 安装：

```bash
sudo apt install ./vocal-subtitle_0.1.0_all.deb
```

扩展和修复：

```bash
sudo vocal-subtitle-setup --profile separation
sudo vocal-subtitle-setup --profile gpu
sudo vocal-subtitle-setup --profile full --download-models
```

基础部署处理纯人声或已分离音频。处理带背景音乐的原始音频前，需要安装 `separation` profile；模型默认按需下载到项目模型缓存目录。

## 错误处理

- 不满足 Python 最低版本时，在安装开始阶段失败并给出系统包安装提示；
- 缺少 ffmpeg 时由源码安装脚本尝试使用系统包管理器安装，`--no-system-deps` 用于容器/CI；
- 可选 profile 安装失败不删除基础环境；
- 模型下载失败不视为包构建失败，运行时输出模型缓存和网络诊断信息；
- systemd 不可用时 Debian 安装仍完成文件和 venv 安装，并输出手动启动命令；
- 基础 GUI 启动失败时通过 `systemctl status`、`journalctl` 和 `vocal-subtitle-setup` 提供修复路径。

## 验证

实现完成后必须执行：

1. `bash -n` 检查所有安装、维护、wrapper 和构建脚本；
2. profile 参数转发和冲突参数测试；
3. 默认配置解析、`none` 分离旁路和 WebRTC 工厂测试；
4. 基础依赖环境下 `vocal-subtitle --help`、Python 导入和 GUI health endpoint smoke test；
5. `build-deb.sh --check`、`dpkg-deb --info`、`dpkg-deb --contents` 和可用时的 `lintian` 检查；
6. 验证包清单不包含 `Vocal_Subtitle.旧项目/`、`venv/`、测试音频或 `gh.tar.gz`；
7. 构建最终 `.deb` 并记录包路径、大小、依赖和安装命令。

不在构包阶段下载大型模型，也不要求完整 GPU、pyannote 或 UVR 环境才能完成基础构建验证。

## 文档更新

- README：增加 Debian 一键安装、源码安装、基础能力边界、profile 命令、GUI 地址和扩展安装说明；
- `docs/DEPLOYMENT.md`：记录安装目录、服务、升级/卸载、模型缓存、profile 和故障排查；
- `docs/ARCHITECTURE.md`：更新部署矩阵和默认依赖描述；
- Debian `control`、`changelog` 和服务说明：改为轻量基础包语义，统一端口为 `7862`；
- 检查并修正相关文档中旧的 `7860`、全量默认安装和 Spleeter 默认支持描述。

## 不在本次范围内

- 不重写现有 ASR、VAD、分离或 Web GUI 业务实现；
- 不删除仍被测试或兼容路径引用的旧引擎源码；
- 不打包模型权重；
- 不制作 Docker 镜像或跨发行版原生包；
- 不改变已有用户配置文件的升级迁移策略之外的业务配置。
