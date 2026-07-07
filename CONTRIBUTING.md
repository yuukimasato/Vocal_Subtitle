# 贡献指南 (Contributing Guide)

感谢你对 vocal-subtitle 项目的关注！

## 开发环境搭建

```bash
# 1. Fork 并 Clone 项目
git clone <your-fork-url>
cd Vocal_Subtitle

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 安装系统依赖
# Ubuntu: sudo apt install ffmpeg
# macOS:  brew install ffmpeg
```

## 代码规范

本项目使用以下工具确保代码质量：

| 工具 | 用途 | 运行方式 |
|------|------|----------|
| **Ruff** | Linting + 格式化 | `ruff check . && ruff format .` |
| **mypy** | 静态类型检查 | `mypy vocal_subtitle/` |
| **pytest** | 单元测试/集成测试 | `pytest` |

### 提交前检查

```bash
# 运行所有检查
ruff check . && mypy vocal_subtitle/ && pytest
```

## 项目结构

```
vocal_subtitle/
├── separation/     # Stage 1: 人声分离引擎
├── vad/            # Stage 2: 语音活动检测
├── merging/        # Stage 3: 片段合并策略
├── asr/            # Stage 4: 语音识别引擎
├── mapping/        # Stage 5: 时间轴映射 + 字幕构建
├── utils/          # 工具层
├── pipeline.py     # 管道编排器
├── config.py       # YAML 配置管理
└── cli.py          # CLI 入口
```

## 如何贡献

### 添加新的分离引擎

1. 在 `vocal_subtitle/separation/` 下创建新文件
2. 继承 `SeparationEngine` 抽象基类
3. 实现 `separate()`, `load_model()`, `name`, `license_info`
4. 在 `separation/__init__.py` 中导出
5. 在 `pipeline.py` 的 `_get_separation_engine()` 中注册
6. 添加单元测试到 `tests/test_separation/`

### 添加新的 VAD 引擎

1. 在 `vocal_subtitle/vad/` 下创建新文件
2. 继承 `VADEngine` 抽象基类
3. 实现 `detect()`, `detect_on_array()`, `load_model()`, `name`
4. 在 `vad/__init__.py` 中导出
5. 在 `pipeline.py` 的 `_get_vad_engine()` 中注册
6. 添加单元测试到 `tests/test_vad/`

### 添加新的 ASR 引擎

1. 在 `vocal_subtitle/asr/` 下创建新文件
2. 继承 `ASREngine` 抽象基类
3. 实现 `transcribe()`, `load_model()`, `name`, `model_name`
4. 在 `asr/__init__.py` 中导出
5. 在 `pipeline.py` 的 `_get_asr_engine()` 中注册
6. 添加单元测试到 `tests/test_asr/`

### 添加新的场景模板

1. 在 `configs/` 下创建 `your_template.yaml`
2. 在 `config.py` 的 `BUILTIN_PROFILES` 中注册
3. 遵循现有模板的结构和命名约定

## 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_separation/
pytest tests/test_merging/
pytest tests/test_mapping/

# 生成覆盖率报告
pytest --cov=vocal_subtitle --cov-report=html

# 运行带详细输出
pytest -v --tb=long
```

## 提交 Pull Request

1. 创建一个描述性的分支名 (如 `feat/add-new-asr-engine`)
2. 编写清晰的 commit message
3. 确保所有测试通过
4. 如果是新功能，添加相应测试
5. 更新文档（如需要）
6. 提交 PR 并在描述中说明改动内容和原因

## 协议合规

添加新依赖时，请确保：
- 依赖的代码协议为 MIT / Apache 2.0 / BSD 类
- 如果是模型权重，验证其商用许可
- 更新 `NOTICE` 文件记录新增依赖
- 在 PR 描述中明确新依赖的协议信息

## 行为准则

- 保持讨论专业且尊重他人
- 提供建设性的代码审查意见
- 帮助新人融入项目

---

再次感谢你的贡献！
