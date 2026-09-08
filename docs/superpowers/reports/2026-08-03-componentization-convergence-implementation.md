# Vocal Subtitle 组件化收口实施报告

日期：2026-08-03  
依据：[组件化收口设计](../specs/2026-08-03-componentization-convergence-design.md)

## 1. 实施内容

### CLI 入口

`vocal_subtitle/cli.py` 从 1697 行收敛为 345 行兼容 façade，保留 `vocal_subtitle.cli:main`、`run`、`batch` 和已有参数入口。

新增：

- `vocal_subtitle/cli_commands/common.py`：公共 Click 参数；
- `vocal_subtitle/cli_commands/administration.py`：模型、profile、系统信息、预检、质量和治理命令；
- `vocal_subtitle/cli_commands/feedback_commands.py`：反馈学习、配置管理和 D3 抽样命令；
- `vocal_subtitle/cli_commands/__init__.py`：命令组件导出；
- `scripts/check_cli_contract.py`：CLI 命令和关键参数契约检查。

命令实现通过 Click group 注册，不在 façade 中保留重复业务实现。

### Pipeline runner

原 `pipeline_runner.py` 的运行生命周期、预检、任务收尾和报告职责已拆分：

- `application/run_lifecycle.py`：原 `run()` 生命周期实现；
- `application/run_preflight.py`：预检与任务启动状态转换；
- `application/run_finalizer.py`：任务历史终态转换；
- `application/run_reporter.py`：统一运行报告构建与落盘；
- `application/pipeline_runner.py`：保留 `PipelineRunMixin` 兼容入口和委托方法。

`Pipeline` 的 Mixin 组合、运行签名、缓存路径、任务状态和报告字段保持不变。

### 测试组件

原 `tests/test_asr/test_evidence_review.py` 按职责拆为：

- `test_evidence_review.py`：证据契约、候选适配、复核服务和可选引擎；
- `test_evidence_review_decision.py`：缓存、风险调度、决策、物理校验和投影来源。

两份测试均保持 pytest 自动收集，不改变测试断言逻辑。

## 2. 验证结果

执行环境：`.venv-production/bin/python`，通过 `PYTHONPATH=. python -m pytest` 运行，避免环境中旧的 pytest shebang 指向已不存在的路径。

```text
1150 passed, 1 skipped, 2 warnings
```

通过的结构检查：

- 模块规模检查：所有变更源文件均为 `ok`；
- 公共导入和兼容入口检查：通过；
- 组件边界和循环依赖检查：通过；
- CLI 命令/关键参数契约检查：通过；
- `compileall`：通过；
- `git diff --check`：通过；
- CLI help 与关键治理命令 smoke：通过。

WebUI method/path 检查发现当前工作区相对于 `5079e43` 基线新增：

- `GET /feedback/review-queue`；
- `GET /feedback/review/{sample_id}`；
- `POST /feedback/d3-sample`；
- `POST /feedback/review/{sample_id}`。

这四条路由来自本轮之前的工作区变更，本次没有修改或回退。既有 WebUI 测试仍通过。

## 3. 范围边界与剩余风险

- 本轮没有修改 ASR 召回策略、模型路由、黄金集阈值或发布结论。
- 未执行真实模型、GPU、多任务并发、长音频生产和外部 LLM 的等价性验收。
- 反馈 CLI 已迁移到独立组件，反馈学习业务仍属于后续后端契约和前端组件阶段的功能范围。
- 下一阶段应以本轮稳定的 application façade、运行报告、任务状态和 CLI/WebUI 兼容入口为基础设计后端组件契约。
