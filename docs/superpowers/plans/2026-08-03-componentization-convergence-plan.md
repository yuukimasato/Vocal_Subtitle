# Vocal Subtitle 组件化收口实施计划

依据：[组件化收口设计](../specs/2026-08-03-componentization-convergence-design.md)

## 实施约束

- 只做结构迁移，不改变业务结果和现有公开契约。
- 不回滚或覆盖工作区已有用户改动。
- 每完成一组迁移，立即执行对应的导入、兼容和单元测试。
- `vocal_subtitle.cli` 与 `vocal_subtitle.pipeline` 继续作为兼容 façade。
- 新增模块不超过 1000 行，1200 行为硬失败。

## 步骤

1. 建立当前基线：运行模块规模、导入边界、公共导入、WebUI 路由和 CLI 帮助检查；记录既有失败与环境限制。
2. 建立 `vocal_subtitle/cli_commands/`，按 run、feedback、quality、governance 分组迁移 CLI 实现；原 `cli.py` 保留 Click 入口和兼容导出。
3. 将 `pipeline_runner.py` 中的预检、运行生命周期、任务收尾和报告生成拆入 application 组件；原 mixin 调用顺序和结果结构保持不变。
4. 将超大证据复核测试按候选准入、决策、诊断和降级拆分，保留原测试文件的可收集性。
5. 扩展结构检查，覆盖新增治理、质量、报告和 ASR 模块；新增 CLI 契约 smoke，冻结命令和参数。
6. 执行定向测试、全量 pytest、compileall、JavaScript 语法检查和 `git diff --check`。
7. 更新组件化实施报告，明确自动化覆盖与真实模型/部署验证的边界。

## 验收

- `cli.py`、`pipeline_runner.py` 和新增实现文件均通过模块规模门禁。
- 领域层无 Pipeline/WebUI 反向依赖，路由无直接引擎构造。
- 公共导入、CLI、WebUI method/path、任务状态和输出 payload 无回归。
- 全量测试无新增失败；既有环境限制单独记录。
- 不修改 ASR 召回、黄金集阈值和发布结论。
