# Vocal Subtitle WebUI 组件工作区实施计划

日期：2026-08-03  
依据：[WebUI 组件工作区设计](../specs/2026-08-03-webui-component-workspace-design.md)  
状态：已完成

## 实施约束

- 继续使用现有静态 HTML/CSS/JavaScript 和 FastAPI，不引入 React、Vite 或新的构建链。
- 保留旧 DOM id、`App` 方法、WebUI route、请求响应字段和任务历史 schema。
- 新工作区只消费现有 API 和共享前端状态，不复制字幕编辑、反馈学习和结果归一化业务规则。
- 质量报告接口只读；报告缺失时允许从历史任务结果生成摘要，但不得改写历史或报告文件。
- 不修改 ASR 召回、模型路由、黄金集阈值和发布结论。

## 步骤与结果

1. 建立顶部导航和 hash 工作区壳层，支持 `process`、`review`、`feedback`、`history`、`quality` 五个视图，并保留处理页原有入口。
2. 建立字幕审核工作区，接入任务选择、字幕加载、音频定位、单条编辑、批量说话人/合并、保存和导出。
3. 建立反馈学习工作区，接入当前编辑结果、修订字幕和音频上传、差异预览、确认学习、profile/health 和 review queue。
4. 建立任务历史工作区，接入状态筛选、分页、详情、审核入口、质量报告入口和显式删除操作。
5. 新增只读 `GET /api/quality/report/{task_id}`，优先读取 `run_report.json`，缺失时回退为 `history_summary`，并保留 404 和降级信息。
6. 建立质量报告工作区，展示运行状态、质量状态、生产路径、阶段耗时、引擎、声学、证据/决策/投影、降级、错误和产物。
7. 补齐 loading/empty/error/degraded 状态、移动端布局、HTML 转义和旧接口兼容测试，并完成浏览器桌面/移动 smoke。

## 验收

- 五个工作区可通过 hash 独立进入，刷新后保持当前视图。
- 旧处理流程、旧 DOM id、旧 `App` 方法和旧 API route 继续可用。
- 字幕审核编辑通过现有接口持久化，反馈学习的预览和写入动作分离。
- 历史任务和质量报告可从当前任务存储或历史数据库恢复；报告缺失时明确标记摘要来源。
- 全量 Python 测试、JavaScript 语法、`compileall` 和 `git diff --check` 通过。
- 未把浏览器 smoke 或无模型测试解释为真实模型、GPU、长音频或黄金集发布验收。
