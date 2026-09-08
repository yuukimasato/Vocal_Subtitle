# Vocal Subtitle WebUI 组件工作区实施报告

日期：2026-08-03  
设计：[WebUI 组件工作区设计](../specs/2026-08-03-webui-component-workspace-design.md)  
计划：[WebUI 组件工作区实施计划](../plans/2026-08-03-webui-component-workspace-plan.md)

## 实施结果

### 工作区壳层

- 新增顶部导航和 hash 单页切换：`#process`、`#review`、`#feedback`、`#history`、`#quality`。
- 处理工作区继续复用原有上传、配置、进度、结果和导出 DOM；旧 `App` 方法和旧 id 未删除。
- 通过共享任务选择同步审核与质量报告，视图切换时按需刷新数据。

### 字幕审核

- 支持从任务历史选择任务并加载字幕事件。
- 支持播放定位、点击字幕跳转、单条文本编辑、保存、导出和批量说话人/连续字幕合并。
- 保留未保存提示、空任务、加载失败和返回反馈/质量报告入口。

### 反馈学习

- 支持使用当前编辑结果，或上传修订字幕与音频进行预览。
- 预览和确认学习是两个明确动作；学习结果展示对齐覆盖率、时间偏移、合并/拆分、文本编辑、参数调整和结构修订信息。
- 接入 profile、health、review queue 和审核操作，失败时保留当前预览。

### 任务历史

- 支持状态筛选、分页、任务详情和状态/质量摘要。
- 提供进入字幕审核、查看质量报告和下载产物的入口。
- 删除仍是显式操作，不随工作区切换触发。

### 质量报告

- 新增只读 `GET /api/quality/report/{task_id}`。
- 优先读取 `cache/reports/{run_id}/run_report.json`；报告不存在时基于任务历史 `result_json` 返回 `history_summary`，不改写任何持久化数据。
- 统一返回 `report_source`、`report_schema_version`、任务/运行标识，并保留状态、质量、阶段、引擎、声学、降级、错误和产物摘要。
- 前端将任务状态和质量状态分开呈现，避免把 `degraded_completed` 误显示为完全成功。

## 文件边界

新增前端模块：

- `vocal_subtitle/webui/static/js/workspace.js`
- `vocal_subtitle/webui/static/js/ui-review.js`
- `vocal_subtitle/webui/static/js/ui-feedback-workspace.js`
- `vocal_subtitle/webui/static/js/ui-history.js`
- `vocal_subtitle/webui/static/js/ui-quality.js`

新增后端路由：

- `vocal_subtitle/webui/routes_quality.py`

兼容扩展集中在 `api-client.js`、`index.html`、`app.css`、反馈/历史/字幕路由和现有任务结果序列化路径。

## 验证结果

已完成以下验证：

- WebUI 定向测试：27 passed。
- 全量 pytest：1161 passed，1 skipped，2 warnings。
- Node JavaScript 语法检查：通过。
- `python -m compileall -q vocal_subtitle tests scripts`：通过。
- `git diff --check`：通过。
- 浏览器 smoke：桌面端五个工作区切换、任务历史、质量报告加载通过；移动端 390x844 无明显重叠；console/errors 为空。

## 兼容边界与剩余风险

- 本阶段没有修改 ASR 召回、引擎路由、黄金集阈值或发布门禁。
- 没有修改任务历史数据库 schema；质量报告缺失时的历史摘要是只读投影。
- 尚未执行真实模型/GPU、多任务并发、长音频生产和外部 LLM 的全样本等价性验收。
- 当前本地服务用于复核：`http://127.0.0.1:8765`。
