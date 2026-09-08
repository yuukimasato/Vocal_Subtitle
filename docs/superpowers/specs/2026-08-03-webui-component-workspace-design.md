# Vocal Subtitle WebUI 组件工作区设计

日期：2026-08-03
状态：已实施
前置阶段：后端组件契约

## 1. 目标与范围

将现有单页处理界面收口为可持续使用的前端组件工作区，重点完善字幕审核、反馈学习、任务历史和质量报告。采用顶部模块导航和单页面视图切换，继续使用当前静态 HTML/CSS/JavaScript 与 FastAPI，不引入 React、Vite 或新的构建链。

本阶段包含：

- 处理工作区：保留现有上传、配置、进度、结果和导出能力；
- 字幕审核工作区：历史任务选择、音频播放、字幕时间轴、单条/批量编辑、说话人处理和审核状态；
- 反馈学习工作区：修订字幕预览、确认学习、当前编辑结果学习、用户配置、健康趋势和反馈审核队列；
- 任务历史工作区：状态筛选、分页、详情、字幕加载、质量摘要和删除；
- 质量报告工作区：运行、版本、阶段、质量、声学、证据/决策/投影、降级、错误和产物摘要；
- 桌面和移动端布局、loading/empty/error/degraded 状态，以及前端契约测试。

本阶段不包含：

- React/Vite 或多页前端迁移；
- ASR 召回、模型路由、黄金集阈值和发布门禁改变；
- 任务历史数据库 schema 变更；
- 旧 WebUI route、旧 DOM id、旧 `App` 方法和旧请求响应字段删除。

## 2. 信息架构

```text
Vocal Subtitle
├── 处理
│   ├── 上传与配置
│   ├── 运行进度
│   └── 当前结果与导出
├── 字幕审核
│   ├── 任务选择
│   ├── 播放与时间定位
│   ├── 字幕编辑/批量操作
│   └── 审核状态与反馈入口
├── 反馈学习
│   ├── 修订版来源
│   ├── 差异预览与学习确认
│   ├── 配置/健康趋势
│   └── review queue
├── 任务历史
│   ├── 状态筛选/分页
│   ├── 任务详情
│   └── 字幕/报告入口
└── 质量报告
    ├── 运行摘要
    ├── 阶段/引擎/版本
    ├── 质量与声学诊断
    ├── evidence/decision/projection
    └── 降级/错误/产物
```

顶部导航使用 hash 路由，例如 `#process`、`#review`、`#feedback`、`#history`、`#quality`。刷新页面后保留当前视图；没有任务时各工作区显示明确的空状态，并提供返回处理入口。

## 3. 组件与数据流

### 3.1 处理工作区

现有处理组件保持原 DOM id 和 `App` 方法，移动到 `process` 视图容器。任务完成后写入共享前端状态：`selectedTaskId`、`taskResult`、`subtitleEvents`、`stats`、`runReport`。导航到审核或报告工作区时直接消费共享状态，必要时通过 API 刷新历史详情。

### 3.2 字幕审核组件

`ReviewWorkspace` 由任务选择器、`SubtitleReviewTable`、音频控制条和审核侧栏组成。数据来源为现有 `/api/history`、`/api/history/{task_id}`、`/api/subtitle/{task_id}`。单条编辑继续使用 `PUT /api/subtitle/{task_id}/{index}`，批量说话人/合并继续使用现有 batch route。播放器时间变化高亮对应字幕，点击字幕跳转时间；未保存编辑、网络错误和无字幕任务必须可见。

### 3.3 反馈学习组件

`FeedbackWorkspace` 调用现有 `/api/feedback/preview`、`/api/feedback/learn`、profile、health、review-queue 接口。预览和写入使用两个明确动作，学习结果展示对齐覆盖率、时间偏移、合并/拆分、文本编辑、参数调整和结构性修订标记。反馈学习只更新反馈配置，不修改任务历史主结果；当前编辑结果学习沿用已有前端入口。

### 3.4 任务历史组件

`HistoryWorkspace` 使用分页查询和状态筛选，列表显示文件名、profile、状态、质量摘要、耗时和时间。选择任务后打开详情面板，提供加载审核、查看报告和下载产物入口。删除操作需要二次确认；清理全部保持原显式操作，不在视图切换时触发。

### 3.5 质量报告组件

新增只读接口 `GET /api/quality/report/{task_id}`。服务端优先读取 `cache/reports/{run_id}/run_report.json`，不存在时使用任务历史 `result_json` 中的 stats/diagnostics 生成兼容摘要。接口响应增加 `report_source` 和 `report_schema_version`，不改写报告文件、不改变历史 schema。

前端 `QualityWorkspace` 将报告分为摘要、阶段与引擎、质量与声学、证据/决策/投影、降级与错误、产物六组。状态颜色按 `completed`、`degraded_completed`、`failed` 显示，质量 `pass/warn/fail` 与任务状态分开表达，避免将降级结果误认为完全成功。

## 4. 前端模块边界

新增静态模块：

```text
vocal_subtitle/webui/static/js/
  workspace.js       # hash 导航、视图生命周期、共享任务选择
  ui-review.js       # 字幕审核工作区
  ui-feedback-workspace.js # 反馈学习工作区
  ui-history.js      # 任务历史工作区
  ui-quality.js      # 质量报告工作区
```

现有 `ui-subtitles.js`、`ui-feedback.js`、`ui-progress.js` 和 `api-client.js` 保留为兼容实现；新工作区通过明确的渲染函数调用它们的稳定 API，不复制编辑和反馈业务逻辑。公共 DOM 生成必须使用现有 escape helper，服务端内容不得未经转义插入 HTML。

## 5. 错误、加载和响应兼容

- API 请求统一显示加载占位和可重试错误；
- 404 任务显示“任务不存在或已清理”，不抛出空白页面；
- 报告缺失时显示基于历史 stats 的摘要，并标记来源为 `history_summary`；
- 反馈学习失败只显示错误和保留预览，不隐藏已有字幕；
- 历史旧字段和旧状态名继续由前端归一化，新增字段全部可选；
- 不在浏览器保存 API key、HF token 或完整异常堆栈。

## 6. 响应式与视觉规则

延续当前深色工作台样式、8px 以内圆角、现有颜色 token 和密集信息布局。桌面端使用主内容区和固定导航；审核工作区采用可收缩双栏，移动端改为上下顺序。图标按钮使用已有 SVG/熟悉符号并提供 `title`/`aria-label`，文字命令保留清晰标签。表格、工具栏和状态胶囊使用稳定尺寸，长文件名和诊断内容截断或换行，不发生布局跳动。

## 7. 测试与验收

- FastAPI：质量报告接口的成功、历史摘要回退、404 和旧报告兼容测试；
- 静态契约：新模块引用、旧 DOM id、旧 route 和关键操作存在性检查；
- JavaScript：模块切换、任务选择、报告归一化和错误状态测试；
- 浏览器 smoke：桌面端加载、五个工作区切换、历史详情、审核编辑入口、质量报告空/有数据状态；
- 移动端截图检查：导航可横向滚动，审核表格/报告无重叠和横向溢出；
- 全量 Python 测试、compileall、`git diff --check`。

验收要求：旧处理流程可继续使用；四类新工作区可独立进入和恢复；字幕审核修改能通过现有接口持久化；反馈学习预览与写入边界清晰；历史和质量报告可从已有任务数据恢复；不改变 ASR 召回和黄金集门禁。
