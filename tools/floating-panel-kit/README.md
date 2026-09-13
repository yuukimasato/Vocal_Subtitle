# floating-panel-kit

可自由**拖动、缩放、吸附停靠**的 React 浮动面板组件。从 `subfx-replica-platform` 的 Ephemeral Panel（浮动面板）系统中提取，去掉了所有业务耦合，可在任意 React ≥ 18 项目中直接复制使用。

## 特性

- **拖动**：按住标题栏拖到任意位置（Pointer Events + 指针捕获，鼠标移出窗口也不丢事件，天然支持触屏）
- **缩放**：四边 + 四角共 8 个方向手柄，强制最小尺寸（默认 240×160）
- **吸附停靠**：拖到工作区边缘或另一个面板边缘 24px 内时高亮提示，松手自动贴齐
- **窗口置顶**：点击任意面板自动提到最高层（zIndex 管理）
- **默认智能摆放**：新面板自动选择不与现有面板重叠的位置，全部冲突时级联偏移
- **键盘支持**：ESC 中断拖拽 → 关闭最顶层可关闭面板；面板可聚焦
- **动画**：打开/关闭过渡，关闭期间内容仍可渲染
- **持久化**：`storageKey` 一行配置即可用 localStorage 记住每个面板的位置和大小；或用 `onLayoutChange` 存到自己的后端
- **容错**：内置错误边界，单个面板崩溃不影响应用其他部分
- **零依赖**：除 React 外无任何运行时依赖（状态用 `useSyncExternalStore`，图标为内联 SVG）

## 目录结构

```
floating-panel-kit/
├── src/
│   ├── index.ts               # 公共导出（含样式副作用导入）
│   ├── types.ts               # PanelLayout / OpenPanelOptions 等类型
│   ├── geometry.ts            # 纯函数：夹取、吸附检测、默认摆放、置顶
│   ├── store.ts               # 框架无关的 store（open/close/dock/…）
│   ├── context.tsx            # Provider + usePanels 等 Hooks + 持久化
│   ├── FloatingPanelLayer.tsx # 渲染层：拖动/缩放交互 + ESC 处理
│   ├── ErrorBoundary.tsx      # 面板级错误边界
│   ├── icons.tsx              # 内联 SVG 图标
│   └── styles.css             # fp- 前缀样式，CSS 变量可定制主题
├── example/AppExample.tsx     # 使用示例
├── tests/geometry.test.ts     # 几何层单元测试（vitest）
└── package.json
```

## 快速开始

把整个 `src/` 复制到目标项目（例如 `src/floating-panel/`），然后：

```tsx
import {
  FloatingPanelProvider,
  FloatingPanelLayer,
  useFloatingPanelStore,
} from '你的项目/floating-panel';

function Toolbar() {
  const store = useFloatingPanelStore();
  return (
    <button
      onClick={() => store.open({
        id: 'player',                    // 唯一 id；重复打开会复用并聚焦
        title: '播放器',
        width: 640,
        height: 400,
        maximizable: true,
        content: <MyPlayer />,           // 内容是任意 ReactNode
      })}
    >
      打开播放器
    </button>
  );
}

export default function App() {
  return (
    <FloatingPanelProvider storageKey="my-app-panels">
      <Toolbar />
      <FloatingPanelLayer />  {/* 放在 JSX 最后，渲染所有浮层 */}
    </FloatingPanelProvider>
  );
}
```

> 样式随 `index.ts` 的 `import './styles.css'` 自动带入；如果你的构建不处理 CSS 副作用导入，删掉该行并在应用入口手动引入。

## API

### `<FloatingPanelProvider>`

| Prop | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `storageKey` | `string` | — | localStorage 键；面板几何按 id 持久化（防抖 300ms） |
| `onLayoutChange` | `(layouts) => void` | — | 布局变化回调，用于存自己的后端 |
| `zIndexBase` | `number` | `1000` | 浮层 z-index 基准，需高于宿主应用 UI |
| `workspacePadding` | `number` | `0` | 工作区四周留白 |
| `getWorkspace` | `() => WorkspaceRect` | 视口 | 自定义可用区域，例如避开侧边栏/顶栏 |
| `closeAnimationMs` | `number` | `200` | 关闭动画时长 |

### `store` 方法（`useFloatingPanelStore()`）

| 方法 | 说明 |
| --- | --- |
| `open(options)` | 打开面板（同 id 复用并聚焦） |
| `close(id)` / `closeAll()` | 关闭（带动画）/ 全部关闭 |
| `focus(id)` | 置顶 |
| `toggleMaximize(id)` | 最大化 / 还原（需 `maximizable: true`） |
| `remove(id)` | 跳过动画立即移除 |
| `getState()` / `subscribe()` | 读取状态 / 订阅变更（框架无关集成入口） |

`open` 的选项：`id`、`title`、`content` 必填；`width`、`height`、`x`、`y`、`closable`、`maximizable`、`minWidth`、`minHeight` 可选。

### Hooks

- `usePanels()` — 面板列表
- `useFloatingPanelStore()` — store 实例
- `useSnapPreview()` — 当前吸附预览
- `usePanelState(selector)` — 自定义订阅

## 主题定制

覆盖 `.fp-panel` 上的 CSS 变量即可：

```css
.fp-panel {
  --fp-bg: #ffffff;
  --fp-header-bg: #f7f8fa;
  --fp-border: #e2e5ea;
  --fp-text: #1a202c;
  --fp-text-soft: #6b7280;
  --fp-accent: #2563eb;
  --fp-radius: 8px;
}
```

## 与原实现（subfx-replica-platform）的对应关系

| 原 | 本组件 | 变化 |
| --- | --- | --- |
| `ephemeral-ui/panelLayout.ts` | `src/geometry.ts` | 去掉 Chat 特判与侧边栏宽度假设，改为可配置工作区 |
| `ephemeral-ui/PanelManager.tsx`（PanelShellInteractive） | `src/FloatingPanelLayer.tsx` | 去掉业务回调/渲染器注册表，内容改为 `content` 节点 |
| `sessionUIStore.ts`（zustand） | `src/store.ts` + `src/context.tsx` | zustand → `useSyncExternalStore`，不可变更新 |
| 布局存后端 `workspace-layout` API | `storageKey` / `onLayoutChange` | 持久化目标可插拔 |
| `lucide-react` 图标 | `src/icons.tsx` | 内联 SVG |

## 开发

```bash
npm install        # 安装 dev 依赖（typescript/vitest）
npm run typecheck  # TS 检查
npm test           # 几何层单元测试
```
