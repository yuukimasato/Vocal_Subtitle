import {
  bringToFront,
  choosePanelPosition,
  clampPanelRect,
  MIN_PANEL_HEIGHT,
  MIN_PANEL_WIDTH,
} from './geometry';
import type {
  DockEdge,
  OpenPanelOptions,
  PanelInstance,
  PanelLayout,
  PanelRect,
  SavedLayout,
  SnapPreview,
  WorkspaceRect,
} from './types';

export interface PanelStoreState {
  panels: PanelInstance[];
  snapPreview: SnapPreview | null;
}

export interface PanelStoreOptions {
  /** 返回可用工作区（默认为视口收缩 padding 后的矩形）；可自定义以避开应用侧边栏等。 */
  workspace?: () => WorkspaceRect;
  /** 浮层的 z-index 基准，需高于宿主应用的常规 UI。默认 1000。 */
  zIndexBase?: number;
  /** 预恢复的面板几何（按 id 匹配），通常来自持久化存储。 */
  savedLayouts?: Record<string, SavedLayout>;
  /** 关闭动画时长（ms），动画结束后才真正移除面板。默认 200。 */
  closeAnimationMs?: number;
}

export interface FloatingPanelStore {
  getState(): PanelStoreState;
  subscribe(listener: () => void): () => void;
  workspace(): WorkspaceRect;
  /** 打开面板；同 id 已存在时复用、更新内容并聚焦。 */
  open(options: OpenPanelOptions): void;
  /** 关闭面板（播放关闭动画后移除）；closable=false 的面板不受影响。 */
  close(id: string): void;
  /** 跳过动画立即移除。 */
  remove(id: string): void;
  closeAll(): void;
  focus(id: string): void;
  /** 更新布局（用于拖动/缩放）；会夹取到工作区并跳过无变化的写入。 */
  patchLayout(id: string, patch: Partial<PanelLayout>): void;
  setSnapPreview(preview: SnapPreview | null): void;
  /** 提交吸附：无目标时记为贴边停靠，有目标时把面板摆到目标边缘。 */
  dock(id: string, targetId: string | null, edge: DockEdge): void;
  undock(id: string): void;
  /** 最大化 / 还原（需要面板 maximizable）。 */
  toggleMaximize(id: string): void;
}

const DEFAULT_WIDTH = 420;
const DEFAULT_HEIGHT = 300;
const DEFAULT_Z_INDEX_BASE = 1000;
const DEFAULT_CLOSE_MS = 200;

function defaultWorkspace(): WorkspaceRect {
  const width = typeof window === 'undefined' ? 1280 : window.innerWidth || 1280;
  const height = typeof window === 'undefined' ? 800 : window.innerHeight || 800;
  return { x: 0, y: 0, width, height };
}

export function createFloatingPanelStore(options: PanelStoreOptions = {}): FloatingPanelStore {
  const workspace = options.workspace ?? defaultWorkspace;
  const zIndexBase = options.zIndexBase ?? DEFAULT_Z_INDEX_BASE;
  const closeMs = options.closeAnimationMs ?? DEFAULT_CLOSE_MS;
  const savedLayouts = options.savedLayouts;

  let state: PanelStoreState = { panels: [], snapPreview: null };
  const listeners = new Set<() => void>();
  const closeTimers = new Map<string, ReturnType<typeof setTimeout>>();

  function notify(): void {
    for (const listener of listeners) listener();
  }

  function setState(next: Partial<PanelStoreState>): void {
    const panels = next.panels ?? state.panels;
    const snapPreview = next.snapPreview !== undefined ? next.snapPreview : state.snapPreview;
    if (panels === state.panels && snapPreview === state.snapPreview) return;
    state = { panels, snapPreview };
    notify();
  }

  function setPanels(next: PanelInstance[]): void {
    setState({ panels: next });
  }

  function replacePanel(id: string, patch: (panel: PanelInstance) => PanelInstance): void {
    setPanels(state.panels.map((panel) => panel.id === id ? patch(panel) : panel));
  }

  function clearCloseTimer(id: string): void {
    const timer = closeTimers.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      closeTimers.delete(id);
    }
  }

  function scheduleRemove(id: string): void {
    clearCloseTimer(id);
    closeTimers.set(id, setTimeout(() => {
      closeTimers.delete(id);
      setPanels(state.panels.filter((panel) => panel.id !== id));
    }, closeMs));
  }

  function maxZIndex(): number {
    return state.panels.reduce((value, panel) => Math.max(value, panel.layout.zIndex), zIndexBase);
  }

  function focus(id: string): void {
    setPanels(bringToFront(state.panels, id));
  }

  function nextLayout(
    id: string,
    rect: PanelRect,
    overrides: Partial<PanelLayout> = {},
  ): PanelLayout {
    return {
      mode: 'floating',
      dockTargetId: null,
      dockEdge: null,
      zIndex: maxZIndex() + 1,
      ...clampPanelRect(rect, workspace(), {
        width: optionsById(id)?.minWidth ?? MIN_PANEL_WIDTH,
        height: optionsById(id)?.minHeight ?? MIN_PANEL_HEIGHT,
      }),
      ...overrides,
    };
  }

  function optionsById(id: string): PanelInstance | undefined {
    return state.panels.find((panel) => panel.id === id);
  }

  const store: FloatingPanelStore = {
    getState: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    workspace,

    open(opts) {
      const existing = optionsById(opts.id);
      if (existing) {
        // 复用：刷新内容、撤销进行中的关闭动画，然后置顶。
        clearCloseTimer(opts.id);
        replacePanel(opts.id, (panel) => ({
          ...panel,
          title: opts.title,
          content: opts.content,
          closable: opts.closable !== false,
          maximizable: opts.maximizable === true,
          closing: false,
          openedAt: Date.now(),
        }));
        focus(opts.id);
        return;
      }

      const minWidth = opts.minWidth ?? MIN_PANEL_WIDTH;
      const minHeight = opts.minHeight ?? MIN_PANEL_HEIGHT;
      const ws = workspace();
      const width = Math.min(Math.max(opts.width ?? DEFAULT_WIDTH, minWidth), ws.width);
      const height = Math.min(Math.max(opts.height ?? DEFAULT_HEIGHT, minHeight), ws.height);
      const min = { width: minWidth, height: minHeight };

      const saved = savedLayouts?.[opts.id];
      let rect: PanelRect;
      if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) {
        rect = clampPanelRect(
          { x: saved.x, y: saved.y, width: saved.width ?? width, height: saved.height ?? height },
          ws, min,
        );
      } else if (opts.x !== undefined && opts.y !== undefined) {
        rect = clampPanelRect({ x: opts.x, y: opts.y, width, height }, ws, min);
      } else {
        const position = choosePanelPosition({ width, height }, state.panels, ws);
        rect = clampPanelRect({ ...position, width, height }, ws, min);
      }

      const panel: PanelInstance = {
        id: opts.id,
        title: opts.title,
        content: opts.content,
        layout: {
          mode: 'floating',
          dockTargetId: null,
          dockEdge: null,
          zIndex: maxZIndex() + 1,
          ...rect,
        },
        closable: opts.closable !== false,
        maximizable: opts.maximizable === true,
        minWidth,
        minHeight,
        restoreRect: null,
        openedAt: Date.now(),
        closing: false,
      };
      setPanels([...state.panels, panel]);
    },

    close(id) {
      const panel = optionsById(id);
      if (!panel || !panel.closable || panel.closing) return;
      replacePanel(id, (item) => ({ ...item, closing: true }));
      scheduleRemove(id);
    },

    remove(id) {
      clearCloseTimer(id);
      setPanels(state.panels.filter((panel) => panel.id !== id));
    },

    closeAll() {
      for (const panel of state.panels) {
        if (!panel.closable || panel.closing) continue;
        replacePanel(panel.id, (item) => ({ ...item, closing: true }));
        scheduleRemove(panel.id);
      }
    },

    focus,

    patchLayout(id, patch) {
      const panel = optionsById(id);
      if (!panel) return;
      const base = panel.layout;
      const rect = clampPanelRect(
        {
          x: patch.x ?? base.x,
          y: patch.y ?? base.y,
          width: patch.width ?? base.width,
          height: patch.height ?? base.height,
        },
        workspace(),
        { width: panel.minWidth, height: panel.minHeight },
      );
      const next = { ...base, ...patch, ...rect };
      // 跳过无变化的写入：拖动中高频触发，避免持续产生新引用驱动无意义的重渲染。
      const keys = Object.keys(next) as Array<keyof PanelLayout>;
      if (keys.every((key) => base[key] === next[key])) return;
      replacePanel(id, (item) => ({ ...item, layout: next }));
    },

    setSnapPreview(preview) {
      setState({ snapPreview: preview });
    },

    dock(id, targetId, edge) {
      const panel = optionsById(id);
      if (!panel || id === targetId) return;
      if (!targetId) {
        replacePanel(id, (item) => ({
          ...item,
          layout: { ...item.layout, mode: 'docked', dockTargetId: null, dockEdge: edge },
        }));
        setState({ snapPreview: null });
        return;
      }
      const target = state.panels.find((item) => item.id === targetId);
      if (!target) return;
      const targetLayout = target.layout;
      replacePanel(id, (item) => ({
        ...item,
        layout: {
          ...item.layout,
          mode: 'docked',
          dockTargetId: targetId,
          dockEdge: edge,
          x: edge === 'right' ? targetLayout.x + targetLayout.width : targetLayout.x,
          y: edge === 'bottom' ? targetLayout.y + targetLayout.height : targetLayout.y,
        },
      }));
      setState({ snapPreview: null });
    },

    undock(id) {
      const panel = optionsById(id);
      if (!panel) return;
      replacePanel(id, (item) => ({
        ...item,
        layout: { ...item.layout, mode: 'floating', dockTargetId: null, dockEdge: null },
      }));
      setState({ snapPreview: null });
    },

    toggleMaximize(id) {
      const panel = optionsById(id);
      if (!panel || !panel.maximizable) return;
      const ws = workspace();
      if (panel.restoreRect) {
        const restored = nextLayout(id, panel.restoreRect);
        replacePanel(id, (item) => ({ ...item, layout: restored, restoreRect: null }));
        return;
      }
      const restoreRect: PanelRect = {
        x: panel.layout.x, y: panel.layout.y,
        width: panel.layout.width, height: panel.layout.height,
      };
      const maximized = nextLayout(id, { x: ws.x, y: ws.y, width: ws.width, height: ws.height });
      replacePanel(id, (item) => ({ ...item, layout: maximized, restoreRect }));
    },
  };

  return store;
}
