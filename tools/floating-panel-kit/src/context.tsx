import { createContext, useContext, useEffect, useMemo, useSyncExternalStore } from 'react';
import type { ReactNode } from 'react';

import { createFloatingPanelStore } from './store';
import type { FloatingPanelStore, PanelStoreState } from './store';
import type { PanelInstance, SavedLayout, SnapPreview, WorkspaceRect } from './types';

const StoreContext = createContext<FloatingPanelStore | null>(null);

export interface FloatingPanelProviderProps {
  children: ReactNode;
  /**
   * localStorage 键名。设置后每次布局变化（防抖 300ms）会把各面板几何
   * 写入 localStorage，并在重新打开同 id 面板时恢复上次位置与尺寸。
   */
  storageKey?: string;
  /** 布局变化回调（防抖 300ms），用于把布局保存到自己的后端。 */
  onLayoutChange?: (layouts: Record<string, SavedLayout>) => void;
  /** 浮层 z-index 基准，默认 1000。 */
  zIndexBase?: number;
  /** 工作区四周留白（仅在未提供 getWorkspace 时生效）。默认 0。 */
  workspacePadding?: number;
  /** 自定义可用工作区，例如避开应用侧边栏和顶栏。 */
  getWorkspace?: () => WorkspaceRect;
  /** 关闭动画时长，默认 200ms。 */
  closeAnimationMs?: number;
}

function defaultWorkspaceWithPadding(padding: number): () => WorkspaceRect {
  return () => {
    const width = typeof window === 'undefined' ? 1280 : window.innerWidth || 1280;
    const height = typeof window === 'undefined' ? 800 : window.innerHeight || 800;
    return {
      x: padding,
      y: padding,
      width: Math.max(320, width - padding * 2),
      height: Math.max(240, height - padding * 2),
    };
  };
}

function loadSavedLayouts(storageKey: string): Record<string, SavedLayout> | undefined {
  if (typeof window === 'undefined') return undefined;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return undefined;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed as Record<string, SavedLayout> : undefined;
  } catch {
    return undefined;
  }
}

export function FloatingPanelProvider({
  children,
  storageKey,
  onLayoutChange,
  zIndexBase,
  workspacePadding = 0,
  getWorkspace,
  closeAnimationMs,
}: FloatingPanelProviderProps) {
  const store = useMemo(
    () => createFloatingPanelStore({
      zIndexBase,
      workspace: getWorkspace ?? defaultWorkspaceWithPadding(workspacePadding),
      savedLayouts: storageKey ? loadSavedLayouts(storageKey) : undefined,
      closeAnimationMs,
    }),
    // Provider 只初始化一次；运行时更换这些 props 需要重新挂载。
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(() => {
    if (!storageKey && !onLayoutChange) return undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = store.subscribe(() => {
      if (timer !== undefined) clearTimeout(timer);
      timer = setTimeout(() => {
        const layouts: Record<string, SavedLayout> = {};
        for (const panel of store.getState().panels) {
          if (panel.closing) continue;
          const { x, y, width, height } = panel.layout;
          layouts[panel.id] = { x, y, width, height };
        }
        if (storageKey) {
          try {
            window.localStorage.setItem(storageKey, JSON.stringify(layouts));
          } catch { /* 隐私模式等场景下写入失败可忽略 */ }
        }
        onLayoutChange?.(layouts);
      }, 300);
    });
    return () => {
      unsubscribe();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [store, storageKey, onLayoutChange]);

  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
}

export function useFloatingPanelStore(): FloatingPanelStore {
  const store = useContext(StoreContext);
  if (!store) throw new Error('useFloatingPanelStore 必须在 <FloatingPanelProvider> 内使用');
  return store;
}

/** 面板列表；store 采用不可变更新，可安全配合 useSyncExternalStore。 */
export function usePanels(): PanelInstance[] {
  const store = useFloatingPanelStore();
  return useSyncExternalStore(
    store.subscribe,
    () => store.getState().panels,
    () => [] as PanelInstance[],
  );
}

export function usePanelState<T>(selector: (state: PanelStoreState) => T): T {
  const store = useFloatingPanelStore();
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.getState()),
  );
}

export function useSnapPreview(): SnapPreview | null {
  return usePanelState((state) => state.snapPreview);
}
