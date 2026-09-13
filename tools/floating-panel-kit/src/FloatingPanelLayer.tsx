import { useCallback, useEffect, useRef, useState } from 'react';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';

import { PanelErrorBoundary } from './ErrorBoundary';
import { getSnapPreview } from './geometry';
import { CloseIcon, MaximizeIcon, RestoreIcon } from './icons';
import { useFloatingPanelStore, usePanels, useSnapPreview } from './context';
import type { FloatingPanelStore } from './store';
import type { PanelInstance, PanelLayout } from './types';

export const SNAP_THRESHOLD_DEFAULT = 24;

type ResizeHandle = 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'nw';

interface PanelInteraction {
  kind: 'drag' | 'resize';
  pointerId: number;
  startX: number;
  startY: number;
  initial: PanelLayout;
  handle?: ResizeHandle;
}

// 全局同一时刻只有一次拖动/缩放；模块级引用让 ESC 能中断进行中的交互。
let activeInteractionCancel: (() => void) | null = null;

const RESIZE_HANDLES: ResizeHandle[] = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'];

/**
 * 浮动面板渲染层。放在 Provider 内、应用 JSX 的最后，
 * 负责把 store 中的面板渲染为可拖动、可缩放的 fixed 浮层。
 */
export function FloatingPanelLayer({ snapThreshold = SNAP_THRESHOLD_DEFAULT }: {
  /** 吸附触发距离（px），默认 24。传 0 关闭吸附。 */
  snapThreshold?: number;
}) {
  const store = useFloatingPanelStore();
  const panels = usePanels();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (activeInteractionCancel) {
        event.preventDefault();
        event.stopPropagation();
        activeInteractionCancel();
        return;
      }
      const { getState, close } = store;
      const top = [...getState().panels]
        .filter((panel) => panel.closable && !panel.closing)
        .sort((a, b) => b.layout.zIndex - a.layout.zIndex)[0];
      if (top) {
        event.preventDefault();
        event.stopPropagation();
        close(top.id);
      }
    };
    document.addEventListener('keydown', handleKeyDown, true);
    return () => document.removeEventListener('keydown', handleKeyDown, true);
  }, [store]);

  if (panels.length === 0) return null;
  return (
    <>
      {panels.map((panel) => (
        <PanelWindow key={panel.id} panel={panel} store={store} snapThreshold={snapThreshold} />
      ))}
    </>
  );
}

function PanelWindow({ panel, store, snapThreshold }: {
  panel: PanelInstance;
  store: FloatingPanelStore;
  snapThreshold: number;
}) {
  const interactionRef = useRef<PanelInteraction | null>(null);
  const [isInteracting, setIsInteracting] = useState(false);
  const snapPreview = useSnapPreview();
  const { id, layout } = panel;

  const clearInteraction = useCallback(() => {
    interactionRef.current = null;
    activeInteractionCancel = null;
    store.setSnapPreview(null);
    setIsInteracting(false);
  }, [store]);

  const cancelInteraction = useCallback(() => {
    const interaction = interactionRef.current;
    if (!interaction) return;
    store.patchLayout(id, interaction.initial);
    clearInteraction();
  }, [clearInteraction, id, store]);

  const releaseCapture = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const updateInteraction = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    const interaction = interactionRef.current;
    if (!interaction || event.pointerId !== interaction.pointerId) return;
    const dx = event.clientX - interaction.startX;
    const dy = event.clientY - interaction.startY;
    const base = interaction.initial;
    if (interaction.kind === 'drag') {
      const rect = { x: base.x + dx, y: base.y + dy, width: base.width, height: base.height };
      const preview = getSnapPreview(
        rect,
        store.workspace(),
        store.getState().panels.filter((item) => item.id !== id),
        snapThreshold,
      );
      store.patchLayout(id, rect);
      store.setSnapPreview(preview);
      return;
    }
    const handle = interaction.handle ?? 'se';
    const x = handle.includes('w') ? base.x + dx : base.x;
    const y = handle.includes('n') ? base.y + dy : base.y;
    const width = handle.includes('w') ? base.width - dx : handle.includes('e') ? base.width + dx : base.width;
    const height = handle.includes('n') ? base.height - dy : handle.includes('s') ? base.height + dy : base.height;
    store.patchLayout(id, { x, y, width, height });
  }, [id, snapThreshold, store]);

  const finishInteraction = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    const interaction = interactionRef.current;
    if (!interaction) return;
    if (interaction.kind === 'drag') {
      const preview = store.getState().snapPreview;
      if (preview) {
        store.patchLayout(id, preview.rect);
        store.dock(id, preview.targetId, preview.dockEdge);
      }
    }
    releaseCapture(event);
    clearInteraction();
  }, [clearInteraction, id, releaseCapture, store]);

  const beginInteraction = useCallback((event: ReactPointerEvent<HTMLElement>, kind: 'drag' | 'resize', handle?: ResizeHandle) => {
    if (event.button !== 0 || panel.closing) return;
    if (kind === 'drag' && isInteractiveTarget(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    store.focus(id);
    const initial: PanelLayout = { ...layout, mode: 'floating', dockTargetId: null, dockEdge: null };
    interactionRef.current = {
      kind, handle, pointerId: event.pointerId,
      startX: event.clientX, startY: event.clientY, initial,
    };
    if (layout.mode === 'docked') store.undock(id);
    activeInteractionCancel = cancelInteraction;
    setIsInteracting(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [cancelInteraction, id, layout, panel.closing, store]);

  const style = {
    '--fp-x': `${layout.x}px`,
    '--fp-y': `${layout.y}px`,
    '--fp-w': `${layout.width}px`,
    '--fp-h': `${layout.height}px`,
    zIndex: layout.zIndex,
  } as CSSProperties;

  const snapping = isInteracting && snapPreview !== null;

  return (
    <div
      style={style}
      tabIndex={0}
      className={`fp-panel fp-float${panel.closing ? ' fp-exit' : ''}${isInteracting ? ' fp-interacting' : ''}`}
      role="dialog"
      aria-modal="false"
      aria-label={panel.title}
      data-panel-id={id}
      data-fp-snapping={snapping || undefined}
      onPointerDown={() => store.focus(id)}
      onFocus={() => store.focus(id)}
    >
      <div
        className="fp-header"
        onPointerDown={(event) => beginInteraction(event, 'drag')}
        onPointerMove={updateInteraction}
        onPointerUp={finishInteraction}
        onPointerCancel={cancelInteraction}
      >
        <span className="fp-title">{panel.title}</span>
        <div className="fp-actions">
          {panel.maximizable && (
            <button
              type="button"
              className="fp-btn fp-btn-max"
              onClick={() => store.toggleMaximize(id)}
              title={panel.restoreRect ? '还原' : '最大化'}
              aria-label={panel.restoreRect ? '还原' : '最大化'}
            >
              {panel.restoreRect ? <RestoreIcon /> : <MaximizeIcon />}
            </button>
          )}
          {panel.closable && (
            <button
              type="button"
              className="fp-btn fp-btn-close"
              onClick={() => store.close(id)}
              title="关闭 (ESC)"
              aria-label="关闭"
            >
              <CloseIcon />
            </button>
          )}
        </div>
      </div>

      {RESIZE_HANDLES.map((handle) => (
        <div
          key={handle}
          className={`fp-resize-handle fp-resize-${handle}`}
          data-resize-handle={handle}
          aria-label={`调整面板大小 ${handle}`}
          onPointerDown={(event) => beginInteraction(event, 'resize', handle)}
          onPointerMove={updateInteraction}
          onPointerUp={finishInteraction}
          onPointerCancel={cancelInteraction}
        />
      ))}

      <div className="fp-body">
        <PanelErrorBoundary label={panel.title}>{panel.content}</PanelErrorBoundary>
      </div>
    </div>
  );
}

/** 标题栏上的按钮/输入框等不应触发拖动。 */
function isInteractiveTarget(target: EventTarget | null): boolean {
  return typeof Element !== 'undefined' &&
    target instanceof Element &&
    Boolean(target.closest('button, input, textarea, select, a, [data-no-panel-drag]'));
}
