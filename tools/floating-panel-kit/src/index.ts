export { FloatingPanelProvider, useFloatingPanelStore, usePanels, useSnapPreview, usePanelState } from './context';
export type { FloatingPanelProviderProps } from './context';
export { FloatingPanelLayer, SNAP_THRESHOLD_DEFAULT } from './FloatingPanelLayer';
export { createFloatingPanelStore } from './store';
export type { FloatingPanelStore, PanelStoreOptions, PanelStoreState } from './store';
export {
  bringToFront,
  choosePanelPosition,
  clampPanelRect,
  getSnapPreview,
  MIN_PANEL_HEIGHT,
  MIN_PANEL_WIDTH,
  SNAP_THRESHOLD,
} from './geometry';
export type { PanelLike, SizeConstraint } from './geometry';
export { PanelErrorBoundary } from './ErrorBoundary';
export type {
  DockEdge,
  OpenPanelOptions,
  PanelInstance,
  PanelLayout,
  PanelMode,
  PanelRect,
  SavedLayout,
  SnapPreview,
  WorkspaceRect,
} from './types';

// 引入组件即带样式；不希望副作用导入的话删掉这一行，
// 改为在应用入口手动 import 'floating-panel-kit/src/styles.css'。
import './styles.css';
