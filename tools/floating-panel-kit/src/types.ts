import type { ReactNode } from 'react';

/** 面板/工作区的矩形几何。坐标为相对视口（position: fixed 坐标系）的像素。 */
export interface PanelRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** 可用工作区（面板允许活动的区域）。 */
export type WorkspaceRect = PanelRect;

export type DockEdge = 'left' | 'right' | 'top' | 'bottom';
export type PanelMode = 'floating' | 'docked';

/** 面板的完整布局状态：几何 + 层级 + 停靠信息。 */
export interface PanelLayout extends PanelRect {
  mode: PanelMode;
  zIndex: number;
  dockTargetId: string | null;
  dockEdge: DockEdge | null;
}

/** 拖动过程中命中的吸附预览；rect 为吸附后的落点矩形。 */
export interface SnapPreview {
  targetId: string | null;
  dockEdge: DockEdge;
  rect: PanelRect;
}

export interface OpenPanelOptions {
  /** 唯一 id。重复打开同一 id 会复用并聚焦已有面板，而不是叠加新窗口。 */
  id: string;
  /** 标题栏文字，同时作为 aria-label。 */
  title: string;
  /** 面板内容（任意 ReactNode）。 */
  content: ReactNode;
  /** 初始尺寸；缺省 420×300，会被夹取到工作区内。 */
  width?: number;
  height?: number;
  /** 初始位置；缺省时自动选择不与现有面板重叠的位置。 */
  x?: number;
  y?: number;
  /** 是否可关闭（标题栏关闭按钮 + ESC）。默认 true。 */
  closable?: boolean;
  /** 是否显示最大化/还原按钮。默认 false。 */
  maximizable?: boolean;
  minWidth?: number;
  minHeight?: number;
}

export interface PanelInstance {
  id: string;
  title: string;
  content: ReactNode;
  layout: PanelLayout;
  closable: boolean;
  maximizable: boolean;
  minWidth: number;
  minHeight: number;
  /** 最大化前的几何，用于还原；null 表示当前未最大化。 */
  restoreRect: PanelRect | null;
  openedAt: number;
  /** 关闭动画进行中；动画结束后由 store 移除。 */
  closing: boolean;
}

/** 持久化时保存的最小几何（不含 zIndex 等瞬态字段）。 */
export type SavedLayout = PanelRect;
