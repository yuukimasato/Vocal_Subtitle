import type { DockEdge, PanelLayout, PanelRect, SnapPreview, WorkspaceRect } from './types';

export const MIN_PANEL_WIDTH = 240;
export const MIN_PANEL_HEIGHT = 160;
export const SNAP_THRESHOLD = 24;
const FLOAT_MARGIN = 16;
const CASCADE_STEP = 28;

export interface SizeConstraint {
  width: number;
  height: number;
}

/** 参与布局计算的任何对象的最小形状（PanelInstance 满足该约束）。 */
export interface PanelLike {
  id: string;
  layout: PanelLayout;
}

function invalidRect(rect: PanelRect): boolean {
  return rect === null || typeof rect !== 'object' || rect.width < 0 || rect.height < 0;
}

/**
 * 把矩形夹取到工作区内：位置不越界、尺寸不小于 min 且不超过工作区。
 * 非法输入（负尺寸）直接抛错，避免把坏数据写进状态。
 */
export function clampPanelRect(
  rect: PanelRect,
  workspace: WorkspaceRect,
  min: SizeConstraint = { width: MIN_PANEL_WIDTH, height: MIN_PANEL_HEIGHT },
): PanelRect {
  if (invalidRect(rect)) throw new Error('panel_geometry_invalid');
  const x = Number.isFinite(rect.x) ? rect.x : workspace.x;
  const y = Number.isFinite(rect.y) ? rect.y : workspace.y;
  const rawWidth = Number.isFinite(rect.width) && rect.width >= 0 ? rect.width : min.width;
  const rawHeight = Number.isFinite(rect.height) && rect.height >= 0 ? rect.height : min.height;
  const width = Math.min(workspace.width, Math.max(min.width, rawWidth));
  const height = Math.min(workspace.height, Math.max(min.height, rawHeight));
  return {
    x: Math.max(workspace.x, Math.min(workspace.x + workspace.width - width, x)),
    y: Math.max(workspace.y, Math.min(workspace.y + workspace.height - height, y)),
    width,
    height,
  };
}

/**
 * 吸附检测：先看四条工作区边，再逐个看其他面板的边缘。
 * 距离 ≤ threshold 且投影有重叠时命中，返回吸附后的落点矩形。
 */
export function getSnapPreview(
  rect: PanelRect,
  workspace: WorkspaceRect,
  panels: PanelLike[],
  threshold: number = SNAP_THRESHOLD,
): SnapPreview | null {
  const candidate = clampPanelRect(rect, workspace);
  const edgeChecks: Array<[DockEdge, number]> = [
    ['left', Math.abs(candidate.x - workspace.x)],
    ['right', Math.abs(candidate.x + candidate.width - (workspace.x + workspace.width))],
    ['top', Math.abs(candidate.y - workspace.y)],
    ['bottom', Math.abs(candidate.y + candidate.height - (workspace.y + workspace.height))],
  ];
  const edge = edgeChecks.find(([, distance]) => distance <= threshold);
  if (edge) {
    const [dockEdge] = edge;
    const snapped = { ...candidate };
    if (dockEdge === 'left') snapped.x = workspace.x;
    if (dockEdge === 'right') snapped.x = workspace.x + workspace.width - snapped.width;
    if (dockEdge === 'top') snapped.y = workspace.y;
    if (dockEdge === 'bottom') snapped.y = workspace.y + workspace.height - snapped.height;
    return { targetId: null, dockEdge, rect: snapped };
  }
  for (const panel of panels) {
    const target = panel.layout;
    const checks: Array<[DockEdge, number, PanelRect]> = [
      ['left', Math.abs(candidate.x + candidate.width - target.x), { ...candidate, x: target.x - candidate.width }],
      ['right', Math.abs(candidate.x - (target.x + target.width)), { ...candidate, x: target.x + target.width }],
      ['top', Math.abs(candidate.y + candidate.height - target.y), { ...candidate, y: target.y - candidate.height }],
      ['bottom', Math.abs(candidate.y - (target.y + target.height)), { ...candidate, y: target.y + target.height }],
    ];
    const match = checks.find(([, distance, snapped]) => distance <= threshold &&
      ((snapped.y < target.y + target.height && snapped.y + snapped.height > target.y) ||
        (snapped.x < target.x + target.width && snapped.x + snapped.width > target.x)));
    if (match) return { targetId: panel.id, dockEdge: match[0], rect: clampPanelRect(match[2], workspace) };
  }
  return null;
}

/** 把 id 对应的面板 zIndex 提到当前最大值 + 1（点击/聚焦置顶）。 */
export function bringToFront<T extends PanelLike>(panels: T[], id: string): T[] {
  const target = panels.find((panel) => panel.id === id);
  if (!target) return panels;
  const max = panels.reduce((value, panel) => Math.max(value, panel.layout.zIndex), 0);
  if (target.layout.zIndex === max) return panels;
  return panels.map((panel) => panel.id === id
    ? { ...panel, layout: { ...panel.layout, zIndex: max + 1 } }
    : panel);
}

function rectsOverlap(a: PanelRect, b: PanelRect): boolean {
  return (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );
}

/**
 * 为新面板挑选打开位置：依次尝试右上、居中、中上、左上四个候选，
 * 全部与现有面板重叠时按 28px 级联偏移。
 */
export function choosePanelPosition(
  size: SizeConstraint,
  existing: PanelLike[],
  workspace: WorkspaceRect,
): { x: number; y: number } {
  const width = Math.min(size.width, workspace.width);
  const height = Math.min(size.height, workspace.height);
  const candidates: Array<{ x: number; y: number }> = [
    { x: workspace.x + workspace.width - width - FLOAT_MARGIN, y: workspace.y + FLOAT_MARGIN },
    { x: workspace.x + Math.round((workspace.width - width) / 2), y: workspace.y + Math.round((workspace.height - height) / 2) },
    { x: workspace.x + Math.round((workspace.width - width) / 2), y: workspace.y + FLOAT_MARGIN },
    { x: workspace.x + FLOAT_MARGIN, y: workspace.y + FLOAT_MARGIN },
  ];
  for (const candidate of candidates) {
    const rect = clampPanelRect({ ...candidate, width, height }, workspace);
    if (!existing.some((panel) => rectsOverlap(rect, panel.layout))) {
      return { x: rect.x, y: rect.y };
    }
  }
  const offset = (existing.length % 4) * CASCADE_STEP;
  const rect = clampPanelRect(
    { x: workspace.x + FLOAT_MARGIN + offset, y: workspace.y + FLOAT_MARGIN + offset, width, height },
    workspace,
  );
  return { x: rect.x, y: rect.y };
}
