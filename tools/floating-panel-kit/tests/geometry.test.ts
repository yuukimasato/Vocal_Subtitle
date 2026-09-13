import { describe, expect, it } from 'vitest';

import { bringToFront, choosePanelPosition, clampPanelRect, getSnapPreview } from '../src/geometry';
import type { PanelLayout, WorkspaceRect } from '../src/types';

const workspace: WorkspaceRect = { x: 0, y: 0, width: 1280, height: 800 };

function layout(patch: Partial<PanelLayout> = {}): PanelLayout {
  return {
    mode: 'floating', x: 0, y: 0, width: 400, height: 300,
    zIndex: 1, dockTargetId: null, dockEdge: null, ...patch,
  };
}

describe('clampPanelRect', () => {
  it('强制最小尺寸并夹取到工作区内', () => {
    const rect = clampPanelRect({ x: -50, y: -50, width: 10, height: 10 }, workspace);
    expect(rect.width).toBe(240);
    expect(rect.height).toBe(160);
    expect(rect.x).toBe(0);
    expect(rect.y).toBe(0);
  });

  it('右下越界时收回到工作区内', () => {
    const rect = clampPanelRect({ x: 1200, y: 700, width: 400, height: 300 }, workspace);
    expect(rect.x).toBe(1280 - 400);
    expect(rect.y).toBe(800 - 300);
  });

  it('尺寸超过工作区时被压到工作区大小', () => {
    const rect = clampPanelRect({ x: 0, y: 0, width: 9999, height: 9999 }, workspace);
    expect(rect.width).toBe(1280);
    expect(rect.height).toBe(800);
  });

  it('负尺寸抛错', () => {
    expect(() => clampPanelRect({ x: 0, y: 0, width: -1, height: 100 }, workspace)).toThrow('panel_geometry_invalid');
  });
});

describe('getSnapPreview', () => {
  it('接近工作区左缘时贴边', () => {
    const preview = getSnapPreview({ x: 10, y: 100, width: 400, height: 300 }, workspace, []);
    expect(preview?.dockEdge).toBe('left');
    expect(preview?.rect.x).toBe(0);
  });

  it('接近其他面板右缘时吸附到其右边缘（dockEdge 指目标边的方位）', () => {
    const panels = [{ id: 'a', layout: layout({ x: 500, y: 100, width: 300, height: 300 }) }];
    const preview = getSnapPreview({ x: 790, y: 120, width: 200, height: 200 }, workspace, panels);
    expect(preview?.targetId).toBe('a');
    expect(preview?.dockEdge).toBe('right');
    expect(preview?.rect.x).toBe(800);
  });

  it('距离超出阈值时不吸附', () => {
    expect(getSnapPreview({ x: 100, y: 100, width: 400, height: 300 }, workspace, [])).toBeNull();
  });
});

describe('choosePanelPosition', () => {
  it('无既有面板时落在右上候选位', () => {
    const pos = choosePanelPosition({ width: 640, height: 520 }, [], workspace);
    expect(pos.x).toBe(1280 - 640 - 16);
    expect(pos.y).toBe(16);
  });

  it('右上被占用时退到不重叠的候选位', () => {
    const existing = [{ id: 'a', layout: layout({ x: 624, y: 16, width: 640, height: 520 }) }];
    const pos = choosePanelPosition({ width: 300, height: 200 }, existing, workspace);
    const overlaps = existing.some((p) =>
      pos.x < p.layout.x + p.layout.width && pos.x + 300 > p.layout.x &&
      pos.y < p.layout.y + p.layout.height && pos.y + 200 > p.layout.y);
    expect(overlaps).toBe(false);
  });

  it('所有候选都被占用时按 28px 级联偏移回退（与原实现一致的兜底）', () => {
    const existing = [{ id: 'a', layout: layout({ x: 624, y: 16, width: 640, height: 520 }) }];
    const pos = choosePanelPosition({ width: 640, height: 520 }, existing, workspace);
    expect(pos.x).toBe(16 + 28);
    expect(pos.y).toBe(16 + 28);
  });
});

describe('bringToFront', () => {
  it('把目标面板提到最高层级', () => {
    const panels = [
      { id: 'a', layout: layout({ zIndex: 3 }) },
      { id: 'b', layout: layout({ zIndex: 7 }) },
    ];
    const next = bringToFront(panels, 'a');
    expect(next.find((p) => p.id === 'a')!.layout.zIndex).toBe(8);
    expect(next.find((p) => p.id === 'b')!.layout.zIndex).toBe(7);
  });

  it('已是最高层时不产生新数组', () => {
    const panels = [
      { id: 'a', layout: layout({ zIndex: 3 }) },
      { id: 'b', layout: layout({ zIndex: 7 }) },
    ];
    expect(bringToFront(panels, 'b')).toBe(panels);
  });
});
