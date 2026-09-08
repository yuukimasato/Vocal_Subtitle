// 可拖拽分隔条：字幕列表宽度（竖条）、波形区高度（横条）。
// 双击分隔条恢复默认；布局记忆存 localStorage（静默降级）。
const LAYOUT_KEY = 'vstEditor.layout';
const DEFAULTS = { waveH: 180, cueW: 400 };
const LIMITS = {
  waveH: { min: 90, maxRatio: 0.6 },  // 上限相对 upper 区高度
  cueW: { min: 280, maxRatio: 0.65 }, // 上限相对窗口宽度
};

export function initSplitters() {
  const root = document.documentElement;
  const layout = loadLayout();
  root.style.setProperty('--wave-h', `${layout.waveH}px`);
  root.style.setProperty('--cue-w', `${layout.cueW}px`);

  wireSplitter({
    el: document.getElementById('split-wave'),
    orientation: 'h',
    layout,
    sizeKey: 'waveH',
    paneSelector: '.wave-wrap',
    cssVar: '--wave-h',
    invert: true, // 分隔条跟随鼠标:向下拖 = 播放器变高(波形变矮)
  });
  wireSplitter({
    el: document.getElementById('split-cue'),
    orientation: 'v',
    layout,
    sizeKey: 'cueW',
    paneSelector: '.cue-pane',
    cssVar: '--cue-w',
    invert: true, // 分隔条跟随鼠标:向左拖 = 播放器变窄(字幕列表变宽)
  });
}

function loadLayout() {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    const saved = raw ? JSON.parse(raw) : {};
    return {
      waveH: clamp(saved.waveH ?? DEFAULTS.waveH, LIMITS.waveH.min, 800),
      cueW: clamp(saved.cueW ?? DEFAULTS.cueW, LIMITS.cueW.min, 1200),
    };
  } catch {
    return { ...DEFAULTS };
  }
}

function saveLayout(layout) {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  } catch {
    // 静默降级
  }
}

function wireSplitter({ el, orientation, layout, sizeKey, paneSelector, cssVar, invert = false }) {
  if (!el) return;
  const root = document.documentElement;
  // 尺寸上限随容器实时变化（窗口缩放后上限跟着变）
  const maxOf = () => (orientation === 'h'
    ? el.closest('.workspace').clientHeight * LIMITS[sizeKey].maxRatio
    : window.innerWidth * LIMITS[sizeKey].maxRatio);
  const apply = (size) => {
    layout[sizeKey] = clamp(size, LIMITS[sizeKey].min, maxOf());
    root.style.setProperty(cssVar, `${Math.round(layout[sizeKey])}px`);
  };
  drag(el, {
    onStart: () => ({
      size: parseInt(getComputedStyle(document.querySelector(paneSelector))[orientation === 'h' ? 'height' : 'width'], 10),
    }),
    onMove: (start, delta) => {
      const deltaSize = orientation === 'h' ? delta.y : delta.x;
      const applied = invert ? -deltaSize : deltaSize;
      apply(start.size + applied);
    },
    onReset: () => {
      layout[sizeKey] = DEFAULTS[sizeKey];
      root.style.setProperty(cssVar, `${DEFAULTS[sizeKey]}px`);
    },
    onSettled: () => saveLayout(layout),
  });
  // 窗口缩小后已保存的尺寸可能超出上限，把 stage 挤到 0 宽：resize 时重钳并落盘
  window.addEventListener('resize', () => {
    apply(layout[sizeKey]);
    saveLayout(layout);
  });
}

function drag(el, { onStart, onMove, onReset, onSettled }) {
  el.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    const start = { ...onStart(), x: event.clientX, y: event.clientY };
    el.classList.add('dragging');
    document.body.style.userSelect = 'none';

    // move/up 绑到 window：指针滑出分隔条或 capture 失败（合成事件）都不脱靶
    const move = (e) => {
      onMove(start, { x: e.clientX - start.x, y: e.clientY - start.y });
    };
    const up = () => {
      el.classList.remove('dragging');
      document.body.style.userSelect = '';
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      onSettled();
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  });
  el.addEventListener('dblclick', () => {
    onReset();
    onSettled();
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
