// 可拖拽分隔条：视频/波形高度、波形/字幕列表宽度。
// 双击分隔条恢复默认；布局记忆存 localStorage（静默降级）。
const LAYOUT_KEY = 'vstEditor.layout';
const DEFAULTS = { waveH: 180, cueW: 400 };
const LIMITS = {
  waveH: { min: 90, maxRatio: 0.6 },   // 上限相对 stage 高度
  cueW: { min: 280, maxRatio: 0.65 },  // 上限相对窗口宽度
};

export function initSplitters() {
  const root = document.documentElement;
  const layout = loadLayout();
  root.style.setProperty('--wave-h', `${layout.waveH}px`);
  root.style.setProperty('--cue-w', `${layout.cueW}px`);

  wireRowSplitter(document.getElementById('split-wave'), layout);
  wireColSplitter(document.getElementById('split-cue'), layout);
}

function loadLayout() {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY);
    const saved = raw ? JSON.parse(raw) : {};
    return {
      waveH: clamp(saved.waveH ?? DEFAULTS.waveH, 90, 800),
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

function wireRowSplitter(el, layout) {
  // 拖动改变下方波形区高度
  if (!el) return;
  const root = document.documentElement;
  drag(el, {
    onStart: () => ({ size: parseInt(getComputedStyle(document.querySelector('.wave-wrap')).height, 10) }),
    onMove: (start, delta) => {
      const max = el.parentElement.clientHeight * LIMITS.waveH.maxRatio;
      layout.waveH = clamp(start.size + delta.y, LIMITS.waveH.min, max);
      root.style.setProperty('--wave-h', `${Math.round(layout.waveH)}px`);
    },
    onReset: () => {
      layout.waveH = DEFAULTS.waveH;
      root.style.setProperty('--wave-h', `${DEFAULTS.waveH}px`);
    },
    onSettled: () => saveLayout(layout),
  });
}

function wireColSplitter(el, layout) {
  // 拖动改变右侧字幕列表宽度（向左拖加宽）
  if (!el) return;
  const root = document.documentElement;
  drag(el, {
    onStart: () => ({ size: parseInt(getComputedStyle(document.querySelector('.cue-pane')).width, 10) }),
    onMove: (start, delta) => {
      const max = window.innerWidth * LIMITS.cueW.maxRatio;
      layout.cueW = clamp(start.size - delta.x, LIMITS.cueW.min, max);
      root.style.setProperty('--cue-w', `${Math.round(layout.cueW)}px`);
    },
    onReset: () => {
      layout.cueW = DEFAULTS.cueW;
      root.style.setProperty('--cue-w', `${DEFAULTS.cueW}px`);
    },
    onSettled: () => saveLayout(layout),
  });
}

function drag(el, { onStart, onMove, onReset, onSettled }) {
  el.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    const start = { ...onStart(), x: event.clientX, y: event.clientY };
    el.classList.add('dragging');
    try {
      el.setPointerCapture(event.pointerId);
    } catch {
      // 合成事件或指针已失效：不捕获也能收到 el 上的后续事件
    }
    document.body.style.userSelect = 'none';

    const move = (e) => {
      onMove(start, { x: e.clientX - start.x, y: e.clientY - start.y });
    };
    const up = (e) => {
      el.classList.remove('dragging');
      document.body.style.userSelect = '';
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      try {
        if (e.pointerId !== undefined) el.releasePointerCapture(e.pointerId);
      } catch {
        // 指针已释放
      }
      onSettled();
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  });
  el.addEventListener('dblclick', () => {
    onReset();
    onSettled();
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
