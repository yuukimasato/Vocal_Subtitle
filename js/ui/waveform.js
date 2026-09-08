// Aegisub 音频显示（audio_display.cpp）的 Web 移植。
// - 波形渲染仍用 wavesurfer（interact 关闭），点击/拖拽由本模块接管并翻译为
//   timing 控制器（js/audio/timing.js，Aegisub 定时控制器语义）的调用；
// - 顶部为时间标尺（拖动平移视图），波形上方覆盖层绘制活动行/多选行/非活动行
//   的选区与红/蓝开始/结束边界、悬停时间标签；
// - 交互对齐 Aegisub：左键扫选新范围/抓取边缘（3px 灵敏度）、右键设结束点、
//   Alt 平移、Ctrl 拖共享边界、Shift 临时吸附（10px）、中键定位播放头、
//   悬停边缘↔光标、鼠标释放在显示边缘时自动滚屏、滚轮滚动视图（Ctrl+滚轮缩放）；
// - 音频盒获得焦点时进入 Aegisub「Audio」键位上下文（见 shortcuts.js）。
import WaveSurfer from '../../vendor/wavesurfer.esm.js';

const SENSITIVITY_PX = 3; // Audio/Start Drag Sensitivity：边缘抓取半径
const SNAP_PX = 10; // Audio/Snap/Distance：吸附半径
const EDGE_FRACTION = 20; // 释放时鼠标位于 1/20 宽度内 → 视图滚过 1/3（Aegisub 同款）
const RULER_H = 22;
const WHEEL_LINE_PX = 16; // deltaMode=1（Firefox 行模式）每行折算像素

// 边界颜色取自 Aegisub 默认主题
const MARKER_START = 'rgb(216,0,0)';
const MARKER_END = 'rgb(0,0,216)';
const MARKER_INACTIVE = 'rgba(190,190,190,0.85)';
const FILL_ACTIVE = 'rgba(124,156,255,0.22)';
const FILL_SELECTED = 'rgba(124,156,255,0.12)';
const FILL_INACTIVE = 'rgba(124,156,255,0.06)';

// 覆盖层样式：wavesurfer 的 DOM 挂在自己的 shadow root 里，主文档样式表
// （css/editor.css）作用不到这棵树，因此 .ab-* 规则只能在这里维护，
// 于 attachOverlay 时注入同一棵 shadow tree。
const OVERLAY_CSS = `
  .ab-overlay { position: absolute; inset: 0; pointer-events: none; }
  .ab-line { position: absolute; top: 0; bottom: 0; }
  .ab-fill-active { background: rgba(124, 156, 255, 0.22); }
  .ab-fill-selected { background: rgba(124, 156, 255, 0.12); }
  .ab-fill-inactive { background: rgba(124, 156, 255, 0.06); }
  .ab-marker { position: absolute; top: 0; bottom: 0; width: 2px; }
  .ab-edge-inactive { background: rgba(190, 190, 190, 0.85); width: 1px; }
  .ab-track-cursor { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(230, 237, 243, 0.55); }
  .ab-track-label {
    position: absolute;
    top: 2px;
    left: 3px;
    font: 10px/1.2 ui-monospace, monospace;
    padding: 0 3px;
    border-radius: 2px;
    background: rgba(0, 0, 0, 0.65);
    color: rgba(230, 237, 243, 0.9);
    white-space: nowrap;
  }
`;

function formatTime(t) {
  const ms = Math.round(t * 1000);
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const frac = ms % 1000;
  return `${m}:${String(s).padStart(2, '0')}.${String(frac).padStart(3, '0')}`;
}

export function createWaveform({
  store,
  actions,
  player,
  timing,
  displayEl,
  containerEl,
  fallbackEl,
  messageEl,
}) {
  let ws = null;
  let fallbackMode = false;
  let pxPerSec = 40;
  let overlay = null; // 波形内容坐标系里的覆盖层（随内容滚动）
  let amplitudeScale = 1; // 纵向缩放（渲染高度倍数）
  let appliedHeight = 0;
  let ruler = displayEl.querySelector('#wave-ruler');
  if (!ruler) {
    ruler = document.createElement('canvas');
    ruler.id = 'wave-ruler';
    displayEl.insertBefore(ruler, displayEl.firstChild);
  }

  // 拖拽会话：{ kind:'markers'|'pan'|'ruler', ... }
  let drag = null;
  let dragButtons = 0; // timing 拖拽期间记录的 event.buttons 位，用于检测和弦按键变化
  let hoverX = null; // 悬停位置（displayEl 内 CSS 像素）
  let autoScrollDir = 0;

  const opts = () => store.state.audioOptions;

  // ---------- 缩放/滚动 ----------
  function visibleWidth() {
    return containerEl.clientWidth || 600;
  }

  function duration() {
    return (ws && !fallbackMode ? ws.getDuration() : 0) || store.state.duration || 0;
  }

  function setZoom(next) {
    pxPerSec = Math.min(1000, Math.max(5, next));
    if (ws && !fallbackMode) {
      try {
        ws.zoom(pxPerSec);
      } catch {
        // 尚未解码完成：只记录数值，ready 后再应用
      }
    }
    store.emit('zoom'); // 通知工具栏回写横向缩放滑条（滚轮/ready 初始缩放时同步）
    syncView();
  }

  function getZoom() {
    return pxPerSec;
  }

  // 当前横向缩放（px/秒）：工具栏滑条做对数反映射用
  function getPxPerSec() {
    return pxPerSec;
  }

  // 纵向缩放：经 wavesurfer 渲染高度体现（容器裁剪），0.125x..4x
  function applyAmplitude() {
    if (!ws || fallbackMode) return;
    const base = containerEl.clientHeight || 150;
    const target = Math.max(24, Math.round(base * amplitudeScale));
    if (target !== appliedHeight) {
      appliedHeight = target;
      try {
        ws.setOptions({ height: target });
      } catch {
        // 未就绪时忽略，ready 后会再次应用
      }
    }
  }

  function setAmplitudeScale(scale) {
    amplitudeScale = Math.min(4, Math.max(0.125, scale));
    applyAmplitude();
  }

  function scrollPx(left) {
    if (!ws || fallbackMode) return;
    ws.setScroll(Math.max(0, left));
  }

  function scrollByViewport(direction) {
    scrollPx((ws?.getScroll?.() ?? 0) + direction * (visibleWidth() / 3));
  }

  // Aegisub audio/scroll/left|right：固定像素滚动（128px）
  function scrollByPixels(delta) {
    scrollPx((ws?.getScroll?.() ?? 0) + delta);
  }

  // 滚动视图使当前选区居中（Aegisub ScrollTimeRangeInView / audio/go_to）
  function scrollToSelection() {
    const range = timing.getPrimaryRange();
    if (!range || fallbackMode || !ws) return;
    const center = ((range.start + range.end) / 2) * pxPerSecNow();
    scrollPx(center - visibleWidth() / 2);
  }

  function selectionVisible(range) {
    if (!ws) return true;
    const left = ws.getScroll();
    const right = left + visibleWidth();
    const scale = pxPerSecNow();
    return range.start * scale >= left && range.end * scale <= right;
  }

  // ---------- 时间 ↔ 像素 ----------
  // 当前有效的像素/秒：波形模式取缩放值，降级视图为整宽铺满。
  // 注意 wavesurfer 在 dur*pxPerSec 不足一屏时会把波形拉伸铺满（fit），
  // 此时实际比例是视口宽/时长；覆盖层/标尺绘制必须与波形同比例，否则错位。
  function pxPerSecNow() {
    if (fallbackMode) {
      const dur = store.state.duration || 0;
      const w = fallbackEl.clientWidth || visibleWidth();
      return dur > 0 ? w / dur : 1;
    }
    const dur = duration() || 0;
    const fit = dur > 0 ? visibleWidth() / dur : 0;
    return Math.max(pxPerSec, fit);
  }

  function timeAtX(clientX) {
    if (fallbackMode || !ws) {
      const rect = fallbackEl.getBoundingClientRect();
      const dur = store.state.duration || 0;
      if (!dur || !rect.width) return 0;
      return Math.min(Math.max(0, ((clientX - rect.left) / rect.width) * dur), dur);
    }
    // wrapper 是滚动内容元素：rect.left 已随滚动平移，无需再加 scrollLeft
    const rect = ws.getWrapper().getBoundingClientRect();
    const dur = ws.getDuration() || store.state.duration || 0;
    if (!dur || !rect.width) return 0;
    const t = ((clientX - rect.left) / rect.width) * dur;
    return Math.min(Math.max(0, t), dur);
  }

  function snapRangeFor(event) {
    // Aegisub：选项开启时默认吸附、按住 Shift 反之
    const enabled = opts().snap !== event.shiftKey;
    return enabled ? SNAP_PX / pxPerSecNow() : 0;
  }

  // ---------- wavesurfer 装配 ----------
  function ensureWs(url) {
    if (ws) return ws;
    ws = WaveSurfer.create({
      container: containerEl,
      media: player.video(),
      url, // 交给构造器自动加载；外部再 load() 会与它竞态触发 AbortError
      waveColor: '#3d4a63',
      progressColor: '#7c9cff',
      cursorColor: '#e6edf3',
      cursorWidth: 1,
      height: containerEl.clientHeight || 150,
      interact: false,
      dragToSeek: false,
      autoCenter: false,
      hideScrollbar: true,
      normalize: true,
    });
    ws.on('error', (err) => {
      // 换媒体时上一个加载被 abort（DOMException: AbortError）不是解码失败，
      // 忽略以免快速连换媒体时误切降级视图
      if (isAbortError(err)) return;
      enterFallback(err);
    });
    ws.on('ready', () => {
      if (!fallbackMode) messageEl.textContent = '';
      // 首次就绪：按整窗约 60 秒选择缩放，并同步应用到 wavesurfer。
      // 只记录不应用的话 wavesurfer 停留在整段铺满的 fit 模式，
      // 覆盖层/标尺按 pxPerSec 绘制会与波形错位（长音频尤为明显）。
      const dur = ws.getDuration() || 0;
      if (dur > 0 && containerEl.clientWidth > 0) {
        setZoom(containerEl.clientWidth / Math.min(60, dur));
      }
      attachOverlay();
      applyAmplitude();
      syncView();
    });
    ws.on('zoom', () => syncView());
    ws.on('scroll', () => drawRuler());
    return ws;
  }

  function attachOverlay() {
    if (!ws || overlay) return;
    const wrapper = ws.getWrapper();
    const root = wrapper.getRootNode();
    if (typeof ShadowRoot !== 'undefined' && root instanceof ShadowRoot && !root.querySelector('style.ab-style')) {
      const style = document.createElement('style');
      style.className = 'ab-style';
      style.textContent = OVERLAY_CSS;
      root.appendChild(style);
    }
    overlay = document.createElement('div');
    overlay.className = 'ab-overlay';
    wrapper.appendChild(overlay);
  }

  // 换媒体时的加载中止（wavesurfer abort 上一个 fetch 后透传的 DOMException）
  const isAbortError = (err) => err?.name === 'AbortError';

  // 加载序号：快速连换媒体时旧加载的 ready/error 已过期，一律忽略（竞态误降级）
  let loadSeq = 0;

  async function loadMedia(url) {
    const seq = ++loadSeq;
    exitFallback();
    messageEl.textContent = '正在解码音频…';
    try {
      const isNew = !ws;
      if (isNew) ensureWs(url);
      else {
        if (overlay) {
          overlay.remove();
          overlay = null;
        }
        await ws.load(url);
      }
      await new Promise((resolve, reject) => {
        if (ws.getDecodedData()) return resolve();
        const ok = () => {
          clean();
          resolve();
        };
        const bad = (err) => {
          clean();
          reject(err ?? new Error('波形解码失败'));
        };
        const clean = () => {
          ws.un('ready', ok);
          ws.un('error', bad);
        };
        ws.on('ready', ok);
        ws.on('error', bad);
      });
      if (seq !== loadSeq) return; // 期间又发起了新的加载：过期结果忽略
      if (!fallbackMode) messageEl.textContent = '';
      syncView();
    } catch (err) {
      // 过期加载的错误与加载中止（换媒体竞态）都不算解码失败
      if (seq !== loadSeq || isAbortError(err)) return;
      enterFallback(err);
    }
  }

  // ---------- 覆盖层绘制 ----------
  // rAF 合并视图重建：同一帧内多次请求只重建一次（pointermove 高频触发的悬停重绘用）
  let syncQueued = false;
  function requestSyncView() {
    if (syncQueued) return;
    syncQueued = true;
    requestAnimationFrame(() => {
      syncQueued = false;
      syncView();
    });
  }

  function syncView() {
    if (fallbackMode) {
      drawFallback();
      drawRuler();
      return;
    }
    if (!ws || !ws.getDecodedData()) return;
    rebuildOverlay();
    drawRuler();
  }

  function makeLine(cls, left, width) {
    const scale = pxPerSecNow();
    const el = document.createElement('div');
    el.className = `ab-line ${cls}`;
    el.style.left = `${Math.max(0, left * scale)}px`;
    el.style.width = `${Math.max(0, (width - left) * scale)}px`;
    return el;
  }

  function addRange(list, range, fillCls, edgeCls) {
    if (!range) return;
    const box = makeLine(fillCls, range.start, range.end);
    if (edgeCls) {
      const start = document.createElement('div');
      start.className = `ab-marker ${edgeCls}`;
      start.style.left = '0';
      const end = document.createElement('div');
      end.className = `ab-marker ${edgeCls}`;
      end.style.right = '0';
      box.append(start, end);
    }
    list.appendChild(box);
  }

  function rebuildOverlay() {
    if (fallbackMode) return; // 降级视图由 drawFallback 绘制
    if (!overlay) attachOverlay();
    if (!overlay) return;
    overlay.textContent = '';
    const dur = duration();
    overlay.style.width = `${Math.max(dur * pxPerSecNow(), visibleWidth())}px`;
    const view = timing.lineView();
    view.inactive.forEach((r) => addRange(overlay, r, 'ab-fill-inactive', 'ab-edge-inactive'));
    view.selected.forEach((r) => addRange(overlay, r, 'ab-fill-selected', 'ab-edge-inactive'));
    if (view.active) {
      const box = makeLine('ab-fill-active', view.active.start, view.active.end);
      const start = document.createElement('div');
      start.className = 'ab-marker';
      start.style.left = '0';
      start.style.background = MARKER_START;
      const end = document.createElement('div');
      end.className = 'ab-marker';
      end.style.right = '0';
      end.style.background = MARKER_END;
      box.append(start, end);
      overlay.appendChild(box);
    }
    // 悬停时间指示线
    if (hoverX != null) {
      const scale = pxPerSecNow();
      const t = timeAtClientX(hoverX);
      const line = document.createElement('div');
      line.className = 'ab-track-cursor';
      line.style.left = `${t * scale}px`;
      const label = document.createElement('div');
      label.className = 'ab-track-label mono';
      label.textContent = formatTime(t);
      line.appendChild(label);
      overlay.appendChild(line);
    }
  }

  function timeAtClientX(clientX) {
    return timeAtX(clientX);
  }

  // ---------- 时间标尺 ----------
  const RULER_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 3000];

  function drawRuler() {
    const width = ruler.clientWidth || displayEl.clientWidth || 600;
    const dpr = window.devicePixelRatio || 1;
    if (ruler.width !== Math.round(width * dpr) || ruler.height !== Math.round(RULER_H * dpr)) {
      ruler.width = Math.round(width * dpr);
      ruler.height = Math.round(RULER_H * dpr);
    }
    const ctx = ruler.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, RULER_H);

    const dur = duration();
    const scale = fallbackMode ? (dur ? width / dur : 1) : pxPerSecNow();
    const scrollLeft = fallbackMode || !ws ? 0 : ws.getScroll();
    const t0 = scrollLeft / scale;
    const t1 = t0 + width / scale;

    const step = RULER_STEPS.find((s) => s * scale >= 70) ?? 6000;
    const minor = step / 5;
    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'top';
    const first = Math.ceil(t0 / minor - 1e-9) * minor;
    for (let t = first; t <= t1 + 1e-9; t += minor) {
      if (t < 0) continue;
      const x = Math.round(t * scale - scrollLeft) + 0.5;
      const isMajor = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      ctx.strokeStyle = 'rgba(230,237,243,0.45)';
      ctx.beginPath();
      ctx.moveTo(x, isMajor ? 6 : 15);
      ctx.lineTo(x, RULER_H);
      ctx.stroke();
      if (isMajor) {
        ctx.fillStyle = '#8b949e';
        ctx.fillText(formatRuler(t, step), x + 3, 2);
      }
    }
  }

  function formatRuler(t, step) {
    const total = Math.round(t * 1000);
    const m = Math.floor(total / 60000);
    const s = Math.floor((total % 60000) / 1000);
    if (step < 1) {
      return `${m}:${String(s).padStart(2, '0')}.${Math.floor((total % 1000) / 100)}`;
    }
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  // ---------- 降级视图 ----------
  function enterFallback(err) {
    if (fallbackMode) return;
    fallbackMode = true;
    containerEl.hidden = true;
    fallbackEl.hidden = false;
    messageEl.textContent =
      '无法解码音频波形（该媒体可能没有音轨，或编码不受支持/文件过大），已切换为时间刻度视图：点击/中键可定位，拖拽定时请在列表或快捷键中完成。';
    console.warn('[waveform] decode failed:', err);
    syncView();
  }

  function exitFallback() {
    if (!fallbackMode) return;
    fallbackMode = false;
    containerEl.hidden = false;
    fallbackEl.hidden = true;
    messageEl.textContent = '';
  }

  function drawFallback() {
    const canvas = fallbackEl.querySelector('canvas');
    const dur = store.state.duration || 1;
    const width = fallbackEl.clientWidth || 600;
    const height = fallbackEl.clientHeight || 150;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const targetTicks = Math.max(4, Math.floor(width / 90));
    const rawStep = dur / targetTicks;
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const step = steps.find((s) => s >= rawStep) ?? 900;
    ctx.fillStyle = '#6e7681';
    ctx.font = '10px monospace';
    for (let t = 0; t <= dur; t += step) {
      const x = (t / dur) * width;
      ctx.fillRect(x, 0, 1, height);
      ctx.fillText(`${Math.round(t)}s`, x + 3, 11);
    }

    // 选区与边界标记（与波形模式同款视觉：活动行红/蓝边界，其余灰）。
    // 画布已经位于标尺下方（css #wave-fallback top:23px），不能再按含标尺的高度
    // 内缩，否则选区带整体下移、与波形模式的铺满效果不一致
    const view = timing.lineView();
    const paint = (range, color, edges) => {
      const x = (range.start / dur) * width;
      const w = Math.max(2, ((range.end - range.start) / dur) * width);
      const top = 2;
      const bandH = height - 4;
      ctx.fillStyle = color;
      ctx.fillRect(x, top, w, bandH);
      if (edges) {
        ctx.fillStyle = edges[0];
        ctx.fillRect(x, top, 2, bandH);
        ctx.fillStyle = edges[1];
        ctx.fillRect(x + w - 2, top, 2, bandH);
      } else {
        ctx.fillStyle = 'rgba(190,190,190,0.85)';
        ctx.fillRect(x, top, 1, bandH);
        ctx.fillRect(x + w - 1, top, 1, bandH);
      }
    };
    view.inactive.forEach((r) => paint(r, FILL_INACTIVE));
    view.selected.forEach((r) => paint(r, FILL_SELECTED));
    if (view.active) paint(view.active, FILL_ACTIVE, [MARKER_START, MARKER_END]);
  }

  // ---------- 指针交互（左键=开始、右键=结束、双键拖扫；波形/降级视图通用） ----------
  // 双键拖扫说明：按 W3C Pointer Events 规范，和弦按键（已有按键按下时再按/松
  // 另一键）不会触发 pointerdown/pointerup，只体现在 pointermove 的 buttons 位上。
  // 因此「左键拖拽中按下右键」不需要额外处理（现有 session 继续拖动即为双键扫选），
  // 而「双键状态下松开任一键」在 onPointerMove 中检测 buttons 变化并提交扫选。
  function sensRange() {
    return SENSITIVITY_PX / pxPerSecNow();
  }

  function onPointerDown(event) {
    // 标尺事件只做视图平移（ruler 自己的 pointerdown 处理），不进入定时逻辑
    if (event.target.closest('#wave-ruler')) return;
    if (!store.state.duration) return;
    displayEl.focus();
    const t = timeAtX(event.clientX);

    if (event.button === 1) {
      event.preventDefault(); // 阻止浏览器中键自动滚动
      player.seek(t);
      return;
    }
    const snapRange = snapRangeFor(event);
    if (event.button === 2) {
      // 右键：设定结束时间并抓起（按住移动可继续调，抬起定结束）。
      // （左键拖拽进行中按下右键不会触发本分支：和弦按键不产生 pointerdown，
      // 双键扫选由 onPointerMove 检测 event.buttons 变化实现，见文件头部说明。）
      const session = store.selectedCue() ? timing.onRightClick(t, snapRange) : null;
      drag = session ? { kind: 'timing', session } : null;
    } else if (event.button === 0) {
      if (!store.selectedCue()) {
        player.seek(t); // 没有可定时的行：退化为定位播放头
        return;
      }
      const session = timing.onLeftClick(t, { ctrl: event.ctrlKey, alt: event.altKey }, sensRange(), snapRange);
      drag = session ? { kind: 'timing', session } : null;
    } else {
      return;
    }
    if (drag?.kind === 'timing' && drag.session) {
      dragButtons = event.buttons; // 记录起始按键位，供 pointermove 检测和弦按键变化
      try {
        displayEl.setPointerCapture(event.pointerId);
      } catch {
        // 合成事件或指针已失效：不捕获也能收到后续事件
      }
      hoverX = null;
      syncView();
      startAutoScroll(); // 边缘自动滚屏只在拖拽期间运转
    } else {
      drag = null;
    }
  }

  function onPointerMove(event) {
    if (drag?.kind === 'ruler') {
      scrollPx(drag.startScroll - (event.clientX - drag.startX));
      return;
    }
    if (drag?.kind === 'timing') {
      if (!drag.session) return;
      // 和弦按键收尾：双键拖扫（dragButtons=3）状态下松开任一键 → 提交扫选
      if (dragButtons === 3 && (event.buttons === 1 || event.buttons === 2)) {
        finishTimingDrag(event.clientX, true);
        return;
      }
      // 预期外情况：拖拽中按键位归零却未收到 pointerup，安全收尾避免拖拽卡死
      if (event.buttons === 0) {
        finishTimingDrag(event.clientX, false);
        return;
      }
      dragButtons = event.buttons;
      // Aegisub 每次事件重算吸附范围（Shift 实时切换吸附开关）
      timing.onMarkerDrag(drag.session, timeAtX(event.clientX), snapRangeFor(event));
      return;
    }
    // 悬停：标记边缘附近显示 ↔ 光标 + 时间标签
    hoverX = event.clientX;
    const nearMarker = timing.isNearbyMarker(timeAtX(hoverX), sensRange(), event.altKey);
    containerEl.style.cursor = nearMarker ? 'ew-resize' : 'crosshair';
    fallbackEl.style.cursor = nearMarker ? 'ew-resize' : 'crosshair';
    if (!player.isPlaying()) requestSyncView(); // 高频悬停重绘按帧合并
  }

  function onPointerUp(event) {
    if (drag?.kind === 'ruler') {
      drag = null;
      return;
    }
    if (drag?.kind === 'timing') finishTimingDrag(event.clientX, true);
  }

  // timing 拖拽收尾：提交扫选、清理状态并同步视图
  // releaseAtEdge=true 时按释放位置做边缘滚屏（Aegisub Audio/Auto/Scroll）
  function finishTimingDrag(clientX, releaseAtEdge) {
    if (drag?.kind !== 'timing') return;
    timing.endDrag();
    drag = null;
    dragButtons = 0;
    stopAutoScroll();
    if (releaseAtEdge && opts().autoScroll && !fallbackMode) {
      const rect = displayEl.getBoundingClientRect();
      const width = rect.width;
      const x = clientX - rect.left;
      if (x < width / EDGE_FRACTION) scrollByViewport(-1);
      else if (width - x < width / EDGE_FRACTION) scrollByViewport(1);
    }
    syncView();
  }

  function onPointerCancel() {
    if (drag?.kind === 'timing') timing.endDrag();
    drag = null;
    dragButtons = 0;
    stopAutoScroll();
  }

  // 拖拽期间鼠标贴近左右边缘时持续滚动（Aegisub scroll_timer）。
  // 仅拖拽期间运转：pointerdown/pointermove 启动，up/cancel/会话作废时停止，
  // 不留常转 rAF 空转循环。
  let autoScrollRaf = 0;
  function autoScrollTick() {
    if (drag?.kind === 'timing' && hoverX != null) {
      const rect = displayEl.getBoundingClientRect();
      const x = hoverX - rect.left;
      const edge = 30;
      autoScrollDir = x < edge ? -1 : rect.width - x < edge ? 1 : 0;
      if (autoScrollDir && !fallbackMode && ws) scrollPx((ws.getScroll() ?? 0) + autoScrollDir * 12);
      autoScrollRaf = requestAnimationFrame(autoScrollTick);
    } else {
      autoScrollDir = 0;
      autoScrollRaf = 0; // 本帧无滚动需求：循环退出，需要时再启动
    }
  }

  function startAutoScroll() {
    if (!autoScrollRaf) autoScrollRaf = requestAnimationFrame(autoScrollTick);
  }

  function stopAutoScroll() {
    if (autoScrollRaf) cancelAnimationFrame(autoScrollRaf);
    autoScrollRaf = 0;
    autoScrollDir = 0;
  }

  displayEl.addEventListener('pointerdown', onPointerDown);
  displayEl.addEventListener('pointermove', (event) => {
    if (drag?.kind === 'timing') {
      hoverX = event.clientX;
      startAutoScroll(); // 按下后未移动过时循环可能尚未启动
    }
    onPointerMove(event);
  });
  displayEl.addEventListener('pointerup', onPointerUp);
  displayEl.addEventListener('pointercancel', onPointerCancel);
  displayEl.addEventListener('pointerleave', () => {
    if (!drag) {
      hoverX = null;
      rebuildOverlay();
    }
  });
  const suppressMenu = (event) => event.preventDefault();
  displayEl.addEventListener('contextmenu', suppressMenu);

  // 标尺拖动 = 平移视图（Aegisub timeline 拖动）；标尺上不做悬停时间标签
  ruler.addEventListener('pointerdown', (event) => {
    // 不冒泡到 displayEl：否则 drag 会话被 timing 覆盖，且 setPointerCapture
    // 转移后 ruler 收不到后续事件（标尺平移失效 + 在标尺上误触发打轴）
    event.stopPropagation();
    if (event.button !== 0 || fallbackMode || !ws) return;
    displayEl.focus();
    drag = { kind: 'ruler', startX: event.clientX, startScroll: ws.getScroll() };
    try {
      ruler.setPointerCapture(event.pointerId);
    } catch {
      // 忽略
    }
  });
  ruler.addEventListener('pointermove', (event) => {
    if (drag?.kind === 'ruler') {
      scrollPx(drag.startScroll - (event.clientX - drag.startX));
    }
  });
  ruler.addEventListener('pointerup', () => {
    if (drag?.kind === 'ruler') drag = null;
  });
  ruler.addEventListener('pointercancel', () => {
    if (drag?.kind === 'ruler') drag = null;
  });
  // 中键辅助点击（Linux 中键粘贴等默认行为）
  displayEl.addEventListener('auxclick', (event) => {
    if (event.button === 1) event.preventDefault();
  });

  // 滚轮：滚动视图；Ctrl(+Meta)+滚轮 = 以光标为锚缩放（Aegisub Wheel Default to Zoom=false）
  displayEl.addEventListener(
    'wheel',
    (event) => {
      event.preventDefault(); // 降级/未加载媒体时也阻止页面滚动
      if (fallbackMode || !ws) return;
      if (event.ctrlKey || event.metaKey) {
        const factor = event.deltaY < 0 ? 1.25 : 0.8;
        const anchorT = timeAtX(event.clientX);
        const rect = containerEl.getBoundingClientRect();
        const anchorX = event.clientX - rect.left;
        setZoom(pxPerSec * factor);
        if (ws) scrollPx(anchorT * pxPerSecNow() - anchorX);
      } else {
        const raw = Math.abs(event.deltaX) >= Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
        const delta = event.deltaMode === 1 ? raw * WHEEL_LINE_PX : raw; // 行模式归一化为像素
        scrollPx((ws.getScroll() ?? 0) + delta);
      }
    },
    { passive: false },
  );

  // ---------- 事件接线 ----------
  store.on('cues', syncView);
  let lastActiveId = store.state.selectedId;
  store.on('selection', () => {
    // Aegisub：仅当活动行变化时丢弃未提交改动（OnFileChanged → Revert 只看换行）；
    // 文本编辑等其余提交不影响 pending
    if (store.state.selectedId !== lastActiveId) {
      lastActiveId = store.state.selectedId;
      timing.revert();
      // 旧拖拽会话指向旧行 id：留着会继续写坏 pending，一并作废
      if (drag?.kind === 'timing') timing.endDrag();
      drag = null;
      dragButtons = 0;
      stopAutoScroll();
    }
    syncView();
    // 拖拽进行中不回拉视图：autoScroll 要求整选区可见会与边缘自动滚动打架（拖拽值抖动）
    if (opts().autoScroll && drag?.kind !== 'timing') {
      const range = timing.getPrimaryRange();
      if (range && !selectionVisible(range)) scrollToSelection();
    }
  });
  // 撤销/重做会改写 cues：残留 pending 指向旧数据，不回收会污染下一次 Ctrl+Z/提交。
  // 时序说明：自动提交拖拽期间自身 commit 也 emit('history')，但彼时 pending 刚被
  // commitPending 清空——据此与外部撤销区分（drag 活跃且仍有 pending 才作废会话；
  // 自动提交下撤销且 pending 恰为空时无法区分，保守不动）。
  store.on('history', () => {
    if (drag?.kind === 'timing') {
      if (timing.hasPending()) {
        timing.endDrag();
        timing.revert();
        drag = null;
        dragButtons = 0;
        stopAutoScroll();
        syncView();
      }
      return;
    }
    if (timing.hasPending()) timing.revert();
  });
  player.onTime((t, playing) => {
    if (playing && hoverX != null) {
      hoverX = null;
      if (!fallbackMode) rebuildOverlay();
    }
  });

  const ro = new ResizeObserver(() => {
    drawRuler();
    // 降级视图的 canvas 尺寸随容器重设，否则缩放窗口后画面拉伸模糊
    if (fallbackMode) drawFallback();
    else rebuildOverlay();
  });
  ro.observe(displayEl);

  // 音频盒是否持有焦点（Aegisub「Audio」键位上下文）
  function isAudioContextActive() {
    const active = document.activeElement;
    return active === displayEl || displayEl.contains(active) || ruler === active;
  }

  // 快进/快退步长随缩放自适应：约 1/4 可见窗口，限制在 5~30s（全局 ←/→ 用）
  function seekStep() {
    if (fallbackMode || !ws) return 5;
    const dur = ws.getDuration() || 0;
    if (!dur) return 5;
    const visible = Math.min(dur, visibleWidth() / pxPerSecNow());
    return Math.max(5, Math.min(30, visible / 4));
  }

  return {
    loadMedia,
    syncView,
    seekStep,
    getZoom,
    getPxPerSec,
    setZoom,
    setAmplitudeScale,
    scrollToSelection,
    scrollByViewport,
    scrollByPixels,
    isAudioContextActive,
  };
}
