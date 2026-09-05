// 波形与字幕区块：wavesurfer v7 + regions 插件。
// 音频解码失败时降级为静态刻度视图（点击定位可用，编辑经列表完成）。
import WaveSurfer from '../../vendor/wavesurfer.esm.js';
import RegionsPlugin from '../../vendor/regions.esm.js';

const CUE_COLOR = 'rgba(124, 156, 255, 0.16)';
const CUE_COLOR_SELECTED = 'rgba(124, 156, 255, 0.42)';

export function createWaveform({
  store,
  actions,
  player,
  containerEl,
  fallbackEl,
  messageEl,
}) {
  let ws = null;
  let regions = null;
  let fallbackMode = false;
  let syncing = false;
  let pxPerSec = 40;
  const regionById = new Map();
  const canvas = fallbackEl.querySelector('canvas');

  function ensureWs(url) {
    if (ws) return ws;
    ws = WaveSurfer.create({
      container: containerEl,
      media: player.video(),
      url, // 交给构造器自动加载；外部再 load() 会与它竞态触发 AbortError
      waveColor: '#3d4a63',
      progressColor: '#7c9cff',
      cursorColor: '#e6edf3',
      cursorWidth: 2,
      height: 150,
      dragToSeek: true,
      hideScrollbar: true,
      normalize: true,
    });
    regions = ws.registerPlugin(RegionsPlugin.create());

    regions.on('region-updated', (region) => {
      if (syncing) return;
      const cue = store.state.cues.find((c) => c.id === region.id);
      if (!cue) return;
      const start = Math.round(region.start * 1000) / 1000;
      const end = Math.round(region.end * 1000) / 1000;
      if (start === cue.start && end === cue.end) return;
      actions.updateCueTimes(cue.id, start, end);
    });
    regions.on('region-clicked', (region, event) => {
      event.stopPropagation();
      actions.select(region.id);
    });
    regions.on('region-double-clicked', (region, event) => {
      event.stopPropagation();
      actions.select(region.id);
      actions.setEditing(region.id);
    });
    ws.on('error', (err) => enterFallback(err));
    return ws;
  }

  async function loadMedia(url) {
    exitFallback();
    messageEl.textContent = '正在解码音频…';
    try {
      const isNew = !ws;
      if (isNew) ensureWs(url);
      else await ws.load(url);
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
      if (!fallbackMode) messageEl.textContent = '';
      syncCues();
    } catch (err) {
      enterFallback(err);
    }
  }

  function syncCues() {
    if (fallbackMode) {
      drawFallback();
      return;
    }
    if (!ws) return;
    syncing = true;
    const state = store.state;
    const seen = new Set();
    state.cues.forEach((cue, index) => {
      seen.add(cue.id);
      const selected = cue.id === state.selectedId;
      const end = Math.max(cue.end, cue.start + 0.01);
      const existing = regionById.get(cue.id);
      if (!existing) {
        const region = regions.addRegion({
          id: cue.id,
          start: cue.start,
          end,
          drag: true,
          resize: true,
          color: selected ? CUE_COLOR_SELECTED : CUE_COLOR,
          content: String(index + 1),
        });
        regionById.set(cue.id, region);
      } else {
        existing.setOptions({
          start: cue.start,
          end,
          color: selected ? CUE_COLOR_SELECTED : CUE_COLOR,
          content: String(index + 1),
        });
      }
    });
    regionById.forEach((region, id) => {
      if (!seen.has(id)) {
        region.remove();
        regionById.delete(id);
      }
    });
    syncing = false;
  }

  // ---------- 降级视图 ----------
  function enterFallback(err) {
    if (fallbackMode) return;
    fallbackMode = true;
    containerEl.hidden = true;
    fallbackEl.hidden = false;
    messageEl.textContent =
      '无法解码音频波形（编码不受支持或文件过大），已切换为时间刻度视图：点击可定位，请在右侧列表编辑时间。';
    console.warn('[waveform] decode failed:', err);
    drawFallback();
  }

  function exitFallback() {
    if (!fallbackMode) return;
    fallbackMode = false;
    containerEl.hidden = false;
    fallbackEl.hidden = true;
    messageEl.textContent = '';
  }

  function drawFallback() {
    const duration = store.state.duration || 1;
    const width = fallbackEl.clientWidth || 600;
    const height = fallbackEl.clientHeight || 150;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // 时间刻度
    const targetTicks = Math.max(4, Math.floor(width / 90));
    const rawStep = duration / targetTicks;
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const step = steps.find((s) => s >= rawStep) ?? 900;
    ctx.fillStyle = '#6e7681';
    ctx.font = '10px monospace';
    for (let t = 0; t <= duration; t += step) {
      const x = (t / duration) * width;
      ctx.fillRect(x, 0, 1, height);
      ctx.fillText(`${Math.round(t)}s`, x + 3, 11);
    }

    // cue 区块
    store.state.cues.forEach((cue, index) => {
      const x = (cue.start / duration) * width;
      const w = Math.max(2, ((cue.end - cue.start) / duration) * width);
      ctx.fillStyle = cue.id === store.state.selectedId ? CUE_COLOR_SELECTED : CUE_COLOR;
      ctx.fillRect(x, 18, w, height - 24);
      ctx.strokeStyle = '#7c9cff';
      ctx.globalAlpha = cue.id === store.state.selectedId ? 0.9 : 0.4;
      ctx.strokeRect(x, 18, w, height - 24);
      ctx.globalAlpha = 1;
      if (w > 18) {
        ctx.fillStyle = '#8b949e';
        ctx.fillText(String(index + 1), x + 3, 30);
      }
    });
  }

  fallbackEl.addEventListener('click', (event) => {
    if (!fallbackMode || !store.state.duration) return;
    const rect = fallbackEl.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    player.seek(Math.min(Math.max(0, ratio), 1) * store.state.duration);
  });

  // Ctrl+滚轮 缩放波形
  containerEl.addEventListener(
    'wheel',
    (event) => {
      if (!event.ctrlKey || !ws) return;
      event.preventDefault();
      pxPerSec = Math.min(1000, Math.max(10, pxPerSec * (event.deltaY < 0 ? 1.25 : 0.8)));
      try {
        ws.zoom(pxPerSec);
      } catch {
        // 未就绪时忽略
      }
    },
    { passive: false },
  );

  store.on('cues', syncCues);
  store.on('selection', syncCues);

  return { loadMedia, syncCues };
}
