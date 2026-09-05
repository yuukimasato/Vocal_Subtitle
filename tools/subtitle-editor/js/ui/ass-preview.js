// ASS 样式预览：JASSUB（WASM libass）。
// JASSUB 无内容更新接口，编辑后防抖重建实例；初始化失败或卡死回退文本预览。
import JASSUB from '../../vendor/jassub.esm.js';

const VENDOR_URL = new URL('../../vendor/', import.meta.url).href;
const REBUILD_DELAY = 600;
// 初始化兜底：worker 偶发卡死（headless/软件渲染环境），超时且无绘制即回退
const INIT_TIMEOUT = 15000;
const DESTROY_LIMIT = 3000;
const PAINT_POLL = 500;

function isCanvasBlank(canvas) {
  if (!canvas || !canvas.width || !canvas.height) return true;
  const probe = document.createElement('canvas');
  probe.width = canvas.width;
  probe.height = canvas.height;
  return canvas.toDataURL() === probe.toDataURL();
}

function paintOrTimeout(current, limit) {
  return new Promise((resolve) => {
    let elapsed = 0;
    const timer = setInterval(() => {
      elapsed += PAINT_POLL;
      const canvas = current && !current._destroyed ? current._canvas : null;
      if (canvas && !isCanvasBlank(canvas)) {
        clearInterval(timer);
        resolve('painted');
      } else if (elapsed >= limit) {
        clearInterval(timer);
        resolve('timeout');
      }
    }, PAINT_POLL);
  });
}

export function createAssPreview({ store, player, serialize, onNotice }) {
  let instance = null;
  let timer = null;
  let active = false;
  let generation = 0;

  function currentContent() {
    const s = store.state;
    return serialize('ass', s.cues, s.subDoc);
  }

  function destroyWithLimit(target) {
    return Promise.race([
      target.destroy(),
      new Promise((resolve) => setTimeout(resolve, DESTROY_LIMIT)),
    ]);
  }

  async function destroyInstance() {
    const current = instance;
    instance = null;
    if (!current) return;
    try {
      await destroyWithLimit(current);
    } catch {
      // 已销毁或渲染器不可用：忽略
    }
  }

  async function start() {
    const gen = ++generation;
    clearTimeout(timer);
    await destroyInstance();
    active = false;
    player.setAssPreviewActive(false);
    const video = player.video();
    if (!video || store.state.subtitleFormat !== 'ass') return false;
    try {
      const created = new JASSUB({
        video,
        subContent: currentContent(),
        workerUrl: VENDOR_URL + 'jassub-worker.js',
        wasmUrl: VENDOR_URL + 'jassub-worker.wasm',
        availableFonts: { 'Liberation Sans': VENDOR_URL + 'jassub-default.woff2' },
        defaultFont: 'Liberation Sans',
      });
      if (gen !== generation) {
        destroyWithLimit(created);
        return false;
      }
      instance = created;
      const verdict = await Promise.race([
        created.ready.then(() => 'ready').catch(() => 'error'),
        paintOrTimeout(created, INIT_TIMEOUT),
      ]);
      if (gen !== generation) return false;
      if (verdict === 'timeout' || verdict === 'error') {
        instance = null;
        await destroyWithLimit(created);
        onNotice?.('ASS 样式预览初始化失败，已回退为纯文本预览');
        return false;
      }
      active = true;
      player.setAssPreviewActive(true);
      return true;
    } catch (err) {
      if (gen === generation) {
        instance = null;
        active = false;
        player.setAssPreviewActive(false);
        onNotice?.('ASS 样式预览初始化失败，已回退为纯文本预览');
      }
      return false;
    }
  }

  function requestRefresh() {
    if (!active) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      start();
    }, REBUILD_DELAY);
  }

  async function setEnabled(enabled) {
    const gen = ++generation;
    clearTimeout(timer);
    if (enabled) {
      await start();
      return;
    }
    const hadInstance = instance !== null;
    await destroyInstance();
    if (gen !== generation) return;
    active = false;
    if (hadInstance) player.setAssPreviewActive(false);
  }

  return {
    requestRefresh,
    setEnabled,
    get active() {
      return active;
    },
  };
}
