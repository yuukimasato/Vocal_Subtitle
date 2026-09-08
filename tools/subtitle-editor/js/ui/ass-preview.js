// ASS 样式预览：JASSUB（WASM libass）。
// 内容刷新优先走 renderer.setTrack 原地换轨（上游 2.5.14 未在主类暴露，但 renderer 代理上有），
// 失败再整体重建实例；初始化失败或卡死回退文本预览。
import JASSUB from '../../vendor/jassub.esm.js';

const VENDOR_URL = new URL('../../vendor/', import.meta.url).href;
// 单文件版（build-standalone.mjs）会把 wasm/worker/字体内嵌为 blob URL 注入到这里
const ASSETS = typeof window !== 'undefined' && window.VstEditorVendorAssets
  ? window.VstEditorVendorAssets
  : {
      jassubWorker: VENDOR_URL + 'jassub-worker.js',
      jassubWasm: VENDOR_URL + 'jassub-worker.wasm',
      jassubFont: VENDOR_URL + 'jassub-default.woff2',
    };
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

function paintOrTimeout(current, limit, isStale) {
  return new Promise((resolve) => {
    let elapsed = 0;
    const timer = setInterval(() => {
      if (isStale()) {
        clearInterval(timer);
        resolve('stale');
        return;
      }
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

  // cue 时间与 video.currentTime 同为秒，直接比较
  function cueVisibleAt(seconds) {
    return store.state.cues.some((cue) => cue.start <= seconds && seconds < cue.end);
  }

  // 渲染由 requestVideoFrameCallback 驱动，只在播放/seek 时触发；
  // 视频暂停时主动补帧，让「改完即见」不依赖播放。首次调用走 resize 分支，需调两次。
  // 可用字体是懒加载（首次绘制触发异步 fetch + reloadFonts），字体就绪前画的是空帧，
  // 暂停时没有后续帧纠偏，因此「应有字幕却仍空白」时限量重试。
  // 注意：依赖闭包里的 store/cueVisibleAt，放模块层会引用不到而静默失败。
  function paintPausedFrame(target, video) {
    const frame = () => ({
      mediaTime: video.currentTime,
      width: video.videoWidth,
      height: video.videoHeight,
      expectedDisplayTime: performance.now(),
    });
    const draw = () => target.manualRender(frame()).catch(() => {});
    return (async () => {
      await draw();
      await draw();
      for (let i = 0; i < 3 && video.paused && cueVisibleAt(video.currentTime) && isCanvasBlank(target._canvas); i++) {
        await new Promise((resolve) => setTimeout(resolve, 250));
        await draw();
      }
    })().catch(() => {
      // 补帧失败无碍：播放或 seek 后 RVFC 会接管渲染
    });
  }

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
    const availableFonts = { 'Liberation Sans': ASSETS.jassubFont };
    try {
      const created = new JASSUB({
        video,
        subContent: currentContent(),
        workerUrl: ASSETS.jassubWorker,
        wasmUrl: ASSETS.jassubWasm,
        // SIMD 分支的 wasm 路径；vendor 只保留一份 wasm，两个分支都指向它
        modernWasmUrl: ASSETS.jassubWasm,
        availableFonts,
        defaultFont: 'Liberation Sans',
      });
      if (gen !== generation) {
        destroyWithLimit(created);
        return false;
      }
      instance = created;
      const verdict = await Promise.race([
        created.ready.then(() => 'ready').catch(() => 'error'),
        paintOrTimeout(created, INIT_TIMEOUT, () => gen !== generation),
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
      // 可用字体默认懒加载，等首次绘制才异步拉取；先显式写入再补帧，暂停状态才能立即出字
      await created.renderer.addFonts(Object.values(availableFonts)).catch(() => {});
      await paintPausedFrame(created, video);
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

  // 原地换轨刷新：'ok' 成功；'stale' 期间实例已被替换（无需动作）；'failed' 失败（降级重建）
  async function refreshTrack() {
    const current = instance;
    const gen = generation;
    const video = player.video();
    if (!current || !video) return 'failed';
    try {
      await current.ready;
      if (gen !== generation || instance !== current) return 'stale';
      await current.renderer.setTrack(currentContent());
      await paintPausedFrame(current, video);
      return 'ok';
    } catch {
      return gen === generation ? 'failed' : 'stale';
    }
  }

  function requestRefresh() {
    if (!active) return;
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const verdict = await refreshTrack();
      if (verdict === 'failed') await start();
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
    // 诊断/测试用：当前 JASSUB 实例（可能为 null）
    get instance() {
      return instance;
    },
  };
}
