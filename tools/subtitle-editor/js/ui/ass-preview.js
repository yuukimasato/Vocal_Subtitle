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
      jassubCjkFont: VENDOR_URL + 'noto-sans-sc-subset.woff2',
    };
// 默认字体必须是 CJK 字体：libass 在样式字体缺少字形时只会回退到 family_default，
// 内置拉丁字体（Liberation Sans）没有汉字，中文字幕会整行渲染成豆腐块。
const CJK_FONT_FAMILY = 'Noto Sans CJK SC';
// 本地字体（Local Font Access，Chrome/Edge 103+）：授权后 libass 能按样式里写的字体名
// 从本机加载，效果与 Aegisub + fontconfig 一致；Firefox/Safari 没有该 API，继续用内置字体。
// queryLocalFonts 必须在用户手势内调用（否则抛 "User activation is required"）；无手势的
// 启动（如 ?subs= 自动加载）只是先跳过，等下一次带手势的启动再问，且只问一次。
let localFontsAsked = false;
async function ensureLocalFonts() {
  if (localFontsAsked) return;
  if (typeof window === 'undefined' || typeof window.queryLocalFonts !== 'function') return;
  if (!navigator.permissions?.query) return;
  try {
    const { state } = await navigator.permissions.query({ name: 'local-fonts' });
    if (state !== 'prompt') {
      localFontsAsked = true; // 已授权（直接可用）或已拒绝（不再打扰）
      return;
    }
    if (!navigator.userActivation?.isActive) return; // 等下一次带用户手势的启动
    localFontsAsked = true;
    await window.queryLocalFonts(); // 触发授权提示；拒绝/忽略则静默回退内置字体
  } catch {
    // 不支持的浏览器或用户拒绝：继续用内置字体
  }
}
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
    // 纯音频媒体没有视频轨（videoWidth/Height 为 0）：libass 按视频尺寸出图，
    // 无尺寸时画布恒为空白，启动预览只会把文本 overlay 挤掉、字幕整个看不见。
    if (!video.videoWidth || !video.videoHeight) return false;
    // 先问本地字体授权再建实例：worker 的字体查找有"查过一次就不再重试"的缓存，
    // 授权晚到就赶不上了
    await ensureLocalFonts();
    const availableFonts = {
      'Liberation Sans': ASSETS.jassubFont,
      [CJK_FONT_FAMILY]: ASSETS.jassubCjkFont,
    };
    try {
      const created = new JASSUB({
        video,
        subContent: currentContent(),
        workerUrl: ASSETS.jassubWorker,
        wasmUrl: ASSETS.jassubWasm,
        // SIMD 分支的 wasm 路径；vendor 只保留一份 wasm，两个分支都指向它
        modernWasmUrl: ASSETS.jassubWasm,
        availableFonts,
        defaultFont: CJK_FONT_FAMILY,
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

  // 换媒体后重新判定可用性：纯音频（无视频轨）关掉预览回退文本，换回视频再启动。
  // 视频→视频不重建：JASSUB 由 requestVideoFrameCallback 感知尺寸变化自行 resize。
  async function onMediaChanged() {
    const video = player.video();
    const canRender = Boolean(video && video.videoWidth > 0 && video.videoHeight > 0);
    const wanted = store.state.assPreview && store.state.subtitleFormat === 'ass';
    if (active && !canRender) {
      await setEnabled(false);
      return;
    }
    if (!active && canRender && wanted) await start();
  }

  return {
    requestRefresh,
    setEnabled,
    onMediaChanged,
    get active() {
      return active;
    },
    // 诊断/测试用：当前 JASSUB 实例（可能为 null）
    get instance() {
      return instance;
    },
  };
}
