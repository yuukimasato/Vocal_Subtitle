// ArtPlayer 装配、播放控制条、试听窗口与文本字幕 overlay。
// 快捷键全部由 shortcuts.js 接管（ArtPlayer 内置热键已关闭）。
import Artplayer from '../../vendor/artplayer.mjs';
import { formatClock } from '../format/time.js';

const RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];
const FPSES = [23.976, 24, 25, 29.97, 30];

export function createPlayer({ store, mountEl, transportEl, onError }) {
  let art = null;
  let mediaUrl = null;
  let auditionEnd = null;
  let lastPlayStart = 0;
  const timeListeners = new Set();
  let lastPreviewText = null;

  const previewEl = document.createElement('div');
  previewEl.className = 'subtitle-preview';
  previewEl.hidden = true;

  // ---------- 控制条 ----------
  function buildTransport() {
    const mk = (tag, className, parent) => {
      const node = document.createElement(tag);
      node.className = className;
      parent.appendChild(node);
      return node;
    };
    const btn = (label, title, onClick) => {
      const b = mk('button', 'tbtn', transportEl);
      b.type = 'button';
      b.textContent = label;
      b.title = title;
      b.setAttribute('aria-label', title);
      b.addEventListener('click', onClick);
      return b;
    };
    const playBtn = btn('▶', '播放 / 暂停（Space）', () => playPause());
    const btnStop = btn('⏹', '停止并回到播放起点（小键盘 8）', () => stop());
    btn('«', '上一帧（,）', () => frameStep(-1));
    btn('»', '下一帧（.）', () => frameStep(+1));
    const timeLabel = mk('span', 'time-label mono', transportEl);
    timeLabel.textContent = '0:00.000 / 0:00.000';

    const rateSel = mk('select', 'tsel', transportEl);
    rateSel.title = '播放倍速';
    RATES.forEach((r) => {
      const opt = document.createElement('option');
      opt.value = String(r);
      opt.textContent = `${r}×`;
      rateSel.appendChild(opt);
    });
    rateSel.value = '1';
    rateSel.addEventListener('change', () => setRate(Number(rateSel.value)));

    const fpsSel = mk('select', 'tsel', transportEl);
    fpsSel.title = '逐帧步长（帧率）';
    FPSES.forEach((f) => {
      const opt = document.createElement('option');
      opt.value = String(f);
      opt.textContent = `${f} fps`;
      fpsSel.appendChild(opt);
    });
    fpsSel.value = '25';
    fpsSel.addEventListener('change', () => store.patch({ fps: Number(fpsSel.value) }));

    const volume = mk('input', 'tvol', transportEl);
    volume.type = 'range';
    volume.min = '0';
    volume.max = '1';
    volume.step = '0.05';
    volume.value = '1';
    volume.title = '音量';
    volume.setAttribute('aria-label', '音量');
    volume.addEventListener('input', () => {
      if (art) art.video.volume = Number(volume.value);
    });

    const loopBtn = btn('⟳ 循环当前句', '循环当前句（L）', () => store.patch({ loopCue: !store.state.loopCue }));
    const previewBtn = btn('预览', '在画面上显示当前字幕文本（P 除外，本开关无快捷键）', () => store.patch({ previewOn: !store.state.previewOn }));
    const assBtn = btn('ASS 样式预览', '使用 libass 按样式渲染 ASS 字幕', () => store.patch({ assPreview: !store.state.assPreview }));

    const sync = () => {
      playBtn.textContent = art && !art.video.paused ? '⏸' : '▶';
      loopBtn.classList.toggle('active', store.state.loopCue);
      previewBtn.classList.toggle('active', store.state.previewOn);
      assBtn.classList.toggle('active', store.state.assPreview);
      assBtn.disabled = store.state.subtitleFormat !== 'ass' || !art;
    };
    store.on('loopCue', sync);
    store.on('previewOn', sync);
    store.on('assPreview', sync);
    store.on('subtitleFormat', sync);
    store.on('mediaLoaded', sync);
    if (art) {
      art.video.addEventListener('play', sync);
      art.video.addEventListener('pause', sync);
    }
    sync();
  }

  // ---------- 播放器 ----------
  function buildPlayer(url) {
    const instance = new Artplayer({
      container: mountEl,
      url,
      autoplay: false,
      setting: false,
      hotkey: false,
      loop: false,
      flip: false,
      playbackRate: false,
      aspectRatio: false,
      screenshot: false,
      pip: false,
      mutex: false,
      autoSize: false,
      autoMini: false,
      fullscreen: true,
      contextmenu: [],
      controls: [],
      icons: {},
      layers: [
        {
          name: 'cuePreview',
          html: previewEl,
          style: {},
        },
      ],
      moreVideoAttr: { playsInline: true, preload: 'auto' },
    });
    instance.on('error', (event) => onError?.(event));
    return instance;
  }

  async function loadFile(file) {
    if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    mediaUrl = URL.createObjectURL(file);
    if (!art) {
      art = buildPlayer(mediaUrl);
      art.video.addEventListener('play', () => {
        lastPlayStart = art.video.currentTime;
      });
    } else {
      await art.switchUrl(mediaUrl);
    }
    const v = art.video;
    if (v.readyState < 1) {
      await new Promise((resolve, reject) => {
        const ok = () => {
          cleanup();
          resolve();
        };
        const bad = () => {
          cleanup();
          reject(new Error('浏览器无法解码该媒体文件'));
        };
        const cleanup = () => {
          v.removeEventListener('loadedmetadata', ok);
          v.removeEventListener('error', bad);
        };
        v.addEventListener('loadedmetadata', ok);
        v.addEventListener('error', bad);
      });
    }
    store.patch({
      mediaName: file.name,
      mediaLoaded: true,
      duration: Number.isFinite(v.duration) ? v.duration : 0,
    });
    return mediaUrl;
  }

  function video() {
    return art?.video ?? null;
  }

  function currentTime() {
    return art?.video?.currentTime ?? 0;
  }

  // ---------- 播放控制 ----------
  function playPause() {
    const v = video();
    if (!v) return;
    if (v.paused) v.play()?.catch(() => {});
    else v.pause();
  }

  function stop() {
    const v = video();
    if (!v) return;
    v.pause();
    auditionEnd = null;
    v.currentTime = lastPlayStart;
  }

  function stopAudition() {
    const v = video();
    auditionEnd = null;
    v?.pause();
  }

  function seek(t) {
    const v = video();
    if (!v) return;
    const max = store.state.duration || Number.POSITIVE_INFINITY;
    v.currentTime = Math.min(Math.max(0, t), max);
  }

  function frameStep(direction) {
    const v = video();
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, v.currentTime + direction / (store.state.fps || 25));
  }

  function playRange(a, b) {
    const v = video();
    if (!v) return;
    lastPlayStart = Math.max(0, a);
    v.currentTime = Math.max(0, a);
    auditionEnd = b;
    v.play()?.catch(() => {});
  }

  function currentTimingCue() {
    const t = video()?.currentTime ?? 0;
    return store.selectedCue() ?? store.cueAt(t);
  }

  function playCurrentCue() {
    const cue = currentTimingCue();
    if (!cue) return;
    playRange(cue.start, cue.end);
  }

  function auditionBefore() {
    const cue = currentTimingCue();
    if (!cue) return;
    playRange(Math.max(0, cue.start - 0.5), cue.start);
  }

  function auditionAfter() {
    const cue = currentTimingCue();
    if (!cue) return;
    playRange(cue.end, Math.min(store.state.duration || cue.end + 0.5, cue.end + 0.5));
  }

  function setRate(rate) {
    const v = video();
    if (v) v.playbackRate = rate;
    store.patch({ rate });
  }

  function onTime(fn) {
    timeListeners.add(fn);
    return () => timeListeners.delete(fn);
  }

  // ---------- rAF 循环：时间分发 / 试听窗口 / 循环当前句 / 文本预览 ----------
  function tick() {
    const v = video();
    if (v) {
      const t = v.currentTime;
      timeListeners.forEach((fn) => fn(t, !v.paused));
      if (!v.paused) {
        if (auditionEnd !== null && t >= auditionEnd - 1e-3) {
          v.pause();
          auditionEnd = null;
        } else if (store.state.loopCue) {
          const cue = store.cueAt(t);
          if (cue && t >= cue.end - 1e-3) v.currentTime = cue.start;
        }
      }
      const cue = store.cueAt(t);
      const text = store.state.previewOn && !assPreviewActive && cue ? cue.text : '';
      if (text !== lastPreviewText) {
        previewEl.textContent = text;
        previewEl.hidden = !text;
        lastPreviewText = text;
      }
    }
    requestAnimationFrame(tick);
  }

  // ass-preview 激活时隐藏文本 overlay，由 libass 接管渲染
  let assPreviewActive = false;
  function setAssPreviewActive(active) {
    assPreviewActive = active;
    if (active) {
      previewEl.hidden = true;
      previewEl.textContent = '';
      lastPreviewText = '';
    }
  }

  buildTransport();
  requestAnimationFrame(tick);

  return {
    loadFile,
    video,
    currentTime,
    playPause,
    stop,
    stopAudition,
    seek,
    frameStep,
    playRange,
    playCurrentCue,
    auditionBefore,
    auditionAfter,
    setRate,
    onTime,
    setAssPreviewActive,
  };
}
