// 装配入口：状态 → 动作 → 播放器/定时控制器/音频盒/预览/列表/工具栏 → 快捷键。
import { createStore } from './state.js';
import { createActions } from './actions.js';
import { createTimingController } from './audio/timing.js';
import { createAudioCommands } from './audio/commands.js';
import { initShortcuts } from './shortcuts.js';
import { createPlayer } from './ui/player.js';
import { createWaveform } from './ui/waveform.js';
import { createAudioToolbar } from './ui/audio-toolbar.js';
import { createAssPreview } from './ui/ass-preview.js';
import { createCueList } from './ui/cue-list.js';
import { createContextMenu } from './ui/context-menu.js';
import { createToolbar } from './ui/toolbar.js';
import { initSplitters } from './ui/splitters.js';
import { createDraftStore } from './draft.js';
import { serializeSubtitle } from './format/index.js';
import { showToast } from './ui/toast.js';

const $ = (selector) => document.querySelector(selector);

initSplitters();

const store = createStore();
const actions = createActions(store);

const player = createPlayer({
  store,
  mountEl: $('#video-mount'),
  transportEl: $('#transport'),
  onError: () => showToast('媒体播放出错：浏览器不支持该格式或文件已损坏', 'error'),
});

// Aegisub 对话定时控制器：改动先进 pending，提交时批量落库
let waveform = null;
const timing = createTimingController({
  cues: () => store.state.cues,
  activeId: () => store.state.selectedId,
  selectedIds: () => store.state.selectedIds ?? [],
  inactiveMode: () => store.state.audioOptions.inactiveMode,
  autoCommit: () => store.state.audioOptions.autoCommit,
  dragTiming: () => store.state.audioOptions.dragTiming,
  duration: () => store.state.duration,
  onApply: (entries, { auto }) => actions.updateCueTimesBulk(entries, {
    coalesceKey: auto ? 'audio-timing' : null,
  }),
  onChange: () => waveform?.syncView(),
});

// 行内编辑落库入口（cueList 稍后装配，这里惰性引用）
let cueList = null;
const flushEdits = () => cueList?.flush();

const audioCommands = createAudioCommands({
  store,
  actions,
  player,
  timing,
  flushEdits,
  waveform: {
    scrollToSelection: () => waveform?.scrollToSelection(),
    scrollByViewport: (dir) => waveform?.scrollByViewport(dir),
  },
});

waveform = createWaveform({
  store,
  actions,
  player,
  timing,
  displayEl: $('#wave-display'),
  containerEl: $('#waveform'),
  fallbackEl: $('#wave-fallback'),
  messageEl: $('#wave-msg'),
});

const audioToolbar = createAudioToolbar({
  store,
  actions,
  player,
  waveform,
  commands: audioCommands,
  toolbarEl: $('#audio-toolbar'),
  sideEl: $('#wave-side'),
});

const assPreview = createAssPreview({
  store,
  player,
  serialize: serializeSubtitle,
  onNotice: (message) => showToast(message),
});

const menu = createContextMenu();
cueList = createCueList({
  store,
  actions,
  player,
  tbodyEl: $('#cue-tbody'),
  wrapEl: $('.cue-table-wrap'),
  countEl: $('#cue-count'),
  toolsEl: $('#cue-tools'),
  statusEl: $('#save-status'),
  menu,
  pasteDialog: $('#paste-dialog'),
});

const draft = createDraftStore(store);

const toolbar = createToolbar({
  store,
  actions,
  player,
  waveform,
  assPreview,
  draft,
  flushEdits,
});

initShortcuts({
  store,
  actions,
  player,
  commands: audioCommands,
  waveform,
  exportCurrent: () => toolbar.exportCurrent(),
  flushEdits,
  seekStep: () => waveform.seekStep(),
});

// ASS 样式预览开关联动
function syncAssPreview() {
  assPreview.setEnabled(store.state.assPreview && store.state.subtitleFormat === 'ass');
}
store.on('assPreview', syncAssPreview);
store.on('subtitleFormat', syncAssPreview);
store.on('mediaLoaded', () => assPreview.onMediaChanged());
store.on('cues', () => assPreview.requestRefresh());

// 空状态与离开提醒（关闭前先提交未落库的编辑并立刻保存草稿）
const stageEmpty = $('#stage-empty');
store.on('mediaLoaded', (s) => {
  stageEmpty.hidden = s.mediaLoaded;
});
window.addEventListener('beforeunload', (e) => {
  flushEdits();
  draft.saveNow();
  if (store.state.dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// 自动化测试/演示辅助：?media=<url>&subs=<url> 启动时直接加载（与打开文件同一路径）
async function autoloadFromQuery() {
  const q = new URLSearchParams(location.search);
  const media = q.get('media');
  const subs = q.get('subs');
  if (!media && !subs) return;
  try {
    if (media) {
      const res = await fetch(media);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const name = decodeURIComponent(media.split('/').pop() || 'media');
      await toolbar.openMediaFile(new File([await res.blob()], name));
    }
    if (subs) {
      const res = await fetch(subs);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const name = decodeURIComponent(subs.split('/').pop() || 'subs.srt');
      await toolbar.openSubtitleFile(new File([await res.text()], name, { type: 'text/plain' }));
    }
  } catch (err) {
    console.warn('[autoload] 失败:', err);
  }
}
autoloadFromQuery();

// 调试句柄（控制台可用：__editor.store.state 等）
window.__editor = { store, actions, player, waveform, timing, audioCommands, assPreview };
