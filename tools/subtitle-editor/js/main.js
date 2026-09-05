// 装配入口：状态 → 动作 → 播放器/波形/预览/列表/工具栏 → 快捷键。
import { createStore } from './state.js';
import { createActions } from './actions.js';
import { initShortcuts } from './shortcuts.js';
import { createPlayer } from './ui/player.js';
import { createWaveform } from './ui/waveform.js';
import { createAssPreview } from './ui/ass-preview.js';
import { createCueList } from './ui/cue-list.js';
import { createToolbar } from './ui/toolbar.js';
import { createDraftStore } from './draft.js';
import { serializeSubtitle } from './format/index.js';
import { showToast } from './ui/toast.js';

const $ = (selector) => document.querySelector(selector);

const store = createStore();
const actions = createActions(store);

const player = createPlayer({
  store,
  mountEl: $('#video-mount'),
  transportEl: $('#transport'),
  onError: () => showToast('媒体播放出错：浏览器不支持该格式或文件已损坏', 'error'),
});

const waveform = createWaveform({
  store,
  actions,
  player,
  containerEl: $('#waveform'),
  fallbackEl: $('#wave-fallback'),
  messageEl: $('#wave-msg'),
});

const assPreview = createAssPreview({
  store,
  player,
  serialize: serializeSubtitle,
  onNotice: (message) => showToast(message),
});

createCueList({
  store,
  actions,
  player,
  tbodyEl: $('#cue-tbody'),
  countEl: $('#cue-count'),
  toolsEl: $('#cue-tools'),
});

const toolbar = createToolbar({
  store,
  actions,
  player,
  waveform,
  assPreview,
  draft: createDraftStore(store),
});

initShortcuts({
  store,
  actions,
  player,
  exportCurrent: () => toolbar.exportCurrent(),
});

// ASS 样式预览开关联动
function syncAssPreview() {
  assPreview.setEnabled(store.state.assPreview && store.state.subtitleFormat === 'ass');
}
store.on('assPreview', syncAssPreview);
store.on('subtitleFormat', syncAssPreview);
store.on('cues', () => assPreview.requestRefresh());

// 空状态与离开提醒
const stageEmpty = $('#stage-empty');
store.on('mediaLoaded', (s) => {
  stageEmpty.hidden = s.mediaLoaded;
});
window.addEventListener('beforeunload', (e) => {
  if (store.state.dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// 调试句柄（控制台可用：__editor.store.state 等）
window.__editor = { store, actions, player, waveform, assPreview };
