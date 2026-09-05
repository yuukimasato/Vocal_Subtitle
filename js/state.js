// 中央状态 + 发布订阅。高频播放进度不走这里（由 player 的 rAF 直接分发）。
import { findCueAt } from './format/cue.js';

export function createStore() {
  const state = {
    cues: [],
    selectedId: null,
    editingId: null,
    duration: 0,
    fps: 25,
    rate: 1,
    follow: true,
    loopCue: false,
    previewOn: true,
    assPreview: true, // 加载 ASS 时是否启用样式预览（可开关）
    mediaName: '',
    subtitleName: '',
    subtitleFormat: '',
    subDoc: null,
    dirty: false,
    mediaLoaded: false,
  };
  const listeners = new Map();

  const store = {
    state,
    on(key, fn) {
      if (!listeners.has(key)) listeners.set(key, new Set());
      listeners.get(key).add(fn);
      return () => listeners.get(key).delete(fn);
    },
    emit(key) {
      (listeners.get(key) ?? new Set()).forEach((fn) => fn(state));
    },
    patch(props) {
      Object.assign(state, props);
      Object.keys(props).forEach((k) => store.emit(k));
    },
    selectedCue() {
      return state.cues.find((c) => c.id === state.selectedId) ?? null;
    },
    cueAt(t) {
      return findCueAt(state.cues, t);
    },
  };
  return store;
}
