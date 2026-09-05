// 字幕变更动作集合。所有可撤销的变更都经过 commit()：先推快照，再写 store。
import { History } from './history.js';
import { sortCues, findCueAt } from './format/cue.js';
import { makeNewCue } from './format/index.js';

const MIN_LEN = 0.05;

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

export function createActions(store) {
  const history = new History();

  function commit(next, extra = {}) {
    history.push(structuredClone(store.state.cues));
    const selectedStill = next.some((c) => c.id === store.state.selectedId)
      ? store.state.selectedId
      : null;
    store.patch({ cues: next, dirty: true, selectedId: selectedStill, ...extra });
    store.emit('history');
  }

  function resolveCue(id) {
    const cue = store.state.cues.find((c) => c.id === (id ?? store.state.selectedId));
    return cue ?? null;
  }

  return {
    history,

    canUndo() {
      return history.canUndo;
    },

    undo() {
      const snap = history.undo(structuredClone(store.state.cues));
      if (!snap) return false;
      store.patch({ cues: snap, dirty: true, editingId: null });
      store.emit('history');
      return true;
    },

    redo() {
      const snap = history.redo(structuredClone(store.state.cues));
      if (!snap) return false;
      store.patch({ cues: snap, dirty: true, editingId: null });
      store.emit('history');
      return true;
    },

    // 打开字幕文件：清空历史
    loadSubtitle(cues, { name, format, doc }) {
      history.clear();
      store.patch({
        cues: sortCues(cues),
        subDoc: doc,
        subtitleFormat: format,
        subtitleName: name,
        dirty: false,
        editingId: null,
        selectedId: sortCues(cues)[0]?.id ?? null,
      });
      store.emit('history');
    },

    select(id) {
      if (!store.state.cues.some((c) => c.id === id)) return;
      store.patch({ selectedId: id });
    },

    selectNeighbour(delta, fallbackTime) {
      const cues = store.state.cues;
      if (!cues.length) return null;
      const current = store.selectedCue() ?? findCueAt(cues, fallbackTime ?? -1);
      let index = current ? cues.indexOf(current) + delta : delta > 0 ? 0 : cues.length - 1;
      index = clamp(index, 0, cues.length - 1);
      const cue = cues[index];
      store.patch({ selectedId: cue.id });
      return cue;
    },

    setEditing(id) {
      store.patch({ editingId: id, selectedId: id ?? store.state.selectedId });
    },

    updateCueText(id, text) {
      const cue = resolveCue(id);
      if (!cue || cue.text === text) return false;
      commit(store.state.cues.map((c) => (c.id === cue.id ? { ...c, text } : c)));
      return true;
    },

    // 行内编辑统一入口：文本与时间合并为一次撤销快照
    updateCue(id, patch = {}) {
      const cue = resolveCue(id);
      if (!cue) return false;
      const next = { ...cue };
      if (patch.text !== undefined && patch.text !== cue.text) next.text = patch.text;
      if (patch.start !== undefined || patch.end !== undefined) {
        const start = patch.start ?? cue.start;
        const end = patch.end ?? cue.end;
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return false;
        next.start = Math.max(0, start);
        next.end = end;
      }
      if (next.start === cue.start && next.end === cue.end && next.text === cue.text) return false;
      commit(sortCues(store.state.cues.map((c) => (c.id === cue.id ? next : c))));
      return true;
    },

    updateCueTimes(id, start, end) {
      const cue = resolveCue(id);
      if (!cue) return false;
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return false;
      if (start < 0) start = 0;
      if (cue.start === start && cue.end === end) return false;
      commit(sortCues(store.state.cues.map((c) => (c.id === cue.id ? { ...c, start, end } : c))));
      return true;
    },

    // 小键盘微调：which = 'start' | 'end'
    nudge(id, which, delta) {
      const cue = resolveCue(id);
      if (!cue || !delta) return false;
      let { start, end } = cue;
      if (which === 'start') {
        start = clamp(start + delta, 0, end - MIN_LEN);
      } else {
        end = Math.max(start + MIN_LEN, end + delta);
      }
      if (start === cue.start && end === cue.end) return false;
      commit(sortCues(store.state.cues.map((c) => (c.id === cue.id ? { ...c, start, end } : c))));
      return true;
    },

    // 播放头处插入；与下一句重叠时缩短
    insertAtTime(t, { format, doc } = {}) {
      const cues = store.state.cues;
      const next = cues.find((c) => c.start > t);
      let end = t + 2;
      if (next && next.start < end) end = Math.max(t + MIN_LEN, next.start);
      const cue = makeNewCue(format ?? store.state.subtitleFormat, doc ?? store.state.subDoc, t, end, '');
      commit(sortCues([...cues, cue]));
      store.patch({ selectedId: cue.id });
      return cue;
    },

    removeCue(id) {
      const cue = resolveCue(id);
      if (!cue) return false;
      const rest = store.state.cues.filter((c) => c.id !== cue.id);
      commit(rest, { editingId: null });
      return true;
    },

    // 在 atTime（通常为播放头）处拆分；时间比例决定文本切分点
    splitCue(id, atTime) {
      const cue = resolveCue(id);
      if (!cue) return false;
      let at = atTime;
      if (!(at > cue.start + MIN_LEN && at < cue.end - MIN_LEN)) {
        at = (cue.start + cue.end) / 2;
      }
      const ratio = (at - cue.start) / (cue.end - cue.start);
      const cut = clamp(Math.round(cue.text.length * ratio), 0, cue.text.length);
      const headText = cue.text.slice(0, cut);
      const tailText = cue.text.slice(cut);
      const tail = makeNewCue(
        store.state.subtitleFormat,
        store.state.subDoc,
        at,
        cue.end,
        tailText,
      );
      const head = { ...cue, end: at, text: headText };
      if (cue.meta) {
        head.meta = { ...cue.meta, parts: [...cue.meta.parts] };
        tail.meta = { ...cue.meta, parts: [...cue.meta.parts] };
      }
      commit(sortCues([
        ...store.state.cues.filter((c) => c.id !== cue.id),
        head,
        tail,
      ]));
      store.patch({ selectedId: tail.id });
      return true;
    },

    mergeWithNext(id) {
      const cues = store.state.cues;
      const cue = resolveCue(id);
      if (!cue) return false;
      const index = cues.indexOf(cue);
      if (index === -1 || index === cues.length - 1) return false;
      const next = cues[index + 1];
      const merged = {
        ...cue,
        end: Math.max(cue.end, next.end),
        text: `${cue.text}${cue.text && next.text ? '\n' : ''}${next.text}`,
      };
      if (cue.meta) merged.meta = { ...cue.meta, parts: [...cue.meta.parts] };
      commit(sortCues(cues.filter((c) => c.id !== next.id).map((c) => (c.id === cue.id ? merged : c))));
      return true;
    },

    // 导出成功后的状态收口
    markExported(format, name) {
      store.patch({ dirty: false, subtitleFormat: format, subtitleName: name });
    },
  };
}
