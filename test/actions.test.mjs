import test from 'node:test';
import assert from 'node:assert/strict';
import { createStore } from '../js/state.js';
import { createActions } from '../js/actions.js';

function setup() {
  const store = createStore();
  const actions = createActions(store);
  actions.loadSubtitle(
    [
      { id: 'a', start: 1, end: 3, text: '第一句' },
      { id: 'b', start: 4, end: 6, text: '第二句' },
      { id: 'c', start: 8, end: 10, text: '第三句' },
    ],
    { name: 'x.srt', format: 'srt', doc: null },
  );
  return { store, actions };
}

test('loadSubtitle 排序并选中第一条', () => {
  const { store } = setup();
  assert.deepEqual(store.state.cues.map((c) => c.id), ['a', 'b', 'c']);
  assert.equal(store.state.selectedId, 'a');
  assert.equal(store.state.dirty, false);
});

test('updateCue 合并提交且可撤销', () => {
  const { store, actions } = setup();
  actions.select('a');
  assert.ok(actions.updateCue('a', { start: 0.5, end: 2.5, text: '改' }));
  const cue = store.state.cues[0];
  assert.equal(cue.text, '改');
  assert.equal(cue.start, 0.5);
  assert.ok(actions.undo());
  assert.equal(store.state.cues[0].text, '第一句');
  assert.equal(store.state.cues[0].start, 1);
  assert.ok(actions.redo());
  assert.equal(store.state.cues[0].text, '改');
});

test('updateCue 拒绝 end<=start', () => {
  const { actions } = setup();
  assert.equal(actions.updateCue('a', { start: 2, end: 2 }), false);
  assert.equal(actions.updateCue('a', { start: 3, end: 1 }), false);
});

test('nudge 约束边界', () => {
  const { store, actions } = setup();
  actions.nudge('a', 'start', -5); // 会被钳到 0
  assert.equal(store.state.cues[0].start, 0);
  actions.nudge('a', 'start', +5); // 越过 end 会被钳到 end-0.05
  assert.equal(store.state.cues[0].start, 2.95);
  actions.nudge('a', 'end', -100); // 收缩到 start+0.05
  assert.equal(store.state.cues[0].end, 3);
});

test('insertAtTime 缩短以避免与下一句重叠', () => {
  const { store, actions } = setup();
  const cue = actions.insertAtTime(2.5);
  const next = store.state.cues.find((c) => c.start > 2.5);
  assert.ok(cue.end <= next.start);
  assert.equal(store.state.selectedId, cue.id);
});

test('splitCue 按播放头拆分并按比例切文本', () => {
  const { store, actions } = setup();
  actions.splitCue('a', 2); // 1..3 的中点，文本 3 字 → 前 1.5 → 取整 2 字（四舍五入）
  const texts = store.state.cues.filter((c) => c.start >= 1 && c.start < 3).map((c) => c.text);
  assert.equal(store.state.cues.length, 4);
  assert.equal(texts.join('|'), '第一|句');
  assert.ok(actions.undo());
  assert.equal(store.state.cues.length, 3);
});

test('mergeWithNext 合并文本与时间', () => {
  const { store, actions } = setup();
  assert.ok(actions.mergeWithNext('a'));
  const merged = store.state.cues[0];
  assert.equal(merged.text, '第一句\n第二句');
  assert.equal(merged.end, 6);
  assert.equal(store.state.cues.length, 2);
  // 最后一行无法向后合并
  const last = store.state.cues[store.state.cues.length - 1];
  assert.equal(actions.mergeWithNext(last.id), false);
});

test('removeCue 后撤销恢复', () => {
  const { store, actions } = setup();
  actions.removeCue('b');
  assert.equal(store.state.cues.length, 2);
  actions.undo();
  assert.equal(store.state.cues.length, 3);
});

test('ASS cue 保留 meta 并在拆分时各自持有副本', () => {
  const store = createStore();
  const actions = createActions(store);
  const meta = { fields: ['layer', 'start', 'end', 'text'], parts: ['0', '0:00:01.00', '0:00:03.00', '文本'], startIdx: 1, endIdx: 2, textIdx: 3 };
  actions.loadSubtitle([{ id: 'a', start: 1, end: 3, text: '文本', meta }], { name: 'x.ass', format: 'ass', doc: { lines: [], fields: meta.fields, appendAt: 0 } });
  actions.splitCue('a', 2);
  const [head, tail] = store.state.cues;
  assert.notEqual(head.meta, tail.meta);
  assert.deepEqual(head.meta.parts, tail.meta.parts);
});
