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

test('updateCueTimesBulk 批量写时间为一次撤销快照', () => {
  const { store, actions } = setup();
  const ok = actions.updateCueTimesBulk([
    { id: 'a', start: 0.5, end: 2.5 },
    { id: 'b', start: 4.5, end: 6.5 },
  ]);
  assert.equal(ok, true);
  assert.deepEqual(
    store.state.cues.map((c) => [c.start, c.end]),
    [[0.5, 2.5], [4.5, 6.5], [8, 10]],
  );
  // 一步撤销即全部恢复，说明两行合并为同一快照
  assert.ok(actions.undo());
  assert.deepEqual(
    store.state.cues.map((c) => [c.start, c.end]),
    [[1, 3], [4, 6], [8, 10]],
  );
});

test('updateCueTimesBulk 忽略空数据与未知行；非有限值被过滤', () => {
  const { store, actions } = setup();
  assert.equal(actions.updateCueTimesBulk([]), false);
  assert.equal(actions.updateCueTimesBulk([{ id: 'zzz', start: 1, end: 2 }]), false);
  assert.equal(actions.updateCueTimesBulk([{ id: 'a', start: NaN, end: 2 }]), false);
  assert.equal(store.state.cues[0].start, 1);
});

test('updateCueTimesBulk 自动提交合并键：连续写入只占一个撤销步骤', () => {
  const { store, actions } = setup();
  actions.updateCueTimesBulk([{ id: 'a', start: 1.1, end: 3 }], { coalesceKey: 'audio-timing' });
  actions.updateCueTimesBulk([{ id: 'a', start: 1.2, end: 3 }], { coalesceKey: 'audio-timing' });
  actions.updateCueTimesBulk([{ id: 'a', start: 1.3, end: 3 }], { coalesceKey: 'audio-timing' });
  assert.equal(store.state.cues[0].start, 1.3);
  assert.ok(actions.undo());
  assert.equal(store.state.cues[0].start, 1); // 一步回到起点
  assert.equal(actions.undo(), false);
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

test('updateCueStyle 写回样式字段且可撤销', () => {
  const store = createStore();
  const actions = createActions(store);
  const meta = {
    fields: ['layer', 'start', 'end', 'style', 'name', 'marginl', 'marginr', 'marginv', 'effect', 'text'],
    parts: ['0', '0:00:01.00', '0:00:03.00', 'Default', '', '0', '0', '0', '', '文本'],
    startIdx: 1, endIdx: 2, textIdx: 9,
  };
  actions.loadSubtitle([{ id: 'a', start: 1, end: 3, text: '文本', meta }], { name: 'x.ass', format: 'ass', doc: { lines: [], fields: meta.fields, appendAt: 0, styles: ['Default', 'OP'] } });
  assert.ok(actions.updateCueStyle(['a'], 'OP'));
  assert.equal(store.state.cues[0].meta.parts[3], 'OP');
  actions.undo();
  assert.equal(store.state.cues[0].meta.parts[3], 'Default');
});

test('updateCueStyle 作用于多选行', () => {
  const store = createStore();
  const actions = createActions(store);
  const mkMeta = () => ({
    fields: ['layer', 'start', 'end', 'style', 'name', 'marginl', 'marginr', 'marginv', 'effect', 'text'],
    parts: ['0', '', '', 'Default', '', '0', '0', '0', '', ''],
    startIdx: 1, endIdx: 2, textIdx: 9,
  });
  actions.loadSubtitle([
    { id: 'a', start: 1, end: 2, text: 'a', meta: mkMeta() },
    { id: 'b', start: 3, end: 4, text: 'b', meta: mkMeta() },
    { id: 'c', start: 5, end: 6, text: 'c', meta: mkMeta() },
  ], { name: 'x.ass', format: 'ass', doc: { lines: [], fields: mkMeta().fields, appendAt: 0, styles: ['Default', 'Sign'] } });
  actions.select('a');
  actions.select('b', { mode: 'range' });
  assert.ok(actions.updateCueStyle(['a', 'b'], 'Sign'));
  assert.deepEqual(store.state.cues.map((c) => c.meta.parts[3]), ['Sign', 'Sign', 'Default']);
});

// ---------- 多选 ----------

test('select 支持 toggle 与 range 多选', () => {
  const { store, actions } = setup();
  actions.select('a');
  actions.select('c', { mode: 'range' });
  assert.deepEqual(store.state.selectedIds, ['a', 'b', 'c']);
  actions.select('a', { mode: 'toggle' });
  assert.deepEqual(store.state.selectedIds, ['b', 'c']);
  actions.select('a', { mode: 'toggle' });
  // 多选始终按 cue 顺序展开
  assert.deepEqual(store.state.selectedIds, ['a', 'b', 'c']);
  actions.select('b');
  assert.deepEqual(store.state.selectedIds, ['b']);
  actions.selectAll();
  assert.equal(store.state.selectedIds.length, 3);
});

test('removeCues 批量删除并把选中移到相邻行', () => {
  const { store, actions } = setup();
  actions.select('a');
  actions.select('b', { mode: 'range' });
  assert.ok(actions.removeCues(store.state.selectedIds));
  assert.deepEqual(store.state.cues.map((c) => c.id), ['c']);
  assert.deepEqual(store.state.selectedIds, ['c']);
  assert.ok(actions.undo());
  assert.equal(store.state.cues.length, 3);
});

// ---------- 插入变体 ----------

test('insertRelativeTo 在参考行前/后插入 5 秒空白行', () => {
  const { store, actions } = setup();
  const before = actions.insertRelativeTo('b', { after: false });
  // 之前:占用参考行开始前的 5 秒(b.start=4 → 0~4,被钳到 0)
  assert.equal(before.start, 0);
  assert.equal(before.end, 4);
  assert.equal(before.text, '');
  const order = store.state.cues.map((c) => c.id);
  assert.deepEqual(order, [before.id, 'a', 'b', 'c']);
  assert.deepEqual(store.state.selectedIds, [before.id]);

  const after = actions.insertRelativeTo('b', { after: true });
  // 之后:占用参考行结束后的 5 秒(b.end=6 → 6~11)
  assert.equal(after.start, 6);
  assert.equal(after.end, 11);
  assert.deepEqual(store.state.cues.map((c) => c.id), [before.id, 'a', 'b', after.id, 'c']);
});

test('insertRelativeTo 以视频时间插入生成零时长行', () => {
  const { store, actions } = setup();
  const cue = actions.insertRelativeTo('b', { after: true, videoTime: 5 });
  assert.equal(cue.start, 5);
  assert.equal(cue.end, 5);
  // 时间排序落在 b(4-6) 内部同起点处，稳定排序保持在 b 之后
  const idxB = store.state.cues.findIndex((c) => c.id === 'b');
  assert.equal(store.state.cues[idxB + 1].id, cue.id);
  // 空列表时也可按视频时间插入
  const { store: store2, actions: actions2 } = setup();
  actions2.removeCues(['a', 'b', 'c']);
  const fresh = actions2.insertRelativeTo(null, { videoTime: 2.5 });
  assert.equal(fresh.start, 2.5);
  assert.equal(store2.state.cues.length, 1);
});

test('duplicateCues 在原行后克隆并选中新行', () => {
  const { store, actions } = setup();
  assert.ok(actions.duplicateCues(['a']));
  assert.equal(store.state.cues.length, 4);
  assert.equal(store.state.cues[0].text, '第一句');
  assert.equal(store.state.cues[1].text, '第一句');
  assert.notEqual(store.state.cues[0].id, store.state.cues[1].id);
  assert.deepEqual(store.state.selectedIds, [store.state.cues[1].id]);
  assert.ok(actions.undo());
  assert.equal(store.state.cues.length, 3);
});

// ---------- 行剪贴板 ----------

test('copy → paste 在参考行后插入剪贴板行', async () => {
  const { store, actions } = setup();
  actions.select('a');
  actions.select('b', { mode: 'range' });
  assert.ok(actions.copyCues(store.state.selectedIds));
  assert.ok(actions.canPaste());
  const inserted = await actions.pasteCues({ refId: 'c', after: true });
  assert.equal(inserted.length, 2);
  assert.equal(store.state.cues.length, 5);
  assert.deepEqual(store.state.selectedIds, inserted.map((c) => c.id));
  // 新行时间与文本来自剪贴板（时间相同者按稳定排序紧邻原行）
  assert.equal(store.state.cues[1].text, '第一句');
  assert.equal(store.state.cues[1].start, 1);
  assert.equal(store.state.cues[3].text, '第二句');
  assert.ok(actions.undo());
});

test('cutCues 复制并删除所选', async () => {
  const { store, actions } = setup();
  actions.select('b');
  assert.ok(actions.cutCues(['b']));
  assert.deepEqual(store.state.cues.map((c) => c.id), ['a', 'c']);
  assert.equal(actions.canPaste(), true);
  const inserted = await actions.pasteCues({ refId: null });
  assert.equal(inserted.length, 1);
  assert.equal(store.state.cues.length, 3);
  assert.ok(store.state.cues.some((c) => c.text === '第二句' && c.start === 4));
});

test('pasteSpecial 按字段覆盖目标行', async () => {
  const { store, actions } = setup();
  actions.select('b');
  assert.ok(actions.copyCues(['a'])); // 1-3 第一句
  assert.ok(await actions.pasteSpecial({ start: true, end: false, text: true }, { refId: 'b' }));
  const b = store.state.cues.find((c) => c.id === 'b');
  assert.equal(b.start, 1); // 来自剪贴板
  assert.equal(b.end, 6); // 保留原值
  assert.equal(b.text, '第一句');
  assert.ok(actions.undo());

  // 剪贴板行多于剩余行时追加新行
  actions.select('a');
  actions.select('c', { mode: 'range' });
  actions.copyCues(store.state.selectedIds); // 3 行
  assert.ok(await actions.pasteSpecial({ start: true, end: true, text: true }, { refId: 'b' }));
  // b 覆盖为「第一句」，c 覆盖为「第二句」，追加「第三句」
  assert.equal(store.state.cues.length, 4);
  assert.equal(store.state.cues.find((c) => c.id === 'b').text, '第一句');
  assert.equal(store.state.cues.find((c) => c.id === 'c').text, '第二句');
  assert.equal(store.state.cues[3].text, '第三句');
});

// ---------- 外部剪贴板（注入读取函数）：纯文本按行新建 / 字幕格式解析粘贴 ----------

function setupWithClipboard(text) {
  const store = createStore();
  const actions = createActions(store, { readClipboardText: async () => text });
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

test('clipboardKind 区分纯文本与字幕剪贴板', async () => {
  assert.equal(await setupWithClipboard('第一行\n第二行').actions.clipboardKind(), 'text');
  assert.equal(
    await setupWithClipboard('1\n00:00:01,000 --> 00:00:03,000\nx').actions.clipboardKind(),
    'subtitle',
  );
  assert.equal(await setupWithClipboard('  \n').actions.clipboardKind(), null);
});

test('pasteCues 纯文本剪贴板按行新建，从参考行结束处顺序占位 5s', async () => {
  const { store, actions } = setupWithClipboard('第一行\n第二行\n\n第三行'); // 空行被跳过
  const inserted = await actions.pasteCues({ refId: 'b', after: true });
  assert.equal(inserted.length, 3);
  // b.end=6 → 6-11 / 11-16 / 16-21，首尾相接互不重叠
  assert.deepEqual(inserted.map((c) => [c.start, c.end]), [[6, 11], [11, 16], [16, 21]]);
  assert.deepEqual(inserted.map((c) => c.text), ['第一行', '第二行', '第三行']);
  assert.deepEqual(store.state.selectedIds, inserted.map((c) => c.id));
  assert.ok(actions.undo());
  assert.equal(store.state.cues.length, 3);
});

test('pasteCues 纯文本剪贴板无参考行时从 videoTime 顺序占位', async () => {
  const { actions } = setupWithClipboard('甲\n乙');
  const inserted = await actions.pasteCues({ refId: null, videoTime: 30 });
  assert.deepEqual(inserted.map((c) => [c.start, c.end]), [[30, 35], [35, 40]]);
});

test('pasteCues 纯文本剪贴板在空文档从 0 开始', async () => {
  const { store, actions } = setupWithClipboard('甲\n乙');
  actions.removeCues(['a', 'b', 'c']);
  const inserted = await actions.pasteCues({ refId: null });
  assert.equal(store.state.cues.length, 2);
  assert.deepEqual(inserted.map((c) => [c.start, c.end]), [[0, 5], [5, 10]]);
});

test('pasteCues SRT 剪贴板按原时间解析插入', async () => {
  const { store, actions } = setupWithClipboard('1\n00:00:20,000 --> 00:00:22,500\n外来句');
  const inserted = await actions.pasteCues({ refId: 'a' });
  assert.equal(inserted.length, 1);
  assert.deepEqual([inserted[0].start, inserted[0].end], [20, 22.5]);
  assert.equal(inserted[0].text, '外来句');
  assert.ok(store.state.cues.includes(inserted[0]));
});

test('pasteCues 裸 Dialogue 剪贴板（Aegisub 复制格式）解析为字幕行', async () => {
  const { actions } = setupWithClipboard(
    'Dialogue: 0,0:00:20.00,0:00:22.00,Default,,0,0,0,,A句\r\nDialogue: 0,0:00:23.00,0:00:25.00,Default,,0,0,0,,B句',
  );
  const inserted = await actions.pasteCues({ refId: 'a' });
  assert.equal(inserted.length, 2);
  assert.deepEqual(inserted.map((c) => [c.start, c.end, c.text]), [
    [20, 22, 'A句'],
    [23, 25, 'B句'],
  ]);
});

test('pasteCues 残缺字幕剪贴板抛出解析错误（不静默贴成文本）', async () => {
  const { actions } = setupWithClipboard('1\n00:00:01,000 --> 00:00:03,000\n好句\n\n2\n残缺块');
  await assert.rejects(() => actions.pasteCues({ refId: 'a' }), (err) => /-->/.test(err.message));
});

test('pasteSpecial 遇纯文本剪贴板委托为按行新建', async () => {
  const { store, actions } = setupWithClipboard('甲');
  assert.ok(await actions.pasteSpecial({ start: true, end: true, text: true }, { refId: 'a' }));
  const inserted = store.state.cues.filter((c) => c.text === '甲');
  assert.equal(inserted.length, 1); // 不覆盖现有行，而是新建
  assert.deepEqual([inserted[0].start, inserted[0].end], [3, 8]); // a.end=3 起 5s
  assert.equal(store.state.cues.length, 4);
});


test('updateCueStyle 对非 ASS 行（无 meta）安全忽略，不产生 null 行', () => {
  const { store, actions } = setup();
  assert.equal(actions.updateCueStyle(['a', 'b'], 'Default'), false);
  assert.ok(store.state.cues.every((c) => c && c.id));
  assert.deepEqual(store.state.cues.map((c) => c.id), ['a', 'b', 'c']);
});

test('markExported 设干净基线：撤销回基线后 dirty 复位、再编辑恢复', () => {
  const { store, actions } = setup();
  actions.select('a');
  actions.updateCue('a', { text: '改' });
  assert.equal(store.state.dirty, true);
  actions.markExported('srt', 'x.srt');
  assert.equal(store.state.dirty, false);
  assert.ok(actions.undo()); // 回到导出前的内容 → 与文件不一致 → 仍为脏
  assert.equal(store.state.dirty, true);
  assert.ok(actions.redo()); // 回到导出内容 → 干净
  assert.equal(store.state.dirty, false);
  actions.updateCue('a', { text: '再改' });
  assert.equal(store.state.dirty, true);
});

test('insertRelativeTo 支持 coalesceKey：多步操作合并为一次撤销', () => {
  const { store, actions } = setup();
  const key = Symbol('k');
  actions.updateCueTimesBulk([{ id: 'a', start: 1, end: 2 }], { coalesceKey: key });
  const cue = actions.insertRelativeTo('a', { after: true, videoTime: 2, coalesceKey: key });
  actions.updateCueTimes(cue.id, 2, 4, { coalesceKey: key });
  const before = store.state.cues.map((c) => [c.start, c.end]);
  assert.ok(actions.undo()); // 一步撤销应同时回滚三步操作
  assert.deepEqual(store.state.cues.map((c) => [c.start, c.end]), [[1, 3], [4, 6], [8, 10]]);
  assert.notDeepEqual(before, [[1, 3], [4, 6], [8, 10]]);
});

test('pasteSpecial 时间非法时返回跳过行数（只贴开始字段且晚于目标结束）', async () => {
  const { store, actions } = setupWithClipboard('1\n00:00:05,000 --> 00:00:07,000\n晚行');
  const result = await actions.pasteSpecial({ start: true, end: false, text: false }, { refId: 'a' });
  assert.equal(result.ok, true);
  assert.equal(result.skipped, 1); // start=5 > 目标 end=3，end<=start 被跳过
  assert.equal(store.state.cues[0].start, 1); // 原值保留
});

test('pasteSpecial 时间被跳过时其余勾选字段仍应用', async () => {
  const { store, actions } = setupWithClipboard('1\n00:00:05,000 --> 00:00:07,000\n晚行');
  const result = await actions.pasteSpecial({ start: true, end: false, text: true }, { refId: 'a' });
  assert.equal(result.skipped, 1);
  assert.equal(store.state.cues[0].start, 1); // 冲突的时间不写入
  assert.equal(store.state.cues[0].text, '晚行'); // 文本字段照常覆盖
});

test('系统剪贴板字幕粘贴保留 ASS meta（样式/字段序）', async () => {
  const ass = '[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Foo\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,OP,,0,0,0,,{\\i1}样式行{\\i0}\n';
  const { store, actions } = setupWithClipboard(ass);
  assert.ok(await actions.pasteCues({ refId: 'a' }));
  const pasted = store.state.cues.find((c) => c.text.includes('样式行'));
  assert.ok(pasted?.meta?.parts, 'meta 应随粘贴保留');
  assert.equal(pasted.meta.fields.indexOf('style') > -1, true);
});
