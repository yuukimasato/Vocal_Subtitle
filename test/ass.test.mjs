import test from 'node:test';
import assert from 'node:assert/strict';
import { parseAss, serializeAss, buildAssDoc, makeAssMeta, DEFAULT_ASS_FIELDS } from '../js/format/ass.js';
import { makeCue } from '../js/format/cue.js';

const SAMPLE = [
  '[Script Info]',
  '; 我的字幕',
  'Title: 测试',
  'PlayResX: 1920',
  '',
  '[V4+ Styles]',
  'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
  'Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1',
  '',
  '[Events]',
  'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
  'Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,你好，世界，带逗号',
  'Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,第二句{\\i1}斜体标签保留{\\i0}\\N换行',
  '',
].join('\n');

test('ASS 解析：文本逗号、标签保留、\\N 转换', () => {
  const { cues, doc } = parseAss(SAMPLE);
  assert.equal(cues.length, 2);
  assert.deepEqual([cues[0].start, cues[0].end], [1, 3.5]);
  assert.equal(cues[0].text, '你好，世界，带逗号');
  assert.equal(cues[1].text, '第二句{\\i1}斜体标签保留{\\i0}\n换行');
  assert.equal(doc.fields.length, 10);
  assert.ok(doc.lines.some((l) => l.t === 'raw' && l.s.includes('Title: 测试')));
});

test('ASS 未编辑时序列化保持原文（除行内空白归一）', () => {
  const { cues, doc } = parseAss(SAMPLE);
  const out = serializeAss(cues, doc);
  assert.ok(out.includes('; 我的字幕'));
  assert.ok(out.includes('Style: Default,Arial,20,'));
  assert.ok(out.includes('Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,你好，世界，带逗号'));
  assert.ok(out.includes('第二句{\\i1}斜体标签保留{\\i0}\\N换行'));
});

test('ASS 编辑时间与文本后其余字段原样保留', () => {
  const { cues, doc } = parseAss(SAMPLE);
  cues[0].start = 1.25;
  cues[0].end = 4;
  cues[0].text = '改过的文本';
  const out = serializeAss(cues, doc);
  const line = out.split('\n').find((l) => l.startsWith('Dialogue:') && l.includes('改过的文本'));
  assert.ok(line, '应包含修改后的 Dialogue 行');
  const parts = line.split(',');
  assert.equal(parts[1], '0:00:01.25');
  assert.equal(parts[2], '0:00:04.00');
  assert.equal(parts[3], 'Default');
  assert.equal(parts[6], '0');
});

test('ASS 删除 cue 后对应行移除', () => {
  const { cues, doc } = parseAss(SAMPLE);
  const rest = cues.filter((c) => c.id !== cues[0].id);
  const out = serializeAss(rest, doc);
  assert.ok(!out.includes('你好，世界'));
  assert.equal(out.split('\n').filter((l) => l.startsWith('Dialogue:')).length, 1);
});

test('ASS 新增 cue（无骨架）以默认字段追加到 Events 末尾', () => {
  const { cues, doc } = parseAss(SAMPLE);
  const fresh = makeCue(8, 10, '新句子');
  fresh.meta = makeAssMeta(doc.fields, fresh);
  const out = serializeAss([...cues, fresh], doc);
  const lines = out.split('\n');
  const freshIdx = lines.findIndex((l) => l.includes('新句子'));
  const oldIdx = lines.findIndex((l) => l.startsWith('Dialogue:'));
  assert.ok(freshIdx > oldIdx, '新句子应追加在原有 Dialogue 之后');
  assert.ok(lines[freshIdx].startsWith('Dialogue: 0,0:00:08.00,0:00:10.00,Default,,0,0,0,,新句子'));
});

test('ASS 缺少 Format 行时使用默认字段序', () => {
  const src = [
    '[Script Info]',
    'Title: x',
    '',
    '[Events]',
    'Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,文本',
  ].join('\n');
  const { cues, doc } = parseAss(src);
  assert.equal(doc.fields, null);
  assert.equal(cues.length, 1);
  assert.deepEqual([cues[0].start, cues[0].end], [1, 2]);
});

test('ASS 字段不足的 Dialogue 原样保留', () => {
  const src = '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00';
  const { cues, doc } = parseAss(src);
  assert.equal(cues.length, 0);
  assert.ok(doc.lines.some((l) => l.t === 'raw' && l.s.startsWith('Dialogue:')));
});

test('ASS 无效时间报错', () => {
  assert.throws(
    () => parseAss('[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:99:01.00,0:00:02.00,Default,,0,0,0,,x'),
    (err) => /无效的 Dialogue 时间/.test(err.message),
  );
});

test('SRT 来源经最小骨架导出 ASS 可往返', () => {
  const cues = [
    makeCue(1, 3.5, '第一句'),
    makeCue(4, 6, '第二句'),
  ];
  const doc = buildAssDoc(cues);
  const text = serializeAss(cues, doc);
  const reparsed = parseAss(text);
  assert.equal(reparsed.cues.length, 2);
  assert.deepEqual(
    reparsed.cues.map((c) => [c.start, c.end, c.text]),
    [[1, 3.5, '第一句'], [4, 6, '第二句']],
  );
});

test('DEFAULT_ASS_FIELDS 为 v4+ 常见字段序', () => {
  assert.deepEqual(DEFAULT_ASS_FIELDS, [
    'layer', 'start', 'end', 'style', 'name',
    'marginl', 'marginr', 'marginv', 'effect', 'text',
  ]);
});
