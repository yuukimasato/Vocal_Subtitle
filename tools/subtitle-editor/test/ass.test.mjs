import test from 'node:test';
import assert from 'node:assert/strict';
import { parseAss, serializeAss, buildAssDoc, makeAssMeta, DEFAULT_ASS_FIELDS } from '../js/format/ass.js';
import { makeNewCue, cueStyle, withCueStyle, ensureDoc } from '../js/format/index.js';
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

test('ASS 裸 Dialogue 行（无段头，Aegisub 复制格式）可解析', () => {
  const src = [
    'Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,第一句',
    'Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,第二句',
  ].join('\r\n');
  const { cues } = parseAss(src);
  assert.equal(cues.length, 2);
  assert.deepEqual([cues[0].start, cues[0].end], [1, 3]);
  assert.deepEqual(cues.map((c) => c.text), ['第一句', '第二句']);
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

test('ASS 解析 [V4+ Styles] 收集样式名', () => {
  const src = [
    '[Script Info]',
    'Title: x',
    '',
    '[V4+ Styles]',
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
    'Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1',
    'Style: OP,思源黑体,60,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,8,10,10,10,1',
    '',
    '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    'Dialogue: 0,0:00:01.00,0:00:02.00,OP,,0,0,0,,文本',
  ].join('\n');
  const { cues, doc } = parseAss(src);
  assert.deepEqual(doc.styles, ['Default', 'OP']);
  assert.equal(doc.defaultStyle, 'Default');
  assert.equal(cueStyle(cues[0]), 'OP');
});

test('样式修改经 withCueStyle 往返序列化', () => {
  const { cues, doc } = parseAss(SAMPLE);
  const updated = withCueStyle(cues[0], 'OP');
  const out = serializeAss([updated, cues[1]], doc);
  const line = out.split('\n').find((l) => l.includes('你好，世界'));
  assert.ok(line.startsWith('Dialogue: 0,0:00:01.00,0:00:03.50,OP,'), line);
});

test('makeAssMeta 接受默认样式，makeNewCue 取文档默认样式', () => {
  const cue = makeCue(1, 2, 'x');
  const meta = makeAssMeta(DEFAULT_ASS_FIELDS, cue, 'OP');
  assert.equal(meta.parts[3], 'OP');
  const doc = buildAssDoc([]);
  doc.defaultStyle = 'Title';
  const fresh = makeNewCue('ass', doc, 1, 2, '新行');
  assert.equal(cueStyle(fresh), 'Title');
});

test('ASS \\N 硬换行与 \\n 软换行语义分开 round-trip', () => {
  const src = [
    '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    'Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,硬\\N换行与软\\n换行',
  ].join('\n');
  const { cues, doc } = parseAss(src);
  assert.equal(cues[0].text, '硬\n换行与软\u2028换行');
  const out = serializeAss(cues, doc);
  assert.ok(out.includes('硬\\N换行与软\\n换行'), out);
  const reparsed = parseAss(out);
  assert.equal(reparsed.cues[0].text, '硬\n换行与软\u2028换行');
});

test('ASS 写出清理 CR 与 U+2028 占位', () => {
  const cue = makeCue(1, 2, '第一行\r\n第二行\u2028第三行');
  const out = serializeAss([cue], buildAssDoc([cue]));
  assert.ok(!out.includes('\r'));
  assert.ok(out.includes('第一行\\N第二行\\n第三行'), out);
});

test('parseAss 空文件 + ensureDoc 后导出完整 ASS 骨架', () => {
  const { cues, doc } = parseAss('');
  assert.equal(cues.length, 0);
  const fresh = makeNewCue('ass', doc, 1, 2, '新行');
  const out = serializeAss([fresh], ensureDoc('ass', [fresh], doc));
  assert.ok(out.includes('[Script Info]'));
  assert.ok(out.includes('[Events]'));
  assert.ok(out.includes('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text'));
  assert.ok(out.includes('Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,新行'));
});

test('ASS 往返字节稳定：反复打开保存不累积尾部空行', () => {
  // SAMPLE 末尾带一个换行（真实文件如此），再补一个「末尾空行」的变体
  for (const src of [SAMPLE, `${SAMPLE}\n`]) {
    const p1 = parseAss(src);
    const once = serializeAss(p1.cues, p1.doc);
    const p2 = parseAss(once);
    const twice = serializeAss(p2.cues, p2.doc);
    assert.equal(twice, once, '第二轮序列化应与第一轮完全一致');
    assert.ok(once.endsWith('\n'), '导出以换行结尾');
  }
});

test('ASS 末尾换行不产生多余的空行骨架项', () => {
  const { doc } = parseAss('Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,x\n');
  assert.equal(doc.lines.at(-1).t, 'dlg');
  assert.equal(doc.appendAt, doc.lines.length);
});

test('ASS Events 段末 Comment 行之后插入新增 cue', () => {
  const src = [
    '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    'Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,第一句',
    'Comment: 这是一段备注',
  ].join('\n');
  const { cues, doc } = parseAss(src);
  const fresh = makeCue(8, 10, '新句子');
  fresh.meta = makeAssMeta(doc.fields, fresh, doc.defaultStyle);
  const out = serializeAss([...cues, fresh], doc);
  const lines = out.split('\n');
  const commentIdx = lines.findIndex((l) => l.startsWith('Comment:'));
  const freshIdx = lines.findIndex((l) => l.includes('新句子'));
  assert.ok(commentIdx !== -1, 'Comment 行应保留');
  assert.ok(freshIdx > commentIdx, '新 cue 应插在 Comment 行之后');
});
