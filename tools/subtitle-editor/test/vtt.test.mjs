import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVtt, stringifyVtt } from '../js/format/vtt.js';
import { ParseError } from '../js/format/time.js';
import { makeCue } from '../js/format/cue.js';

const SAMPLE = [
  'WEBVTT',
  '',
  'NOTE 这是注释块',
  '多行注释',
  '',
  'STYLE',
  '::cue { color: red }',
  '',
  '1',
  '00:00:01.000 --> 00:00:03.500 align:start line:0%',
  '你好 <b>世界</b>',
  '',
  '00:04.000 --> 00:00:06.000',
  '无 id 且省略小时的 cue',
  '',
].join('\n');

test('VTT 解析：头、注释/样式块跳过、settings 保留', () => {
  const { cues } = parseVtt(SAMPLE);
  assert.equal(cues.length, 2);
  assert.deepEqual([cues[0].start, cues[0].end], [1, 3.5]);
  assert.equal(cues[0].settings, 'align:start line:0%');
  assert.equal(cues[0].text, '你好 <b>世界</b>');
  assert.deepEqual([cues[1].start, cues[1].end], [4, 6]);
  assert.equal(cues[1].settings, undefined);
});

test('VTT 缺少 WEBVTT 头时报错', () => {
  assert.throws(() => parseVtt('1\n00:00:01.000 --> 00:00:02.000\nx'), ParseError);
});

test('VTT 往返一致（含 settings）', () => {
  const once = parseVtt(SAMPLE);
  const text = stringifyVtt(once.cues);
  const twice = parseVtt(text);
  assert.equal(twice.cues.length, 2);
  assert.equal(twice.cues[0].settings, 'align:start line:0%');
  assert.equal(twice.cues[1].text, once.cues[1].text);
  assert.ok(text.startsWith('WEBVTT\n'));
});

test('VTT 逗号毫秒可解析（SRT 改名 .vtt），写出仍用点号', () => {
  const { cues } = parseVtt('WEBVTT\n\n00:00:01,000 --> 00:00:03,500\n你好');
  assert.equal(cues.length, 1);
  assert.deepEqual([cues[0].start, cues[0].end], [1, 3.5]);
  assert.ok(stringifyVtt(cues).includes('00:00:01.000 --> 00:00:03.500'));
});

test('VTT STYLE/REGION 块与 cue id 行 round-trip', () => {
  const src = [
    'WEBVTT - 示例',
    'Kind: captions',
    '',
    'STYLE',
    '::cue { color: red }',
    '',
    'REGION',
    'id: bottom width:90%',
    '',
    'intro',
    '00:00:01.000 --> 00:00:03.500',
    '你好',
    '',
    '00:00:04.000 --> 00:00:06.000',
    '无 id 的 cue',
    '',
  ].join('\n');
  const { cues, doc } = parseVtt(src);
  assert.deepEqual(doc.header, ['Kind: captions']);
  assert.equal(doc.blocks.length, 2);
  assert.equal(cues[0].vttId, 'intro');
  assert.equal(cues[1].vttId, undefined);

  const text = stringifyVtt(cues, doc);
  assert.ok(text.startsWith('WEBVTT - 示例\nKind: captions\n'));
  assert.ok(text.includes('::cue { color: red }'));
  assert.ok(text.includes('id: bottom width:90%'));
  assert.ok(text.includes('intro\n00:00:01.000'));

  const reparsed = parseVtt(text);
  assert.equal(reparsed.cues.length, 2);
  assert.equal(reparsed.cues[0].vttId, 'intro');
  assert.equal(reparsed.cues[1].vttId, '2'); // 无 id 的 cue 写出默认序号
  assert.deepEqual([reparsed.cues[1].start, reparsed.cues[1].end], [4, 6]);
});

test('VTT NOTE 块内含 --> 不误判为 cue', () => {
  const src = [
    'WEBVTT',
    '',
    'NOTE',
    '这一段 --> 像时间轴但其实只是注释',
    '',
    '00:00:01.000 --> 00:00:02.000',
    '真 cue',
    '',
  ].join('\n');
  const { cues } = parseVtt(src);
  assert.equal(cues.length, 1);
  assert.equal(cues[0].text, '真 cue');
});

test('VTT 文本含空行 round-trip：导出折叠为单换行', () => {
  const text = stringifyVtt([makeCue(1, 2, 'a\n\nb')]);
  // 全文只应有"WEBVTT 头 | cue"一个块分隔空行
  assert.equal(text.split('\n\n').length, 2);
  const reparsed = parseVtt(text).cues;
  assert.equal(reparsed.length, 1);
  assert.equal(reparsed[0].text, 'a\nb');
});
