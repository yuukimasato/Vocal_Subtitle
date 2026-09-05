import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVtt, stringifyVtt } from '../js/format/vtt.js';
import { ParseError } from '../js/format/time.js';

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
