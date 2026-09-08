import test from 'node:test';
import assert from 'node:assert/strict';
import { detectFormat, parseSubtitle, serializeSubtitle, ParseError } from '../js/format/index.js';

test('按扩展名识别', () => {
  assert.equal(detectFormat('', 'a.srt'), 'srt');
  assert.equal(detectFormat('', 'a.vtt'), 'vtt');
  assert.equal(detectFormat('', 'a.ass'), 'ass');
  assert.equal(detectFormat('', 'a.ssa'), 'ass');
});

test('按内容识别', () => {
  assert.equal(detectFormat('WEBVTT\n\n1\n00:01.000 --> 00:02.000\nx'), 'vtt');
  assert.equal(detectFormat('[Script Info]\nTitle: x'), 'ass');
  assert.equal(detectFormat('Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,x'), 'ass');
  assert.equal(detectFormat('1\n00:00:01,000 --> 00:00:02,000\nx'), 'srt');
  assert.equal(detectFormat('随便文本'), null);
});

test('普通文本含 WEBVTT 字样不误判为 VTT', () => {
  assert.equal(detectFormat('会议记录\n\nWEBVTT 相关讨论'), null);
  assert.equal(detectFormat('明天讨论 WEBVTT 规范'), null);
});

test('普通文本行中含时间戳不误判为 SRT', () => {
  assert.equal(detectFormat('会议在 1:23:45,678 --> 1:23:50,000 之间召开'), null);
  assert.equal(detectFormat('记录：Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,x'), null);
});

test('parseSubtitle 分发并附加 id', () => {
  const { format, cues } = parseSubtitle('1\n00:00:01,000 --> 00:00:02,000\n你好', { filename: 'x.srt' });
  assert.equal(format, 'srt');
  assert.equal(cues.length, 1);
  assert.ok(cues[0].id);
});

test('parseSubtitle 内容嗅探兜底（扩展名错误）', () => {
  const { format } = parseSubtitle('WEBVTT\n\n00:01.000 --> 00:02.000\n你好', { filename: 'x.txt' });
  assert.equal(format, 'vtt');
});

test('无法识别时报 ParseError', () => {
  assert.throws(() => parseSubtitle('无法识别的内容'), ParseError);
});

test('serializeSubtitle 按格式分发', () => {
  const { cues } = parseSubtitle('1\n00:00:01,000 --> 00:00:02,000\n你好', { filename: 'x.srt' });
  assert.ok(serializeSubtitle('srt', cues).includes('00:00:01,000'));
  assert.ok(serializeSubtitle('vtt', cues).startsWith('WEBVTT'));
});
