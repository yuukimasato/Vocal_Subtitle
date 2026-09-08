import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSrt, stringifySrt } from '../js/format/srt.js';
import { ParseError } from '../js/format/time.js';
import { makeCue } from '../js/format/cue.js';

const SAMPLE = [
  '1',
  '00:00:01,000 --> 00:00:03,500',
  '你好，世界',
  '',
  '2',
  '00:00:04,000 --> 00:00:06,000',
  '第二行',
  '换行内容',
  '',
].join('\n');

test('SRT 基础解析：多行文本、序号、CRLF', () => {
  const { cues } = parseSrt(SAMPLE.replace(/\n/g, '\r\n'));
  assert.equal(cues.length, 2);
  assert.deepEqual(
    [cues[0].start, cues[0].end, cues[0].text],
    [1, 3.5, '你好，世界'],
  );
  assert.equal(cues[1].text, '第二行\n换行内容');
});

test('SRT 容忍缺失序号与点号毫秒', () => {
  const { cues } = parseSrt('00:00:01.000 --> 00:00:02.000\n没有序号');
  assert.equal(cues.length, 1);
  assert.equal(cues[0].text, '没有序号');
});

test('SRT 去除 BOM 与首尾空行', () => {
  const { cues } = parseSrt('\uFEFF\n\n' + SAMPLE);
  assert.equal(cues.length, 2);
});

test('SRT 缺少时间轴行时报错并带行号', () => {
  assert.throws(
    () => parseSrt('第一段\n随便文本\n\n第二段没时间轴'),
    (err) => err instanceof ParseError && /第 1 行/.test(err.message),
  );
});

test('SRT 结束早于开始时报错', () => {
  assert.throws(
    () => parseSrt('1\n00:00:05,000 --> 00:00:01,000\nx'),
    ParseError,
  );
});

test('SRT 往返一致且按开始时间排序', () => {
  const { cues } = parseSrt(SAMPLE);
  cues.reverse();
  const text = stringifySrt(cues);
  const reparsed = parseSrt(text).cues;
  assert.equal(reparsed.length, 2);
  assert.deepEqual(
    reparsed.map((c) => [c.start, c.end, c.text]),
    [[1, 3.5, '你好，世界'], [4, 6, '第二行\n换行内容']],
  );
});

test('SRT 导出为 UTF-8 无 BOM、块间空行分隔', () => {
  const text = stringifySrt(parseSrt(SAMPLE).cues);
  assert.ok(!text.startsWith('\uFEFF'));
  assert.equal(text.split('\n\n').length, 2);
  assert.ok(text.endsWith('\n'));
});

test('SRT 文本含空行 round-trip：导出折叠为单换行', () => {
  const text = stringifySrt([makeCue(1, 2, 'a\n\nb\n\nc')]);
  // 空行是块分隔符：cue 文本内不得出现空行，否则读回会被拆成多个块
  assert.equal(text.split('\n\n').length, 1);
  const reparsed = parseSrt(text).cues;
  assert.equal(reparsed.length, 1);
  assert.equal(reparsed[0].text, 'a\nb\nc');
});
