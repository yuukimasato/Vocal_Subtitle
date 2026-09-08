import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyClipboard } from '../js/format/index.js';
import { ParseError } from '../js/format/time.js';

test('classifyClipboard 识别 SRT / VTT / ASS 剪贴板', () => {
  const srt = classifyClipboard(
    '1\n00:00:01,000 --> 00:00:03,000\n第一句\n\n2\n00:00:04,000 --> 00:00:06,000\n第二句',
  );
  assert.equal(srt.kind, 'subtitle');
  assert.equal(srt.format, 'srt');
  assert.equal(srt.cues.length, 2);
  assert.equal(srt.cues[0].text, '第一句');

  const vtt = classifyClipboard('WEBVTT\n\n00:01.000 --> 00:03.000\n你好');
  assert.equal(vtt.kind, 'subtitle');
  assert.equal(vtt.format, 'vtt');

  const ass = classifyClipboard(
    [
      '[Script Info]',
      '',
      '[Events]',
      'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
      'Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,台词',
    ].join('\n'),
  );
  assert.equal(ass.kind, 'subtitle');
  assert.equal(ass.format, 'ass');
  assert.equal(ass.cues[0].text, '台词');
});

test('classifyClipboard 兼容 Aegisub 复制的裸 Dialogue 行（无段头）', () => {
  const parsed = classifyClipboard(
    [
      'Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,第一句',
      'Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,第二句',
    ].join('\r\n'),
  );
  assert.equal(parsed.kind, 'subtitle');
  assert.equal(parsed.format, 'ass');
  assert.deepEqual(parsed.cues.map((c) => c.text), ['第一句', '第二句']);
});

test('classifyClipboard 纯多行文本：CRLF 归一、trim、跳过空行', () => {
  const parsed = classifyClipboard('第一行\r\n第二行  \n\r\n\n  第三行\r');
  assert.deepEqual(parsed, { kind: 'text', lines: ['第一行', '第二行', '第三行'] });
});

test('classifyClipboard 嗅探到格式但 0 条 cue 时回落为纯文本', () => {
  const parsed = classifyClipboard('WEBVTT\n\n');
  assert.deepEqual(parsed, { kind: 'text', lines: ['WEBVTT'] });
});

test('classifyClipboard 嗅探到格式但解析失败时抛 ParseError', () => {
  // 块 1 合法（保证嗅探为 SRT），块 2 缺 "-->" 时间轴行
  const broken = '1\n00:00:01,000 --> 00:00:03,000\n好句\n\n2\n残缺块';
  assert.throws(() => classifyClipboard(broken), ParseError);
});

test('classifyClipboard 空白文本返回 null', () => {
  assert.equal(classifyClipboard(''), null);
  assert.equal(classifyClipboard('  \n \r\n\t'), null);
});
