// WebVTT 解析与序列化。cue settings 原样保留。
import { ParseError, parseTimestamp, formatVttTime } from './time.js';
import { makeCue, sortCues } from './cue.js';

const TIME_LINE = /((?:\d{1,3}:)?\d{1,2}:\d{2}\.\d{1,3})\s*-->\s*((?:\d{1,3}:)?\d{1,2}:\d{2}\.\d{1,3})(.*)/;

export function parseVtt(text) {
  const lines = String(text).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split('\n');
  const first = lines.findIndex((line) => line.trim() !== '');
  if (first === -1 || !lines[first].startsWith('WEBVTT')) {
    throw new ParseError('VTT 第 1 行：缺少 "WEBVTT" 文件头');
  }
  const cues = [];
  let block = [];
  let blockStart = 0;
  let inHeader = true;

  const flush = () => {
    const hadContent = block.some((line) => line.trim() !== '');
    const current = block;
    const currentStart = blockStart;
    block = [];
    if (!hadContent) return;
    if (inHeader) {
      inHeader = false; // WEBVTT 头块（可能带元数据行），跳过
      return;
    }
    const timeIdx = current.findIndex((line) => TIME_LINE.test(line));
    if (timeIdx === -1) {
      const head = current[0].trim();
      if (head.startsWith('NOTE') || head.startsWith('STYLE') || head.startsWith('REGION')) return;
      return; // 其他非 cue 块宽容跳过
    }
    const m = TIME_LINE.exec(current[timeIdx]);
    const start = parseTimestamp(m[1]);
    const end = parseTimestamp(m[2]);
    if (start === null || end === null) {
      throw new ParseError(`VTT 第 ${currentStart + timeIdx + 1} 行：无效时间戳`);
    }
    if (end < start) {
      throw new ParseError(`VTT 第 ${currentStart + timeIdx + 1} 行：结束时间早于开始时间`);
    }
    const settings = m[3].trim();
    const textLines = current.slice(timeIdx + 1);
    const extra = settings ? { settings } : {};
    cues.push(makeCue(start, end, textLines.join('\n'), extra));
  };

  lines.forEach((line, i) => {
    if (line.trim() === '') flush();
    else {
      if (block.length === 0) blockStart = i;
      block.push(line);
    }
  });
  flush();
  return { cues };
}

export function stringifyVtt(cues) {
  const sorted = sortCues(cues);
  const body = sorted
    .map((cue, i) => {
      const settings = cue.settings ? ` ${cue.settings}` : '';
      const text = String(cue.text ?? '').replace(/\r/g, '');
      return `${i + 1}\n${formatVttTime(cue.start)} --> ${formatVttTime(cue.end)}${settings}\n${text}`;
    })
    .join('\n\n');
  return sorted.length ? `WEBVTT\n\n${body}\n` : 'WEBVTT\n';
}
