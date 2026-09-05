// SubRip (SRT) 解析与序列化。UTF-8 无 BOM、LF。
import { ParseError, parseTimestamp, formatSrtTime } from './time.js';
import { makeCue, sortCues } from './cue.js';

const TIME_LINE = /(\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3})\s*-->\s*(\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3})/;

export function parseSrt(text) {
  const lines = String(text).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split('\n');
  const cues = [];
  let block = [];
  let blockStart = 0;

  const flush = () => {
    if (!block.some((line) => line.trim() !== '')) {
      block = [];
      return;
    }
    const timeIdx = block.findIndex((line) => TIME_LINE.test(line));
    if (timeIdx === -1) {
      // 记录块内第一个非空行的行号，便于定位
      const firstNonEmpty = blockStart + block.findIndex((line) => line.trim() !== '');
      throw new ParseError(`SRT 第 ${firstNonEmpty + 1} 行附近：找不到 "-->" 时间轴行`);
    }
    const m = TIME_LINE.exec(block[timeIdx]);
    const start = parseTimestamp(m[1]);
    if (start === null) {
      throw new ParseError(`SRT 第 ${blockStart + timeIdx + 1} 行：无效的开始时间 "${m[1]}"`);
    }
    const end = parseTimestamp(m[2]);
    if (end === null) {
      throw new ParseError(`SRT 第 ${blockStart + timeIdx + 1} 行：无效的结束时间 "${m[2]}"`);
    }
    if (end < start) {
      throw new ParseError(`SRT 第 ${blockStart + timeIdx + 1} 行：结束时间早于开始时间`);
    }
    const textLines = block.slice(timeIdx + 1).filter((line, i, arr) => {
      // 去掉首尾空行，保留中间的（多行字幕的空行分隔极少见，原样保留）
      if (line.trim() !== '') return true;
      return i > 0 && i < arr.length - 1;
    });
    // 时间行之前的序号行（纯数字）忽略；其他内容也忽略（宽容处理）
    cues.push(makeCue(start, end, textLines.join('\n')));
    block = [];
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

export function stringifySrt(cues) {
  const sorted = sortCues(cues);
  return (
    sorted
      .map((cue, i) => {
        const text = String(cue.text ?? '').replace(/\r/g, '');
        return `${i + 1}\n${formatSrtTime(cue.start)} --> ${formatSrtTime(cue.end)}\n${text}`;
      })
      .join('\n\n') + (sorted.length ? '\n' : '')
  );
}
