// WebVTT 解析与序列化。cue settings 原样保留；STYLE/REGION 块与
// 头部元数据行、cue id 行存入 doc 骨架，写出时还原。
import { ParseError, parseTimestamp, formatVttTime } from './time.js';
import { makeCue, sortCues } from './cue.js';

// 时间戳：可选小时 + 分:秒[.,]毫秒（SRT 改名 .vtt 的逗号毫秒同样放行，写出恒用点号）。
// (?=…) 把可选小时组原子化：避免 "10:30.000" 被回溯误读成 "1:0:30.000"（10 分钟 → 1 小时）。
const TS = '(?:(?=\\d+:\\d{1,2}:\\d{1,2}[,.])\\d+:)?\\d{1,2}:\\d{1,2}[,.]\\d+';
const TIME_LINE = new RegExp(`(${TS})\\s*-->\\s*(${TS})(.*)`);

export function parseVtt(text) {
  const lines = String(text).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split('\n');
  const first = lines.findIndex((line) => line.trim() !== '');
  if (first === -1 || !lines[first].startsWith('WEBVTT')) {
    throw new ParseError('VTT 第 1 行：缺少 "WEBVTT" 文件头');
  }
  const cues = [];
  // doc 骨架：head 为 WEBVTT 首行，header 为头块内其余元数据行，blocks 为 STYLE/REGION 块原文
  const doc = { head: lines[first], header: [], blocks: [] };
  let block = [];
  let blockStart = 0;
  let headerDone = false;

  const flush = () => {
    const hadContent = block.some((line) => line.trim() !== '');
    const current = block;
    const currentStart = blockStart;
    block = [];
    if (!hadContent) return;
    if (!headerDone) {
      // WEBVTT 头块：首行是文件头，其余为元数据行
      headerDone = true;
      doc.header = current.slice(1);
      return;
    }
    const head = current[0].trim();
    if (head === 'STYLE' || head === 'REGION') {
      doc.blocks.push(current.join('\n')); // 块级定义原文存骨架
      return;
    }
    if (head === 'NOTE' || head.startsWith('NOTE ') || head.startsWith('NOTE\t')) {
      return; // 注释块整体跳到空行为止，其中的 "-->" 不当作 cue
    }
    const timeIdx = current.findIndex((line) => TIME_LINE.test(line));
    if (timeIdx === -1) return; // 其他非 cue 块宽容跳过
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
    const extra = {};
    if (timeIdx > 0) extra.vttId = current[0]; // cue id 行原样保留
    if (settings) extra.settings = settings;
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
  return { cues, doc };
}

export function stringifyVtt(cues, doc) {
  const sorted = sortCues(cues);
  const head = typeof doc?.head === 'string' && doc.head.startsWith('WEBVTT') ? doc.head : 'WEBVTT';
  const header = Array.isArray(doc?.header) ? doc.header : [];
  const blocks = Array.isArray(doc?.blocks) ? doc.blocks : [];
  const chunks = [header.length ? `${head}\n${header.join('\n')}` : head];
  chunks.push(...blocks);
  sorted.forEach((cue, i) => {
    // 有存储 id 用存储 id，否则用序号
    const id = cue.vttId ?? String(i + 1);
    const settings = cue.settings ? ` ${cue.settings}` : '';
    // 空行是 VTT 的块分隔符：文本内连续空行折叠为单换行、去掉首尾空行；
    // U+2028（ASS 软换行占位）规范为换行，避免跨格式导出残留私有占位符
    const text = String(cue.text ?? '')
      .replace(/\r/g, '')
      .replace(/\u2028/g, '\n')
      .replace(/\n{2,}/g, '\n')
      .replace(/^\n+|\n+$/g, '');
    chunks.push(`${id}\n${formatVttTime(cue.start)} --> ${formatVttTime(cue.end)}${settings}\n${text}`);
  });
  return chunks.join('\n\n') + '\n';
}
