// ASS/SSA 骨架式解析与序列化。
// 原则：非 Dialogue 行原样保留；Dialogue 只重写 时间与文本 字段，
// {\...} 标签在文本中原样保留；\N / \n 与换行双向转换。
import { ParseError, parseTimestamp, formatAssTime } from './time.js';
import { makeCue, sortCues } from './cue.js';

export const DEFAULT_ASS_FIELDS = [
  'layer', 'start', 'end', 'style', 'name',
  'marginl', 'marginr', 'marginv', 'effect', 'text',
];

const MINIMAL_HEADER = [
  '[Script Info]',
  '; Exported by Vocal Subtitle Timing Editor',
  'ScriptType: v4.00+',
  'PlayResX: 1920',
  'PlayResY: 1080',
  'WrapStyle: 0',
  'ScaledBorderAndShadow: yes',
  '',
  '[V4+ Styles]',
  'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
  'Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1',
  '',
  '[Events]',
  'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
];

function dialoguePrefix(line) {
  const m = /^\s*Dialogue:\s?/i.exec(line);
  return m ? line.slice(m[0].length) : null;
}

// 按逗号分割且保留最后一段中的逗号（JS split 的 limit 会截断丢弃，不能直接用）
function splitWithRest(body, count) {
  const parts = body.split(',');
  if (parts.length <= count) return parts;
  return [...parts.slice(0, count - 1), parts.slice(count - 1).join(',')];
}

export function parseAss(text) {
  const lines = String(text).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split('\n');
  const docLines = [];
  const cues = [];
  let section = '';
  let fields = null;
  let sawEvents = false;
  let lastEventsLen = 0;

  lines.forEach((line) => {
    const sec = /^\s*\[(.+)\]\s*$/.exec(line);
    if (sec) {
      section = sec[1].trim().toLowerCase();
      // 宽容匹配：[Events] / [V4+ Events] / [V4 Events] 等变体
      if (section.includes('event')) {
        section = 'events';
        sawEvents = true;
      }
      docLines.push({ t: 'raw', s: line });
      if (section === 'events') lastEventsLen = docLines.length;
      return;
    }
    if (section === 'events') {
      if (/^\s*Format\s*:/i.test(line)) {
        fields = line.replace(/^\s*Format\s*:/i, '').split(',').map((s) => s.trim().toLowerCase());
        docLines.push({ t: 'raw', s: line });
        lastEventsLen = docLines.length;
        return;
      }
      const body = dialoguePrefix(line);
      if (body !== null) {
        const fmt = fields ?? DEFAULT_ASS_FIELDS;
        const startIdx = fmt.indexOf('start');
        const endIdx = fmt.indexOf('end');
        const textIdx = fmt.indexOf('text');
        // 最后一段为 Text，允许其中包含逗号
        const parts = splitWithRest(body, fmt.length);
        const maxIdx = Math.max(startIdx, endIdx, textIdx);
        const usable = startIdx >= 0 && endIdx >= 0 && textIdx >= 0 && parts.length >= maxIdx + 1;
        if (!usable) {
          docLines.push({ t: 'raw', s: line });
        } else {
          const start = parseTimestamp(parts[startIdx]);
          const end = parseTimestamp(parts[endIdx]);
          if (start === null || end === null) {
            throw new ParseError(`ASS：无效的 Dialogue 时间 "${parts[startIdx]}" / "${parts[endIdx]}"`);
          }
          if (end < start) {
            throw new ParseError(`ASS：Dialogue 结束时间早于开始时间`);
          }
          const cue = makeCue(start, end, String(parts[textIdx]).replace(/\\[Nn]/g, '\n'), {
            meta: { fields: fmt, parts, startIdx, endIdx, textIdx },
          });
          cues.push(cue);
          docLines.push({ t: 'dlg', id: cue.id });
        }
        lastEventsLen = docLines.length;
        return;
      }
    }
    docLines.push({ t: 'raw', s: line });
  });

  const doc = { lines: docLines, fields, appendAt: lastEventsLen || docLines.length };
  return { cues, doc };
}

// 从 SRT/VTT 等“纯 cue”来源导出 ASS 时，构造最小合法骨架
export function buildAssDoc(cues) {
  const fields = DEFAULT_ASS_FIELDS;
  const lines = MINIMAL_HEADER.map((s) => ({ t: 'raw', s }));
  cues.forEach((cue) => lines.push({ t: 'dlg', id: cue.id }));
  return { lines, fields, appendAt: lines.length };
}

function defaultPart(field, cue) {
  switch (field) {
    case 'start': return formatAssTime(cue.start);
    case 'end': return formatAssTime(cue.end);
    case 'text': return String(cue.text ?? '').replace(/\n/g, '\\N');
    case 'layer': return '0';
    case 'style': return 'Default';
    case 'marginl':
    case 'marginr':
    case 'marginv': return '0';
    case 'name':
    case 'effect': return '';
    default: return '';
  }
}

export function serializeAss(cues, doc) {
  const lines = doc?.lines ?? buildAssDoc(cues).lines;
  const fmt = doc?.fields ?? DEFAULT_ASS_FIELDS;
  const byId = new Map(cues.map((c) => [c.id, c]));
  const out = [];
  let insertAt = out.length;
  let sawAppendPoint = false;

  lines.forEach((entry, i) => {
    if (!sawAppendPoint && i >= (doc?.appendAt ?? lines.length)) {
      sawAppendPoint = true;
      insertAt = out.length;
    }
    if (entry.t === 'raw') {
      out.push(entry.s);
      return;
    }
    const cue = byId.get(entry.id);
    if (!cue) return; // 已删除：整行移除
    const meta = cue.meta ?? { fields: fmt, startIdx: fmt.indexOf('start'), endIdx: fmt.indexOf('end'), textIdx: fmt.indexOf('text') };
    const parts = [...(cue.meta?.parts ?? meta.parts ?? defaultPartsFor(meta.fields ?? fmt, cue))];
    parts[meta.startIdx] = formatAssTime(cue.start);
    parts[meta.endIdx] = formatAssTime(cue.end);
    parts[meta.textIdx] = String(cue.text ?? '').replace(/\n/g, '\\N');
    out.push(`Dialogue: ${parts.join(',')}`);
  });
  if (!sawAppendPoint) insertAt = out.length;

  // 新增（骨架中不存在）的 cue：按时间序插入到 Events 段末尾
  const known = new Set(lines.filter((e) => e.t === 'dlg').map((e) => e.id));
  const fresh = sortCues(cues.filter((c) => !known.has(c.id)));
  const freshLines = fresh.map((cue) => {
    const fieldsFor = cue.meta?.fields ?? fmt;
    const parts = cue.meta?.parts
      ? [...cue.meta.parts]
      : defaultPartsFor(fieldsFor, cue);
    parts[fieldsFor.indexOf('start')] = formatAssTime(cue.start);
    parts[fieldsFor.indexOf('end')] = formatAssTime(cue.end);
    parts[fieldsFor.indexOf('text')] = String(cue.text ?? '').replace(/\n/g, '\\N');
    return `Dialogue: ${parts.join(',')}`;
  });
  out.splice(insertAt, 0, ...freshLines);
  return out.join('\n') + '\n';
}

function defaultPartsFor(fields, cue) {
  return fields.map((field) => defaultPart(field, cue));
}

// 为“新建 cue”生成 ASS 元数据（其余格式为空）
export function makeAssMeta(fields, cue) {
  const fmt = fields ?? DEFAULT_ASS_FIELDS;
  return {
    fields: fmt,
    parts: defaultPartsFor(fmt, cue),
    startIdx: fmt.indexOf('start'),
    endIdx: fmt.indexOf('end'),
    textIdx: fmt.indexOf('text'),
  };
}
