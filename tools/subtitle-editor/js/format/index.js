// 字幕格式嗅探与分发。解析器返回 {cues, doc?}；ASS 的 doc 为骨架。
import { ParseError } from './time.js';
import { makeCue } from './cue.js';
import { parseSrt, stringifySrt } from './srt.js';
import { parseVtt, stringifyVtt } from './vtt.js';
import { parseAss, serializeAss, buildAssDoc, makeAssMeta, DEFAULT_ASS_FIELDS } from './ass.js';

export { ParseError };

const EXT_MAP = new Map([
  ['srt', 'srt'],
  ['vtt', 'vtt'],
  ['ass', 'ass'],
  ['ssa', 'ass'],
]);

// 首个非空行（嗅探 WEBVTT 用：\s 可跨行，/m 锚定挡不住正文出现该字样的普通文本）
function firstNonEmptyLine(body) {
  return body.split('\n').find((line) => line.trim() !== '') ?? '';
}

export function detectFormat(text, filename = '') {
  const ext = (/\.([a-z0-9]+)$/i.exec(filename)?.[1] ?? '').toLowerCase();
  if (EXT_MAP.has(ext)) return EXT_MAP.get(ext);
  const body = String(text).replace(/^\uFEFF/, '');
  // 全部行首严格锚定：时间戳 / Dialogue 出现在句中、行中不误判
  const first = firstNonEmptyLine(body);
  if (first.startsWith('WEBVTT')) return 'vtt';
  if (/^[ \t]*\[Script Info\]/m.test(body) || /^[ \t]*Dialogue\s*:/m.test(body)) return 'ass';
  if (/^\d+:\d{1,2}:\d{1,2}[,.]\d+\s*-->\s*\d+:\d{1,2}:\d{1,2}[,.]\d+/m.test(body)) return 'srt';
  return null;
}

const PARSERS = { srt: parseSrt, vtt: parseVtt, ass: parseAss };
const SERIALIZERS = { srt: stringifySrt, vtt: stringifyVtt, ass: serializeAss };

export function parseSubtitle(text, { filename = '', format } = {}) {
  const fmt = format ?? detectFormat(text, filename);
  if (!fmt) throw new ParseError('无法识别字幕格式（支持 SRT / VTT / ASS/SSA）');
  const { cues, doc } = PARSERS[fmt](text);
  return { format: fmt, cues, doc: doc ?? null };
}

// 剪贴板内容分类：能解析出 cue 的字幕 → subtitle；其余按纯文本逐行拆分（跳过空行）→ text。
// 嗅探到字幕格式但解析失败时抛出 ParseError（残缺字幕应报错而非贴成散文行）。
export function classifyClipboard(text) {
  const body = String(text).replace(/^\uFEFF/, '');
  if (!body.trim()) return null;
  const fmt = detectFormat(body);
  if (fmt) {
    const { cues, doc } = PARSERS[fmt](body);
    if (cues.length) return { kind: 'subtitle', format: fmt, cues, doc: doc ?? null };
  }
  const lines = body
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length ? { kind: 'text', lines } : null;
}

export function serializeSubtitle(format, cues, doc) {
  const fn = SERIALIZERS[format];
  if (!fn) throw new ParseError(`未知格式：${format}`);
  return fn(cues, doc);
}

// 在播放头处新建 cue；ASS 来源时附上与其骨架一致的元数据（样式取文档默认样式）
export function makeNewCue(format, doc, start, end, text) {
  const cue = makeCue(start, end, text);
  if (format === 'ass') {
    cue.meta = makeAssMeta(doc?.fields ?? DEFAULT_ASS_FIELDS, cue, doc?.defaultStyle);
  }
  return cue;
}

// 修改 ASS cue 的样式字段（写回原骨架字段序）；非 ASS cue 或无 style 字段返回 null
export function cueStyle(cue) {
  const idx = cue?.meta?.fields?.indexOf('style') ?? -1;
  if (!cue?.meta?.parts || idx < 0) return null;
  return cue.meta.parts[idx] ?? '';
}

export function withCueStyle(cue, style) {
  const idx = cue?.meta?.fields?.indexOf('style') ?? -1;
  if (!cue?.meta?.parts || idx < 0) return null;
  const meta = { ...cue.meta, parts: [...cue.meta.parts] };
  meta.parts[idx] = style;
  return { ...cue, meta };
}

// ASS 骨架是否带有 [Events] 段（doc.lines 中段头以 raw 行保存）
function assDocHasEvents(doc) {
  return !!doc?.lines?.some((e) => e.t === 'raw' && /^\s*\[[^\]]*event/i.test(e.s));
}

export function ensureDoc(format, cues, doc) {
  // 骨架缺失或没有 [Events] 段（如 parseAss('') 后新增 cue）时，重建最小合法骨架
  if (format === 'ass' && !assDocHasEvents(doc)) return buildAssDoc(cues);
  return doc;
}

export const FORMATS = [
  { id: 'srt', label: 'SRT', ext: 'srt' },
  { id: 'vtt', label: 'WebVTT', ext: 'vtt' },
  { id: 'ass', label: 'ASS/SSA', ext: 'ass' },
];
