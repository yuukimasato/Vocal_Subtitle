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

export function detectFormat(text, filename = '') {
  const ext = (/\.([a-z0-9]+)$/i.exec(filename)?.[1] ?? '').toLowerCase();
  if (EXT_MAP.has(ext)) return EXT_MAP.get(ext);
  const body = String(text).replace(/^\uFEFF/, '');
  if (/^\s*WEBVTT/m.test(body)) return 'vtt';
  if (/^\s*\[Script Info\]/im.test(body) || /^\s*Dialogue\s*:/im.test(body)) return 'ass';
  if (/\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3}\s*-->\s*\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3}/.test(body)) return 'srt';
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

export function serializeSubtitle(format, cues, doc) {
  const fn = SERIALIZERS[format];
  if (!fn) throw new ParseError(`未知格式：${format}`);
  return fn(cues, doc);
}

// 在播放头处新建 cue；ASS 来源时附上与其骨架一致的元数据
export function makeNewCue(format, doc, start, end, text) {
  const cue = makeCue(start, end, text);
  if (format === 'ass') {
    cue.meta = makeAssMeta(doc?.fields ?? DEFAULT_ASS_FIELDS, cue);
  }
  return cue;
}

export function ensureDoc(format, cues, doc) {
  if (format === 'ass' && !doc) return buildAssDoc(cues);
  return doc;
}

export const FORMATS = [
  { id: 'srt', label: 'SRT', ext: 'srt' },
  { id: 'vtt', label: 'WebVTT', ext: 'vtt' },
  { id: 'ass', label: 'ASS/SSA', ext: 'ass' },
];
