// 时间解析与格式化。全部纯函数，可独立单测。
// 统一约定：cue 内时间为秒（float），毫秒精度。

export class ParseError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ParseError';
  }
}

// "H:MM:SS,mmm" | "H:MM:SS.mmm" | "MM:SS.mmm" | "M:SS.cc" → 秒；非法返回 null
export function parseTimestamp(text) {
  const m = /^(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})$/.exec(String(text).trim());
  if (!m) return null;
  const hours = m[1] !== undefined ? Number(m[1]) : 0;
  const minutes = Number(m[2]);
  const seconds = Number(m[3]);
  if (minutes > 59 || seconds > 59) return null;
  const ms = Number(m[4].padEnd(3, '0'));
  return hours * 3600 + minutes * 60 + seconds + ms / 1000;
}

function splitHms(t) {
  const total = Math.max(0, Math.round(t * 1000));
  const h = Math.floor(total / 3600000);
  const m = Math.floor((total % 3600000) / 60000);
  const s = Math.floor((total % 60000) / 1000);
  const ms = total % 1000;
  return { h, m, s, ms };
}

const pad = (n, w = 2) => String(n).padStart(w, '0');

export function formatSrtTime(t) {
  const { h, m, s, ms } = splitHms(t);
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
}

export function formatVttTime(t) {
  const { h, m, s, ms } = splitHms(t);
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(ms, 3)}`;
}

// ASS 时间只有厘秒精度：H:MM:SS.cc
export function formatAssTime(t) {
  const cs = Math.max(0, Math.round(t * 100));
  const h = Math.floor(cs / 360000);
  const m = Math.floor((cs % 360000) / 6000);
  const s = Math.floor((cs % 6000) / 100);
  const c = cs % 100;
  return `${h}:${pad(m)}:${pad(s)}.${pad(c)}`;
}

// 界面显示：m:ss.mmm / h:mm:ss.mmm
export function formatClock(t) {
  if (!Number.isFinite(t)) return '--:--.---';
  const { h, m, s, ms } = splitHms(t);
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}.${pad(ms, 3)}` : `${m}:${pad(s)}.${pad(ms, 3)}`;
}

export function formatDuration(t) {
  return Number.isFinite(t) ? `${Math.max(0, Math.round(t * 1000))}ms` : '--';
}

// 行内编辑用：接受 m:ss.mmm / mm:ss / 纯秒数；非法返回 null
export function parseFlexibleTime(text) {
  const raw = String(text).trim();
  if (!raw) return null;
  const ts = parseTimestamp(raw);
  if (ts !== null) return ts;
  let m = /^(\d+(?:\.\d+)?)$/.exec(raw);
  if (m) return Math.max(0, Number(m[1]));
  m = /^(\d{1,3}):(\d{1,2}(?:\.\d{1,3})?)$/.exec(raw);
  if (m) {
    const minutes = Number(m[1]);
    const seconds = Number(m[2]);
    if (seconds >= 60) return null;
    return minutes * 60 + seconds;
  }
  return null;
}
