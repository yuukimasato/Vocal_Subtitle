// CLI 数据面的纯逻辑：字幕变换与结构校验。操作 plain cue 数组，不碰 DOM/store。
// 变换语义与 js/actions.js 保持一致（test/cli.test.mjs 有逐字段对齐单测）；
// 词汇表见 docs/开发文档 §4.1：结构检查归编辑器（overlaps/gaps/tooShort/…），声学裁决归 provider。
import {
  parseSubtitle,
  serializeSubtitle,
  ensureDoc,
  makeNewCue,
  detectFormat,
  ParseError,
} from '../js/format/index.js';
import { sortCues } from '../js/format/cue.js';

export const MIN_LEN = 0.05; // 与 js/actions.js 的 MIN_LEN 一致：编辑器允许的最短行时长
export const SET_MIN_END_GAP = 0.001; // 与 actions.updateCueTimesBulk 的 end 下限一致

// ---------- 载入 / 保存 ----------

export function loadSubtitle(text, { filename = '', format } = {}) {
  return parseSubtitle(text, { filename, format });
}

export function detectFormatOrThrow(text, filename) {
  const fmt = detectFormat(text, filename);
  if (!fmt) throw new ParseError('无法识别字幕格式（支持 SRT / VTT / ASS/SSA）');
  return fmt;
}

export function saveSubtitle(format, cues, doc) {
  return serializeSubtitle(format, cues, ensureDoc(format, cues, doc));
}

// ---------- 变换（语义对齐 actions.js） ----------

// 整体平移。actions 没有全局平移命令，此处定义：start 钳到 ≥0，end 不短于 start+MIN_LEN。
export function shiftCues(cues, delta) {
  if (!Number.isFinite(delta)) throw new ParseError('shift 需要数字秒数');
  return sortCues(
    cues.map((c) => {
      const start = Math.max(0, c.start + delta);
      const end = Math.max(start + MIN_LEN, c.end + delta);
      return { ...c, start, end };
    }),
  );
}

// 批量改时间，plan = [{index(1-based), start, end}]。语义对齐 actions.updateCueTimesBulk：
// 非法条目跳过；start 钳到 ≥0；end 下限 start+0.001；无变化不改写。
// 返回 {cues, changed}。
export function setCueTimes(cues, plan) {
  if (!Array.isArray(plan)) throw new ParseError('set 需要数组形式的 plan：[{"index":1,"start":1.0,"end":3.0}]');
  const wanted = new Map(
    plan
      .filter((e) => e && Number.isFinite(e.start) && Number.isFinite(e.end) && Number.isInteger(e.index))
      .map((e) => [e.index, e]),
  );
  if (!wanted.size) return { cues: sortCues(cues), changed: 0 };
  let changed = 0;
  const sorted = sortCues(cues);
  const next = sorted.map((cue, i) => {
    const patch = wanted.get(i + 1);
    if (!patch) return cue;
    const start = Math.max(0, patch.start);
    const end = Math.max(start + SET_MIN_END_GAP, patch.end);
    if (start === cue.start && end === cue.end) return cue;
    changed += 1;
    return { ...cue, start, end };
  });
  return { cues: next, changed };
}

// 在 atTime 处拆行。语义对齐 actions.splitCue：
// at 不在 (start+MIN_LEN, end-MIN_LEN) 开区间内时取中点；文本按字符比例切分；meta 双份拷贝。
// format/doc 供 makeNewCue 构造与骨架一致的新行（ASS）。返回 {cues, head, tail}（tail 为新行）。
export function splitCue(cues, index, atTime, { format = '', doc = null } = {}) {
  const sorted = sortCues(cues);
  const cue = sorted[index - 1];
  if (!cue) throw new ParseError(`行不存在：${index}（共 ${sorted.length} 行）`);
  let at = atTime;
  if (!(at > cue.start + MIN_LEN && at < cue.end - MIN_LEN)) {
    at = (cue.start + cue.end) / 2;
  }
  const ratio = (at - cue.start) / (cue.end - cue.start);
  const cut = Math.round(cue.text.length * ratio);
  const tail = makeNewCue(format, doc, at, cue.end, cue.text.slice(cut));
  const head = { ...cue, end: at, text: cue.text.slice(0, cut) };
  if (cue.meta) {
    head.meta = { ...cue.meta, parts: [...cue.meta.parts] };
    tail.meta = { ...cue.meta, parts: [...cue.meta.parts] };
  }
  const next = sortCues([...sorted.filter((c) => c.id !== cue.id), head, tail]);
  return { cues: next, head, tail };
}

// 与下一行合并。语义对齐 actions.mergeWithNext：end 取 max，文本以 \n 连接，保留本行 meta。
export function mergeCueWithNext(cues, index) {
  const sorted = sortCues(cues);
  const cue = sorted[index - 1];
  if (!cue) throw new ParseError(`行不存在：${index}（共 ${sorted.length} 行）`);
  const next = sorted[index]; // 合并对象是排序后的下一行
  if (!next) throw new ParseError(`第 ${index} 行已是最后一行，无下一行可合并`);
  const merged = {
    ...cue,
    end: Math.max(cue.end, next.end),
    text: `${cue.text}${cue.text && next.text ? '\n' : ''}${next.text}`,
  };
  return sortCues(sorted.filter((c) => c !== cue && c.id !== next.id).concat(merged));
}

// 重编号：排序后重写 VTT 标识行 id（cue.vttId）；SRT 序号在序列化时天然隐式重编，无需处理。
export function renumberCues(cues, format) {
  const sorted = sortCues(cues);
  if (format !== 'vtt') return sorted;
  return sorted.map((cue, i) => ({ ...cue, vttId: String(i + 1) }));
}

// ---------- 结构校验（check 的词汇表见开发文档 §4.1） ----------

// severity："error" 参与退出码 3；"warn" 仅提示（--strict 时也参与）。
export function checkCues(cues, { minLen = MIN_LEN, maxGap = 10 } = {}) {
  const sorted = sortCues(cues);
  const problems = [];
  sorted.forEach((cue, i) => {
    const idx = i + 1;
    if (!(cue.end > cue.start)) {
      problems.push({
        severity: 'error',
        kind: 'endBeforeStart',
        index: idx,
        start: cue.start,
        end: cue.end,
        detail: '结束时间不晚于开始时间',
      });
    } else if (cue.end - cue.start < minLen) {
      problems.push({
        severity: 'error',
        kind: 'tooShort',
        index: idx,
        start: cue.start,
        end: cue.end,
        duration: cue.end - cue.start,
        detail: `行时长 ${round(cue.end - cue.start)}s 低于最短行时长 ${minLen}s`,
      });
    }
    if (i > 0) {
      const prev = sorted[i - 1];
      if (cue.start < prev.end) {
        problems.push({
          severity: 'error',
          kind: 'overlaps',
          index: idx,
          withIndex: i,
          start: cue.start,
          end: cue.end,
          overlap: round(prev.end - cue.start),
          detail: `与第 ${i} 行重叠 ${round(prev.end - cue.start)}s`,
        });
      } else if (maxGap != null && cue.start - prev.end > maxGap) {
        problems.push({
          severity: 'warn',
          kind: 'gaps',
          index: idx,
          gap: round(cue.start - prev.end),
          detail: `与第 ${i} 行之间有 ${round(cue.start - prev.end)}s 空隙（阈值 ${maxGap}s）`,
        });
      }
    }
  });
  return {
    problems,
    summary: {
      cues: sorted.length,
      errors: problems.filter((p) => p.severity === 'error').length,
      warnings: problems.filter((p) => p.severity === 'warn').length,
      speechDuration: round(sorted.reduce((acc, c) => acc + Math.max(0, c.end - c.start), 0)),
      firstStart: sorted.length ? sorted[0].start : null,
      lastEnd: sorted.length ? sorted[sorted.length - 1].end : null,
    },
    clean: problems.every((p) => p.severity !== 'error'),
  };
}

function round(v) {
  return Math.round(v * 1000) / 1000;
}

// ---------- info ----------

export function subtitleInfo(cues) {
  const sorted = sortCues(cues);
  const durations = sorted.map((c) => Math.max(0, c.end - c.start)).sort((a, b) => a - b);
  const median = durations.length ? durations[Math.floor(durations.length / 2)] : null;
  return {
    count: sorted.length,
    firstStart: sorted.length ? sorted[0].start : null,
    lastEnd: sorted.length ? sorted[sorted.length - 1].end : null,
    speechDuration: round(durations.reduce((a, b) => a + b, 0)),
    duration: {
      min: durations.length ? round(durations[0]) : null,
      median: median != null ? round(median) : null,
      max: durations.length ? round(durations[durations.length - 1]) : null,
    },
  };
}
