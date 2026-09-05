// cue 构造与排序约定。

let counter = 0;

export function makeCue(start, end, text, extra = {}) {
  counter += 1;
  return {
    id: `cue-${counter}-${Math.random().toString(36).slice(2, 8)}`,
    start,
    end,
    text: String(text ?? ''),
    ...extra,
  };
}

// cues 恒按 start 稳定排序（同 start 按.end）
export function sortCues(cues) {
  return [...cues].sort((a, b) => a.start - b.start || a.end - b.end);
}

export function findCueAt(cues, t) {
  let lo = 0;
  let hi = cues.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const cue = cues[mid];
    if (t < cue.start) hi = mid - 1;
    else if (t >= cue.end) lo = mid + 1;
    else return cue;
  }
  return null;
}

export function indexAfter(cues, t) {
  // 第一个 start > t 的下标
  let lo = 0;
  let hi = cues.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].start <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
