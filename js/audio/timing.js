// Aegisub 3.2「对话定时控制器」（audio_timing_dialogue.cpp）的语义移植。
// 纯逻辑、无 DOM：波形显示层把指针事件翻译成这里的调用。
//
// 核心模型（与 Aegisub 一致）：
// - 当前行的开始/结束是两个可拖动的「标记」；多选行（不含锚点行）的标记跟随移动；
// - 左键点在远离两标记处：把所有左标记设到点击处并抓起拖动，锚定当前结束点
//   （结束固定、拖动调开始；点在结束点右侧时 CheckMarkers 交换角色，
//   行变为 [原结束, 点击处]，拖动即扫选新范围）；
//   拖过锚点时选区跟随交换（开始/结束互换，min/max 归一）；
// - 左键点在标记附近（抓取灵敏度内）：抓取最近的标记；Ctrl 同时抓取所有行上共位的标记
//   （拖动共享边界）；左标记被点击时立即移到点击处，右标记只在拖动时移动；
// - 右键：把右标记设到点击处并抓起（设结束点；越过原开始点时 CheckMarkers
//   交换角色，行变为 [点击处, 原开始]）；
// - Alt+拖：整段平移（活动行 + 多选行的两端一起移动）；
// - 吸附（Shift 反转开关）：把被抓标记整体吸到其他行边界上（10px 内）；
// - 改动先进 pending（未提交预览），手动提交（G/Enter）才写入行并算一步撤销；
//   自动提交开启时每次改动立即写入（撤销合并）；切换行丢弃未提交改动（revert）；
// - 前置/延后：开始 -200ms / 结束 +300ms（Aegisub 默认 lead-in/lead-out）。

export const LEAD_IN_MS = 200;
export const LEAD_OUT_MS = 300;
export const NUDGE_MS = 10; // Aegisub ModifyLength/ModifyStart 的步长

// 非活动行显示模式（Aegisub Audio/Inactive Lines Display Mode）
export const INACTIVE_MODE = { NONE: 0, PREV: 1, PREV_NEXT: 2, ALL: 3 };

const APPLY_EPS = 0.001; // 写入行时保证 end > start 的最小间隔（秒）

export function createTimingController(deps) {
  // deps: {
  //   cues: () => [{id, start, end}]        已提交且按时间排序的行
  //   activeId: () => id|null               锚点行（当前定时行）
  //   selectedIds: () => id[]               多选行
  //   inactiveMode: () => 0|1|2|3
  //   autoCommit: () => bool
  //   dragTiming: () => bool                关闭时扫选拖动开始标记而非结束标记
  //   onApply: (entries, {auto}) => void    写入行（bulk）；auto=true 时合并撤销
  //   onChange: () => void                  预览变化（重绘）
  //   duration: () => number|null           可选，音频总时长：Alt 平移钳制上界（缺省只钳下界 0）
  // }
  const pending = new Map(); // id -> {start,end}（未提交的预览时间）

  const cueById = (id) => deps.cues().find((c) => c.id === id) ?? null;

  // id→cue 索引：批量场景（lineView）每次调用重建一次，避免每行 O(n) find
  const cueMap = () => new Map(deps.cues().map((c) => [c.id, c]));

  // 行的生效时间：pending 优先，否则用已提交值；byId 传 cueMap() 预建索引
  function effIn(byId, id) {
    const p = pending.get(id);
    if (p) return { ...p };
    const cue = byId.get(id);
    return cue ? { start: cue.start, end: cue.end } : null;
  }

  function eff(id) {
    return effIn(cueMap(), id);
  }

  // 参与定时的行：锚点行 + 多选行（不含锚点）
  function selectedLines() {
    const activeId = deps.activeId();
    const wanted = new Set(deps.selectedIds());
    wanted.delete(activeId);
    return deps.cues().filter((c) => wanted.has(c.id));
  }

  // 非活动行（仅用于显示与吸附目标，不可直接拖动）
  function inactiveLines() {
    const cues = deps.cues();
    const activeId = deps.activeId();
    const idx = cues.findIndex((c) => c.id === activeId);
    const excluded = new Set([activeId]);
    selectedLines().forEach((c) => excluded.add(c.id));
    const pick = (cue) => (cue && !excluded.has(cue.id) ? [cue] : []);
    switch (deps.inactiveMode()) {
      case INACTIVE_MODE.PREV:
        return idx > 0 ? pick(cues[idx - 1]) : [];
      case INACTIVE_MODE.PREV_NEXT:
        return [...(idx > 0 ? pick(cues[idx - 1]) : []), ...(idx >= 0 && idx + 1 < cues.length ? pick(cues[idx + 1]) : [])];
      case INACTIVE_MODE.ALL:
        return cues.filter((c) => !excluded.has(c.id));
      default:
        return [];
    }
  }

  // 全部可定时行（活动 + 多选）及其生效范围
  function timeables() {
    const active = cueById(deps.activeId());
    const rest = selectedLines();
    return active ? [active, ...rest] : rest;
  }

  function markPending(id, start, end) {
    const cur = eff(id);
    if (!cur) return false;
    if (cur.start === start && cur.end === end) return false;
    pending.set(id, { start, end });
    return true;
  }

  // 写入某一端的标记位置（另一端固定），选区按 min/max 归一（Aegisub CheckMarkers 交换）
  function writeMarker(id, edge, pos) {
    const cur = eff(id);
    if (!cur) return false;
    const start = edge === 'left' ? pos : cur.start;
    const end = edge === 'right' ? pos : cur.end;
    return markPending(id, Math.min(start, end), Math.max(start, end));
  }

  function writeRange(id, start, end) {
    return markPending(id, Math.min(start, end), Math.max(start, end));
  }

  // 每批标记改动后调用：自动提交开启时立即落库（撤销合并），否则通知预览重绘
  function flush() {
    if (!pending.size) return;
    if (deps.autoCommit()) commitPending({ auto: true });
    else deps.onChange();
  }

  // 吸附：把各被拖位置整体移向最近的「其他行边界」，距离限制 snapMs 内（SnapMarkers）。
  // 与 Aegisub 一致：所有被抓标记应用同一个位移。
  function movedKeySet(items) {
    const keys = new Set();
    items.forEach(({ id, edge }) => {
      if (edge) keys.add(`${id}:${edge}`);
    });
    return keys;
  }

  function snap(positions, movedKeys, snapMs) {
    if (snapMs <= 0 || !positions.length) return null;
    const candidates = [];
    const collect = (line) => {
      if (!line) return;
      const t = eff(line.id);
      if (!t) return;
      if (!movedKeys.has(`${line.id}:left`)) candidates.push(t.start);
      if (!movedKeys.has(`${line.id}:right`)) candidates.push(t.end);
    };
    collect(cueById(deps.activeId()));
    selectedLines().forEach(collect);
    inactiveLines().forEach(collect);
    let best = null;
    let bestDist = snapMs;
    for (const p of positions) {
      for (const c of candidates) {
        const dist = Math.abs(c - p);
        if (dist > 0 && dist <= bestDist) {
          bestDist = dist;
          best = c - p;
        }
      }
    }
    return best;
  }

  // ---------- Aegisub: AudioTimingController 接口 ----------

  function isNearbyMarker(pos, sensitivity, altDown) {
    if (altDown) return true;
    const active = cueById(deps.activeId());
    if (!active) return false;
    const t = eff(active.id);
    return Math.abs(t.start - pos) <= sensitivity || Math.abs(t.end - pos) <= sensitivity;
  }

  // 返回拖动会话（非空表示按下后进入拖拽），可能已写入一次位置
  function onLeftClick(pos, { ctrl = false, alt = false } = {}, sensitivity, snapRange) {
    const active = cueById(deps.activeId());
    if (!active) return null;

    // Alt：平移活动行 + 多选行（两端一起动）
    if (alt) {
      const items = timeables().map((c) => {
        const t = eff(c.id);
        return { id: c.id, left: t.start, right: t.end };
      });
      return { mode: 'pan', clickedPos: pos, items }; // 平移基准（秒）
    }

    const t = eff(active.id);
    const distL = Math.abs(t.start - pos);
    const distR = Math.abs(t.end - pos);

    // 点在两个标记都远的位置：左键 = 把开始标记设到点击处（Aegisub SetPosition），
    // 随后 CheckMarkers 交换两端角色：点击越过原结束点时行变为 [原结束, 点击处]
    // （不并拢成零长选区），未越过时即 [点击处, 原结束]——两种情况统一为
    // 「选区 = 点击处与原结束点之间」。拖拽锚点取交换后的结束点：越过时锚=点击处
    // （拖动即重新扫选新范围），否则锚=原结束点（结束固定、拖动调开始，
    // 多次左键扫选时结束时间不跟随）。dragTiming 关闭时抓的是开始标记，锚不变。
    if (distL > sensitivity && distR > sensitivity) {
      const lines = timeables();
      // 吸附候选排除活动行自身两端：点击落在结束点右侧 3~10px 环带时，
      // 开始点不得吸到自身结束点（否则写成零长选区）。
      // snap() 返回位移量，须叠加在点击处之上（?? 0 即不吸附时落点击处）
      const own = new Set([`${active.id}:left`, `${active.id}:right`]);
      const snapped = pos + (snap([pos], own, snapRange) ?? 0);
      lines.forEach((c) => {
        const t = eff(c.id);
        writeRange(c.id, Math.min(snapped, t.end), Math.max(snapped, t.end));
      });
      const edge = deps.dragTiming() === false ? 'left' : 'right';
      const items = lines.map((c) => ({ id: c.id, edge, other: eff(c.id).end }));
      const session = { mode: 'marker', items };
      flush();
      return session;
    }

    // 点在标记附近：抓取较近者；Ctrl 扩大到所有行上共位的标记（含非活动行）
    const clickedEdge = distL <= distR ? 'left' : 'right';
    const clickedPos = clickedEdge === 'left' ? t.start : t.end;
    const grabbed = [{ id: active.id, edge: clickedEdge }];
    if (ctrl) {
      const wantPos = clickedPos;
      const scan = (c) => {
        if (!c || c.id === active.id) return;
        const ct = eff(c.id);
        if (Math.abs(ct.start - wantPos) < 1e-9) grabbed.push({ id: c.id, edge: 'left' });
        if (Math.abs(ct.end - wantPos) < 1e-9) grabbed.push({ id: c.id, edge: 'right' });
      };
      selectedLines().forEach(scan);
      inactiveLines().forEach(scan);
    }

    // 左标记被点击时立即移到点击处（含吸附）；右标记只在拖动时移动。
    // 吸附以点击处（即将写入的位置）为基准、位移叠加其上，而非标记原位置
    if (clickedEdge === 'left') {
      const snapped = pos + (snap([pos], movedKeySet(grabbed), snapRange) ?? 0);
      grabbed.forEach((item) => writeMarker(item.id, 'left', snapped));
    }

    const session = {
      mode: 'marker',
      items: grabbed.map((item) => {
        const it = eff(item.id);
        return { id: item.id, edge: item.edge, other: item.edge === 'left' ? it.end : it.start };
      }),
    };
    flush();
    return session;
  }

  function onRightClick(pos, snapRange) {
    const active = cueById(deps.activeId());
    if (!active) return null;
    const lines = timeables();
    // snap() 返回位移量，叠加在点击处之上（同 onLeftClick 的 SetPosition 语义）
    const snapped = pos + (snap([pos], new Set(), snapRange) ?? 0);
    // 右键 = 把结束标记设到点击处（Aegisub SetPosition），CheckMarkers 交换角色：
    // 点击越过原开始点时行变为 [点击处, 原开始]（不并拢成零长选区），
    // 否则即 [原开始, 点击处]——选区 = 点击处与原开始点之间。
    // 拖拽锚点取原开始点（Aegisub 抓的是右标记对象，交换后被拖标记的另一端恒为原开始）。
    const items = lines.map((c) => {
      const start = eff(c.id).start;
      writeRange(c.id, Math.min(start, snapped), Math.max(start, snapped));
      return { id: c.id, edge: 'right', other: start };
    });
    const session = { mode: 'marker', items };
    flush();
    return session;
  }

  function onMarkerDrag(session, pos, snapRange) {
    if (!session) return;
    if (session.mode === 'pan') {
      // Alt 平移：每步按增量移动当前生效位置（Aegisub clicked_ms 增量语义）。
      // 平移量钳制使平移后 start>=0 且（duration 可用时）end<=duration，
      // 否则提交时会被钳成变形区间。
      const rawDur = deps.duration?.() ?? 0;
      const dur = Number.isFinite(rawDur) && rawDur > 0 ? rawDur : null;
      const clampShift = (s) => {
        let out = s;
        if (out < 0) {
          const minStart = Math.min(...session.items.map((item) => item.left));
          out = Math.max(out, -minStart);
        }
        if (out > 0 && dur != null) {
          const maxEnd = Math.max(...session.items.map((item) => item.right));
          out = Math.min(out, dur - maxEnd);
        }
        return out;
      };
      const raw = pos - session.clickedPos;
      const shift = clampShift(raw);
      // 保留未生效的越界量：贴边停住后指针回移可立即继续平移
      session.clickedPos = pos - (raw - shift);
      session.items.forEach((item) => {
        writeRange(item.id, item.left + shift, item.right + shift);
        item.left += shift;
        item.right += shift;
      });
      const positions = session.items.flatMap((item) => {
        const t = eff(item.id);
        return t ? [t.start, t.end] : [];
      });
      const delta = snap(positions, new Set(), snapRange);
      if (delta) {
        const d = clampShift(delta); // 吸附位移同样不得把行推出边界
        if (d) {
          session.items.forEach((item) => {
            const t = eff(item.id);
            writeRange(item.id, t.start + d, t.end + d);
          });
          session.items.forEach((item) => {
            item.left += d;
            item.right += d;
          });
        }
      }
      flush();
      return;
    }
    // 被拖的物理标记始终跟随指针；另一端固定在抓取时的位置（session.other）。
    // 选区按 min/max 归一——拖过另一端即交换（Aegisub CheckMarkers）。
    session.items.forEach((item) => {
      writeRange(item.id, Math.min(item.other, pos), Math.max(item.other, pos));
    });
    // 吸附候选排除：被拖动的边（声明端）+ 活动行自身另一端。扫选拖过锚点后
    // 指针值会落到另一端上，两端都不得成为自身的吸附目标（避免自我回拉/零长）
    const movedKeys = movedKeySet(session.items);
    const activeId = deps.activeId();
    const mine = activeId != null ? session.items.find((item) => item.id === activeId) : null;
    if (mine) movedKeys.add(`${activeId}:${mine.edge === 'left' ? 'right' : 'left'}`);
    const delta = snap([pos], movedKeys, snapRange);
    if (delta) {
      const snappedPos = pos + delta;
      session.items.forEach((item) => {
        writeRange(item.id, Math.min(item.other, snappedPos), Math.max(item.other, snappedPos));
      });
    }
    flush();
  }

  // 会话状态由显示层（waveform.js）持有：无内部状态需清理，保留以满足既有调用
  function endDrag() {}

  // ---------- Aegisub: AddLeadIn / AddLeadOut / ModifyLength / ModifyStart ----------

  function withActive(fn) {
    const active = cueById(deps.activeId());
    if (!active) return false;
    fn(active);
    flush();
    return true;
  }

  function addLeadIn() {
    return withActive((active) => {
      const t = eff(active.id);
      writeMarker(active.id, 'left', t.start - LEAD_IN_MS / 1000);
    });
  }

  function addLeadOut() {
    return withActive((active) => {
      const t = eff(active.id);
      writeMarker(active.id, 'right', t.end + LEAD_OUT_MS / 1000);
    });
  }

  // delta: ±1（每步 NUDGE_MS）；缩短不许越过开始点
  function modifyLength(delta) {
    return withActive((active) => {
      const t = eff(active.id);
      writeMarker(active.id, 'right', Math.max(t.end + delta * (NUDGE_MS / 1000), t.start));
    });
  }

  function modifyStart(delta) {
    return withActive((active) => {
      const t = eff(active.id);
      writeMarker(active.id, 'left', Math.min(t.start + delta * (NUDGE_MS / 1000), t.end));
    });
  }

  // ---------- 提交 / 放弃 ----------

  function hasPending() {
    return pending.size > 0;
  }

  function pendingEntries() {
    const entries = [];
    pending.forEach((t, id) => {
      entries.push({
        id,
        start: Math.max(0, t.start),
        end: Math.max(Math.max(0, t.start) + APPLY_EPS, t.end),
      });
    });
    return entries;
  }

  // 手动提交：返回待写入行并清空 pending（调用方负责落库）
  function commitPending({ auto = false } = {}) {
    if (!pending.size) return [];
    const entries = pendingEntries();
    pending.clear();
    if (auto) deps.onApply(entries, { auto: true });
    else deps.onChange();
    return entries;
  }

  function revert() {
    if (!pending.size) return false;
    pending.clear();
    deps.onChange();
    return true;
  }

  // ---------- 查询 ----------

  function getPrimaryRange() {
    const active = cueById(deps.activeId());
    if (!active) return null;
    return eff(active.id);
  }

  // 渲染用行视图（全部为生效时间）；用预建索引避免数千行时 O(n²) 查找
  function lineView() {
    const byId = cueMap();
    const activeId = deps.activeId();
    const active = activeId ? effIn(byId, activeId) : null;
    return {
      active: active && activeId ? { id: activeId, ...active } : null,
      selected: selectedLines().map((c) => ({ id: c.id, ...effIn(byId, c.id) })),
      inactive: inactiveLines().map((c) => ({ id: c.id, ...effIn(byId, c.id) })),
    };
  }

  return {
    isNearbyMarker,
    onLeftClick,
    onRightClick,
    onMarkerDrag,
    endDrag,
    addLeadIn,
    addLeadOut,
    modifyLength,
    modifyStart,
    hasPending,
    pendingEntries,
    commitPending,
    revert,
    getPrimaryRange,
    lineView,
    eff,
    // 调试/测试：未提交预览的快照
    pendingSnapshot: () => Object.fromEntries(pending),
  };
}
