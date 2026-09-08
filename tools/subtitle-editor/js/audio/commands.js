// Aegisub 音频命令层（command/audio.cpp + command/time.cpp 语义）。
// 所有播放范围都取「生效时间」（含未提交的 pending），与 Aegisub 一致；
// 提交写入行并算一步撤销，「提交后自动转至下一行」开启时前进到下一行（不定位、不播放）。
const PLAY_WINDOW = 0.5; // Q/W/E/D 的 500ms

export function createAudioCommands({ store, actions, player, timing, waveform, flushEdits }) {
  function primaryRange() {
    const range = timing.getPrimaryRange();
    if (range) return range;
    // 无活动行时退而求其次：播放头所在行（无字幕则无范围）
    const cue = store.cueAt(player.currentTime());
    return cue ? { start: cue.start, end: cue.end } : null;
  }

  function clampEnd(t) {
    return Math.min(t, store.state.duration || t);
  }

  const commands = {
    // audio/play/selection（S；音频盒内 Space；小键盘 5）
    'play/selection': () => {
      const r = primaryRange();
      if (r) player.playRange(r.start, r.end);
    },
    // audio/play/line（R）：对话模式下与播放选区同范围
    'play/line': () => {
      const r = primaryRange();
      if (r) player.playRange(r.start, r.end);
    },
    // audio/stop（H；小键盘 8）：Aegisub 同时停视频控制器——回到本次播放起点
    'stop': () => {
      player.stop();
    },
    // audio/play/toggle（B）：Aegisub 停止分支只停音频（原地暂停，不回退）
    'play/toggle': () => {
      if (player.isPlaying()) player.stopAudition();
      else commands['play/selection']();
    },
    // audio/play/selection/before（Q；小键盘 1）
    'play/selection/before': () => {
      const r = primaryRange();
      if (r) player.playRange(Math.max(0, r.start - PLAY_WINDOW), r.start);
    },
    // audio/play/selection/after（W；小键盘 3）
    'play/selection/after': () => {
      const r = primaryRange();
      if (r) player.playRange(r.end, clampEnd(r.end + PLAY_WINDOW));
    },
    // audio/play/selection/begin（E）
    'play/selection/begin': () => {
      const r = primaryRange();
      if (r) player.playRange(r.start, r.start + Math.min(PLAY_WINDOW, r.end - r.start));
    },
    // audio/play/selection/end（D）
    'play/selection/end': () => {
      const r = primaryRange();
      if (r) player.playRange(r.end - Math.min(PLAY_WINDOW, r.end - r.start), r.end);
    },
    // audio/play/to_end（T）
    'play/to_end': () => {
      const r = primaryRange();
      if (r) player.playRange(r.start, clampEnd(store.state.duration || r.end));
    },
    // time/lead/in（C）：开始点提前 200ms
    'time/lead/in': () => {
      timing.addLeadIn();
    },
    // time/lead/out（V）：结束点延后 300ms
    'time/lead/out': () => {
      timing.addLeadOut();
    },
    // time/length/increase|decrease（小键盘 + / -）：每步 10ms
    'time/length/increase': () => {
      timing.modifyLength(1);
    },
    'time/length/decrease': () => {
      timing.modifyLength(-1);
    },
    // 小键盘 4/6/7/9 的微调：dir ±1 步 = ModifyStart/ModifyLength ±10ms，
    // 与 Aegisub 及小键盘 +/- 完全同步（对话模式下无粗/细两挡之分）
    'time/start/nudge': (dir) => {
      timing.modifyStart(dir);
    },
    'time/length/nudge': (dir) => {
      timing.modifyLength(dir);
    },
    // time/prev（←/Z）
    'time/prev': () => {
      actions.selectNeighbour(-1, player.currentTime());
    },
    // time/next（→/X）
    'time/next': () => {
      actions.selectNeighbour(1, player.currentTime());
    },
    // audio/commit（G；音频盒内 Enter；小键盘 Enter）
    'commit': () => {
      flushEdits?.();
      const entries = timing.commitPending();
      if (entries.length) actions.updateCueTimesBulk(entries);
      if (store.state.audioOptions.autoNext) actions.selectNeighbour(1, player.currentTime());
    },
    // audio/commit/default（Shift+G）：提交后新建一行，默认时长 2s，起点为当前结束点。
    // 一次性 coalesceKey 把「提交改动 / 插入新行 / 新行定时」三步写库合并为一步撤销
    'commit/default': () => {
      flushEdits?.();
      const ref = store.selectedCue();
      const start = timing.getPrimaryRange()?.end ?? ref?.end;
      const entries = timing.commitPending();
      const key = Symbol('commit-default');
      if (entries.length) actions.updateCueTimesBulk(entries, { coalesceKey: key });
      if (!ref || start == null) return;
      // 新行时长钳到媒体末尾内（duration 未知时不钳上界），且保证 end > start
      const dur = store.state.duration || 0;
      const end = Math.max(start + 0.05, Math.min(start + 2, dur || Infinity));
      const cue = actions.insertRelativeTo(ref.id, { after: true, videoTime: start, coalesceKey: key });
      if (cue) actions.updateCueTimes(cue.id, start, end, { coalesceKey: key });
    },
    // audio/commit/stay：仅提交不前进
    'commit/stay': () => {
      flushEdits?.();
      const entries = timing.commitPending();
      if (entries.length) actions.updateCueTimesBulk(entries);
    },
    // audio/go_to：滚动视图使选区可见
    'go_to': () => {
      waveform?.scrollToSelection();
    },
    // audio/scroll/left|right（A/F）：Aegisub 固定滚动 128 像素（audio.cpp）
    'scroll/left': () => {
      waveform?.scrollByPixels(-128);
    },
    'scroll/right': () => {
      waveform?.scrollByPixels(128);
    },
  };

  function run(name, ...args) {
    const fn = commands[name];
    if (!fn) return false;
    fn(...args);
    return true;
  }

  return { run, primaryRange };
}
