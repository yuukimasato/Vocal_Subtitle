// 全局键盘，按 Aegisub 的键位上下文分层（hotkey::check 的 Audio / Always 上下文）：
// - Always：小键盘时间轴键位（仅 Numpad 系列，对齐 Aegisub 3.2 默认；不在输入框内即生效，NumLock 任意状态，e.code 识别）；
// - Audio：音频盒（波形区）持有焦点时接管一套 Aegisub 默认键位（S/Q/W/E/D/R/T/H/B/G/C/V/Z/X/A/F/←/→/Space/Enter）；
// - 其余全局键位（视频播放、行编辑）在音频盒无焦点时保持原样。
// 输入框聚焦时放行浏览器原生编辑行为（含 Ctrl+Z/Y 文本撤销；Ctrl+D/M 等行操作不劫持）；
// <dialog> 打开时全局键位全部让路。

import { showToast } from './ui/toast.js';

const EDITING_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

export function initShortcuts({ store, actions, player, commands, waveform, exportCurrent, flushEdits, seekStep }) {
  window.addEventListener('keydown', (event) => {
    // 模态 <dialog>（帮助/关于/粘贴）打开时全部让路：Enter 关闭、Esc 原生处理、
    // 输入框内打字不被 Space/方向键劫持。
    // 除 target 判定外再查一次 open 状态，兜住焦点意外落在 body 的边缘情况
    if (event.target?.closest?.('dialog') || document.querySelector('dialog[open]')) return;
    // 只认 Ctrl/Cmd：单按 Alt 或 Alt+X 误入 handleCombo 会误触发删行/拆分/撤销
    if (event.ctrlKey || event.metaKey) {
      handleCombo(event, { store, actions, player, exportCurrent, flushEdits });
      return;
    }
    const target = event.target;
    if (target && (EDITING_TAGS.has(target.tagName) || target.isContentEditable)) return;
    handlePlain(event, { store, actions, player, commands, waveform, seekStep });
  });
}

function inEditingField(event) {
  const target = event.target;
  return Boolean(target && (EDITING_TAGS.has(target.tagName) || target.isContentEditable));
}

async function pasteFromClipboard({ store, actions, player, flushEdits }) {
  flushEdits?.();
  const ids = store.state.selectedIds ?? [];
  const refId = ids[ids.length - 1] ?? null;
  try {
    const inserted = await actions.pasteCues({ refId, after: true, videoTime: player.currentTime() });
    if (!inserted) showToast('剪贴板中没有可粘贴的字幕行', 'error');
  } catch (err) {
    showToast(err?.message ?? '粘贴失败', 'error'); // 残缺字幕剪贴板的解析错误
  }
}

function handleCombo(event, { store, actions, player, exportCurrent, flushEdits }) {
  const code = event.code;
  const shift = event.shiftKey;

  // 撤销/重做/导出在非输入焦点下都接管；剪贴板与全选在输入框内交给浏览器原生行为；
  // 行破坏性操作（拆分/合并）在输入框内一律屏蔽
  switch (code) {
    case 'KeyZ':
      // 输入框内放行浏览器原生文本撤销/重做：若走全局 flush+undo，焦点输入框
      // 不会被回写，继续输入会把刚撤销的内容重新提交（无声逆转撤销）
      if (inEditingField(event)) return;
      event.preventDefault();
      flushEdits?.();
      if (shift) actions.redo();
      else actions.undo();
      return;
    case 'KeyY':
      if (inEditingField(event)) return;
      event.preventDefault();
      flushEdits?.();
      actions.redo();
      return;
    case 'KeyS':
      event.preventDefault();
      flushEdits?.();
      exportCurrent?.();
      return;
    case 'KeyX':
      if (inEditingField(event)) return;
      event.preventDefault();
      flushEdits?.();
      actions.cutCues(store.state.selectedIds);
      return;
    case 'KeyC':
      if (inEditingField(event)) return;
      event.preventDefault();
      flushEdits?.();
      actions.copyCues(store.state.selectedIds);
      return;
    case 'KeyV':
      if (inEditingField(event)) return;
      event.preventDefault();
      pasteFromClipboard({ store, actions, player, flushEdits });
      return;
    case 'KeyA':
      if (inEditingField(event)) return;
      event.preventDefault();
      actions.selectAll();
      return;
    case 'KeyD': {
      if (inEditingField(event)) return;
      event.preventDefault();
      const cue = store.selectedCue() ?? store.cueAt(player.currentTime());
      if (cue) actions.splitCue(cue.id, player.currentTime());
      return;
    }
    case 'KeyM': {
      if (inEditingField(event)) return;
      event.preventDefault();
      const cue = store.selectedCue() ?? store.cueAt(player.currentTime());
      if (cue) actions.mergeWithNext(cue.id);
      return;
    }
    default:
  }
}

// Aegisub「Audio」上下文：音频盒持有焦点时的键位（default_hotkey.json Audio 组）。
// 返回 true 表示按键已被音频盒消费。
function handleAudioContext(event, { commands }) {
  const code = event.code;
  const shift = event.shiftKey;
  const run = (name, ...args) => {
    event.preventDefault();
    commands.run(name, ...args);
    return true;
  };

  switch (code) {
    case 'KeyS':
    case 'Space':
      return run('play/selection');
    case 'KeyQ':
      return run('play/selection/before');
    case 'KeyW':
      return run('play/selection/after');
    case 'KeyE':
      return run('play/selection/begin');
    case 'KeyD':
      return run('play/selection/end');
    case 'KeyR':
      return run('play/line');
    case 'KeyT':
      return run('play/to_end');
    case 'KeyB':
      return run('play/toggle');
    case 'KeyH':
      return run('stop');
    case 'KeyG':
      return run(shift ? 'commit/default' : 'commit');
    case 'KeyC':
      return run('time/lead/in');
    case 'KeyV':
      return run('time/lead/out');
    case 'KeyZ':
      return run('time/prev');
    case 'KeyX':
      return run('time/next');
    case 'KeyA':
      return run('scroll/left');
    case 'KeyF':
      return run('scroll/right');
    case 'ArrowLeft':
      return run('time/prev');
    case 'ArrowRight':
      return run('time/next');
    case 'Enter':
      return run('commit');
    default:
      return false;
  }
}

function handlePlain(event, { store, actions, player, commands, waveform, seekStep }) {
  const code = event.code;
  const shift = event.shiftKey;
  const audioActive = waveform?.isAudioContextActive?.() ?? false;
  // Alt 组合不映射任何键位（Alt+拖动是波形整段平移，键盘侧无 Alt 键位），避免误触发
  if (event.altKey) return;

  // 音频盒小键盘键位（Aegisub Always 上下文 + 小键盘微调），任何焦点下可用
  switch (code) {
    case 'NumpadEnter':
      event.preventDefault();
      commands.run('commit');
      return;
    case 'Numpad5':
      event.preventDefault();
      commands.run('play/selection');
      return;
    case 'Numpad1':
      event.preventDefault();
      commands.run('play/selection/before');
      return;
    case 'Numpad3':
      event.preventDefault();
      commands.run('play/selection/after');
      return;
    case 'Numpad8':
      event.preventDefault();
      commands.run('stop');
      return;
    case 'NumpadAdd':
      event.preventDefault();
      commands.run('time/length/increase'); // +10ms
      return;
    case 'NumpadSubtract':
      event.preventDefault();
      commands.run('time/length/decrease'); // -10ms
      return;
    case 'Numpad4':
      // Aegisub：ModifyStart ±1 步 = ±10ms，无粗/细两挡（Shift 无绑定）
      event.preventDefault();
      commands.run('time/start/nudge', -1);
      return;
    case 'Numpad6':
      event.preventDefault();
      commands.run('time/start/nudge', 1);
      return;
    case 'Numpad7':
      event.preventDefault();
      commands.run('time/length/nudge', -1); // 缩短 10ms
      return;
    case 'Numpad9':
      event.preventDefault();
      commands.run('time/length/nudge', 1);
      return;
    case 'Numpad2':
      // Aegisub time/next：只切换活动行，不动播放位置
      event.preventDefault();
      actions.selectNeighbour(+1, player.currentTime());
      return;
    case 'Numpad0':
      // Aegisub time/prev：只切换活动行，不动播放位置
      event.preventDefault();
      actions.selectNeighbour(-1, player.currentTime());
      return;
    default:
      break;
  }

  // 音频盒持有焦点：Aegisub Audio 上下文键位优先
  if (audioActive && handleAudioContext(event, { commands })) return;

  switch (code) {
    case 'Space':
      event.preventDefault();
      player.playPause();
      return;
    case 'Enter':
      event.preventDefault();
      document.activeElement?.blur?.();
      player.stopAudition();
      return;
    case 'KeyN': {
      event.preventDefault();
      const cue = actions.insertAtTime(player.currentTime());
      if (cue) actions.setEditing(cue.id);
      return;
    }
    case 'Delete':
    case 'Backspace': {
      const ids = store.state.selectedIds ?? [];
      if (ids.length) {
        event.preventDefault();
        actions.removeCues(ids);
      }
      return;
    }
    case 'KeyL':
      event.preventDefault();
      store.patch({ loopCue: !store.state.loopCue });
      return;
    case 'KeyF':
      event.preventDefault();
      store.patch({ follow: !store.state.follow });
      return;
    case 'ArrowLeft':
    case 'ArrowRight': {
      event.preventDefault();
      // 步长随波形缩放自适应（约 1/4 可见窗口，5~30s）；Shift 细步
      const step = Math.max(1, seekStep ? seekStep() : 5);
      const delta = shift ? Math.max(0.5, step / 5) : step;
      player.seek(player.currentTime() + (code === 'ArrowLeft' ? -delta : delta));
      return;
    }
    case 'Comma':
      event.preventDefault();
      player.frameStep(-1);
      return;
    case 'Period':
      event.preventDefault();
      player.frameStep(+1);
      return;
    default:
  }
}
