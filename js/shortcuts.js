// 全局键盘。时间轴键位为小键盘方案（REAPER 打轴式），主键盘数字排做镜像。
// 用 e.code 识别，不受 NumLock / 输入法影响。输入框聚焦时仅放行编辑器自身按键。

const EDITING_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

export function initShortcuts({ store, actions, player, exportCurrent }) {
  window.addEventListener('keydown', (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) {
      handleCombo(event, { store, actions, player, exportCurrent });
      return;
    }
    const target = event.target;
    if (target && (EDITING_TAGS.has(target.tagName) || target.isContentEditable)) return;
    handlePlain(event, { store, actions, player });
  });
}

function handleCombo(event, { store, actions, player, exportCurrent }) {
  const code = event.code;
  const shift = event.shiftKey;
  if (code === 'KeyZ') {
    event.preventDefault();
    if (shift) actions.redo();
    else actions.undo();
    return;
  }
  if (code === 'KeyY') {
    event.preventDefault();
    actions.redo();
    return;
  }
  if (code === 'KeyD') {
    event.preventDefault();
    const cue = currentCue({ store, player });
    if (cue) actions.splitCue(cue.id, player.currentTime());
    return;
  }
  if (code === 'KeyM') {
    event.preventDefault();
    const cue = currentCue({ store, player });
    if (cue) actions.mergeWithNext(cue.id);
    return;
  }
  if (code === 'KeyS') {
    event.preventDefault();
    exportCurrent?.();
  }
}

function currentCue({ store, player }) {
  return store.selectedCue() ?? store.cueAt(player.currentTime());
}

function handlePlain(event, { store, actions, player }) {
  const code = event.code;
  const shift = event.shiftKey;
  const fine = shift ? 0.01 : 0.1;

  // 小键盘时间轴键位 + 主键盘镜像
  switch (code) {
    case 'NumpadEnter':
    case 'Enter':
      event.preventDefault();
      document.activeElement?.blur?.();
      player.stopAudition();
      return;
    case 'Numpad5':
    case 'Digit5':
      event.preventDefault();
      player.playCurrentCue();
      return;
    case 'Numpad1':
    case 'Digit1':
      event.preventDefault();
      player.auditionBefore();
      return;
    case 'Numpad3':
    case 'Digit3':
      event.preventDefault();
      player.auditionAfter();
      return;
    case 'Numpad8':
    case 'Digit8':
      event.preventDefault();
      player.stop();
      return;
    case 'Numpad4':
    case 'Digit4': {
      event.preventDefault();
      const cue = currentCue({ store, player });
      if (cue) actions.nudge(cue.id, 'start', -fine);
      return;
    }
    case 'Numpad6':
    case 'Digit6': {
      event.preventDefault();
      const cue = currentCue({ store, player });
      if (cue) actions.nudge(cue.id, 'start', +fine);
      return;
    }
    case 'Numpad7':
    case 'Digit7': {
      event.preventDefault();
      const cue = currentCue({ store, player });
      if (cue) actions.nudge(cue.id, 'end', -fine);
      return;
    }
    case 'Numpad9':
    case 'Digit9': {
      event.preventDefault();
      const cue = currentCue({ store, player });
      if (cue) actions.nudge(cue.id, 'end', +fine);
      return;
    }
    case 'Numpad2':
    case 'Digit2': {
      event.preventDefault();
      const cue = actions.selectNeighbour(+1, player.currentTime());
      if (cue) player.seek(cue.start);
      return;
    }
    case 'Numpad0':
    case 'Digit0': {
      event.preventDefault();
      const cue = actions.selectNeighbour(-1, player.currentTime());
      if (cue) player.seek(cue.start);
      return;
    }
    default:
      break;
  }

  switch (code) {
    case 'Space':
      event.preventDefault();
      player.playPause();
      return;
    case 'KeyN': {
      event.preventDefault();
      const cue = actions.insertAtTime(player.currentTime());
      if (cue) actions.setEditing(cue.id);
      return;
    }
    case 'Delete':
    case 'Backspace': {
      const cue = store.selectedCue();
      if (cue) {
        event.preventDefault();
        actions.removeCue(cue.id);
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
      event.preventDefault();
      player.seek(player.currentTime() - (shift ? 1 : 5));
      return;
    case 'ArrowRight':
      event.preventDefault();
      player.seek(player.currentTime() + (shift ? 1 : 5));
      return;
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
