// 字幕表格（底部面板）：文本/时间即点即编、防抖自动提交与草稿保存、
// 多选（Ctrl/Shift）与右键行操作菜单、播放高亮与跟随滚动。
// 渲染采用行级增量更新，编辑中不整表重建，避免打断输入焦点。
import { formatClock, formatDuration, parseFlexibleTime } from '../format/time.js';
import { cueStyle } from '../format/index.js';
import { showToast } from './toast.js';

const TEXT_COMMIT_DELAY = 700;
const COLS = 7;

export function createCueList({
  store,
  actions,
  player,
  tbodyEl,
  wrapEl,
  countEl,
  toolsEl,
  statusEl,
  menu,
  pasteDialog,
}) {
  const rows = new Map(); // id → 行视图
  const pending = new Map(); // id → 文本字段防抖定时器
  const timeDirty = new Set(); // id → 时间字段已改动，待 blur/Enter 提交
  let orderedIds = [];
  let focusedId = null;
  let lastActiveId = null;
  let playerPlaying = false;
  let menuRefId = null;

  // ---------- 工具条 ----------
  // 按 id 获取，避免依赖 DOM 顺序的解构
  const insertBtn = document.getElementById('cue-insert');
  const deleteBtn = document.getElementById('cue-delete');
  const splitBtn = document.getElementById('cue-split');
  const mergeBtn = document.getElementById('cue-merge');
  const followBox = toolsEl.querySelector('input[type="checkbox"]');
  insertBtn.addEventListener('click', () => {
    const cue = actions.insertAtTime(player.currentTime());
    if (cue) actions.setEditing(cue.id); // 新建后直接进入文本输入
  });
  deleteBtn.addEventListener('click', () => actions.removeCues(store.state.selectedIds));
  splitBtn.addEventListener('click', () => {
    const cue = store.selectedCue();
    if (cue) actions.splitCue(cue.id, player.currentTime());
  });
  mergeBtn.addEventListener('click', () => {
    const cue = store.selectedCue();
    if (cue) actions.mergeWithNext(cue.id);
  });
  followBox.checked = store.state.follow;
  followBox.addEventListener('change', () => store.patch({ follow: followBox.checked }));
  store.on('follow', (s) => {
    followBox.checked = s.follow;
  });

  // 草稿自动保存状态提示
  store.on('savedAt', (s) => {
    if (statusEl && s.savedAt) {
      statusEl.textContent = `已自动保存 ${new Date(s.savedAt).toLocaleTimeString()}`;
    }
  });
  // 草稿保存失败（draft.js 配额不足时经 draftSaveFailedAt 通知）
  store.on('draftSaveFailedAt', () => {
    if (statusEl) statusEl.textContent = '草稿保存失败（存储配额不足？）';
  });

  // ---------- 编辑提交 ----------
  function autosize(el) {
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }

  function scheduleCommit(id) {
    clearTimeout(pending.get(id));
    pending.set(id, setTimeout(() => commitRow(id), TEXT_COMMIT_DELAY));
  }

  // 文本字段的防抖提交（只管文本；时间改走 blur/Enter 提交，避免半输入中间值写库）
  function commitRow(id) {
    clearTimeout(pending.get(id));
    if (!pending.delete(id)) return; // 该行没有等待提交的文本编辑
    const view = rows.get(id);
    const cue = store.state.cues.find((c) => c.id === id);
    if (!view || !cue) return;
    if (view.text.value !== cue.text) actions.updateCue(id, { text: view.text.value });
  }

  // 时间字段的提交：blur/Enter（或 flushEdits）时执行；无效值不落库（blur 校验已提示）
  function commitTimes(id) {
    if (!timeDirty.delete(id)) return;
    const view = rows.get(id);
    const cue = store.state.cues.find((c) => c.id === id);
    if (!view || !cue) return;
    const start = parseFlexibleTime(view.start.value);
    const end = parseFlexibleTime(view.end.value);
    if (start === null || end === null || end <= start) return;
    const patch = {};
    if (start !== cue.start) patch.start = start;
    if (end !== cue.end) patch.end = end;
    if (Object.keys(patch).length) actions.updateCue(id, patch);
  }

  // 外部操作前的统一落库入口：时间字段提交未 blur 的改动，文本字段照旧防抖提交
  function flushEdits() {
    [...timeDirty].forEach((id) => commitTimes(id));
    [...pending.keys()].forEach((id) => commitRow(id));
  }

  function validateTimeInput(input) {
    const text = input.value.trim();
    if (text !== '' && parseFlexibleTime(text) === null) {
      input.classList.add('invalid');
      showToast(`无法识别时间 "${input.value}"`, 'error');
      return false;
    }
    input.classList.remove('invalid');
    return true;
  }

  function checkTimes(id) {
    const view = rows.get(id);
    if (!view) return;
    const start = parseFlexibleTime(view.start.value);
    const end = parseFlexibleTime(view.end.value);
    if (start !== null && end !== null && end <= start) {
      view.end.classList.add('invalid');
      showToast('结束时间必须晚于开始时间', 'error');
    }
  }

  // ---------- 行构建 ----------
  function focusText(id) {
    const view = rows.get(id);
    if (!view) return;
    view.text.focus();
    const len = view.text.value.length;
    view.text.setSelectionRange(len, len);
  }

  function playCue(id) {
    const cue = store.state.cues.find((c) => c.id === id);
    if (!cue) return;
    actions.select(id);
    const end = cue.end > cue.start
      ? cue.end
      : Math.min(store.state.duration || cue.start + 2, cue.start + 2); // 零时长行试听 2s
    player.playRange(cue.start, end);
  }

  function buildRow(cue) {
    const id = cue.id;
    const tr = document.createElement('tr');
    tr.dataset.id = id;
    tr.className = 'cue-row';
    tr.innerHTML = `
      <td class="num mono"></td>
      <td class="time"><input class="edit-time mono" data-field="start" title="开始时间（回车或移开焦点提交）"></td>
      <td class="time"><input class="edit-time mono" data-field="end" title="结束时间（回车或移开焦点提交）"></td>
      <td class="dur mono"></td>
      <td class="style-cell"><select class="edit-style" title="ASS 样式（Aegisub 样式列）"></select></td>
      <td class="text-cell"><textarea class="edit-text" rows="1" title="字幕文本（修改后自动保存）"></textarea></td>
      <td class="ops"><button type="button" class="op play" title="播放此句">▶</button></td>`;
    const view = {
      tr,
      num: tr.querySelector('.num'),
      start: tr.querySelector('[data-field="start"]'),
      end: tr.querySelector('[data-field="end"]'),
      dur: tr.querySelector('.dur'),
      style: tr.querySelector('.edit-style'),
      text: tr.querySelector('.edit-text'),
      play: tr.querySelector('.op.play'),
    };
    view.style.addEventListener('change', () => {
      actions.updateCueStyle([id], view.style.value);
    });
    view.play.addEventListener('click', (e) => {
      e.stopPropagation();
      playCue(id);
    });
    [view.start, view.end].forEach((input) => {
      // 时间字段不做逐键防抖：停顿即提交会把半输入的中间值（如「0:0」=0s）写进数据，
      // 只在 blur/Enter 时经 commitTimes 落库
      input.addEventListener('input', () => {
        input.classList.remove('invalid');
        timeDirty.add(id);
      });
      input.addEventListener('blur', () => {
        validateTimeInput(input);
        commitTimes(id);
        checkTimes(id);
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          input.blur(); // 走 blur 流程统一提交并校验
        }
      });
    });
    view.text.addEventListener('input', () => {
      autosize(view.text);
      scheduleCommit(id);
    });
    view.text.addEventListener('blur', () => commitRow(id));
    tr.addEventListener('click', (e) => {
      if (e.target.closest('button, select')) return;
      const mode = e.shiftKey ? 'range' : (e.ctrlKey || e.metaKey) ? 'toggle' : 'single';
      actions.select(id, { mode });
      // 时间可能已被编辑，实时取最新值
      if (mode === 'single') {
        const fresh = store.state.cues.find((c) => c.id === id);
        if (fresh) player.seek(fresh.start);
      }
    });
    tr.addEventListener('dblclick', (e) => {
      if (e.target.closest('input, textarea, button, select')) return;
      focusText(id);
    });
    rows.set(id, view);
    return view;
  }

  // ---------- 样式列（ASS） ----------
  // 样式清单来自 subDoc.styles（解析 [V4+ Styles] 所得）；非 ASS 或无样式表时整列隐藏
  function styleCatalog() {
    const s = store.state;
    return s.subtitleFormat === 'ass' ? (s.subDoc?.styles ?? []) : [];
  }

  function syncStyleSelects() {
    const styles = styleCatalog();
    wrapEl.classList.toggle('hide-style', !styles.length);
    if (!styles.length) return;
    rows.forEach((view) => {
      const current = view.style.value;
      view.style.replaceChildren(...styles.map((name) => new Option(name, name)));
      if (current && !styles.includes(current)) view.style.appendChild(new Option(current, current));
      view.style.value = current;
    });
  }

  function paintStyle(view, cue) {
    if (pending.has(cue.id)) return;
    const value = cueStyle(cue);
    if (value === null) return;
    // 属性选择器对含特殊字符的样式名不可靠，直接遍历比较 value
    const exists = [...view.style.options].some((opt) => opt.value === value);
    if (!exists) {
      view.style.appendChild(new Option(value, value));
    }
    if (view.style.value !== value) view.style.value = value;
  }

  // ---------- 渲染（增量） ----------
  function updateRow(view, cue) {
    if (pending.has(cue.id)) return; // 编辑中的行等提交后再回写
    const active = document.activeElement;
    const startText = formatClock(cue.start);
    const endText = formatClock(cue.end);
    if (active !== view.start && view.start.value !== startText) view.start.value = startText;
    if (active !== view.end && view.end.value !== endText) view.end.value = endText;
    if (active !== view.text && view.text.value !== cue.text) {
      view.text.value = cue.text;
      autosize(view.text); // 仅文本变化时量高：每行一次强制重排，大表下开销显著
    }
    // 时间已是一致值（提交后回写）：清掉上次非法输入或起止冲突留下的红框，
    // 否则修正数据后红框会一直挂着（checkTimes 标的是另一个输入框）
    if (cue.end > cue.start) {
      view.start.classList.remove('invalid');
      view.end.classList.remove('invalid');
    }
    paintStyle(view, cue);
    view.dur.textContent = formatDuration(cue.end - cue.start);
  }

  function reconcile() {
    const cues = store.state.cues;
    const cueIds = new Set(cues.map((c) => c.id)); // 集合化避免逐行 some() 的 O(n²)

    rows.forEach((view, id) => {
      if (!cueIds.has(id)) {
        view.tr.remove();
        clearTimeout(pending.get(id));
        pending.delete(id);
        timeDirty.delete(id);
        rows.delete(id);
      }
    });
    cues.forEach((cue) => {
      let view = rows.get(cue.id);
      if (!view) {
        view = buildRow(cue);
        tbodyEl.appendChild(view.tr);
      }
      updateRow(view, cue);
    });

    const ids = cues.map((c) => c.id);
    const orderChanged = ids.length !== orderedIds.length
      || ids.some((id, i) => id !== orderedIds[i]);
    // 该行还在编辑（文本防抖或时间未 blur 提交）时推迟重排，提交后自然归位；
    // 移动 DOM 会让聚焦输入框 blur，避免半输入的时间值被立即提交
    if (orderChanged && !(focusedId && (pending.has(focusedId) || timeDirty.has(focusedId)))) {
      let cursor = tbodyEl.firstChild;
      ids.forEach((id) => {
        const tr = rows.get(id).tr;
        if (tr === cursor) {
          cursor = cursor.nextSibling;
          return;
        }
        tbodyEl.insertBefore(tr, cursor);
      });
      orderedIds = ids;
    }

    cues.forEach((cue, index) => {
      rows.get(cue.id).num.textContent = String(index + 1);
    });

    if (!cues.length) {
      if (!tbodyEl.querySelector('.cue-empty')) {
        tbodyEl.textContent = '';
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = COLS;
        td.className = 'cue-empty';
        td.textContent = '暂无字幕：拖入字幕文件、按 N 在播放头处新建，或右键插入新行';
        tr.appendChild(td);
        tbodyEl.appendChild(tr);
      }
    } else {
      tbodyEl.querySelector('.cue-empty')?.parentElement?.remove();
    }
    countEl.textContent = cues.length ? `共 ${cues.length} 条` : '';
    highlightActive(player.currentTime(), true);
  }

  function paintSelection() {
    const selected = new Set(store.state.selectedIds ?? []);
    rows.forEach((view, id) => {
      view.tr.classList.toggle('selected', selected.has(id));
    });
  }

  // ---------- 播放高亮与跟随 ----------
  function highlightActive(t, force) {
    const cue = store.cueAt(t);
    const id = cue?.id ?? null;
    if (id === lastActiveId && !force) return;
    if (lastActiveId) rows.get(lastActiveId)?.tr.classList.remove('active');
    if (id) {
      const view = rows.get(id);
      if (view) {
        view.tr.classList.add('active');
        if (playerPlaying && store.state.follow) view.tr.scrollIntoView({ block: 'nearest' });
      }
    }
    lastActiveId = id;
  }

  player.onTime((t, playing) => {
    playerPlaying = playing;
    highlightActive(t, false);
  });

  tbodyEl.addEventListener('focusin', (event) => {
    focusedId = event.target.closest('tr[data-id]')?.dataset.id ?? null;
  });
  tbodyEl.addEventListener('focusout', () => {
    focusedId = null;
  });

  // ---------- 右键行操作菜单 ----------
  // 捕获阶段记录右键目标行；点在多选行上时保留多选，否则单选该行。
  // 右键落在输入框内时不改选中、放行浏览器原生菜单（剪切/粘贴等）。
  const inEditField = (event) => Boolean(event.target.closest('input, textarea, select'));
  wrapEl.addEventListener(
    'contextmenu',
    (event) => {
      if (inEditField(event)) return;
      const tr = event.target.closest('tr[data-id]');
      if (tr) {
        menuRefId = tr.dataset.id;
        if (!(store.state.selectedIds ?? []).includes(menuRefId)) actions.select(menuRefId);
      } else {
        menuRefId = store.state.selectedId ?? store.state.cues.at(-1)?.id ?? null;
      }
    },
    true,
  );

  const tryPaste = (result) => {
    Promise.resolve(result)
      .then((res) => {
        // 兼容两种契约：false = 无可粘贴行；{ ok, skipped } = 部分行时间未粘贴
        if (!res) showToast('剪贴板中没有可粘贴的字幕行', 'error');
        else if (res?.skipped > 0) showToast(`${res.skipped} 行时间未粘贴：剪贴板时间非法，或与目标行保留的时间冲突`, 'warn');
      })
      .catch((err) => showToast(err?.message ?? '粘贴失败', 'error'));
  };
  // 选择性粘贴：纯文本剪贴板（无时间轴）直接按行新建字幕行，不弹字段对话框；
  // 字幕格式剪贴板才进入字段覆盖对话框。
  const smartPasteSpecial = async (refId) => {
    let kind = null;
    try {
      kind = await actions.clipboardKind();
    } catch (err) {
      showToast(err?.message ?? '无法读取剪贴板', 'error');
      return;
    }
    if (!kind) {
      showToast('剪贴板中没有可粘贴的字幕行', 'error');
      return;
    }
    if (kind === 'text') {
      tryPaste(actions.pasteCues({ refId, after: true, videoTime: player.currentTime() }));
      return;
    }
    openPasteDialog(refId);
  };

  function buildMenuItems() {
    const refId = menuRefId;
    const inSelection = refId && (store.state.selectedIds ?? []).includes(refId);
    const selIds = inSelection ? store.state.selectedIds : refId ? [refId] : [];
    const hasRef = Boolean(refId);
    const insert = (opts) => {
      const cue = actions.insertRelativeTo(refId, opts);
      if (cue) actions.setEditing(cue.id);
    };
    return [
      { label: '插入(之前)', enabled: hasRef, onClick: () => insert({ after: false }) },
      { label: '插入(之后)', enabled: hasRef, onClick: () => insert({ after: true }) },
      { label: '以视频时间插入(之前)', onClick: () => insert({ after: false, videoTime: player.currentTime() }) },
      { label: '以视频时间插入(之后)', onClick: () => insert({ after: true, videoTime: player.currentTime() }) },
      { type: 'separator' },
      { label: '重复行', enabled: selIds.length > 0, onClick: () => actions.duplicateCues(selIds) },
      { type: 'separator' },
      { label: '剪切行', enabled: selIds.length > 0, shortcut: 'Ctrl+X', onClick: () => actions.cutCues(selIds) },
      { label: '复制行', enabled: selIds.length > 0, shortcut: 'Ctrl+C', onClick: () => actions.copyCues(selIds) },
      {
        label: '粘贴行',
        shortcut: 'Ctrl+V',
        onClick: () => tryPaste(actions.pasteCues({ refId, after: true, videoTime: player.currentTime() })),
      },
      {
        label: '选择性粘贴…',
        onClick: () => smartPasteSpecial(refId),
      },
      { type: 'separator' },
      { label: '删除行', danger: true, enabled: selIds.length > 0, shortcut: 'Del', onClick: () => actions.removeCues(selIds) },
    ];
  }

  function openPasteDialog(refId) {
    if (pasteDialog?.open) return; // 防重入：对话框已打开时忽略本次请求
    if (!pasteDialog) {
      tryPaste(actions.pasteSpecial({ start: true, end: true, text: true }, { refId }));
      return;
    }
    pasteDialog.returnValue = '';
    pasteDialog.showModal();
    pasteDialog.addEventListener(
      'close',
      () => {
        if (pasteDialog.returnValue !== 'ok') return;
        const fields = { start: false, end: false, text: false };
        pasteDialog.querySelectorAll('input[data-field]').forEach((box) => {
          fields[box.dataset.field] = box.checked;
        });
        tryPaste(actions.pasteSpecial(fields, { refId, videoTime: player.currentTime() }));
      },
      { once: true },
    );
  }

  menu.attach(wrapEl, () => buildMenuItems(), (event) => !inEditField(event));

  store.on('cues', reconcile);
  store.on('selection', paintSelection);
  store.on('subDoc', syncStyleSelects);
  store.on('subtitleFormat', syncStyleSelects);
  // setEditing(id) → 聚焦该行文本框（插入新行 / 波形双击的入口）
  store.on('editingId', (s) => {
    const id = s.editingId;
    if (!id) return;
    store.patch({ editingId: null });
    const view = rows.get(id);
    if (view) {
      focusText(id);
      view.tr.scrollIntoView({ block: 'nearest' });
    }
  });

  syncStyleSelects();
  reconcile();

  return { flush: flushEdits };
}
