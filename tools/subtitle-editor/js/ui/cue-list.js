// 字幕表格：选择/跳转/行内编辑/播放高亮与跟随滚动。
import { formatClock, formatDuration, parseFlexibleTime } from '../format/time.js';
import { showToast } from './toast.js';

export function createCueList({ store, actions, player, tbodyEl, countEl, toolsEl }) {
  const rows = new Map(); // id → <tr>
  let editing = null; // { id, committed }
  let lastActiveId = null;
  let playerPlaying = false;

  // ---------- 工具条 ----------
  const [insertBtn, deleteBtn, splitBtn, mergeBtn] = toolsEl.querySelectorAll('button');
  const followBox = toolsEl.querySelector('input[type="checkbox"]');
  insertBtn.addEventListener('click', () => actions.insertAtTime(player.currentTime()));
  deleteBtn.addEventListener('click', () => {
    const cue = store.selectedCue();
    if (cue) actions.removeCue(cue.id);
  });
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

  // ---------- 渲染 ----------
  function render() {
    const cues = store.state.cues;
    if (editing && !cues.some((c) => c.id === editing.id)) editing = null;
    tbodyEl.textContent = '';
    rows.clear();

    if (!cues.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 6;
      td.className = 'cue-empty';
      td.textContent = '暂无字幕：拖入字幕文件，或把播放头移到目标位置后按 N 新建';
      tr.appendChild(td);
      tbodyEl.appendChild(tr);
      countEl.textContent = '';
      return;
    }
    const frag = document.createDocumentFragment();
    cues.forEach((cue, index) => {
      frag.appendChild(cue.id === store.state.editingId ? buildEditingRow(cue, index) : buildRow(cue, index));
    });
    tbodyEl.appendChild(frag);
    countEl.textContent = `共 ${cues.length} 条`;
    highlightActive(player.currentTime(), true);
  }

  function opBtn(label, title, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'op';
    b.textContent = label;
    b.title = title;
    b.setAttribute('aria-label', title);
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      onClick();
    });
    return b;
  }

  function buildRow(cue, index) {
    const tr = document.createElement('tr');
    tr.dataset.id = cue.id;
    tr.className = 'cue-row';
    if (cue.id === store.state.selectedId) tr.classList.add('selected');
    tr.innerHTML = `
      <td class="num mono"></td>
      <td class="time mono"></td>
      <td class="time mono"></td>
      <td class="dur mono"></td>
      <td class="text"></td>
      <td class="ops"></td>`;
    tr.querySelector('.num').textContent = index + 1;
    tr.querySelector('.time').textContent = formatClock(cue.start);
    tr.querySelectorAll('.time')[1].textContent = formatClock(cue.end);
    tr.querySelector('.dur').textContent = formatDuration(cue.end - cue.start);
    tr.querySelector('.text').textContent = cue.text;
    const ops = tr.querySelector('.ops');
    ops.appendChild(opBtn('✂', '在播放头处拆分（Ctrl+D）', () => actions.splitCue(cue.id, player.currentTime())));
    ops.appendChild(opBtn('⤓', '与下一句合并（Ctrl+M）', () => actions.mergeWithNext(cue.id)));
    ops.appendChild(opBtn('✕', '删除（Delete）', () => actions.removeCue(cue.id)));

    tr.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      actions.select(cue.id);
      player.seek(cue.start);
    });
    tr.addEventListener('dblclick', (e) => {
      if (e.target.closest('button')) return;
      actions.setEditing(cue.id);
    });
    rows.set(cue.id, tr);
    return tr;
  }

  function buildEditingRow(cue, index) {
    const tr = document.createElement('tr');
    tr.dataset.id = cue.id;
    tr.className = 'cue-row editing';
    tr.innerHTML = `
      <td class="num mono"></td>
      <td class="time"><input class="edit-time mono" data-field="start"></td>
      <td class="time"><input class="edit-time mono" data-field="end"></td>
      <td class="dur mono"></td>
      <td class="edit-main"><textarea class="edit-text" rows="2"></textarea></td>
      <td class="ops">
        <button type="button" class="op ok" title="提交（Enter / Ctrl+Enter）">✓</button>
        <button type="button" class="op cancel" title="取消（Esc）">✕</button>
      </td>`;
    tr.querySelector('.num').textContent = index + 1;
    tr.querySelector('.dur').textContent = formatDuration(cue.end - cue.start);
    const startInput = tr.querySelector('[data-field="start"]');
    const endInput = tr.querySelector('[data-field="end"]');
    const textarea = tr.querySelector('.edit-text');
    startInput.value = formatClock(cue.start);
    endInput.value = formatClock(cue.end);
    textarea.value = cue.text;
    editing = { id: cue.id, committed: false };

    const markInvalid = (input) => input.classList.add('invalid');
    const commit = () => {
      if (!editing || editing.committed) return;
      const start = parseFlexibleTime(startInput.value);
      const end = parseFlexibleTime(endInput.value);
      if (start === null) {
        markInvalid(startInput);
        showToast(`无法识别开始时间 "${startInput.value}"`, 'error');
        return;
      }
      if (end === null) {
        markInvalid(endInput);
        showToast(`无法识别结束时间 "${endInput.value}"`, 'error');
        return;
      }
      if (end <= start) {
        markInvalid(endInput);
        showToast('结束时间必须晚于开始时间', 'error');
        return;
      }
      editing.committed = true;
      actions.setEditing(null);
      actions.updateCue(cue.id, { start, end, text: textarea.value });
    };
    const cancel = () => {
      if (editing) editing.committed = true;
      actions.setEditing(null);
    };

    [startInput, endInput].forEach((input) => {
      input.addEventListener('input', () => input.classList.remove('invalid'));
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          commit();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          cancel();
        }
      });
    });
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        cancel();
      } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        commit();
      }
    });
    textarea.addEventListener('blur', () => commit());
    startInput.addEventListener('blur', () => commit());
    endInput.addEventListener('blur', () => commit());
    tr.querySelector('.op.ok').addEventListener('click', () => commit());
    tr.querySelector('.op.cancel').addEventListener('mousedown', () => {
      if (editing) editing.committed = true; // 阻止 blur 先触发提交
    });
    tr.querySelector('.op.cancel').addEventListener('click', () => cancel());

    setTimeout(() => startInput.focus(), 0);
    rows.set(cue.id, tr);
    return tr;
  }

  // ---------- 高亮与跟随 ----------
  function highlightActive(t, force) {
    const cue = store.cueAt(t);
    const id = cue?.id ?? null;
    if (id === lastActiveId && !force) return;
    if (lastActiveId) rows.get(lastActiveId)?.classList.remove('active');
    if (id) {
      const tr = rows.get(id);
      if (tr) {
        tr.classList.add('active');
        if (playerPlaying && store.state.follow) tr.scrollIntoView({ block: 'nearest' });
      }
    }
    lastActiveId = id;
  }

  player.onTime((t, playing) => {
    playerPlaying = playing;
    highlightActive(t, false);
  });

  store.on('cues', render);
  store.on('editingId', render);
  store.on('selection', () => {
    rows.forEach((tr, id) => {
      tr.classList.toggle('selected', id === store.state.selectedId);
    });
  });

  render();
}
