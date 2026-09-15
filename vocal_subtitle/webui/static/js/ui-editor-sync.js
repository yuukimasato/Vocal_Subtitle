// 编辑台同步字幕面板：字幕打轴工作台（8631）每次编辑/导出后自动把最新稿同步到处理台
// （POST /api/editor-sync/subtitle），此处列出全部同步稿并提供"下载最新稿"入口。
// 纯展示与取数：文件名等来自同步载荷的内容一律经 textContent 渲染，防注入。
(function() {
  'use strict';

  var refreshTimer = null;

  function el(id) {
    return document.getElementById(id);
  }

  function formatTime(iso) {
    if (!iso) return '—';
    var t = new Date(iso);
    return isNaN(t.getTime()) ? String(iso) : t.toLocaleString();
  }

  function row(meta) {
    var row = document.createElement('div');
    row.className = 'editor-sync-row';
    row.style.cssText = 'display:flex;align-items:center;gap:10px;padding:8px 10px;border-bottom:1px solid var(--border);flex-wrap:wrap;';

    var name = document.createElement('a');
    name.className = 'editor-sync-name';
    name.href = '/api/editor-sync/subtitles/' + encodeURIComponent(meta.id) + '/download';
    name.textContent = meta.name || meta.id;
    name.style.cssText = 'color:var(--accent);text-decoration:none;font-weight:500;word-break:break-all;';
    name.title = '下载最新稿';

    var chip = document.createElement('span');
    chip.className = 'status-chip';
    chip.textContent = String(meta.format || '?').toUpperCase();

    var info = document.createElement('span');
    info.style.cssText = 'font-size:0.72rem;color:var(--text-tertiary);';
    var bits = [];
    if (meta.cue_count) bits.push(meta.cue_count + ' 条');
    if (meta.media_name) bits.push('媒体：' + meta.media_name);
    if (meta.include_speakers === false) bits.push('已剥离说话人');
    bits.push('同步于 ' + formatTime(meta.saved_at));
    info.textContent = bits.join(' · ');

    var download = document.createElement('a');
    download.className = 'btn-export';
    download.href = '/api/editor-sync/subtitles/' + encodeURIComponent(meta.id) + '/download';
    download.textContent = '下载';
    download.style.cssText = 'text-decoration:none;margin-left:auto;';

    row.append(name, chip, info, download);
    return row;
  }

  function render(items) {
    var list = el('editor-sync-list');
    if (!list) return;
    list.textContent = '';
    if (!items.length) {
      var empty = document.createElement('div');
      empty.className = 'workspace-empty';
      empty.textContent = '暂无同步：在字幕打轴工作台（8631）编辑字幕后自动出现在这里，可随时下载最新稿';
      list.appendChild(empty);
    } else {
      items.forEach(function(meta) { list.appendChild(row(meta)); });
    }
    var count = el('editor-sync-count');
    if (count) count.textContent = items.length + ' 份';
  }

  async function refresh() {
    var list = el('editor-sync-list');
    if (!list) return;
    try {
      var res = await fetch('/api/editor-sync/subtitles');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var data = await res.json();
      render(Array.isArray(data.items) ? data.items : []);
    } catch (err) {
      list.textContent = '';
      var tip = document.createElement('div');
      tip.className = 'workspace-empty';
      tip.textContent = '同步服务不可用（' + (err && err.message ? err.message : '未知错误') + '）';
      list.appendChild(tip);
    }
  }

  window.EditorSync = { refresh: refresh };
  document.addEventListener('DOMContentLoaded', function() {
    refresh();
    el('btn-editor-sync-refresh')?.addEventListener('click', refresh);
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(refresh, 30000); // 编辑台持续编辑时保持最新
  });
})();
