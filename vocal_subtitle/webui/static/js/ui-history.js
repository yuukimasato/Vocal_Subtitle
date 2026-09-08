(function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const statusLabel = {completed: '已完成', degraded: '降级完成', degraded_completed: '降级完成', failed: '失败', running: '运行中', pending: '等待中', preflight: '预检中'};
  const HistoryUI = {
    initialized: false, items: [], page: 0, pageSize: 20, total: 0, selected: null,
    init() {
      if (this.initialized) return;
      this.initialized = true;
      document.getElementById('history-status-filter')?.addEventListener('change', () => { this.page = 0; this.refresh(); });
      document.getElementById('history-refresh')?.addEventListener('click', () => this.refresh(true));
      document.getElementById('history-prev')?.addEventListener('click', () => { if (this.page) { this.page--; this.refresh(); } });
      document.getElementById('history-next')?.addEventListener('click', () => { if ((this.page + 1) * this.pageSize < this.total) { this.page++; this.refresh(); } });
    },
    async refresh(force = false) {
      this.init();
      const filter = document.getElementById('history-status-filter')?.value || '';
      const list = document.getElementById('history-workspace-list');
      if (list && (!this.items.length || force)) list.innerHTML = '<div class="workspace-loading">加载历史...</div>';
      try {
        const data = await VocalSubtitleApi.getHistoryFiltered(this.pageSize, this.page * this.pageSize, filter);
        this.items = data.items || []; this.total = data.total || 0;
        this.renderList(); this.fillTaskSelectors();
        if (this.selected && this.items.some((item) => item.id === this.selected)) await this.loadDetail(this.selected);
      } catch (error) { if (list) list.innerHTML = '<div class="workspace-error">历史加载失败，可重试</div>'; toast('历史加载失败: ' + error.message, 'error'); }
    },
    fillTaskSelectors() {
      ['review-task-select', 'quality-task-select'].forEach((id) => {
        const select = document.getElementById(id); if (!select) return;
        const current = select.value || Workspace.taskId() || '';
        select.innerHTML = '<option value="">选择任务</option>' + this.items.map((item) => '<option value="' + esc(item.id) + '">' + esc(item.input_file_name || item.id) + ' · ' + esc(statusLabel[item.status] || item.status || '') + '</option>').join('');
        if (current) select.value = current;
      });
    },
    renderList() {
      const list = document.getElementById('history-workspace-list');
      if (!list) return;
      document.getElementById('history-summary').textContent = this.total + ' 条记录 · 第 ' + (this.page + 1) + ' 页';
      document.getElementById('history-page-label').textContent = (this.page + 1) + ' / ' + Math.max(1, Math.ceil(this.total / this.pageSize));
      document.getElementById('history-prev').disabled = this.page === 0;
      document.getElementById('history-next').disabled = (this.page + 1) * this.pageSize >= this.total;
      if (!this.items.length) { list.innerHTML = '<div class="workspace-empty">暂无任务历史</div>'; return; }
      list.innerHTML = this.items.map((item) => '<button type="button" class="history-workspace-item ' + (item.id === this.selected ? 'active' : '') + '" data-task-id="' + esc(item.id) + '">' +
        '<span class="history-status-dot ' + esc(item.status || '') + '"></span><span class="history-file">' + esc(item.input_file_name || item.id) + '</span><span class="history-item-meta">' + esc(statusLabel[item.status] || item.status || '—') + '<br>' + esc((item.created_at || '').slice(0, 16).replace('T', ' ')) + '</span></button>').join('');
      list.querySelectorAll('[data-task-id]').forEach((button) => button.addEventListener('click', () => this.loadDetail(button.dataset.taskId)));
    },
    async loadDetail(taskId) {
      if (!taskId) return;
      this.selected = taskId; Workspace.setTask(taskId); this.renderList();
      const body = document.getElementById('history-detail-body');
      if (body) body.innerHTML = '<div class="workspace-loading">加载任务详情...</div>';
      try {
        const detail = await VocalSubtitleApi.getHistoryDetail(taskId);
        const status = document.getElementById('history-detail-status'); if (status) { status.textContent = statusLabel[detail.status] || detail.status || '—'; status.dataset.status = detail.status || ''; }
        const summary = detail.result_summary || {}; const stats = summary.stats || {};
        if (body) body.innerHTML = '<dl class="detail-list">' + [['文件', detail.input_file_name], ['配置', detail.profile], ['字幕', summary.subtitle_count || detail.events?.length || 0], ['语音片段', summary.segment_count || stats.segment_count || '—'], ['质量', stats.quality_status || '—'], ['耗时', detail.total_duration_seconds ? Number(detail.total_duration_seconds).toFixed(1) + 's' : '—']].map((row) => '<div><dt>' + esc(row[0]) + '</dt><dd>' + esc(row[1]) + '</dd></div>').join('') + '</dl><div class="detail-actions"><button type="button" class="workspace-button primary" data-open="review">字幕审核</button><button type="button" class="workspace-button" data-open="quality">质量报告</button></div>' + (detail.error ? '<div class="workspace-error">' + esc(detail.error) + '</div>' : '');
        body?.querySelectorAll('[data-open]').forEach((button) => button.addEventListener('click', () => Workspace.switchTo(button.dataset.open)));
      } catch (error) { if (body) body.innerHTML = '<div class="workspace-error">任务不存在或已清理</div>'; toast('任务详情加载失败: ' + error.message, 'error'); }
    }
  };
  window.HistoryUI = HistoryUI;
  document.addEventListener('DOMContentLoaded', () => HistoryUI.init());
})();
