(function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const statusLabel = {completed: '已完成', degraded: '降级完成', degraded_completed: '降级完成', failed: '失败', running: '运行中', pending: '等待中', preflight: '预检中'};
  // 双界面收敛（2026-09-10）：历史列表在侧栏，任务详情与质量报告内联进处理工作区。
  const HistoryUI = {
    initialized: false, selected: null,
    init() {
      if (this.initialized) return;
      this.initialized = true;
    },
    async showDetail(taskId) {
      if (!taskId) return;
      this.init();
      this.selected = taskId;
      const panel = document.getElementById('task-detail-panel');
      const body = document.getElementById('task-detail-body');
      const status = document.getElementById('task-detail-status');
      const quality = document.getElementById('quality-report-content');
      if (panel) panel.hidden = false;
      if (quality) quality.hidden = true;
      if (body) body.innerHTML = '<div class="workspace-loading">加载任务详情...</div>';
      if (status) { status.textContent = '加载中'; status.dataset.status = 'running'; }
      try {
        const detail = await VocalSubtitleApi.getHistoryDetail(taskId);
        if (this.selected !== taskId) return; // 用户已切换到其他任务
        if (status) { status.textContent = statusLabel[detail.status] || detail.status || '—'; status.dataset.status = detail.status || ''; }
        const summary = detail.result_summary || {}; const stats = summary.stats || {};
        if (body) body.innerHTML = '<dl class="detail-list">' + [['文件', detail.input_file_name], ['配置', detail.profile], ['字幕', summary.subtitle_count || detail.events?.length || 0], ['语音片段', summary.segment_count || stats.segment_count || '—'], ['质量', stats.quality_status || '—'], ['耗时', detail.total_duration_seconds ? Number(detail.total_duration_seconds).toFixed(1) + 's' : '—']].map((row) => '<div><dt>' + esc(row[0]) + '</dt><dd>' + esc(row[1]) + '</dd></div>').join('') + '</dl><div class="detail-actions"><button type="button" class="workspace-button primary" data-action="open-editor" title="带音频/字幕/评审清单直达 subtitle-editor">在编辑器中打开</button><button type="button" class="workspace-button" data-action="delete-task">删除此记录</button></div>' + (detail.error ? '<div class="workspace-error">' + esc(detail.error) + '</div>' : '');
        body?.querySelector('[data-action="open-editor"]')?.addEventListener('click', () => window.App && App.openInEditor(taskId));
        body?.querySelector('[data-action="delete-task"]')?.addEventListener('click', (event) => window.App && App.deleteHistoryItem(taskId, event));
        if (window.QualityUI) await QualityUI.load(taskId);
        if (this.selected === taskId && quality) quality.hidden = false;
      } catch (error) {
        if (this.selected !== taskId) return;
        if (body) body.innerHTML = '<div class="workspace-error">任务详情不可用（可能刚完成尚未入史，或已被清理）</div>';
        if (status) { status.textContent = '—'; status.dataset.status = ''; }
      }
    },
    resetDetail(taskId) {
      if (taskId && this.selected && this.selected !== taskId) return;
      this.selected = null;
      const panel = document.getElementById('task-detail-panel');
      const body = document.getElementById('task-detail-body');
      const status = document.getElementById('task-detail-status');
      const quality = document.getElementById('quality-report-content');
      if (panel) panel.hidden = true;
      if (quality) { quality.hidden = true; quality.innerHTML = ''; }
      if (body) body.innerHTML = '<div class="workspace-empty">完成任务或点击侧栏历史记录后，在此显示任务详情与质量报告</div>';
      if (status) { status.textContent = '未选择'; status.dataset.status = ''; }
    }
  };
  window.HistoryUI = HistoryUI;
  document.addEventListener('DOMContentLoaded', () => HistoryUI.init());
})();
