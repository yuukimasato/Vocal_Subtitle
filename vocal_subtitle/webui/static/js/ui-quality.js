(function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const json = (value) => { try { return JSON.stringify(value, null, 2); } catch (_) { return String(value || ''); } };
  const stateLabel = {completed: '已完成', degraded_completed: '降级完成', degraded: '降级完成', failed: '失败', running: '运行中'};
  // 双界面收敛（2026-09-10）：质量报告不再独立成工作区，由任务详情面板驱动，渲染进内联容器。
  const QualityUI = {
    initialized: false, taskId: null,
    init() {
      if (this.initialized) return;
      this.initialized = true;
    },
    async load(taskId) {
      this.init();
      if (!taskId) { this.renderEmpty(); return; }
      this.taskId = taskId;
      const container = document.getElementById('quality-report-content'); if (!container) return;
      container.innerHTML = '<div class="workspace-panel workspace-loading">加载质量报告...</div>';
      try { const data = await VocalSubtitleApi.getQualityReport(taskId); if (this.taskId === taskId) this.render(data); }
      catch (error) { if (this.taskId === taskId) this.renderError(error); }
    },
    render(data) {
      const report = data.report || {}; const pipeline = report.pipeline_path || {}; const quality = report.quality || {};
      const taskState = report.status || report.task_status || '';
      const container = document.getElementById('quality-report-content'); if (!container) return;
      const stageRows = Object.entries(report.stages || {}).map(([name, value]) => '<div class="quality-stage-row"><span>' + esc(name) + '</span><span>' + esc(value.status || 'completed') + '</span><strong>' + (value.duration_seconds !== undefined ? Number(value.duration_seconds).toFixed(2) + 's' : '—') + '</strong></div>').join('') || '<div class="workspace-empty">暂无阶段数据</div>';
      const engines = Object.entries(report.engine_availability || {}).map(([name, value]) => '<div class="quality-engine-row"><span>' + esc(name) + '</span><span class="status-chip" data-status="' + esc(value.status || '') + '">' + esc(value.status || '—') + '</span></div>').join('') || '<div class="workspace-empty">暂无引擎快照</div>';
      const acoustic = quality.acoustic_report || {};
      const diagnostics = report.diagnostics || {};
      const sourceMeta = '来源: ' + (data.report_source === 'run_report' ? 'run_report' : '历史摘要') + ' · ' + (data.report_schema_version || 'run-report-v1');
      container.innerHTML = '<section class="workspace-panel quality-summary-panel"><div class="quality-summary-grid"><div><span class="quality-kicker">任务状态</span><strong class="quality-big-value" data-status="' + esc(taskState) + '">' + esc(stateLabel[taskState] || taskState || '—') + '</strong></div><div><span class="quality-kicker">质量状态</span><strong class="quality-big-value" data-status="' + esc(quality.status || '') + '">' + esc(quality.status || '—') + '</strong></div><div><span class="quality-kicker">生产路径</span><strong class="quality-big-value">' + esc(pipeline.production_path || '—') + '</strong></div><div><span class="quality-kicker">运行 ID</span><strong class="quality-code">' + esc(data.run_id || report.run_id || '—') + '</strong></div></div><div class="quality-meta">' + esc(sourceMeta) + '</div></section>' +
        '<section class="quality-two-col"><div class="workspace-panel"><div class="workspace-panel-header"><div class="panel-title">阶段与引擎</div></div><div class="quality-stage-list">' + stageRows + '</div><div class="quality-engine-list">' + engines + '</div></div><div class="workspace-panel"><div class="workspace-panel-header"><div class="panel-title">质量与声学</div></div><div class="quality-stat-list">' + this.metric('声学健康度', acoustic.health_score !== undefined ? Number(acoustic.health_score).toFixed(0) + '%' : '—') + this.metric('复核状态', diagnostics.review_status || '—') + this.metric('决策数量', diagnostics.decision_count || quality.decision_count || 0) + '</div></div></section>' +
        '<section class="quality-two-col"><div class="workspace-panel"><div class="workspace-panel-header"><div class="panel-title">降级与错误</div></div><div class="quality-stat-list">' + this.metric('降级类别', report.degradation?.category || '—') + this.metric('降级原因', report.degradation?.reason || '—') + this.metric('错误', (report.errors || []).length ? (report.errors || []).map((item) => item.message || item.category).join('；') : '无') + '</div></div><div class="workspace-panel"><div class="workspace-panel-header"><div class="panel-title">证据 / 决策 / 投影</div></div><pre class="quality-json">' + esc(json({evidence: report.evidence, decision: report.decision, projection: report.projection})) + '</pre></div></section>' +
        '<section class="workspace-panel"><div class="workspace-panel-header"><div class="panel-title">产物</div></div><pre class="quality-json">' + esc(json(report.output || {})) + '</pre></section>';
    },
    metric(label, value) { return '<div class="quality-stat-row"><span>' + esc(label) + '</span><strong>' + esc(value) + '</strong></div>'; },
    renderEmpty() { const container = document.getElementById('quality-report-content'); if (container) container.innerHTML = '<div class="workspace-panel workspace-empty">暂无质量报告</div>'; },
    renderError(error) { const container = document.getElementById('quality-report-content'); if (container) container.innerHTML = '<div class="workspace-panel workspace-error">质量报告加载失败，可重试</div>'; toast('质量报告加载失败: ' + (error.message || error), 'error'); }
  };
  window.QualityUI = QualityUI;
  document.addEventListener('DOMContentLoaded', () => QualityUI.init());
})();
