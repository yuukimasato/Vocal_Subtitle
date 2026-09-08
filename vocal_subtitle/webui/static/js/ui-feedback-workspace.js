 (function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const FeedbackWorkspace = {
    initialized: false, refFile: null, audioFile: null, profiles: [],
    init() {
      if (this.initialized) return;
      this.initialized = true;
      document.getElementById('feedback-workspace-ref')?.addEventListener('change', (event) => { this.refFile = event.target.files[0] || null; this.renderSource(); });
      document.getElementById('feedback-workspace-audio')?.addEventListener('change', (event) => { this.audioFile = event.target.files[0] || null; this.renderSource(); });
      document.getElementById('feedback-use-current')?.addEventListener('click', () => this.useCurrentEdits());
      document.getElementById('feedback-workspace-preview')?.addEventListener('click', () => this.run(false));
      document.getElementById('feedback-workspace-learn')?.addEventListener('click', () => this.run(true));
      document.getElementById('feedback-workspace-refresh')?.addEventListener('click', () => this.refresh(true));
      document.getElementById('feedback-queue-refresh')?.addEventListener('click', () => this.loadQueue());
      document.getElementById('feedback-workspace-profile')?.addEventListener('change', () => this.loadHealth());
    },
    async refresh(force = false) {
      this.init();
      try {
        if (force || !this.profiles.length) await this.loadProfiles();
        this.syncLegacyFiles(); this.renderSource();
        await Promise.all([this.loadHealth(), this.loadQueue()]);
      } catch (error) { toast('反馈数据加载失败: ' + (error.message || error), 'error'); }
    },
    syncLegacyFiles() {
      if (!this.refFile && window.App) this.refFile = App.state._fbRefFile || null;
      if (!this.audioFile && window.App) this.audioFile = App.state._fbAudioFile || App.state.selectedFile || null;
    },
    async loadProfiles() {
      const data = await VocalSubtitleApi.feedbackProfiles(); this.profiles = data.profiles || [];
      const select = document.getElementById('feedback-workspace-profile'); if (!select) return;
      const current = select.value || 'user_default';
      select.innerHTML = this.profiles.length ? this.profiles.map((profile) => '<option value="' + esc(profile.profile_id) + '">' + esc(profile.profile_id) + ' · ' + Number(profile.feedback_count || 0) + ' 次</option>').join('') : '<option value="user_default">user_default</option>';
      if (Array.from(select.options).some((option) => option.value === current)) select.value = current;
    },
    useCurrentEdits() {
      const events = window.App && App.state ? App.state.subtitleEvents : [];
      if (!events || !events.length) { toast('当前没有字幕编辑结果', 'info'); return; }
      const text = events.map((event, index) => (index + 1) + '\n' + (window.formatEventAsSRT ? formatEventAsSRT(event) : event.start + ' --> ' + event.end + '\n' + event.text) + '\n').join('\n');
      this.refFile = new File([text], 'edited.srt', {type: 'text/plain'});
      if (window.App) { App.state._fbRefFile = this.refFile; this.audioFile = this.audioFile || App.state.selectedFile || null; App.state._fbAudioFile = this.audioFile; }
      this.renderSource();
    },
    renderSource() {
      const status = document.getElementById('feedback-source-status'); const detail = document.getElementById('feedback-source-detail');
      if (status) { status.textContent = this.refFile && this.audioFile ? '可预览' : '缺少文件'; status.dataset.status = this.refFile && this.audioFile ? 'ready' : 'pending'; }
      if (detail) detail.innerHTML = '<div>字幕: <strong>' + esc(this.refFile?.name || '未选择') + '</strong></div><div>音频: <strong>' + esc(this.audioFile?.name || '未选择') + '</strong></div>';
    },
    async run(write) {
      this.syncLegacyFiles(); this.renderSource();
      if (!this.refFile || !this.audioFile) { toast('请先提供修订字幕和音频文件', 'info'); return; }
      const profile = document.getElementById('feedback-workspace-profile')?.value || 'user_default';
      const status = document.getElementById('feedback-workspace-status'); if (status) status.textContent = write ? '学习中...' : '分析中...';
      try {
        const report = write ? await VocalSubtitleApi.feedbackLearn(this.audioFile, this.refFile, App.state.selectedProfile || 'default', profile, true, false) : await VocalSubtitleApi.feedbackPreview(this.audioFile, this.refFile, App.state.selectedProfile || 'default');
        this.renderReport(report); if (write && report.status === 'ok') await this.loadProfiles();
        toast(write ? (report.status === 'ok' ? '反馈学习完成' : '学习未写入') : '差异预览完成', report.status === 'ok' ? 'success' : 'info');
      } catch (error) { if (status) status.textContent = '失败'; toast((write ? '学习' : '预览') + '失败: ' + (error.message || error), 'error'); }
    },
    renderReport(report) {
      const status = document.getElementById('feedback-workspace-status'); if (status) { status.textContent = report.status === 'ok' ? '完成' : (report.status || '未完成'); status.dataset.status = report.status || ''; }
      const target = document.getElementById('feedback-workspace-report'); if (!target) return;
      const metrics = [['对齐覆盖率', report.alignment_coverage !== undefined ? (Number(report.alignment_coverage) * 100).toFixed(1) + '%' : '—'], ['总配对', report.total_pairs || 0], ['时间偏移', report.time_shifts_count || 0], ['合并/拆分', report.merge_actions_count || 0], ['文本编辑', report.text_edits_count || 0]];
      const params = Object.entries(report.param_adjustments || {}).map(([key, value]) => '<div class="feedback-adjustment"><span>' + esc(key) + '</span><strong>' + esc(typeof value === 'object' ? JSON.stringify(value) : value) + '</strong></div>').join('');
      target.innerHTML = '<div class="feedback-result-metrics">' + metrics.map((row) => '<div><span>' + esc(row[0]) + '</span><strong>' + esc(row[1]) + '</strong></div>').join('') + '</div>' + (params ? '<div class="feedback-adjustments-title">参数调整</div>' + params : '<div class="workspace-empty">无需调整参数</div>') + (report.structural_revision ? '<div class="feedback-warning">检测到结构性修订，参数权重将降低</div>' : '') + (report.message ? '<div class="feedback-message">' + esc(report.message) + '</div>' : '');
    },
    async loadHealth() {
      const profile = document.getElementById('feedback-workspace-profile')?.value || 'user_default'; const target = document.getElementById('feedback-health-chart'); if (!target) return;
      try { const data = await VocalSubtitleApi.feedbackHealth(profile, 12); const points = data.trend || []; if (!points.length) { target.innerHTML = '<div class="workspace-empty">暂无趋势数据</div>'; return; } const max = Math.max(...points.map((point) => Number(point.health_after || 0)), 1); target.innerHTML = '<div class="health-bars">' + points.map((point) => '<div class="health-bar-item" title="' + esc(point.timestamp || '') + '"><div class="health-bar" style="height:' + Math.max(8, Number(point.health_after || 0) / max * 100) + '%"></div><span>' + Number(point.health_after || 0).toFixed(0) + '</span></div>').join('') + '</div>'; }
      catch (error) { target.innerHTML = '<div class="workspace-empty">趋势暂不可用</div>'; }
    },
    async loadQueue() {
      const target = document.getElementById('feedback-review-queue'); if (!target) return;
      try { const data = await VocalSubtitleApi.getReviewQueue('pending'); const samples = data.samples || []; if (!samples.length) { target.innerHTML = '<div class="workspace-empty">暂无待审核样本</div>'; return; } target.innerHTML = samples.map((sample) => '<div class="review-queue-item"><div><strong>' + esc(sample.sample_id || 'sample') + '</strong><span>' + esc(sample.submitted_at || '') + '</span></div><div class="review-queue-summary">' + esc(typeof sample.diff_summary === 'object' ? JSON.stringify(sample.diff_summary) : sample.diff_summary || '—') + '</div><div class="review-queue-actions"><button type="button" data-review="accepted" data-sample="' + esc(sample.sample_id) + '">接受</button><button type="button" data-review="rejected" data-sample="' + esc(sample.sample_id) + '">拒绝</button></div></div>').join(''); target.querySelectorAll('[data-review]').forEach((button) => button.addEventListener('click', () => this.reviewSample(button.dataset.sample, button.dataset.review))); }
      catch (error) { target.innerHTML = '<div class="workspace-empty">审核队列暂不可用</div>'; }
    },
    async reviewSample(sampleId, result) { const fd = new FormData(); fd.append('result', result); try { await fetch('/api/feedback/review/' + encodeURIComponent(sampleId), {method: 'POST', body: fd}).then(async (response) => { if (!response.ok) throw new Error(await response.text()); return response.json(); }); await this.loadQueue(); toast('反馈样本已' + (result === 'accepted' ? '接受' : '拒绝'), 'success'); } catch (error) { toast('审核失败: ' + (error.message || error), 'error'); } }
  };
  window.FeedbackWorkspace = FeedbackWorkspace;
  document.addEventListener('DOMContentLoaded', () => FeedbackWorkspace.init());
})();
