 (function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  // 双界面收敛（2026-09-10）：学习入口迁至 subtitle-editor（8631）"学习"面板；
  // 此处仅保留反馈档案治理：档案健康趋势 + 复审队列。
  const FeedbackWorkspace = {
    initialized: false, profiles: [],
    init() {
      if (this.initialized) return;
      this.initialized = true;
      document.getElementById('feedback-workspace-profile')?.addEventListener('change', () => this.loadHealth());
      // D38：应用学习参数开关（状态持久化在服务端用户配置目录）
      document.getElementById('feedback-apply-overrides-cb')?.addEventListener('change', (event) => this.saveApplyOverrides(event.target.checked));
    },
    async refresh(force = false) {
      this.init();
      try {
        if (force || !this.profiles.length) await this.loadProfiles();
        await Promise.all([this.loadHealth(), this.loadQueue(), this.loadApplyOverrides()]);
        // 上传学习窗口（V2，D23）：任务历史列表随工作区进入一并刷新
        if (window.UploadLearn) UploadLearn.refresh();
        // 档案管理（回滚/重置入口）随工作区进入一并刷新
        if (window.App) await App.refreshFeedbackConfigs();
      } catch (error) { toast('反馈数据加载失败: ' + (error.message || error), 'error'); }
    },
    async loadApplyOverrides() {
      const input = document.getElementById('feedback-apply-overrides-cb'); if (!input) return;
      try {
        const data = await VocalSubtitleApi.feedbackOverridesApply();
        input.checked = !!data.apply_overrides_on_run;
      } catch (error) { input.checked = false; }
    },
    async saveApplyOverrides(enabled) {
      const input = document.getElementById('feedback-apply-overrides-cb');
      try {
        const data = await VocalSubtitleApi.feedbackOverridesApplyToggle(enabled);
        if (input) input.checked = !!data.apply_overrides_on_run;
        toast(data.message || (enabled ? '已开启：新任务将应用学习参数' : '已关闭：新任务保持默认参数'), 'success');
      } catch (error) {
        if (input) input.checked = !enabled; // 失败回滚 UI 状态
        toast('保存开关状态失败: ' + (error.message || error), 'error');
      }
    },
    async loadProfiles() {
      const data = await VocalSubtitleApi.feedbackProfiles(); this.profiles = data.profiles || [];
      const select = document.getElementById('feedback-workspace-profile'); if (!select) return;
      const current = select.value || 'user_default';
      select.innerHTML = this.profiles.length ? this.profiles.map((profile) => '<option value="' + esc(profile.profile_id) + '">' + esc(profile.profile_id) + ' · ' + Number(profile.feedback_count || 0) + ' 次</option>').join('') : '<option value="user_default">user_default</option>';
      if (Array.from(select.options).some((option) => option.value === current)) select.value = current;
    },
    async loadHealth() {
      const profile = document.getElementById('feedback-workspace-profile')?.value || 'user_default'; const target = document.getElementById('feedback-health-chart'); if (!target) return;
      try { const data = await VocalSubtitleApi.feedbackHealth(profile, 12); const points = data.trend || []; if (!points.length) { target.innerHTML = '<div class="workspace-empty">暂无趋势数据</div>'; return; } const max = Math.max(...points.map((point) => Number(point.health_after || 0)), 1); target.innerHTML = '<div class="health-bars">' + points.map((point) => '<div class="health-bar-item" title="' + esc(point.timestamp || '') + '"><div class="health-bar" style="height:' + Math.max(8, Number(point.health_after || 0) / max * 100) + '%"></div><span>' + Number(point.health_after || 0).toFixed(0) + '</span></div>').join('') + '</div>'; }
      catch (error) { target.innerHTML = '<div class="workspace-empty">趋势暂不可用</div>'; }
    },
    // diff_summary 的机器键名转用户可读文案
    diffSummaryText(summary) {
      const labelMap = [['text_correction', '文本修改'], ['time_adjustment', '时间调整'], ['format_preference', '格式偏好'], ['merge', '合并'], ['split', '拆分']];
      if (typeof summary === 'string') return summary || '—';
      if (!summary || typeof summary !== 'object') return '—';
      const parts = labelMap.map(([key, label]) => summary[key] !== undefined ? label + ' ' + summary[key] : null).filter(Boolean);
      if (parts.length) return parts.join(' · ');
      const rest = Object.entries(summary).map(([key, value]) => key + ' ' + value);
      return rest.length ? rest.join(' · ') : '—';
    },

    formatTime(ts) {
      if (!ts) return '';
      const match = String(ts).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
      return match ? match[2] + '-' + match[3] + ' ' + match[4] + ':' + match[5] : String(ts);
    },

    async loadQueue() {
      const target = document.getElementById('feedback-review-queue'); if (!target) return;
      try { const data = await VocalSubtitleApi.getReviewQueue('pending'); const samples = data.samples || []; if (!samples.length) { target.innerHTML = '<div class="workspace-empty">暂无待审核样本</div>'; return; } target.innerHTML = samples.map((sample) => '<div class="review-queue-item"><div><strong>' + esc(sample.sample_id || 'sample') + '</strong><span>' + esc(this.formatTime(sample.submitted_at)) + '</span></div><div class="review-queue-summary">' + esc(this.diffSummaryText(sample.diff_summary)) + '</div><div class="review-queue-actions"><button type="button" data-review="accepted" data-sample="' + esc(sample.sample_id) + '">接受</button><button type="button" data-review="rejected" data-sample="' + esc(sample.sample_id) + '">拒绝</button></div></div>').join(''); target.querySelectorAll('[data-review]').forEach((button) => button.addEventListener('click', () => this.reviewSample(button.dataset.sample, button.dataset.review))); }
      catch (error) { target.innerHTML = '<div class="workspace-empty">审核队列暂不可用</div>'; }
    },
    async reviewSample(sampleId, result) { const fd = new FormData(); fd.append('result', result); try { await fetch('/api/feedback/review/' + encodeURIComponent(sampleId), {method: 'POST', body: fd}).then(async (response) => { if (!response.ok) throw new Error(await response.text()); return response.json(); }); await this.loadQueue(); toast('反馈样本已' + (result === 'accepted' ? '接受' : '拒绝'), 'success'); } catch (error) { toast('审核失败: ' + (error.message || error), 'error'); } }
  };
  window.FeedbackWorkspace = FeedbackWorkspace;
  document.addEventListener('DOMContentLoaded', () => FeedbackWorkspace.init());
})();
