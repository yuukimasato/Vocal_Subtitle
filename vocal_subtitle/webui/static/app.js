(function() {
'use strict';

// ---- Utility Functions ----
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = (seconds % 60).toFixed(1);
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(4,'0')}`;
  return `${m}:${String(s).padStart(4,'0')}`;
}

function toast(msg, type) {
  type = type || 'info';
  const container = $('#toast-container');
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 4000);
}

// ---- Application State ----
const App = {
  state: window.VocalSubtitleState.create(),

  saveSettings() {
    return window.VocalSubtitleSettings.save(this);
  },

  loadSettings() {
    return window.VocalSubtitleSettings.load();
  },

  // ---- API Client ----
  api: window.VocalSubtitleApi,
  // ---- WebSocket ----
  ws: window.VocalSubtitleWs,
  // ---- Business Logic ----
  async handleRun() {
    if (App.state.isRunning || !App.state.selectedFile) return;
    if (App.state.funasrPreparing) {
      toast('FunASR 正在准备，请等待本地模型就绪后再开始处理', 'info');
      return;
    }

    App.state.isRunning = true;
    App.ui.setRunning(true);

    try {
      // Build form data
      const formData = new FormData();
      formData.append('file', App.state.selectedFile);
      formData.append('profile', App.state.selectedProfile);
      formData.append('output_format', App.state.outputFormat);
      formData.append('skip_separation', App.state.skipSeparation);
      formData.append('overrides', JSON.stringify(App.collectOverrides()));

      const resp = await fetch('/api/run', { method: 'POST', body: formData });
      if (!resp.ok) { const t = await resp.text(); throw new Error(t); }
      const data = await resp.json();
      App.state.taskId = data.task_id;

      // 缓存命中：直接获取结果，无需等待 WebSocket
      if ((data.status === 'completed' || data.status === 'degraded') && data.from_cache) {
        toast('✅ 缓存命中！检测到相同文件的历史处理结果，直接加载', 'success');
        const taskData = await App.api.getTaskStatus(data.task_id);
        if (taskData.result) {
          $('#empty-state').style.display = 'none';
          $('#results-content').style.display = 'block';
          App.ui.pipelineComplete(taskData.result);
          // 在统计区域显示缓存标记
          var cacheTag = document.getElementById('cache-hit-tag');
          if (!cacheTag) {
            cacheTag = document.createElement('div');
            cacheTag.id = 'cache-hit-tag';
            cacheTag.style.cssText = 'display:inline-block;padding:3px 10px;border-radius:12px;background:rgba(63,185,80,0.15);color:var(--accent-green);font-size:0.78rem;font-weight:600;margin-left:8px;';
            cacheTag.textContent = '⚡ 缓存结果';
            var statsGrid = document.getElementById('stats-grid');
            if (statsGrid) statsGrid.appendChild(cacheTag);
          }
        }
        return;
      }

      // Connect WebSocket for progress
      App.ws.connect(data.task_id);

      // Show results area
      $('#empty-state').style.display = 'none';
      $('#results-content').style.display = 'block';

    } catch (ex) {
      toast('启动失败: ' + ex.message, 'error');
      App.ui.setRunning(false);
      App.state.isRunning = false;
    }
  },

  collectOverrides() {
    return window.VocalSubtitleSettings.collectOverrides();
  },

  // ---- History & Cache Management ----

	  async refreshHistory() {
	    try {
	      const data = await App.api.getHistory(50, 0);
	      App.ui.renderHistory(data.items || []);
	    } catch (ex) {
	      console.error('Failed to load history:', ex);
	    }
	  },

	  async deleteHistoryItem(taskId, event) {
	    if (event) event.stopPropagation();
	    if (!confirm('删除此历史记录？关联的缓存文件将保留。')) return;
	    try {
	      await App.api.deleteHistory(taskId);
	      if (window.HistoryUI) HistoryUI.resetDetail(taskId);
	      App.refreshHistory();
	      toast('历史记录已删除', 'info');
	    } catch (ex) {
	      toast('删除失败: ' + ex.message, 'error');
	    }
	  },

	  async clearHistory() {
	    if (!confirm('确认清除全部历史记录？\n\n这将同时删除：\n• 所有任务历史记录\n• 持久化的字幕和分离音频文件\n• 上传目录中的临时文件\n\n此操作不可恢复。')) return;
	    try {
	      const resp = await App.api.clearAllHistory();
	      var msg = '已清除 ' + resp.deleted_count + ' 条历史记录';
	      if (resp.persistent_files_cleaned) msg += '，' + resp.persistent_files_cleaned + ' 个持久化文件';
	      if (resp.uploads_cleaned) msg += '，' + resp.uploads_cleaned + ' 个上传目录';
	      toast(msg, 'info');
	      App.refreshHistory();
	      if (window.HistoryUI) HistoryUI.resetDetail();
	      App.refreshCacheInfo();
	    } catch (ex) {
	      toast('清除失败: ' + ex.message, 'error');
	    }
	  },

	  async refreshCacheInfo() {
	    try {
	      const info = await App.api.getCacheInfo();
	      // 显示 uploads 目录大小（用户上传文件缓存）
      $('#cache-size').textContent = (info.total_mb || 0).toFixed(1) + ' MB';
	      if (info.task_history_count > 0) {
	        $('#cache-size').textContent += ' | ' + info.task_history_count + ' 条历史';
	      }
	    } catch (ex) {
	      console.error('Failed to load cache info:', ex);
	    }
	  },

	  async clearCache() {
	    if (!confirm('确认清除上传缓存？\n\n将清理历史上传的临时文件。\n计算缓存和模型文件将保留，以便加速后续处理。')) return;
	    try {
	      const resp = await App.api.clearAllCache();
	      var msg = '上传缓存已清除';
	      if (resp.uploads_cleaned) msg += '（含 ' + resp.uploads_cleaned + ' 个上传目录）';
	      toast(msg, 'success');
	      App.refreshCacheInfo();
	    } catch (ex) {
	      toast('清除缓存失败: ' + ex.message, 'error');
	    }
	  },

	  handleHistoryClick(item) {
	    if (item.status !== 'completed' && item.status !== 'degraded') {
	      toast('该任务未完成，无法查看结果', 'info');
	      return;
	    }
	    // 设置 taskId 以启用导出/下载按钮（在异步加载详情前就设置）
	    App.state.taskId = item.id;
	    Workspace.setTask(item.id);

    // 先用列表摘要快速显示统计信息（与 renderStats 保持同一组核心指标）
    if (item.result_summary && item.result_summary.stats) {
      const stats = item.result_summary.stats;
      $('#stats-grid').innerHTML = `
        <div class="stat-card"><div class="stat-value">${stats.total_time ? stats.total_time.toFixed(1)+'s' : '缓存'}</div><div class="stat-label">总耗时</div></div>
        <div class="stat-card"><div class="stat-value">${item.result_summary.subtitle_count || 0}</div><div class="stat-label">字幕条数</div></div>
        <div class="stat-card"><div class="stat-value">${item.total_duration_seconds ? (item.total_duration_seconds/60).toFixed(1)+'min' : '—'}</div><div class="stat-label">音频时长</div></div>
      `;
	      $('#empty-state').style.display = 'none';
	      $('#results-content').style.display = 'block';

	      // 快速显示音频导出栏（如果列表摘要中已有路径信息）
	      App.ui.updateAudioExportBar(item.result_summary);
	      // 更新导出栏（根据是否有 LLM 优化版本切换按钮布局）
	      App.ui.updateExportBar(item.result_summary || {});
	    }

	    // 获取完整详情，完善统计与导出栏，并联动任务详情/质量面板
	    App.api.getHistoryDetail(item.id).then(detail => {
	      const summary = detail.result_summary || {};
	      if (summary.stats) {
	        App.ui.renderStats(summary.stats);
	        $('#empty-state').style.display = 'none';
	        $('#results-content').style.display = 'block';
	        App.ui.renderDiagnosticReport({ stats: summary.stats });
	      }
	      // 使用详情中的完整 result_summary（含 vocals_path / accompaniment_path）更新导出栏
	      App.ui.updateAudioExportBar(detail.result_summary || {});
	      App.ui.updateExportBar(detail.result_summary || {});
	    }).catch(ex => {
	      console.error('Failed to load history detail:', ex);
	    });

	    // 任务详情 + 质量报告内联面板
	    if (window.HistoryUI) HistoryUI.showDetail(item.id);
	  },

    async exportFormat(fmt) {
    if (!App.state.taskId) { toast('请先完成处理', 'error'); return; }
    try {
      const resp = await fetch('/api/subtitle/' + App.state.taskId + '/export?format=' + fmt);
      if (!resp.ok) throw new Error('Export failed');
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'subtitle.' + fmt;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
      toast('已下载 ' + fmt.toUpperCase() + ' 字幕文件', 'success');
    } catch (ex) {
      toast('导出失败: ' + ex.message, 'error');
    }
  },

  async downloadSubtitleFile(version) {
    if (!App.state.taskId) { toast('请先完成处理', 'error'); return; }
    var label = version === 'clean' ? '干净版' : 'LLM优化版';
    try {
      var resp = await fetch('/api/tasks/' + App.state.taskId + '/subtitle-file?version=' + version);
      if (!resp.ok) {
        var errText = '';
        try { var err = await resp.json(); errText = err.detail || ''; } catch(e) {}
        throw new Error(errText || 'Download failed');
      }
      var blob = await resp.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      // 从 Content-Disposition 获取文件名，或使用默认名称
      var disposition = resp.headers.get('content-disposition');
      var filename = 'subtitle_' + label + '.srt';
      if (disposition) {
        var match = disposition.match(/filename="?(.+?)"?$/);
        if (match) filename = match[1];
      }
      a.download = filename;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
      toast('已下载' + label + '字幕文件', 'success');
    } catch (ex) {
      toast('下载' + label + '失败: ' + ex.message, 'error');
    }
  },

  async downloadAudio(type) {
    if (!App.state.taskId) { toast('请先完成处理', 'error'); return; }
    var label = type === 'vocals' ? '人声' : '背景声';
    try {
      var resp = await fetch('/api/tasks/' + App.state.taskId + '/audio?type=' + type);
      if (!resp.ok) {
        var errText = '';
        try { var err = await resp.json(); errText = err.detail || ''; } catch(e) {}
        throw new Error(errText || 'Download failed');
      }
      var blob = await resp.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = label + '.wav';
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
      toast('已下载' + label + '文件', 'success');
    } catch (ex) {
      toast('下载' + label + '失败: ' + ex.message, 'error');
    }
  },

  // ---- UI Rendering ----
  ui: window.VocalSubtitleUi,

  // ---- 双界面收敛：深链 subtitle-editor + 任务详情面板 ----

  // subtitle-editor（8631）地址；localStorage 键 vst.editorBase 可覆盖默认值
  editorBase() {
    return localStorage.getItem('vst.editorBase') || 'http://127.0.0.1:8631/';
  },

  openInEditor(taskId) {
    taskId = taskId || App.state.taskId;
    if (!taskId) { toast('请先完成处理或从历史选择任务', 'error'); return; }
    var origin = location.origin;
    var qs = new URLSearchParams({
      media: origin + '/api/tasks/' + taskId + '/audio/stream?type=input',
      subs: origin + '/api/tasks/' + taskId + '/subtitle-file?version=clean',
      manifest: origin + '/api/tasks/' + taskId + '/manifest'
    });
    window.open(App.editorBase() + '?' + qs.toString(), '_blank');
  },

  showTaskDetail() {
    // 任务详情面板在完成/选择历史时自动展示；保留空实现避免旧调用报错
    var panel = document.getElementById('task-detail-panel');
    if (!panel) return;
    panel.hidden = false;
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (App.state.taskId && window.HistoryUI) HistoryUI.showDetail(App.state.taskId);
  },

  // ---- 反馈档案治理（学习入口在 8631 编辑器） ----

  async refreshFeedbackConfigs() {
    try {
      var data = await App.api.feedbackProfiles();
      App.ui.renderFeedbackConfigMgmt(data.profiles || []);
      // Also check for conflicts
      if (data.profiles && data.profiles.length > 0) {
        try {
          var conflicts = await App.api.feedbackConflicts(data.profiles[0].profile_id);
          App.ui.renderFeedbackConflicts(conflicts);
        } catch(ex) { /* conflicts check failed silently */ }
      }
    } catch (ex) {
      console.error('Failed to load feedback configs:', ex);
    }
  },

  async showFeedbackProfile(name) {
    try {
      var profile = await App.api.feedbackProfile(name);
      var mgmtEl = $('#feedback-config-mgmt');
      if (!mgmtEl) return;
      var html = '<div style="font-size:0.78rem;font-weight:600;color:var(--text-primary);margin-top:4px;">' + App.ui.escapeHtml(name) + '</div>';
      html += '<div style="font-size:0.74rem;color:var(--text-tertiary);">学习次数: ' + (profile.feedback_count || 0) + ' | Few-shot: ' + (profile.few_shot_examples_count || 0) + '</div>';
      if (profile.overrides && Object.keys(profile.overrides).length > 0) {
        html += '<div style="font-size:0.74rem;color:var(--text-secondary);margin-top:2px;">参数覆盖:</div>';
        Object.entries(profile.overrides).forEach(function(e) {
          html += '<div class="feedback-metric-row"><span class="fm-label" style="font-family:var(--font-mono);font-size:0.62rem;">' + App.ui.escapeHtml(e[0]) + '</span><span class="fm-value">' + (typeof e[1] === 'number' ? e[1].toFixed(3) : e[1]) + '</span></div>';
        });
      }
      html += '<div class="feedback-btn-row" style="margin-top:4px;">';
      html += '<button class="feedback-mini-btn" onclick="App.handleFeedbackRollback(\'' + name + '\')">↩ 回滚</button>';
      html += '<button class="feedback-mini-btn danger" onclick="App.handleFeedbackReset(\'' + name + '\')">🗑 重置</button>';
      html += '</div>';
      mgmtEl.innerHTML = html;
    } catch (ex) {
      toast('加载配置失败: ' + ex.message, 'error');
    }
  },

  async handleFeedbackRollback(profileName) {
    if (!confirm('确认回滚配置 "' + profileName + '" 到上一个备份版本？')) return;
    try {
      var result = await App.api.feedbackProfileRollback(profileName);
      toast('已回滚到 ' + result.updated_at + ' 的版本', 'success');
      App.showFeedbackProfile(profileName);
    } catch (ex) {
      toast('回滚失败: ' + ex.message, 'error');
    }
  },

  async handleFeedbackReset(profileName) {
    if (profileName === 'user_default') {
      if (!confirm('确认重置默认配置？所有学习到的参数将丢失。')) return;
    }
    try {
      await App.api.feedbackProfileDelete(profileName);
      toast('配置 "' + profileName + '" 已重置', 'success');
      App.refreshFeedbackConfigs();
    } catch (ex) {
      if (ex.message && ex.message.indexOf('Cannot delete the default profile') >= 0) {
        toast('无法删除默认配置文件，请使用回滚功能', 'error');
      } else {
        toast('重置失败: ' + ex.message, 'error');
      }
    }
  },

  renderFeedbackConflicts(conflicts) {
    if (!conflicts || !conflicts.conflicts || conflicts.conflicts.length === 0) return;
    var activeConflicts = conflicts.conflicts.filter(function(c) { return c.is_oscillating; });
    if (activeConflicts.length === 0) return;

    var html = '<div style="margin-top:6px;">';
    html += '<div style="font-size:0.72rem;font-weight:600;color:var(--accent-orange);">⚠️ 参数冲突</div>';
    activeConflicts.forEach(function(c) {
      html += '<div class="feedback-conflict-warn">';
      html += '<div class="fc-header">' + App.ui.escapeHtml(c.param_path) + ' (' + c.severity + ')</div>';
      html += '<div style="font-size:0.72rem;color:var(--text-secondary);">震荡 ' + c.oscillation_count + ' 次 · 建议: ' + (c.recommended_action || 'review') + '</div>';
      if (c.suggested_actions && c.suggested_actions.length > 0) {
        html += '<div class="feedback-btn-row" style="margin-top:2px;">';
        c.suggested_actions.forEach(function(a) {
          html += '<button class="feedback-mini-btn" onclick="App.resolveFeedbackConflict(\'' + conflicts.profile_id + '\',\'' + c.param_path + '\',\'' + a.id + '\')">' + App.ui.escapeHtml(a.label) + '</button>';
        });
        html += '</div>';
      }
      html += '</div>';
    });
    html += '</div>';
    var configEl = $('#feedback-config-mgmt');
    if (configEl) {
      configEl.innerHTML = html + (configEl.innerHTML || '');
    }
  },

  async resolveFeedbackConflict(profileName, paramPath, action) {
    try {
      var result = await App.api.feedbackConflictResolve(profileName, paramPath, action);
      toast(result.message || '冲突已解决', 'success');
      App.refreshFeedbackConfigs();
    } catch (ex) {
      toast('解决失败: ' + ex.message, 'error');
    }
  },

  // ---- Init ----
  init() {
    document.addEventListener('DOMContentLoaded', () => {
      App.ui.init();
    });
  }
};

// 暴露到全局作用域，使内联事件处理器 (onclick/oninput/onchange) 能找到 App
window.App = App;

// Shared helpers used by the split ui-* modules.  They used to live in the
// inline application script, so publish them before App.init invokes a UI
// module loaded earlier by index.html.
window.$ = $;
window.$$ = $$;
window.formatTime = formatTime;
window.toast = toast;

App.init();

})();
