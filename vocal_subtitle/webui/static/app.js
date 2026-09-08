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

/**
 * 计算原始文本和优化后文本的词级差异，返回高亮 HTML。
 * 算法：对 token 序列做 LCS (Longest Common Subsequence)，
 * 回溯分类每个 token 为 same / added，用不同颜色标记。
 * CJK 字符逐字分词，拉丁语言逐词分词。
 */
function diffAndHighlight(original, optimized) {
  if (!original || original === optimized) return App.ui.escapeHtml(optimized);

  // 分词：CJK 单字、拉丁单词、空白、标点
  function tokenize(text) {
    var tokens = [];
    var re = /([一-鿿㐀-䶿]|[\w]+|[^\w\s]|\s+)/g;
    var m;
    while ((m = re.exec(text)) !== null) {
      tokens.push({ text: m[1], pos: m.index });
    }
    return tokens;
  }

  var origTokens = tokenize(original);
  var optTokens = tokenize(optimized);
  var origWords = origTokens.map(function(t) { return t.text; });
  var optWords = optTokens.map(function(t) { return t.text; });
  var m = origWords.length;
  var n = optWords.length;

  // LCS DP 表
  var dp = new Array(m + 1);
  for (var i = 0; i <= m; i++) {
    dp[i] = new Array(n + 1);
    for (var j = 0; j <= n; j++) dp[i][j] = 0;
  }
  for (var i = 1; i <= m; i++) {
    for (var j = 1; j <= n; j++) {
      if (origWords[i - 1] === optWords[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // 回溯分类每个 optimized token
  var tags = new Array(n);
  for (var k = 0; k < n; k++) tags[k] = 'added';
  var i = m, j = n;
  while (i > 0 && j > 0) {
    if (origWords[i - 1] === optWords[j - 1]) {
      tags[j - 1] = 'same';
      i--; j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }

  // 构建高亮 HTML
  var result = [];
  for (var k = 0; k < n; k++) {
    var word = App.ui.escapeHtml(optWords[k]);
    if (tags[k] === 'same') {
      result.push(word);
    } else {
      result.push('<span class="diff-added">' + word + '</span>');
    }
  }
  return result.join('');
}

/**
 * 将秒数格式化为 SRT 时间戳 (HH:MM:SS,mmm)
 */
function formatSRTTime(seconds) {
  var h = Math.floor(seconds / 3600);
  var m = Math.floor((seconds % 3600) / 60);
  var s = Math.floor(seconds % 60);
  var ms = Math.floor((seconds % 1) * 1000);
  return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0') + ',' + String(ms).padStart(3, '0');
}

/**
 * 将秒数格式化为 ASS 时间戳 (H:MM:SS.cc)
 */
function formatASSTime(seconds) {
  var h = Math.floor(seconds / 3600);
  var m = Math.floor((seconds % 3600) / 60);
  var s = Math.floor(seconds % 60);
  var cs = Math.floor((seconds % 1) * 100);
  return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0') + '.' + String(cs).padStart(2, '0');
}

/**
 * 将单条字幕事件格式化为 SRT 条目
 */
function formatEventAsSRT(event) {
  return event.index + '\n' + formatSRTTime(event.start) + ' --> ' + formatSRTTime(event.end) + '\n' + event.text;
}

/**
 * 将单条字幕事件格式化为 ASS Dialogue 行
 */
function formatEventAsASS(event) {
  return 'Dialogue: 0,' + formatASSTime(event.start) + ',' + formatASSTime(event.end) + ',Default,,0,0,0,,' + event.text;
}

/**
 * 检测字幕事件是否存在冲突
 *
 * 仅将以下情况视为真正的冲突（需要人工确认）：
 * 1. LLM 将其他条目的内容复制到了当前条目（跨条目内容泄漏）
 * 2. 不同说话人的文本被合并到同一条字幕
 *
 * 普通的 LLM 修正（拼写、标点、语序、语气词去除）不再视为冲突，
 * 最终版本直接使用 LLM 优化后的文本。
 *
 * @param {Object} event - 字幕事件
 * @param {Array} allEvents - 所有字幕事件（用于相邻条目对比）
 * @returns {boolean} 是否存在需要人工确认的冲突
 */
function hasConflict(event, allEvents) {
  // 无原始文本 = LLM 未修改 → 无冲突
  if (!event.original_text) return false;
  // 文本未变 → 无冲突
  if (event.original_text === event.text) return false;

  // LLM 做了修改 → 检查是否为跨条目内容泄漏
  if (allEvents && allEvents.length > 0) {
    var idx = event.index;

    // 检查前一相邻条目：前一条是否吸收了当前条目的内容
    var prev = null;
    for (var i = 0; i < allEvents.length; i++) {
      if (allEvents[i].index === idx - 1) { prev = allEvents[i]; break; }
    }
    if (prev && prev.text && event.text.trim().length >= 3) {
      if (prev.text.indexOf(event.text.trim()) >= 0) {
        return true;  // 当前条目的内容被前一条吸收了
      }
    }

    // 检查后一相邻条目：当前条目是否吸收了后一条目的内容
    var next = null;
    for (var i = 0; i < allEvents.length; i++) {
      if (allEvents[i].index === idx + 1) { next = allEvents[i]; break; }
    }
    if (next && next.text && next.text.trim().length >= 3) {
      if (event.text.indexOf(next.text.trim()) >= 0) {
        return true;  // 当前条目吸收了后一条的内容
      }
    }
  }

  // 普通 LLM 修正（拼写、标点、语气词去除等）→ 自动接受
  return false;
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

    App.state.selectedSubtitleIndexes.clear();
    App.state.selectionAnchorIndex = null;
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
	      const data = await App.api.getHistory(20, 0);
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
	    App.ui.clearSubtitleSelection(false);

	    // 先用列表摘要快速显示统计信息
	    if (item.result_summary && item.result_summary.stats) {
	      const stats = item.result_summary.stats;
	      $('#stats-grid').innerHTML = `
	        <div class="stat-card"><div class="stat-value">${stats.total_time ? stats.total_time.toFixed(1)+'s' : '缓存'}</div><div class="stat-label">总耗时</div></div>
	        <div class="stat-card"><div class="stat-value">${item.result_summary.segment_count || 0}</div><div class="stat-label">语音片段</div></div>
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

	    // 获取完整详情（含字幕事件 + 音频路径），完善显示
	    App.api._fetch('/api/history/' + item.id).then(detail => {
	      const summary = detail.result_summary || {};
	      if (summary.stats) {
	        App.ui.renderStats(summary.stats);
	        $('#empty-state').style.display = 'none';
	        $('#results-content').style.display = 'block';
	        App.ui.renderDiagnosticReport({ stats: summary.stats });
	      }
      if (detail.events && detail.events.length > 0) {
        App.state.subtitleEvents = detail.events;
        App.ui.renderTimeline(detail.events);
        if (window.WaveformUI) {
          WaveformUI.setEvents(detail.events);
          WaveformUI.setTask(App.state.taskId, summary, detail.events);
        }
      }
	      // 使用详情中的完整 result_summary（含 vocals_path / accompaniment_path）更新导出栏
	      App.ui.updateAudioExportBar(detail.result_summary || {});
	      App.ui.updateExportBar(detail.result_summary || {});
	    }).catch(ex => {
	      console.error('Failed to load history detail:', ex);
	    });
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

  // ---- Feedback Business Logic (Phase 5) ----
  _fbRefFile: null,
  _fbAudioFile: null,

  pickRefFile() {
    var input = $('#fb-ref-input');
    if (input) input.click();
  },

  pickAudioFile() {
    var input = $('#fb-audio-input');
    if (input) input.click();
  },

  async handleFeedbackLearn() {
    var refFile = App.state._fbRefFile || (App.state._fbRefBlob ? new File([App.state._fbRefBlob], 'edited.srt', {type: 'text/plain'}) : null);
    var audioFile = App.state._fbAudioFile;
    if (!refFile || !audioFile) {
      toast('请先选择修订字幕文件和音频文件', 'error');
      return;
    }
    var learnBtn = $('#fb-learn-btn');
    if (learnBtn) { learnBtn.disabled = true; learnBtn.textContent = '学习中...'; }
    try {
      var report = await App.api.feedbackLearn(audioFile, refFile, App.state.selectedProfile, 'user_default', true, false);
      App.ui.renderFeedbackReport(report);
      App.refreshFeedbackConfigs();
      if (report.status === 'ok') {
        toast('反馈学习完成！已更新 ' + Object.keys(report.param_adjustments || {}).length + ' 个参数', 'success');
      } else {
        toast('学习未完成: ' + (report.message || '未知错误'), 'info');
      }
    } catch (ex) {
      toast('学习失败: ' + ex.message, 'error');
    } finally {
      if (learnBtn) { learnBtn.disabled = false; learnBtn.textContent = '📚 开始学习'; }
    }
  },

  async handleFeedbackPreview() {
    var refFile = App.state._fbRefFile;
    var audioFile = App.state._fbAudioFile;
    if (!refFile || !audioFile) {
      toast('请先选择修订字幕文件和音频文件', 'error');
      return;
    }
    var previewBtn = $('#fb-preview-btn');
    if (previewBtn) { previewBtn.disabled = true; previewBtn.textContent = '分析中...'; }
    try {
      var report = await App.api.feedbackPreview(audioFile, refFile, App.state.selectedProfile);
      App.ui.renderFeedbackReport(report);
      toast('差异预览完成 — 未实际更新配置', 'info');
    } catch (ex) {
      toast('预览失败: ' + ex.message, 'error');
    } finally {
      if (previewBtn) { previewBtn.disabled = false; previewBtn.textContent = '🔍 预览差异'; }
    }
  },

  async handleFeedbackLearnFromEdits() {
    var events = App.state.subtitleEvents;
    if (!events || events.length === 0) {
      toast('当前没有字幕数据，请先处理音频', 'error');
      return;
    }
    // Build SRT from current subtitle events
    var srt = '';
    events.forEach(function(e, i) {
      srt += (i + 1) + '\n';
      srt += formatEventAsSRT(e) + '\n\n';
    });
    var blob = new Blob([srt], {type: 'text/plain'});
    App.state._fbRefBlob = blob;
    App.state._fbRefFile = new File([blob], 'edited.srt', {type: 'text/plain'});

    // Try to get audio from current task
    if (App.state.taskId) {
      try {
        var audioUrl = '/api/tasks/' + App.state.taskId + '/audio?type=input';
        var resp = await fetch(audioUrl);
        if (resp.ok) {
          var audioBlob = await resp.blob();
          App.state._fbAudioFile = new File([audioBlob], 'audio.bin', {type: audioBlob.type || 'audio/wav'});
        }
      } catch(ex) { /* audio download failed, user will need to pick manually */ }
    }

    var statusEl = $('#fb-file-status');
    if (statusEl) {
      statusEl.textContent = '修订字幕: 从 ' + events.length + ' 条当前编辑结果生成' + (App.state._fbAudioFile ? ' | 音频: 已复用' : ' | ⚠️ 请手动选择音频文件');
    }
    App.ui._updateFbButtons();

    if (!App.state._fbAudioFile) {
      toast('修订字幕已就绪！请再选择音频文件后点击"开始学习"', 'info');
    } else {
      toast('修订字幕和音频已就绪！点击"开始学习"提交', 'success');
    }
  },

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
      var html = '<div style="font-size:0.72rem;font-weight:600;color:var(--text-primary);margin-top:4px;">' + App.ui.escapeHtml(name) + '</div>';
      html += '<div style="font-size:0.65rem;color:var(--text-tertiary);">学习次数: ' + (profile.feedback_count || 0) + ' | Few-shot: ' + (profile.few_shot_examples_count || 0) + '</div>';
      if (profile.overrides && Object.keys(profile.overrides).length > 0) {
        html += '<div style="font-size:0.66rem;color:var(--text-secondary);margin-top:2px;">参数覆盖:</div>';
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
    var reportEl = $('#feedback-report');
    if (!conflicts || !conflicts.conflicts || conflicts.conflicts.length === 0) return;
    var activeConflicts = conflicts.conflicts.filter(function(c) { return c.is_oscillating; });
    if (activeConflicts.length === 0) return;

    var html = '<div style="margin-top:6px;">';
    html += '<div style="font-size:0.72rem;font-weight:600;color:var(--accent-orange);">⚠️ 参数冲突</div>';
    activeConflicts.forEach(function(c) {
      html += '<div class="feedback-conflict-warn">';
      html += '<div class="fc-header">' + App.ui.escapeHtml(c.param_path) + ' (' + c.severity + ')</div>';
      html += '<div style="font-size:0.62rem;color:var(--text-secondary);">震荡 ' + c.oscillation_count + ' 次 · 建议: ' + (c.recommended_action || 'review') + '</div>';
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
    // Append to report or config area
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

  // ---- 波形打轴：时间轴保存 ----
  async saveWaveformTiming(index, start, end) {
    if (!App.state.taskId) throw new Error('当前没有已加载的任务');
    await App.api.updateSubtitleTiming(App.state.taskId, index, start, end);
    var event = App.state.subtitleEvents.find(function(e) { return Number(e.index) === Number(index); });
    if (event) {
      event.start = start;
      event.end = end;
      if (App.ui.updateTimelineRowTimes) App.ui.updateTimelineRowTimes(event);
    }
  },

  // ---- Init ----
  init() {
    document.addEventListener('DOMContentLoaded', () => {
      App.ui.init();
      if (window.WaveformUI) {
        WaveformUI.onTimingChange = function(index, start, end) {
          return App.saveWaveformTiming(index, start, end);
        };
      }
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
window.formatSRTTime = formatSRTTime;
window.formatASSTime = formatASSTime;
window.formatEventAsSRT = formatEventAsSRT;
window.formatEventAsASS = formatEventAsASS;
window.diffAndHighlight = diffAndHighlight;
window.hasConflict = hasConflict;
window.toast = toast;

App.init();

})();
