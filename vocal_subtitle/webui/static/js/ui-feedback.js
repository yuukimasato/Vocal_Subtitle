(function() {
'use strict';
const part = {
    renderHistory(items) {
      App.state.historyItems = items;
      const panel = $('#history-panel');
      if (!items || items.length === 0) {
        panel.innerHTML = '<div class="history-empty">暂无历史记录</div>';
        return;
      }
      panel.innerHTML = items.map((item, idx) => {
        const date = item.created_at ? item.created_at.slice(0, 16).replace('T', ' ') : '';
        const statusClass = item.status === 'completed' ? 'ok' : (item.status === 'failed' ? 'err' : '');
        const statusText = item.status === 'completed' ? '✓' : (item.status === 'failed' ? '✗' : '…');
        const duration = item.total_duration_seconds > 0 ? (item.total_duration_seconds / 60).toFixed(1) + 'min' : '';
        const isClickable = item.status === 'completed';
        return '<div class="history-item" data-idx="' + idx + '" style="' + (isClickable ? '' : 'cursor:default;opacity:0.6;') + '">' +
          '<span class="h-status ' + statusClass + '">' + statusText + '</span>' +
          '<span class="h-name" title="' + App.ui.escapeHtml(item.input_file_name) + '">' + App.ui.escapeHtml(item.input_file_name) + '</span>' +
          '<span class="h-meta">' + duration + ' ' + date + '</span>' +
          '<span class="h-delete" data-delete="' + item.id + '" title="删除">×</span>' +
          '</div>';
      }).join('');

      // Bind click events
      panel.querySelectorAll('.history-item[data-idx]').forEach(function(el) {
        el.addEventListener('click', function() {
          var idx = parseInt(this.dataset.idx);
          if (idx >= 0 && App.state.historyItems[idx]) {
            App.handleHistoryClick(App.state.historyItems[idx]);
          }
        });
      });
      panel.querySelectorAll('.h-delete[data-delete]').forEach(function(el) {
        el.addEventListener('click', function(e) {
          e.stopPropagation();
          App.deleteHistoryItem(this.dataset.delete, e);
        });
      });
    },

    // ---- Diagnostic Report Rendering ----
    renderDiagnosticReport(result) {
      var stats = result.stats || {};
      var report = stats.diagnostic_report;
      var panel = $('#diagnostic-panel');
      if (!panel) return;

      if (!report || report.health_score === undefined) {
        panel.style.display = 'none';
        return;
      }

      panel.style.display = 'block';
      var score = report.health_score;
      var circle = $('#health-score-circle');
      circle.textContent = score.toFixed(0) + '%';
      circle.className = 'score-circle ' + (score >= 90 ? 'good' : (score >= 70 ? 'warn' : 'bad'));

      // Render metrics
      var metricsHtml = '';
      var details = [];
      if (report.start_in_silence > 0) {
        metricsHtml += '<div class="diagnostic-metric issue"><span class="dm-icon">▶</span><span>开头误入静音:</span><span class="dm-value">' + report.start_in_silence + '</span></div>';
      }
      if (report.end_in_silence > 0) {
        metricsHtml += '<div class="diagnostic-metric ' + (report.end_truncated > 0 ? 'bad' : 'issue') + '"><span class="dm-icon">⏹</span><span>结尾误入静音:</span><span class="dm-value">' + report.end_in_silence + '</span></div>';
      }
      if (report.end_truncated > 0) {
        metricsHtml += '<div class="diagnostic-metric bad"><span class="dm-icon">✂</span><span>切尾事件:</span><span class="dm-value">' + report.end_truncated + '</span></div>';
      }
      if (report.snapped_starts !== undefined) {
        metricsHtml += '<div class="diagnostic-metric ok"><span class="dm-icon">↔</span><span>吸附修正:</span><span class="dm-value">' + (report.snapped_starts + (report.snapped_ends || 0)) + '</span></div>';
      }
      if (report.total_events) {
        metricsHtml += '<div class="diagnostic-metric"><span class="dm-icon">📋</span><span>校验事件:</span><span class="dm-value">' + report.total_events + '</span></div>';
      }
      $('#diagnostic-metrics').innerHTML = metricsHtml || '<span style="font-size:0.74rem;color:var(--text-tertiary);">全部通过 ✓</span>';

      // Render flagged events detail
      var flagged = report.events_flagged || [];
      var detailEl = $('#diagnostic-detail');
      if (flagged.length > 0) {
        detailEl.style.display = 'block';
        var detailHtml = '';
        flagged.forEach(function(f) {
          var tagClass = f.issue === 'end_truncated' || f.issue === 'possible_truncation' ? 'trunc' : 'deviate';
          var tagText = f.issue === 'end_truncated' ? '切尾' : (f.issue === 'possible_truncation' ? '疑似切尾' : (f.issue === 'start_deviation' ? '偏移' : f.issue));
          detailHtml += '<div class="dd-item" onclick="App.ui._scrollToEvent(' + f.id + ')">' +
            '<span class="dd-tag ' + tagClass + '">' + tagText + '</span>' +
            '<span>#' + f.id + '</span>' +
            '<span style="color:var(--text-secondary);margin-left:4px;">' + (f.text_preview ? App.ui.escapeHtml(f.text_preview) : '') + '</span>' +
            '<span style="margin-left:auto;font-family:var(--font-mono);font-size:0.7rem;color:var(--accent-orange);">' + (f.deviation_ms ? '+' + f.deviation_ms + 'ms' : '') + '</span>' +
            '</div>';
        });
        detailEl.innerHTML = detailHtml;
      } else {
        detailEl.style.display = 'none';
      }
    },

    _scrollToEvent(index) {
      // Find and highlight the subtitle row
      var rows = document.querySelectorAll('#subtitle-tbody tr');
      rows.forEach(function(r) { r.style.background = ''; });
      var target = document.querySelector('#subtitle-tbody .col-idx');
      // Search by text content
      rows.forEach(function(r) {
        var idxCell = r.querySelector('.col-idx');
        if (idxCell && idxCell.textContent.trim() === String(index)) {
          r.style.background = 'rgba(88,166,255,0.15)';
          r.scrollIntoView({ behavior: 'smooth', block: 'center' });
          setTimeout(function() { r.style.background = ''; }, 3000);
        }
      });
    },

    escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    },

    // ---- Feedback UI Rendering (Phase 5) ----
    showFeedbackEntry(result) {
      var emptyEl = $('#feedback-empty-state');
      var controlsEl = $('#feedback-controls');
      if (emptyEl) emptyEl.textContent = '处理后，可在此提交修订版进行反馈学习';
      if (controlsEl) controlsEl.style.display = 'block';
      // Show "learn from edits" button if we have subtitle events
      var learnEditsBtn = $('#fb-learn-edits');
      if (learnEditsBtn && App.state.subtitleEvents && App.state.subtitleEvents.length > 0) {
        learnEditsBtn.style.display = '';
      }
      // Show learn/preview buttons if files are selected
      App.ui._updateFbButtons();
    },

    renderFeedbackReport(report) {
      var reportEl = $('#feedback-report');
      var dividerEl = $('#feedback-divider');
      if (!reportEl) return;
      reportEl.style.display = 'block';
      if (dividerEl) dividerEl.style.display = 'block';

      var statusClass = report.status === 'ok' ? 'ok' : 'err';
      var statusText = report.status === 'ok' ? '✓ 学习完成' : '✗ ' + (report.message || '错误');
      var structuralNote = report.structural_revision ? ' <span style="color:var(--accent-orange);font-size:0.6rem;">(结构性修订, 权重降低)</span>' : '';

      var html = '<div class="feedback-report">';
      html += '<div class="fr-header"><span class="fr-title">学习报告</span><span class="fr-status ' + statusClass + '">' + statusText + '</span></div>';
      if (report.status === 'ok') {
        html += '<div class="feedback-metric-row"><span class="fm-label">对齐覆盖率</span><span class="fm-value">' + (report.alignment_coverage * 100).toFixed(1) + '% (' + report.total_pairs + ' 对)</span></div>';
        html += '<div class="feedback-metric-row"><span class="fm-label">时间偏移</span><span class="fm-value">' + (report.time_shifts_count || 0) + ' 处</span></div>';
        html += '<div class="feedback-metric-row"><span class="fm-label">合并/拆分</span><span class="fm-value">' + (report.merge_actions_count || 0) + ' 处</span></div>';
        html += '<div class="feedback-metric-row"><span class="fm-label">文本编辑</span><span class="fm-value">' + (report.text_edits_count || 0) + ' 处</span></div>';
      } else if (report.auto_event_count !== undefined) {
        html += '<div class="feedback-metric-row"><span class="fm-label">对齐覆盖率</span><span class="fm-value" style="color:var(--danger)">' + (report.alignment_coverage * 100).toFixed(1) + '% (' + report.total_pairs + ' 对)</span></div>';
        html += '<div class="feedback-metric-row"><span class="fm-label">事件数</span><span class="fm-value">自动: ' + report.auto_event_count + ' 条, 修订: ' + report.manual_event_count + ' 条</span></div>';
      }
      html += structuralNote;

      if (report.param_adjustments && Object.keys(report.param_adjustments).length > 0) {
        html += '<div style="font-size:0.72rem;font-weight:600;color:var(--text-primary);margin-top:6px;">参数调整建议</div>';
        Object.entries(report.param_adjustments).forEach(function(entry) {
          var path = entry[0], adj = entry[1];
          html += '<div class="feedback-param-row">';
          html += '<div class="fp-header"><span class="fp-path">' + App.ui.escapeHtml(path) + '</span>';
          html += '<span class="feedback-badge ' + (adj.direction || '') + '">' + (adj.direction === 'increase' ? '↑ 增大' : '↓ 减小') + '</span>';
         html += '<span class="feedback-badge ' + (adj.param_tier || '') + '">' + (adj.param_tier || '').replace('_', ' ') + '</span>';
          if (adj.confidence !== undefined) {
            var confClass = adj.confidence >= 0.7 ? 'high' : (adj.confidence >= 0.4 ? 'medium' : 'low');
            html += '<span class="feedback-badge ' + confClass + '">置信度 ' + (adj.confidence * 100).toFixed(0) + '%</span>';
          }
          html += '</div>';
          if (adj.reason) html += '<div class="fp-reason">' + App.ui.escapeHtml(adj.reason) + '</div>';
          html += '</div>';
        });
      } else if (report.status === 'ok') {
        html += '<div style="font-size:0.68rem;color:var(--text-tertiary);margin-top:4px;">无需调整参数 — 当前参数已匹配用户偏好</div>';
      }
      html += '</div>';
      reportEl.innerHTML = html;
    },

    renderFeedbackConfigMgmt(profiles) {
      var mgmtEl = $('#feedback-config-mgmt');
      var dividerEl = $('#feedback-divider');
      if (!mgmtEl) return;
      mgmtEl.style.display = 'block';
      if (dividerEl) dividerEl.style.display = 'block';

      if (!profiles || profiles.length === 0) {
        mgmtEl.innerHTML = '<div class="feedback-empty">暂无用户配置</div>';
        return;
      }

      var html = '<div style="font-size:0.72rem;font-weight:600;color:var(--text-primary);margin-bottom:4px;">用户配置</div>';
      html += '<div class="feedback-btn-row" style="margin-bottom:4px;">';
      html += '<button class="feedback-mini-btn" onclick="App.refreshFeedbackConfigs()">🔄 刷新</button>';
      html += '</div>';

      profiles.forEach(function(p) {
        html += '<div class="feedback-config-item" onclick="App.showFeedbackProfile(\'' + p.profile_id + '\')">';
        html += '<span class="fci-name">' + App.ui.escapeHtml(p.profile_id) + '</span>';
        html += '<span class="fci-meta">' + (p.feedback_count || 0) + ' 次学习</span>';
        html += '<span class="fci-meta">' + (p.overrides ? Object.keys(p.overrides).length : 0) + ' 参数</span>';
        html += '</div>';
      });
      mgmtEl.innerHTML = html;
    },

    renderFeedbackHealth(healthData) {
      var reportEl = $('#feedback-report');
      if (!reportEl || !healthData || !healthData.trend || healthData.trend.length === 0) return;
      var latest = healthData.trend[healthData.trend.length - 1];
      if (!latest || latest.health_after === undefined) return;
      var score = latest.health_after;
      var grade = score >= 85 ? 'excellent' : (score >= 70 ? 'good' : (score >= 50 ? 'fair' : 'poor'));
      var gradeColors = {excellent: 'var(--accent-green)', good: 'var(--accent-green)', fair: 'var(--accent-orange)', poor: 'var(--accent-red)'};
      var html = '<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">';
      html += '<span style="font-size:0.68rem;color:var(--text-secondary);">健康度</span>';
      html += '<span style="font-weight:700;font-size:0.82rem;color:' + (gradeColors[grade] || 'var(--text-primary)') + ';">' + score.toFixed(1) + '%</span>';
      html += '<span class="feedback-badge ' + grade + '" style="font-size:0.6rem;">' + grade + '</span>';
      html += '</div>';
      reportEl.innerHTML = html + (reportEl.innerHTML || '');
    },

    _updateFbButtons() {
      var refFile = App.state._fbRefFile;
      var audioFile = App.state._fbAudioFile;
      var learnBtn = $('#fb-learn-btn');
      var previewBtn = $('#fb-preview-btn');
      if (learnBtn) learnBtn.style.display = (refFile && audioFile) ? '' : 'none';
      if (previewBtn) previewBtn.style.display = (refFile && audioFile) ? '' : 'none';
    },
};
window.VocalSubtitleUi = Object.assign(window.VocalSubtitleUi || {}, part);
})();

