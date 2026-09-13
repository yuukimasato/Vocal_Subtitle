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
        const terminal = item.status === 'completed' || item.status === 'degraded';
        const statusClass = terminal ? (item.status === 'degraded' ? 'warn' : 'ok') : (item.status === 'failed' ? 'err' : '');
        const statusText = terminal ? (item.status === 'degraded' ? '!' : '✓') : (item.status === 'failed' ? '✗' : '…');
        const duration = item.total_duration_seconds > 0 ? (item.total_duration_seconds / 60).toFixed(1) + 'min' : '';
        const isClickable = terminal;
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
          detailHtml += '<div class="dd-item">' +
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

    escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    },

    renderFeedbackConfigMgmt(profiles) {
      var mgmtEl = $('#feedback-config-mgmt');
      if (!mgmtEl) return;
      mgmtEl.style.display = 'block';

      if (!profiles || profiles.length === 0) {
        mgmtEl.innerHTML = '<div class="feedback-empty">暂无用户配置</div>';
        return;
      }

      var html = '<div style="font-size:0.78rem;font-weight:600;color:var(--text-primary);margin-bottom:4px;">用户配置</div>';
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

};
window.VocalSubtitleUi = Object.assign(window.VocalSubtitleUi || {}, part);
})();
