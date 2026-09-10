(function() {
'use strict';
const part = {
    onDeviceChange() {
      App.ui.updateDeviceHint();
    },

    // 更新设备提示文字（显示 "auto" 解析到的实际设备）
    updateDeviceHint() {
      var sel = document.querySelector('#options-panel [data-key="device"]');
      var hint = $('#device-hint');
      if (!sel || !hint) return;
      if (sel.value === 'auto') {
        var badge = $('#device-label');
        var text = badge ? badge.textContent : '检测中...';
        var isGpu = text.indexOf('GPU') >= 0 || text.indexOf('Silicon') >= 0 || text.indexOf('✓') >= 0;
        hint.innerHTML = '→ <span class="' + (isGpu ? 'gpu' : 'cpu') + '">' + text + '</span>';
        hint.style.display = 'inline';
      } else {
        hint.style.display = 'none';
      }
    },

    // Progress Updates
    stageStart(stage, total, description) {
      const row = $('#stage-' + stage);
      if (!row) return;
      row.style.display = 'flex';  // Auto-show hidden stages
      row.classList.add('active');
      row.classList.remove('completed', 'error');
      const desc = row.querySelector('.stage-desc');
      if (desc) desc.textContent = description || '处理中...';
      const bar = $('#bar-' + stage);
      if (bar) {
        bar.style.width = '0%';
        bar.classList.remove('completed', 'error');
        // Add indeterminate shimmer for stages that may not report granular progress
        bar.classList.add('indeterminate');
      }
      // 不重置时间显示 — 多块处理时保留已累计的时间
      // 仅首次显示 '...'
      if (!row._stageAccumulatedTime) {
        $('#time-' + stage).textContent = '...';
      }
      // Record start time for minimum visible duration
      row._stageStartTime = performance.now();
    },

    stageProgress(stage, current, total, extra) {
      const bar = $('#bar-' + stage);
      if (bar && total > 0) {
        // Real granular progress arriving — remove indeterminate shimmer
        bar.classList.remove('indeterminate');
        bar.style.width = Math.round((current / total) * 100) + '%';
      }
      // Update stage description with detailed progress info (e.g., "处理音频块: 26/53")
      if (extra && extra.detail) {
        const row = $('#stage-' + stage);
        if (row) {
          const desc = row.querySelector('.stage-desc');
          if (desc) desc.textContent = extra.detail;
        }
      }
    },

    stageFinish(stage, elapsed) {
      const row = $('#stage-' + stage);
      if (!row) return;

      // 累加耗时（多块处理场景：每个块完成都发一次 stage_finish）
      if (!row._stageAccumulatedTime) row._stageAccumulatedTime = 0;
      row._stageAccumulatedTime += (elapsed || 0);

      const MIN_DISPLAY_MS = 350; // Minimum visible time so fast stages are perceivable

      const doFinish = () => {
        row.classList.remove('active');
        row.classList.add('completed');
        const bar = $('#bar-' + stage);
        if (bar) {
          bar.classList.remove('indeterminate');
          bar.style.width = '100%';
          bar.classList.add('completed');
        }
        // 显示累计耗时
        $('#time-' + stage).textContent = row._stageAccumulatedTime.toFixed(1) + 's';
        // Preserve informative detail text (e.g. "检测到 52 个语音段")
        // only override generic placeholders
        const desc = row.querySelector('.stage-desc');
        if (desc) {
          const cur = desc.textContent;
          if (!cur || cur === '处理中...' || cur === '等待开始') {
            desc.textContent = '完成';
          }
        }
      };

      const startTime = row._stageStartTime || 0;
      const visibleDuration = performance.now() - startTime;

      if (visibleDuration > 0 && visibleDuration < MIN_DISPLAY_MS) {
        setTimeout(doFinish, MIN_DISPLAY_MS - visibleDuration);
      } else {
        doFinish();
      }
    },

    renderStats(stats) {
      stats = stats || {};
      $('#stats-grid').innerHTML = `
        <div class="stat-card"><div class="stat-value">${(stats.total_time || 0).toFixed(1)}s</div><div class="stat-label">总耗时</div></div>
        <div class="stat-card"><div class="stat-value">${stats.segment_count || 0}</div><div class="stat-label">语音片段</div></div>
        <div class="stat-card"><div class="stat-value">${stats.subtitle_count || 0}</div><div class="stat-label">字幕条数</div></div>
        <div class="stat-card"><div class="stat-value">${stats.duration_seconds ? (stats.duration_seconds/60).toFixed(1)+'min' : '—'}</div><div class="stat-label">音频时长</div></div>
        <div class="stat-card"><div class="stat-value">${stats.speaker_count || 0}</div><div class="stat-label">说话人数</div></div>
        <div class="stat-card"><div class="stat-value">${stats.local_speaker_split_count || 0}</div><div class="stat-label">局部换人切分</div></div>
        <div class="stat-card"><div class="stat-value">${stats.speaker_conflict_count || 0}</div><div class="stat-label">说话人冲突</div></div>
        <div class="stat-card"><div class="stat-value">${stats.unknown_speaker_count || 0}</div><div class="stat-label">未确认</div></div>
        <div class="stat-card"><div class="stat-value">${App.ui.escapeHtml(stats.status || 'completed')}</div><div class="stat-label">运行状态</div></div>
        <div class="stat-card"><div class="stat-value">${App.ui.escapeHtml(stats.quality_status || 'pass')}</div><div class="stat-label">质量状态</div></div>
        <div class="stat-card"><div class="stat-value">${App.ui.escapeHtml((stats.final_engine || stats.selected_engine || '—'))}</div><div class="stat-label">最终引擎</div></div>
        <div class="stat-card"><div class="stat-value">${App.ui.escapeHtml(stats.detected_language || 'unknown')}</div><div class="stat-label">检测语言</div></div>
        <div class="stat-card"><div class="stat-value" title="${App.ui._escapeAttr(stats.fallback_reason || '')}">${App.ui.escapeHtml(stats.fallback_reason ? '已回退' : '—')}</div><div class="stat-label">回退原因</div></div>
      `;
    },

    async pipelineComplete(result) {
      App.ws.disconnect();
      App.state.isRunning = false;
      App.ui.setRunning(false);
      App.ui.clearPipelineError();

      const stats = result.stats;
      const eventCount = (result.events || []).length;

      // Show stats
      App.ui.renderStats(stats);

      // Render diagnostic report if available
      App.ui.renderDiagnosticReport(result);

      // Update button
      $('#btn-run-text').textContent = '处理完成 ✓';
      setTimeout(() => {
        if (!App.state.isRunning) {
          $('#btn-run-text').textContent = '开始处理';
        }
      }, 3000);

      const runtimeStatus = stats.status || result.status || 'completed';
      const qualityStatus = stats.quality_status || 'pass';
      const railStatus = document.getElementById('process-rail-status');
      if (railStatus) {
        const degraded = runtimeStatus === 'degraded' || runtimeStatus === 'degraded_completed' || qualityStatus !== 'pass';
        railStatus.textContent = degraded ? '降级完成' : '已完成';
        railStatus.dataset.status = degraded ? 'degraded_completed' : 'completed';
      }
      const qualityMessage = qualityStatus === 'pass'
        ? '字幕生成完成'
        : '字幕已生成，但质量状态为 ' + qualityStatus;
      const statusMessage = runtimeStatus === 'degraded' || runtimeStatus === 'degraded_completed' ? '，运行状态为降级完成' : '';
      toast(qualityMessage + statusMessage + '，共 ' + eventCount + ' 条', runtimeStatus === 'completed' && qualityStatus === 'pass' ? 'success' : 'warning');

      // Refresh history and cache info
      App.refreshHistory();
      App.refreshCacheInfo();

      // 刷新反馈档案治理数据（学习入口在 subtitle-editor 8631）
      App.refreshFeedbackConfigs();

      // Display LLM stage if enabled
      if (stats.stage_timings && stats.stage_timings.llm) {
        $('#stage-llm').style.display = 'flex';
        App.ui.stageFinish('llm', stats.stage_timings.llm);
      }

      // Apply backend cumulative stage timings (authoritative, covers multi-chunk accumulation)
      if (stats.stage_timings) {
        var st = stats.stage_timings;
        // Helper: apply cumulative time from backend, overriding frontend accumulator
        var applyTiming = function(stageKey, displayName) {
          if (st[stageKey] !== undefined) {
            var row = $('#stage-' + stageKey);
            if (row) {
              row.style.display = 'flex';
              if (!row.classList.contains('completed')) {
                App.ui.stageFinish(stageKey, 0);  // mark completed
              }
              // Override accumulated time with authoritative backend value
              row._stageAccumulatedTime = st[stageKey];
              $('#time-' + stageKey).textContent = st[stageKey].toFixed(1) + 's';
            }
          }
        };
        applyTiming('macro_chunk');
        applyTiming('separation');
        applyTiming('vad');
        applyTiming('ffmpeg_vad');
        applyTiming('merging');
        applyTiming('asr');
        applyTiming('boundary_refine');
        applyTiming('mapping');
        applyTiming('acoustic');
        applyTiming('llm');
        applyTiming('llm_merge');
      }

      // Show/hide audio + subtitle export bars based on result
      App.ui.updateAudioExportBar(result);
      App.ui.updateExportBar(result);

      // 任务详情 + 质量报告内联面板（双界面收敛：历史/质量并入处理台）
      if (window.HistoryUI) HistoryUI.showDetail(App.state.taskId);
    },

    pipelineError(message) {
      App.ws.disconnect();
      App.state.isRunning = false;
      App.ui.setRunning(false);
      const errorMessage = message || '处理失败';
      const errorPanel = $('#process-error');
      if (errorPanel) {
        errorPanel.textContent = '处理失败: ' + errorMessage;
        errorPanel.hidden = false;
      }
      toast('处理失败: ' + errorMessage, 'error');
      $('#btn-run-text').textContent = '重试';
      const railStatus = document.getElementById('process-rail-status');
      if (railStatus) {
        railStatus.textContent = '处理失败';
        railStatus.dataset.status = 'failed';
      }
    },

    clearPipelineError() {
      const errorPanel = $('#process-error');
      if (errorPanel) {
        errorPanel.textContent = '';
        errorPanel.hidden = true;
      }
    },

    setRunning(running) {
      const btn = $('#btn-run');
      if (running) {
        App.ui.clearPipelineError();
        const railStatus = document.getElementById('process-rail-status');
        if (railStatus) {
          railStatus.textContent = '处理中';
          railStatus.dataset.status = 'running';
        }
        btn.classList.add('running');
        btn.disabled = true;
        $('#btn-run-text').textContent = '处理中...';
        // Reset all stages
        var allStages = ['macro_chunk','separation','vad','ffmpeg_vad','merging','asr','boundary_refine','mapping','acoustic','llm'];
        allStages.forEach(function(s) {
          var row = $('#stage-' + s);
          if (row) {
            row.classList.remove('active','completed','error');
            row._stageAccumulatedTime = 0;     // 重置累计耗时
            row._stageStartTime = 0;
          }
          var bar = $('#bar-' + s);
          if (bar) { bar.style.width = '0%'; bar.classList.remove('completed','error'); }
          var timeEl = $('#time-' + s);
          if (timeEl) timeEl.textContent = '—';
          var desc = row ? row.querySelector('.stage-desc') : null;
          if (desc) desc.textContent = '等待开始';
        });
        // Hide optional stages
        var optionalStages = ['macro_chunk','ffmpeg_vad','boundary_refine','acoustic','llm'];
        optionalStages.forEach(function(s) {
          var row = $('#stage-' + s);
          if (row) row.style.display = 'none';
        });
        // Hide diagnostic report during processing
        var diagPanel = document.getElementById('diagnostic-panel');
        if (diagPanel) diagPanel.style.display = 'none';
        // Hide audio export bar during processing
        var audioBar = document.getElementById('audio-export-bar');
        if (audioBar) audioBar.style.display = 'none';
      } else {
        btn.classList.remove('running');
        btn.disabled = !App.state.selectedFile;
      }
    },

    // 从已删除的 ui-subtitles.js 迁入：仅保留导出栏控制（表格/波形已随双界面收敛移除）

    updateAudioExportBar(result) {
      var bar = document.getElementById('audio-export-bar');
      if (!bar) return;
      var hasVocals = !!(result && result.vocals_path);
      var hasAccomp = !!(result && result.accompaniment_path);
      if (hasVocals || hasAccomp) {
        bar.style.display = 'flex';
        // 如果没有人声或伴奏，隐藏对应按钮
        var btns = bar.querySelectorAll('.btn-export');
        if (btns[0]) btns[0].style.display = hasVocals ? '' : 'none';
        if (btns[1]) btns[1].style.display = hasAccomp ? '' : 'none';
      } else {
        bar.style.display = 'none';
      }
    },

    // 根据 LLM 优化是否生效，调整字幕下载栏的按钮布局
    // - LLM 启用时：隐藏主按钮，显示「干净版」+「LLM 优化版」
    // - LLM 未启用时：只显示主按钮「字幕文件」
    updateExportBar(result) {
      var mainBtn = document.getElementById('btn-srt-main');
      var cleanBtn = document.getElementById('btn-clean-srt');
      var llmBtn = document.getElementById('btn-llm-srt');
      var hasLLM = !!(result && result.llm_subtitle_path);

      if (mainBtn) mainBtn.style.display = hasLLM ? 'none' : '';
      if (cleanBtn) cleanBtn.style.display = hasLLM ? '' : 'none';
      if (llmBtn) llmBtn.style.display = hasLLM ? '' : 'none';
    },
};
window.VocalSubtitleUi = Object.assign(window.VocalSubtitleUi || {}, part);
})();
