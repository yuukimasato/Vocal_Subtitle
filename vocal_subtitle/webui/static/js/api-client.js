  // ---- API Client ----
  api: {
    async _fetch(url, opts) {
      const res = await fetch(url, opts);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const ct = res.headers.get('content-type') || '';
      return ct.includes('application/json') ? res.json() : res.text();
    },

    async getProfiles() { return this._fetch('/api/profiles'); },
    async getProfileConfig(name) { return this._fetch('/api/profiles/' + name); },
    async getDeviceInfo() { return this._fetch('/api/device'); },
    async getFunASRStatus(model) {
      return this._fetch('/api/asr/funasr/status?model=' + encodeURIComponent(model || ''));
    },
    async prepareFunASR(model) {
      return this._fetch('/api/asr/funasr/prepare', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: model || ''})
      });
    },
    async getSubtitles(taskId) { return this._fetch('/api/subtitle/' + taskId); },
    async updateSubtitle(taskId, index, text) {
      return this._fetch('/api/subtitle/' + taskId + '/' + index, {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: index, text: text})
      });
    },
    async batchEditSubtitles(taskId, payload) {
      return this._fetch('/api/subtitle/' + taskId + '/batch', {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
    },
    async getTaskStatus(taskId) { return this._fetch('/api/tasks/' + taskId); },
	    async getHistory(limit, offset) {
	      const params = new URLSearchParams();
	      if (limit) params.set('limit', limit);
	      if (offset) params.set('offset', offset);
	      const qs = params.toString();
	      return this._fetch('/api/history' + (qs ? '?' + qs : ''));
	    },
	    async deleteHistory(taskId) { return this._fetch('/api/history/' + taskId, { method: 'DELETE' }); },
	    async clearAllHistory() { return this._fetch('/api/history', { method: 'DELETE' }); },
	    async getCacheInfo() { return this._fetch('/api/cache/info'); },
	    async clearAllCache(stage) {
	      const qs = stage ? '?stage=' + encodeURIComponent(stage) : '';
	      return this._fetch('/api/cache' + qs, { method: 'DELETE' });
	    },
    // ---- Feedback APIs (Phase 5) ----
    async feedbackLearn(audioFile, refFile, profile, fbProfile, runFirst, dryRun) {
      const fd = new FormData();
      fd.append('audio', audioFile);
      fd.append('reference', refFile);
      fd.append('profile', profile);
      fd.append('feedback_profile', fbProfile);
      fd.append('run_pipeline_first', runFirst ? 'true' : 'false');
      fd.append('dry_run', dryRun ? 'true' : 'false');
      const res = await fetch('/api/feedback/learn', { method: 'POST', body: fd });
      if (!res.ok) { const t = await res.text(); throw new Error(t); }
      return res.json();
    },
    async feedbackPreview(audioFile, refFile, profile) {
      return this.feedbackLearn(audioFile, refFile, profile, 'user_default', true, true);
    },
    async feedbackProfiles() { return this._fetch('/api/feedback/profiles'); },
    async feedbackProfile(name) { return this._fetch('/api/feedback/profile/' + name); },
    async feedbackProfileRollback(name) {
      return this._fetch('/api/feedback/profile/' + name + '/rollback', { method: 'POST' });
    },
    async feedbackProfileDelete(name) {
      return this._fetch('/api/feedback/profile/' + name, { method: 'DELETE' });
    },
    async feedbackFingerprints() { return this._fetch('/api/feedback/fingerprints'); },
    async feedbackFingerprintMatch(audioFile) {
      const fd = new FormData();
      fd.append('audio', audioFile);
      const res = await fetch('/api/feedback/fingerprints/match', { method: 'POST', body: fd });
      if (!res.ok) { const t = await res.text(); throw new Error(t); }
      return res.json();
    },
    async feedbackFingerprintDelete(fpId) {
      return this._fetch('/api/feedback/fingerprints/' + fpId, { method: 'DELETE' });
    },
    async feedbackHealth(profile, limit) {
      const qs = limit ? '?limit=' + limit : '';
      return this._fetch('/api/feedback/health/' + profile + qs);
    },
    async feedbackHealthCompute(autoSubFile, refSubFile) {
      const fd = new FormData();
      fd.append('auto_subtitle', autoSubFile);
      fd.append('reference_subtitle', refSubFile);
      const res = await fetch('/api/feedback/health/compute', { method: 'POST', body: fd });
      if (!res.ok) { const t = await res.text(); throw new Error(t); }
      return res.json();
    },
    async feedbackShadow(profile) { return this._fetch('/api/feedback/shadow/' + profile); },
    async feedbackShadowToggle(profile, enabled) {
      return this._fetch('/api/feedback/shadow/' + profile + '/toggle', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: enabled})
      });
    },
    async feedbackShadowRecord(profile, healthCur, healthShadow) {
      const fd = new FormData();
      fd.append('health_current', String(healthCur));
      fd.append('health_shadow', String(healthShadow));
      const res = await fetch('/api/feedback/shadow/' + profile + '/record', { method: 'POST', body: fd });
      if (!res.ok) { const t = await res.text(); throw new Error(t); }
      return res.json();
    },
    async feedbackConflicts(profile) { return this._fetch('/api/feedback/conflicts/' + profile); },
    async feedbackConflictResolve(profile, paramPath, action) {
      return this._fetch('/api/feedback/conflicts/' + profile + '/resolve', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({param_path: paramPath, action: action})
      });
    },
    async feedbackImpactPreview(profileName) {
      const qs = profileName ? '?profile_name=' + encodeURIComponent(profileName) : '';
      return this._fetch('/api/feedback/impact/preview' + qs, { method: 'POST' });
    },
  },

  // ---- WebSocket ----
  ws: {
    connect(taskId) {
      if (App.state.ws) {
        App.state.ws._intentionalClose = true;
        App.state.ws.close();
      }
      App.ws.stopTaskPolling();
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(protocol + '//' + location.host + '/ws/tasks/' + taskId);
      App.state.ws = ws;

      ws.onmessage = function(e) {
        try {
          const msg = JSON.parse(e.data);
          App.ws.handleMessage(msg);
        } catch(ex) { console.error('WS parse error:', ex); }
      };

      ws.onclose = function() {
        if (!ws._intentionalClose && App.state.isRunning) {
          // The task runs in a server thread. Recover the terminal result via
          // REST when the progress socket disappears instead of losing a
          // completed task or showing a false failure immediately.
          App.ws.startTaskPolling(taskId);
        }
      };

      ws.onerror = function() {
        console.error('WebSocket error');
      };

      // Heartbeat
      App.state._wsHeartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: 'ping'}));
      }, 25000);
    },

    disconnect() {
      if (App.state._wsHeartbeat) { clearInterval(App.state._wsHeartbeat); }
      App.ws.stopTaskPolling();
      if (App.state.ws) {
        App.state.ws._intentionalClose = true;
        App.state.ws.close();
        App.state.ws = null;
      }
    },

    startTaskPolling(taskId) {
      App.ws.stopTaskPolling();
      let attempts = 0;
      App.state._taskPollTimer = setInterval(async function() {
        if (!App.state.isRunning || App.state.taskId !== taskId) {
          App.ws.stopTaskPolling();
          return;
        }
        attempts += 1;
        try {
          const status = await App.api.getTaskStatus(taskId);
          if (status.status === 'completed' && status.result) {
            App.ws.stopTaskPolling();
            App.ui.pipelineComplete(status.result);
          } else if (status.status === 'failed') {
            App.ws.stopTaskPolling();
            App.ui.pipelineError(status.error || '处理失败');
          }
        } catch (ex) {
          // Keep polling briefly through transient REST failures. After the
          // timeout, expose the connection issue rather than hanging forever.
          if (attempts >= 15) {
            App.ws.stopTaskPolling();
            App.ui.pipelineError('实时连接已断开，且无法获取任务状态');
          }
        }
      }, 1000);
    },

    stopTaskPolling() {
      if (App.state._taskPollTimer) {
        clearInterval(App.state._taskPollTimer);
        App.state._taskPollTimer = null;
      }
    },

    handleMessage(msg) {
      switch (msg.type) {
        case 'stage_start':
          App.ui.stageStart(msg.stage, msg.total, msg.description);
          break;
        case 'progress':
          App.ui.stageProgress(msg.stage, msg.current, msg.total, msg.extra);
          break;
        case 'stage_finish':
          App.ui.stageFinish(msg.stage, msg.elapsed_seconds);
          break;
        case 'complete':
          App.ui.pipelineComplete(msg.result);
          break;
        case 'error':
          App.ui.pipelineError(msg.message);
          break;
      }
    }
  },

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
      if (data.status === 'completed' && data.from_cache) {
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
    const overrides = {};
    document.querySelectorAll('#options-panel [data-key]').forEach(el => {
      const key = el.dataset.key;
      let val;
      if (el.type === 'checkbox') val = el.checked;
      else if (el.type === 'range') val = parseFloat(el.value);
      else if (el.type === 'number') val = el.value === '' ? '' : parseFloat(el.value);
      else val = el.value;
      if (val !== '' && val !== null && val !== undefined) {
        // Map key names to config paths
        const keyMap = {
          'separator': 'separator',
          'uvr_model': 'uvr_model',
          'vad_engine': 'vad_engine',
          'vad_threshold': 'vad_threshold',
          'vad_ffmpeg_enabled': 'ffmpeg_enabled',
          'vad_ffmpeg_noise_db': 'ffmpeg_noise_db',
          'asr_engine': 'asr_engine',
          'asr_model': 'asr_model',
          'language': 'language',
          'device': 'device',
          'diarization_enabled': 'diarization',
          'speaker_fusion': 'speaker_fusion',
          'global_diarization_model': 'global_diarization_model',
          'speaker_diarization_scope': 'speaker_diarization_scope',
          'local_speaker_refinement': 'local_speaker_refinement',
          'expected_speakers': 'expected_speakers',
          'diarization_local_context': 'diarization_local_context',
          'diarization_min_local_segment': 'diarization_min_local_segment',
          'diarization_min_change_confidence': 'diarization_min_change_confidence',
          'diarization_distance_threshold': 'diarization_distance_threshold',
          'diarization_min_speakers': 'diarization_min_speakers',
          'diarization_max_speakers': 'diarization_max_speakers',
          'speaker_role_enabled': 'speaker_role',
          'speaker_role_context_hint': 'speaker_role_context_hint',
          'speaker_embedding_enabled': 'speaker_embedding',
          'speaker_embedding_model_ref': 'speaker_embedding_model',
          'llm_enabled': 'llm_optimize',
          'llm_model': 'llm_model',
          'macro_chunk_enabled': 'macro_chunking.enabled',
          'macro_chunk_target_duration': 'macro_chunking.target_chunk_duration',
          'macro_chunk_max_duration': 'macro_chunking.max_chunk_duration',
          'merge_fast_gap': 'merge_decision.fast_merge_max_gap',
          'merge_llm_min_gap': 'merge_decision.llm_decision_min_gap',
          'merge_llm_max_gap': 'merge_decision.llm_decision_max_gap',
          'merge_hard_split_gap': 'merge_decision.hard_split_min_gap',
          'boundary_refine_enabled': 'boundary_refinement.enabled',
          'boundary_refine_max_shrink': 'boundary_refinement.max_shrink_ms',
          'acoustic_enabled': 'acoustic_validation.enabled',
          'acoustic_max_snap': 'acoustic_validation.max_snap_distance',
          'acoustic_generate_report': 'acoustic_validation.generate_report',
          'acoustic_skeleton_mode': 'skeleton_mode',
        };
        const mapped = keyMap[key] || key;
        overrides[mapped] = val;
      }
    });

    // Also collect LLM base_url and api_key from localStorage
    const savedUrl = localStorage.getItem('vocal_llm_url');
    const savedKey = localStorage.getItem('vocal_llm_key');
    if (savedUrl) overrides['llm_base_url'] = savedUrl;
    if (savedKey) overrides['llm_api_key'] = savedKey;

    return overrides;
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
	    if (item.status !== 'completed') {
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
