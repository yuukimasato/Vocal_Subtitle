(function() {
'use strict';
window.VocalSubtitleApi = {
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
    async getTaskStatus(taskId) { return this._fetch('/api/tasks/' + taskId); },
	    async getHistory(limit, offset) {
	      const params = new URLSearchParams();
	      if (limit) params.set('limit', limit);
	      if (offset) params.set('offset', offset);
	      const qs = params.toString();
	      return this._fetch('/api/history' + (qs ? '?' + qs : ''));
	    },
	    async getHistoryFiltered(limit, offset, status) {
	      const params = new URLSearchParams();
	      if (limit) params.set('limit', limit);
	      if (offset) params.set('offset', offset);
	      if (status) params.set('status', status);
	      const qs = params.toString();
	      return this._fetch('/api/history' + (qs ? '?' + qs : ''));
	    },
	    async getHistoryDetail(taskId) { return this._fetch('/api/history/' + encodeURIComponent(taskId)); },
	    async getHistoryByHash(hash, limit) {
	      // 按输入文件内容哈希查询可复用任务（V2 上传学习绑定来源任务，D26）
	      const qs = limit ? '?limit=' + encodeURIComponent(limit) : '';
	      return this._fetch('/api/history/by-hash/' + encodeURIComponent(hash) + qs);
	    },
	    async getQualityReport(taskId) {
	      return this._fetch('/api/quality/report/' + encodeURIComponent(taskId));
	    },
	    async getReviewQueue(status) {
	      const qs = status ? '?status=' + encodeURIComponent(status) : '';
	      return this._fetch('/api/feedback/review-queue' + qs);
	    },
	    async deleteHistory(taskId) { return this._fetch('/api/history/' + taskId, { method: 'DELETE' }); },
	    async clearAllHistory() { return this._fetch('/api/history', { method: 'DELETE' }); },
	    async getCacheInfo() { return this._fetch('/api/cache/info'); },
	    async clearAllCache(stage) {
	      const qs = stage ? '?stage=' + encodeURIComponent(stage) : '';
	      return this._fetch('/api/cache' + qs, { method: 'DELETE' });
	    },
    async feedbackProfiles() { return this._fetch('/api/feedback/profiles'); },
    async feedbackLearn(formData) {
      // multipart 调用 /api/feedback/learn（task_id + scenario + reference 等）
      const res = await fetch('/api/feedback/learn', { method: 'POST', body: formData });
      if (!res.ok) { const t = await res.text(); throw new Error(t); }
      return res.json();
    },
    async feedbackOverridesApply() { return this._fetch('/api/feedback/overrides-apply'); },
    async feedbackOverridesApplyToggle(enabled) {
      return this._fetch('/api/feedback/overrides-apply', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: enabled})
      });
    },
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
    async datasetPreview(scenarios, outDir, datasetName) {
      // 数据集导出统计预览（与导出同一服务层过滤器，预览数字 = 导出条目数）
      const params = new URLSearchParams();
      if (scenarios && scenarios.length) params.set('scenarios', scenarios.join(','));
      if (datasetName) params.set('dataset_name', datasetName);
      if (outDir) params.set('out_dir', outDir);
      const qs = params.toString();
      return this._fetch('/api/feedback/dataset/preview' + (qs ? '?' + qs : ''));
    },
    async datasetExport(payload) {
      // 执行导出（许可必选）：{license, scenarios, out_dir, bundle_audio, dataset_name}
      return this._fetch('/api/feedback/dataset/export', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
    },
};
})();
