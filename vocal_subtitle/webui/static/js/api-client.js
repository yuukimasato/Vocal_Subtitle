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
    async getSubtitles(taskId) { return this._fetch('/api/subtitle/' + taskId); },
    // payload: {text?, start?, end?}；传字符串时视为纯文本更新（旧用法兼容）
    async updateSubtitle(taskId, index, payload) {
      if (typeof payload === 'string') payload = {text: payload};
      return this._fetch('/api/subtitle/' + taskId + '/' + index, {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.assign({index: index}, payload))
      });
    },
    async updateSubtitleTiming(taskId, index, start, end) {
      return this.updateSubtitle(taskId, index, {start: start, end: end});
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
	    async getHistoryFiltered(limit, offset, status) {
	      const params = new URLSearchParams();
	      if (limit) params.set('limit', limit);
	      if (offset) params.set('offset', offset);
	      if (status) params.set('status', status);
	      const qs = params.toString();
	      return this._fetch('/api/history' + (qs ? '?' + qs : ''));
	    },
	    async getHistoryDetail(taskId) { return this._fetch('/api/history/' + encodeURIComponent(taskId)); },
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
};
})();
