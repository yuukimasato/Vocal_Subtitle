(function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const time = (value) => window.formatTime ? formatTime(Number(value || 0)) : Number(value || 0).toFixed(1) + 's';

  const ReviewUI = {
    initialized: false,
    tasks: [],
    events: [],
    taskId: null,
    wave: null,
    _loadedAudioTaskId: null,

    init() {
      if (this.initialized) return;
      this.initialized = true;
      document.getElementById('review-task-select')?.addEventListener('change', (event) => this.load(event.target.value));
      document.getElementById('review-refresh')?.addEventListener('click', () => this.refresh(true));
      document.getElementById('review-export')?.addEventListener('click', () => {
        if (this.taskId && window.App) App.exportFormat('srt');
      });
      document.getElementById('review-go-feedback')?.addEventListener('click', () => Workspace.switchTo('feedback'));
      document.getElementById('review-go-quality')?.addEventListener('click', () => Workspace.switchTo('quality'));
      document.addEventListener('workspace:task-change', (event) => {
        if (Workspace.current === 'review' && event.detail.taskId !== this.taskId) this.load(event.detail.taskId);
      });
      this.initWave();
    },

    // 审核工作区使用独立的波形打轴实例（与主面板同一交互：左键设开始、
    // 右键设结束、拖动标记线、键盘微调、防抖保存）。
    initWave() {
      if (!window.WaveformView) return;
      this.wave = new WaveformView({
        name: 'review',
        canvasId: 'review-waveform-canvas',
        audioId: 'review-audio-player',
        playheadId: 'review-waveform-playhead',
        statusId: 'review-save-status',
        emptyId: 'review-waveform-empty',
        timeId: 'review-waveform-time',
        zoomId: 'review-waveform-zoom',
        playId: 'review-waveform-play',
        stopId: 'review-waveform-stop',
        resetId: 'review-waveform-reset',
        onTimingChange: (index, start, end) => this.saveTiming(index, start, end),
      });
      this.wave.init();

      document.addEventListener('waveform:active-changed', (event) => {
        if (event.detail.name !== 'review') return;
        const index = Number(event.detail.index);
        document.querySelectorAll('#review-tbody tr.review-row-active').forEach((row) => {
          row.classList.remove('review-row-active');
        });
        if (index == null || Number.isNaN(index)) return;
        const row = document.querySelector('#review-tbody tr[data-review-index="' + index + '"]');
        if (row) {
          row.classList.add('review-row-active');
          if (event.detail.source !== 'events' && event.detail.source !== 'load') {
            row.scrollIntoView({block: 'nearest', behavior: 'smooth'});
          }
        }
      });
      document.addEventListener('waveform:timing-changed', (event) => {
        if (event.detail.name !== 'review') return;
        const detail = event.detail;
        const ev = this.events.find((item) => Number(item.index) === Number(detail.index));
        if (ev) { ev.start = detail.start; ev.end = detail.end; this.updateRowTime(ev); }
      });
    },

    async refresh(force = false) {
      this.init();
      try {
        if (force || !this.tasks.length) {
          const data = await VocalSubtitleApi.getHistory(50, 0);
          this.tasks = data.items || [];
          this.fillTasks();
        }
        const taskId = Workspace.taskId() || this.taskId || this.tasks[0]?.id;
        if (taskId) await this.load(taskId);
      } catch (error) {
        this.showError(error);
      }
    },

    fillTasks() {
      const select = document.getElementById('review-task-select');
      if (!select) return;
      const selected = this.taskId || Workspace.taskId() || '';
      select.innerHTML = '<option value="">选择任务</option>' + this.tasks.map((task) =>
        '<option value="' + esc(task.id) + '">' + esc(task.input_file_name || task.id) + ' · ' + esc(task.status || '') + '</option>'
      ).join('');
      if (selected) select.value = selected;
    },

    async load(taskId) {
      if (!taskId) { this.renderEmpty('暂无可审核字幕'); return; }
      this.taskId = taskId;
      Workspace.setTask(taskId);
      this.setStatus('加载中...', 'pending');
      try {
        const detail = await VocalSubtitleApi.getHistoryDetail(taskId);
        this.events = detail.events || [];
        this.render(detail);
      } catch (error) {
        this.renderEmpty('任务不存在或已清理');
        this.showError(error);
      }
    },

    render(detail) {
      const summary = detail.result_summary || {};
      const stats = summary.stats || {};
      document.getElementById('review-task-meta').textContent = (detail.input_file_name || detail.id || '') + ' · ' + (detail.profile || 'default');
      this.setStatus(detail.status === 'degraded_completed' || detail.status === 'degraded' ? '降级完成' : '已加载', detail.status);
      this.loadWaveAudio(detail.id);
      const metrics = document.getElementById('review-metrics');
      if (metrics) metrics.innerHTML = [
        ['状态', detail.status || '—'], ['字幕', summary.subtitle_count || this.events.length],
        ['语音片段', summary.segment_count || stats.segment_count || '—'],
        ['质量', stats.quality_status || '—'], ['引擎', stats.final_engine || stats.selected_engine || '—']
      ].map((row) => '<div class="metric-row"><span>' + esc(row[0]) + '</span><strong>' + esc(row[1]) + '</strong></div>').join('');
      this.renderEvents();
    },

    // The stream route falls back to the uploaded input when separation
    // was skipped or degraded, so review remains usable without vocals_path.
    async loadWaveAudio(taskId) {
      if (!this.wave) return;
      if (this._loadedAudioTaskId === taskId) {
        this.wave.setEvents(this.events);
        return;
      }
      const base = '/api/tasks/' + encodeURIComponent(taskId) + '/audio/stream?type=';
      const loaded = await this.wave.setAudioUrl(base + 'vocals');
      if (!loaded) await this.wave.setAudioUrl(base + 'input');
      this._loadedAudioTaskId = taskId;
      this.wave.setEvents(this.events);
    },

    renderEvents() {
      const body = document.getElementById('review-tbody');
      const empty = document.getElementById('review-empty');
      if (!body) return;
      if (!this.events.length) { body.innerHTML = ''; if (empty) empty.hidden = false; return; }
      if (empty) empty.hidden = true;
      body.innerHTML = this.events.map((event, index) => '<tr data-review-index="' + esc(event.index ?? index + 1) + '">' +
        '<td class="review-index">' + esc(event.index ?? index + 1) + '</td>' +
        '<td class="review-time"><button type="button" class="review-play" data-start="' + Number(event.start || 0) + '" title="播放此条" aria-label="播放此条">▶</button><span class="review-time-text">' + time(event.start) + '<br>' + time(event.end) + '</span></td>' +
        '<td class="review-speaker">' + esc(event.speaker_label || (event.speaker_id === null || event.speaker_id === undefined ? '—' : 'S' + event.speaker_id)) + '</td>' +
        '<td><textarea class="review-text" aria-label="字幕文本">' + esc(event.text || '') + '</textarea></td>' +
        '<td><button type="button" class="review-save" title="保存此条">保存</button></td></tr>').join('');
      body.querySelectorAll('.review-play').forEach((button) => button.addEventListener('click', () => {
        const row = button.closest('tr');
        const index = Number(row.dataset.reviewIndex);
        if (this.wave) {
          this.wave.setActiveEvent(index, {source: 'review-click'});
          this.wave.playActive();
        } else {
          this.play(Number(button.dataset.start));
        }
      }));
      body.querySelectorAll('tr[data-review-index]').forEach((row) => {
        row.addEventListener('click', (event) => {
          if (event.target && event.target.closest && event.target.closest('button, textarea, input, select')) return;
          if (this.wave) this.wave.setActiveEvent(Number(row.dataset.reviewIndex), {source: 'review-click'});
        });
      });
      body.querySelectorAll('.review-save').forEach((button) => button.addEventListener('click', () => {
        const row = button.closest('tr');
        const index = Number(row.dataset.reviewIndex);
        const text = row.querySelector('.review-text').value;
        this.save(index, text, button);
      }));
      // 同步活动行高亮
      if (this.wave && this.wave.activeIndex != null) {
        const activeRow = body.querySelector('tr[data-review-index="' + this.wave.activeIndex + '"]');
        if (activeRow) activeRow.classList.add('review-row-active');
      }
    },

    updateRowTime(event) {
      const row = document.querySelector('#review-tbody tr[data-review-index="' + event.index + '"]');
      if (!row) return;
      const textEl = row.querySelector('.review-time-text');
      if (textEl) textEl.innerHTML = time(event.start) + '<br>' + time(event.end);
      const playBtn = row.querySelector('.review-play');
      if (playBtn) playBtn.dataset.start = String(Number(event.start) || 0);
    },

    async saveTiming(index, start, end) {
      if (!this.taskId) throw new Error('未选择任务');
      await VocalSubtitleApi.updateSubtitleTiming(this.taskId, index, start, end);
      const event = this.events.find((item) => Number(item.index) === Number(index));
      if (event) { event.start = start; event.end = end; this.updateRowTime(event); }
    },

    async save(index, text, button) {
      button.disabled = true;
      try {
        const result = await VocalSubtitleApi.updateSubtitle(this.taskId, index, {text: text});
        const event = this.events.find((item) => Number(item.index) === index);
        if (event) event.text = text;
        if (result.events) this.events = result.events;
        this.setStatus('已保存', 'completed');
        toast('字幕已保存', 'success');
      } catch (error) { this.showError(error); this.setStatus('保存失败', 'failed'); }
      finally { button.disabled = false; }
    },

    play(start) {
      if (this.wave) {
        this.wave.seekTo(start);
        this.wave.play();
        return;
      }
      const player = document.getElementById('review-audio-player');
      if (!player) { toast('当前任务没有可播放音频', 'info'); return; }
      player.currentTime = start;
      player.play().catch(() => {});
    },

    setStatus(text, status) {
      const el = document.getElementById('review-save-status');
      if (el) { el.textContent = text; el.dataset.status = status || ''; }
    },
    renderEmpty(text) {
      const body = document.getElementById('review-tbody'); if (body) body.innerHTML = '';
      const empty = document.getElementById('review-empty'); if (empty) { empty.textContent = text; empty.hidden = false; }
      if (this.wave) { this.wave.clear(false); this.wave.showEmpty('选择任务后在此显示打轴波形'); }
    },
    showError(error) { toast('审核数据加载失败: ' + (error.message || error), 'error'); }
  };
  window.ReviewUI = ReviewUI;
  document.addEventListener('DOMContentLoaded', () => ReviewUI.init());
})();
