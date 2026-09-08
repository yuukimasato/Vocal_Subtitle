// Aegisub 风格的音频打轴视图。
// 活动字幕行在波形上以红色粗线（开始）、蓝色粗线（结束）与高亮区间呈现：
// 左键点击设开始、右键点击设结束、拖动边线微调、拖动区间整体平移，
// 改动经防抖自动保存（onTimingChange 回调），Enter 立即提交并跳到下一行。
(function() {
  'use strict';

  const ZOOM_MIN = 1;
  const ZOOM_MAX = 64;
  const MIN_WINDOW = 0.05;     // 最窄可视窗口（秒）
  const MIN_EVENT_DUR = 0.02;  // 字幕最短时长（秒）
  const HANDLE_PX = 7;         // 边线命中半宽（像素）
  const NUDGE_STEP = 0.01;     // 方向键步长（秒），Alt 细调 1ms
  const SAVE_DEBOUNCE_MS = 500;
  const START_COLOR = '#ff5f56';
  const END_COLOR = '#59a7ff';

  const notify = (message, kind) => {
    if (typeof window.toast === 'function') window.toast(message, kind);
    else console.warn(message);
  };

  class WaveformView {
    constructor(options) {
      this.name = options.name || 'waveform';
      this.el = {};
      const get = (id) => (id ? document.getElementById(id) : null);
      this.el.canvas = get(options.canvasId);
      this.el.audio = get(options.audioId);
      this.el.playhead = get(options.playheadId);
      this.el.status = get(options.statusId);
      this.el.empty = get(options.emptyId);
      this.el.time = get(options.timeId);
      this.el.zoom = get(options.zoomId);
      this.el.play = get(options.playId);
      this.el.stop = get(options.stopId);
      this.el.reset = get(options.resetId);
      this.el.help = get(options.helpId);
      this.el.helpPanel = get(options.helpPanelId);

      this.context = this.el.canvas ? this.el.canvas.getContext('2d') : null;
      this.audioContext = null;
      this.buffer = null;
      this.samples = null;
      this.objectUrl = null;
      this.duration = 0;
      this.zoom = 1;
      this.viewStart = 0;
      this.events = [];
      this.activeIndex = null;
      this.savedTiming = null;   // {start, end} 最近一次已保存值
      this.pendingSave = false;
      this.saveTimer = null;
      this.playStopAt = null;
      this.dragging = null;
      this.followPlayback = true;
      this.lastView = { start: 0, duration: 1, width: 1 };
      // 外部装配的保存回调：(index, start, end) => Promise
      this.onTimingChange = options.onTimingChange || null;
      this.initialized = false;
    }

    init() {
      if (this.initialized || !this.el.canvas || !this.el.audio) return;
      this.initialized = true;
      this.el.play?.addEventListener('click', () => this.togglePlay());
      this.el.stop?.addEventListener('click', () => this.stop());
      this.el.reset?.addEventListener('click', () => this.resetView());
      this.el.help?.addEventListener('click', () => this.toggleHelp());
      this.el.zoom?.addEventListener('input', (event) => {
        this.setZoom(Number(event.target.value) || 1, this.viewCenter());
      });

      const canvas = this.el.canvas;
      canvas.addEventListener('mousedown', (event) => this.onMouseDown(event));
      window.addEventListener('mousemove', (event) => this.onMouseMove(event));
      window.addEventListener('mouseup', (event) => this.onMouseUp(event));
      canvas.addEventListener('contextmenu', (event) => this.onContextMenu(event));
      canvas.addEventListener('dblclick', (event) => this.onDoubleClick(event));
      canvas.addEventListener('wheel', (event) => this.onWheel(event), { passive: false });
      // Canvas width/height attributes can resize the observed parent in some
      // Chromium builds; use a plain window listener to avoid a redraw loop.
      window.addEventListener('resize', () => this.draw());
      document.addEventListener('keydown', (event) => this.onKeyDown(event));

      this.el.audio.addEventListener('loadedmetadata', () => {
        if (!this.buffer && Number.isFinite(this.el.audio.duration)) {
          this.duration = Number(this.el.audio.duration) || this.duration;
        }
        this.updateTime();
        this.draw();
      });
      this.el.audio.addEventListener('timeupdate', () => {
        this.updateTime();
        this.followPlayingView();
        this.highlightEvent();
        this.draw();
      });
      this.el.audio.addEventListener('play', () => this.setPlayLabel(true));
      this.el.audio.addEventListener('pause', () => this.setPlayLabel(false));
      this.el.audio.addEventListener('ended', () => {
        this.playStopAt = null;
        this.setPlayLabel(false);
      });
      this.clear();
    }

    // ------------------------------------------------------------------
    // 数据装载
    // ------------------------------------------------------------------

    async setTask(taskId, result, events) {
      this.init();
      if (!this.el.audio || !taskId) return;
      if (Array.isArray(events)) this.setEvents(events, { activate: false });
      const sourceType = result && result.vocals_path ? 'vocals' : 'input';
      const base = '/api/tasks/' + encodeURIComponent(taskId) + '/audio/stream?type=';
      const loaded = await this.loadUrl(base + sourceType);
      if (!loaded && sourceType === 'vocals') {
        await this.loadUrl(base + 'input');
      }
    }

    async setAudioUrl(url) {
      this.init();
      return this.loadUrl(url);
    }

    async setLocalFile(file) {
      this.init();
      if (!file) return;
      if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = URL.createObjectURL(file);
      await this.loadUrl(this.objectUrl, { revokeOnFailure: false });
      this.setStatus('本地预览');
    }

    async loadUrl(url, options) {
      this.clear(false);
      this.setStatus('加载波形...');
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('audio ' + response.status);
        const bytes = await response.arrayBuffer();
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) throw new Error('Web Audio API unavailable');
        if (!this.audioContext) this.audioContext = new AudioContextCtor();
        this.buffer = await this.audioContext.decodeAudioData(bytes.slice(0));
        this.samples = this.buildPeaks(this.buffer);
        this.duration = this.buffer.duration || 0;
        this.el.audio.src = url;
        this.el.audio.load();
        this.hideEmpty();
        this.setStatus('已加载');
        this.ensureActiveEvent();
        this.draw();
      } catch (error) {
        this.buffer = null;
        this.samples = null;
        if (Array.isArray(this.events) && this.events.length) {
          // 无解码波形时退化为「标尺 + 字幕区间 + 标记线」，仍可打轴。
          this.duration = Math.max(...this.events.map((e) => Number(e.end) || 0)) + 0.5;
          this.el.audio.removeAttribute('src');
          this.draw();
          this.setStatus('波形不可用，可点击设时间');
          this.showEmpty('音频无法解码，仍可编辑字幕时间轴');
          this.ensureActiveEvent();
        } else {
          this.duration = 0;
          this.setStatus('波形不可用');
          this.showEmpty('音频无法解码，仍可编辑字幕时间轴');
        }
        if (options?.revokeOnFailure !== false && this.objectUrl) {
          URL.revokeObjectURL(this.objectUrl);
          this.objectUrl = null;
        }
        return false;
      }
      return true;
    }

    buildPeaks(buffer) {
      const channel = buffer.getChannelData(0);
      const bucketSize = Math.max(1, Math.floor(channel.length / 12000));
      const peaks = new Float32Array(Math.ceil(channel.length / bucketSize));
      for (let index = 0; index < peaks.length; index += 1) {
        const start = index * bucketSize;
        const end = Math.min(channel.length, start + bucketSize);
        let peak = 0;
        for (let sample = start; sample < end; sample += 1) peak = Math.max(peak, Math.abs(channel[sample]));
        peaks[index] = peak;
      }
      return peaks;
    }

    setEvents(events, options) {
      this.events = Array.isArray(events) ? events.slice() : [];
      const exists = this.activeIndex != null
        && this.events.some((e) => Number(e.index) === Number(this.activeIndex));
      if (!exists) this.activeIndex = null;
      if (this.activeIndex == null && this.events.length && options?.activate !== false) {
        this.activeIndex = this.events[0].index;
      }
      this.refreshSavedTiming();
      if (options?.activate !== false) this.emitActiveChanged('events');
      this.draw();
    }

    clear(showEmpty = true) {
      this.buffer = null;
      this.samples = null;
      this.duration = 0;
      this.viewStart = 0;
      this.activeIndex = null;
      this.savedTiming = null;
      this.pendingSave = false;
      clearTimeout(this.saveTimer);
      if (this.el.audio) {
        this.el.audio.pause();
        this.el.audio.removeAttribute('src');
        this.el.audio.load();
      }
      if (showEmpty) this.showEmpty('完成任务后将在此显示可缩放波形');
      this.updateTime();
      this.draw();
    }

    // ------------------------------------------------------------------
    // 活动行与保存
    // ------------------------------------------------------------------

    get activeEvent() {
      if (this.activeIndex == null) return null;
      return this.events.find((e) => Number(e.index) === Number(this.activeIndex)) || null;
    }

    setActiveEvent(index, options = {}) {
      const event = this.events.find((e) => Number(e.index) === Number(index));
      if (!event) return;
      this.activeIndex = event.index;
      this.refreshSavedTiming();
      if (options.center !== false) this.centerOnEvent(event);
      if (options.seek) this.seekTo(Number(event.start) || 0);
      this.emitActiveChanged(options.source || 'api');
      this.draw();
    }

    emitActiveChanged(source) {
      document.dispatchEvent(new CustomEvent('waveform:active-changed', {
        detail: { index: this.activeIndex, source, name: this.name }
      }));
    }

    refreshSavedTiming() {
      const event = this.activeEvent;
      this.savedTiming = event
        ? { start: Number(event.start) || 0, end: Number(event.end) || 0 }
        : null;
    }

    // 音频/事件加载完成后确保存在默认活动行（红/蓝标记线随之出现）
    ensureActiveEvent() {
      if (this.activeIndex == null && this.events.length) {
        this.activeIndex = this.events[0].index;
        this.refreshSavedTiming();
        this.emitActiveChanged('load');
      } else {
        this.draw();
      }
    }

    setTiming(start, end, options = {}) {
      const event = this.activeEvent;
      if (!event) {
        notify('请先在字幕列表中选择一行（或按 ↑/↓）', 'info');
        return false;
      }
      const nextStart = Math.max(0, Number(start));
      const nextEnd = Math.max(nextStart + MIN_EVENT_DUR, Number(end));
      if (nextStart === Number(event.start) && nextEnd === Number(event.end)) return false;
      event.start = Number(nextStart.toFixed(3));
      event.end = Number(nextEnd.toFixed(3));
      document.dispatchEvent(new CustomEvent('waveform:timing-changed', {
        detail: { index: event.index, start: event.start, end: event.end, name: this.name }
      }));
      this.draw();
      if (options.save !== false) this.scheduleSave();
      return true;
    }

    // Aegisub 语义：左键设开始、右键设结束；越过另一端时两端交换。
    setStartAt(time) {
      const event = this.activeEvent;
      if (!event) return false;
      const t = Math.max(0, Number(time));
      if (t < Number(event.end) - MIN_EVENT_DUR) return this.setTiming(t, event.end);
      return this.setTiming(event.end, t + MIN_EVENT_DUR);
    }

    setEndAt(time) {
      const event = this.activeEvent;
      if (!event) return false;
      const t = Math.max(0, Number(time));
      if (t > Number(event.start) + MIN_EVENT_DUR) return this.setTiming(event.start, t);
      return this.setTiming(Math.max(0, t - MIN_EVENT_DUR), event.start);
    }

    scheduleSave() {
      if (!this.onTimingChange || !this.activeEvent) return;
      this.pendingSave = true;
      this.updateSaveBadge();
      clearTimeout(this.saveTimer);
      this.saveTimer = setTimeout(() => this.flushSave(), SAVE_DEBOUNCE_MS);
    }

    async flushSave() {
      clearTimeout(this.saveTimer);
      const event = this.activeEvent;
      if (!this.pendingSave || !event || !this.onTimingChange) return;
      const snapshot = { index: event.index, start: Number(event.start), end: Number(event.end) };
      try {
        await this.onTimingChange(snapshot.index, snapshot.start, snapshot.end);
        this.savedTiming = { start: snapshot.start, end: snapshot.end };
        this.pendingSave = false;
        this.updateSaveBadge();
      } catch (error) {
        // 保存失败：回滚到最近一次已保存的时间轴。
        this.pendingSave = false;
        if (this.savedTiming) {
          const rollback = this.events.find((e) => Number(e.index) === Number(snapshot.index));
          if (rollback) {
            rollback.start = this.savedTiming.start;
            rollback.end = this.savedTiming.end;
            document.dispatchEvent(new CustomEvent('waveform:timing-changed', {
              detail: { index: rollback.index, start: rollback.start, end: rollback.end, name: this.name }
            }));
          }
        }
        this.updateSaveBadge();
        this.draw();
        notify('时间轴保存失败: ' + (error && error.message ? error.message : error), 'error');
      }
    }

    updateSaveBadge() {
      const node = this.el.status;
      if (!node) return;
      if (this.pendingSave) node.textContent = '时间轴未保存…';
      else if (node.textContent === '时间轴未保存…') node.textContent = '时间轴已保存';
    }

    // ------------------------------------------------------------------
    // 视图坐标
    // ------------------------------------------------------------------

    get viewDuration() {
      if (!this.duration) return 1;
      return Math.min(this.duration, Math.max(MIN_WINDOW, this.duration / this.zoom));
    }

    get canvasWidth() {
      const rect = this.el.canvas?.getBoundingClientRect();
      return Math.max(320, Math.floor(rect?.width || 640));
    }

    viewCenter() {
      return this.viewStart + this.viewDuration / 2;
    }

    clampView() {
      const duration = this.viewDuration;
      this.viewStart = Math.max(0, Math.min(this.viewStart, Math.max(0, this.duration - duration)));
    }

    setZoom(zoom, anchorTime) {
      const anchor = anchorTime == null ? this.viewCenter() : anchorTime;
      const anchorRatio = (anchor - this.viewStart) / this.viewDuration;
      this.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom));
      this.viewStart = anchor - anchorRatio * this.viewDuration;
      this.clampView();
      if (this.el.zoom) this.el.zoom.value = String(this.zoom);
      this.draw();
    }

    centerOnEvent(event) {
      const start = Number(event.start) || 0;
      const end = Number(event.end) || 0;
      const pad = Math.max(0.25, (end - start) * 0.8);
      const focusStart = Math.max(0, start - pad);
      const focusEnd = end + pad;
      if (focusEnd - focusStart < this.viewDuration) {
        this.viewStart = focusStart - Math.max(0, (this.viewDuration - (focusEnd - focusStart)) / 2);
      } else {
        this.viewStart = focusStart;
      }
      this.clampView();
    }

    timeToX(time) {
      return ((time - this.viewStart) / this.viewDuration) * this.lastView.width;
    }

    xToTime(x) {
      return this.viewStart + (x / this.lastView.width) * this.viewDuration;
    }

    // ------------------------------------------------------------------
    // 绘制
    // ------------------------------------------------------------------

    draw() {
      if (!this.el.canvas || !this.context) return;
      const width = this.canvasWidth;
      const height = 170;
      const rulerHeight = 18;
      const ratio = window.devicePixelRatio || 1;
      this.el.canvas.width = width * ratio;
      this.el.canvas.height = height * ratio;
      this.el.canvas.style.height = height + 'px';
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);

      this.lastView = { start: this.viewStart, duration: this.viewDuration, width };

      const ctx = this.context;
      ctx.fillStyle = '#0b1018';
      ctx.fillRect(0, 0, width, height);

      this.drawRuler(width, height, rulerHeight);

      const waveTop = rulerHeight + 2;
      const waveHeight = height - waveTop;
      ctx.strokeStyle = 'rgba(147,167,210,.3)';
      ctx.beginPath();
      ctx.moveTo(0, waveTop + waveHeight / 2 + .5);
      ctx.lineTo(width, waveTop + waveHeight / 2 + .5);
      ctx.stroke();

      if (this.samples && this.duration) {
        this.drawWaveBars(width, waveTop, waveHeight);
      }

      this.drawEventRegions(width, waveTop, waveHeight);
      this.updatePlayhead();
    }

    drawRuler(width, height, rulerHeight) {
      const ctx = this.context;
      const step = this.rulerStep(width);
      ctx.strokeStyle = 'rgba(124,156,255,.16)';
      ctx.lineWidth = 1;
      ctx.fillStyle = 'rgba(147,167,210,.75)';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textBaseline = 'top';
      const first = Math.floor(this.viewStart / step) * step;
      for (let t = first; t <= this.viewStart + this.viewDuration + step; t += step) {
        if (t < 0) continue;
        const x = Math.round(this.timeToX(t)) + .5;
        if (x < 0 || x > width) continue;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.fillText(this.formatRulerLabel(t, step), x + 3, 3);
      }
      ctx.strokeStyle = 'rgba(124,156,255,.4)';
      ctx.beginPath();
      ctx.moveTo(0, rulerHeight + .5);
      ctx.lineTo(width, rulerHeight + .5);
      ctx.stroke();
    }

    rulerStep(width) {
      const target = this.viewDuration / Math.max(4, width / 96);
      const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
      for (const step of steps) if (step >= target) return step;
      return 1200;
    }

    formatRulerLabel(time, step) {
      if (step < 1) return time.toFixed(step < 0.1 ? 2 : 1) + 's';
      const mins = Math.floor(time / 60);
      const secs = Math.round(time % 60);
      return mins > 0 ? mins + ':' + String(secs).padStart(2, '0') : secs + 's';
    }

    drawWaveBars(width, top, height) {
      const ctx = this.context;
      const startIndex = Math.floor(this.viewStart / this.duration * this.samples.length);
      const endIndex = Math.max(startIndex + 1, Math.ceil((this.viewStart + this.viewDuration) / this.duration * this.samples.length));
      const step = Math.max(1, (endIndex - startIndex) / width);
      ctx.fillStyle = '#7c9cff';
      for (let x = 0; x < width; x += 1) {
        const index = Math.min(endIndex - 1, startIndex + Math.floor(x * step));
        const amplitude = this.samples[index] || 0;
        const bar = Math.max(2, amplitude * (height * .82));
        ctx.fillRect(x, top + (height - bar) / 2, 1, bar);
      }
    }

    drawEventRegions(width, top, height) {
      const ctx = this.context;
      const active = this.activeEvent;
      this.events.forEach((event) => {
        if (active && Number(event.index) === Number(active.index)) return;
        const left = this.timeToX(Number(event.start));
        const right = this.timeToX(Number(event.end));
        if (right <= 0 || left >= width) return;
        ctx.fillStyle = 'rgba(124,156,255,.16)';
        ctx.fillRect(Math.max(0, left), top, Math.max(2, right - Math.max(0, left)), height);
      });

      if (!active) return;
      const startX = this.timeToX(Number(active.start));
      const endX = this.timeToX(Number(active.end));
      const left = Math.max(-4, startX);
      const right = Math.min(width + 4, endX);
      if (right >= 0 && left <= width) {
        ctx.fillStyle = 'rgba(240,180,90,.14)';
        ctx.fillRect(Math.max(0, left), top, Math.max(2, right - Math.max(0, left)), height);
      }
      this.drawMarker(startX, START_COLOR, top, height, '开始');
      this.drawMarker(endX, END_COLOR, top, height, '结束');
      this.drawMarkerLabel(startX, active.start, START_COLOR, top, right - left < 90);
      this.drawMarkerLabel(endX, active.end, END_COLOR, top, false, true);
    }

    drawMarker(x, color, top, height, label) {
      const ctx = this.context;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + .5, top);
      ctx.lineTo(x + .5, top + height);
      ctx.stroke();
      // 顶部抓手
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x - 5, top);
      ctx.lineTo(x + 6, top);
      ctx.lineTo(x + 6, top + 10);
      ctx.lineTo(x + .5, top + 15);
      ctx.lineTo(x - 5, top + 10);
      ctx.closePath();
      ctx.fill();
      ctx.save();
      ctx.fillStyle = '#0b1018';
      ctx.font = 'bold 8px ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      ctx.fillText('⇔', x - 3.5, top + 6);
      ctx.restore();
      void label;
    }

    drawMarkerLabel(x, time, color, top, hide, alignRight) {
      if (hide || !Number.isFinite(Number(time))) return;
      const ctx = this.context;
      const text = this.formatTime(time);
      ctx.font = '10px ui-monospace, monospace';
      const width = ctx.measureText(text).width + 8;
      const x0 = alignRight ? x - width - 4 : x + 4;
      ctx.fillStyle = 'rgba(11,16,24,.85)';
      ctx.fillRect(x0, top + 2, width, 14);
      ctx.fillStyle = color;
      ctx.textBaseline = 'top';
      ctx.fillText(text, x0 + 4, top + 4);
    }

    // ------------------------------------------------------------------
    // 鼠标交互
    // ------------------------------------------------------------------

    eventAt(x) {
      const time = this.xToTime(x);
      return this.events.find((event) =>
        time >= Number(event.start) && time <= Number(event.end)) || null;
    }

    hitTest(x) {
      const active = this.activeEvent;
      if (active) {
        const startX = this.timeToX(Number(active.start));
        const endX = this.timeToX(Number(active.end));
        if (Math.abs(x - startX) <= HANDLE_PX && startX >= -HANDLE_PX && startX <= this.lastView.width + HANDLE_PX) return 'start';
        if (Math.abs(x - endX) <= HANDLE_PX && endX >= -HANDLE_PX && endX <= this.lastView.width + HANDLE_PX) return 'end';
        if (x > startX + HANDLE_PX && x < endX - HANDLE_PX) return 'region';
      }
      return 'empty';
    }

    onMouseDown(event) {
      if (event.button !== 0 || !this.duration) return;
      const rect = this.el.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const time = this.xToTime(x);
      const hit = this.hitTest(x);
      if (hit === 'start' || hit === 'end') {
        this.dragging = { mode: hit };
      } else if (hit === 'region') {
        const active = this.activeEvent;
        this.dragging = {
          mode: 'region',
          offset: time - Number(active.start),
          duration: Number(active.end) - Number(active.start),
          startX: x,
          moved: false,
          clickTime: time,
        };
      } else {
        const other = this.eventAt(x);
        this.dragging = {
          mode: 'pan',
          startX: x,
          startView: this.viewStart,
          moved: false,
          clickEvent: other && (!this.activeEvent || other.index !== this.activeEvent.index) ? other : null,
          clickTime: time,
        };
      }
      event.preventDefault();
    }

    onMouseMove(event) {
      if (!this.el.canvas) return;
      const rect = this.el.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      if (!this.dragging) {
        this.updateCursor(x);
        return;
      }
      const time = Math.max(0, Math.min(this.duration || Infinity, this.xToTime(x)));
      const active = this.activeEvent;
      if (this.dragging.mode === 'start' && active) {
        const end = Number(active.end);
        const start = Math.min(time, end - MIN_EVENT_DUR);
        if (Math.abs(start - Number(active.start)) > 1e-4) {
          active.start = Number(Math.max(0, start).toFixed(3));
          this.draw();
        }
      } else if (this.dragging.mode === 'end' && active) {
        const start = Number(active.start);
        const end = Math.max(time, start + MIN_EVENT_DUR);
        if (Math.abs(end - Number(active.end)) > 1e-4) {
          active.end = Number(end.toFixed(3));
          this.draw();
        }
      } else if (this.dragging.mode === 'region' && active) {
        if (Math.abs(x - this.dragging.startX) > 3) this.dragging.moved = true;
        if (!this.dragging.moved) return;
        const duration = this.dragging.duration;
        let start = Math.max(0, time - this.dragging.offset);
        if (this.duration) start = Math.min(start, this.duration - duration);
        const end = start + duration;
        if (Math.abs(start - Number(active.start)) > 1e-4) {
          active.start = Number(start.toFixed(3));
          active.end = Number(end.toFixed(3));
          this.draw();
        }
      } else if (this.dragging.mode === 'pan') {
        const dx = x - this.dragging.startX;
        if (Math.abs(dx) > 3) this.dragging.moved = true;
        this.viewStart = this.dragging.startView - (dx / this.lastView.width) * this.viewDuration;
        this.clampView();
        this.draw();
      }
    }

    onMouseUp() {
      const drag = this.dragging;
      this.dragging = null;
      if (!drag) return;
      if (drag.mode === 'start' || drag.mode === 'end') {
        this.afterTimingDrag();
      } else if (drag.mode === 'region') {
        // 区域内点击（无拖动）= Aegisub 左键设开始；拖动 = 整体平移
        if (!drag.moved) {
          if (this.setStartAt(drag.clickTime)) this.afterTimingDrag();
          if (this.el.audio && this.el.audio.src) {
            this.seekTo(Math.max(0, drag.clickTime));
            this.play();
          }
        } else {
          this.afterTimingDrag();
        }
      } else if (drag.mode === 'pan') {
        if (!drag.moved) {
          if (drag.clickEvent) {
            this.setActiveEvent(drag.clickEvent.index, { source: 'waveform-click' });
          } else if (drag.clickTime != null) {
            // Aegisub：左键点击设开始，并把播放头移到点击处试听。
            if (this.setStartAt(drag.clickTime)) this.afterTimingDrag();
            if (this.el.audio && this.el.audio.src) {
              this.seekTo(Math.max(0, drag.clickTime));
              this.play();
            }
          }
        }
      }
    }

    afterTimingDrag() {
      const event = this.activeEvent;
      if (!event) return;
      document.dispatchEvent(new CustomEvent('waveform:timing-changed', {
        detail: { index: event.index, start: event.start, end: event.end, name: this.name }
      }));
      this.scheduleSave();
    }

    onContextMenu(event) {
      if (!this.duration) return;
      event.preventDefault();
      const rect = this.el.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const time = this.xToTime(x);
      const hit = this.hitTest(x);
      if (hit === 'start' || hit === 'end') return;
      // Aegisub：右键点击设结束。
      if (this.setEndAt(time)) this.afterTimingDrag();
    }

    onDoubleClick(event) {
      const rect = this.el.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const target = this.eventAt(x);
      if (target) this.setActiveEvent(target.index, { source: 'waveform-dblclick' });
    }

    onWheel(event) {
      if (!this.duration) return;
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) {
        const rect = this.el.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const anchor = this.xToTime(x);
        const factor = event.deltaY < 0 ? 1.2 : 1 / 1.2;
        this.setZoom(this.zoom * factor, anchor);
      } else {
        const delta = (event.shiftKey ? 3 : 1) * (event.deltaY || event.deltaX);
        this.viewStart += (delta / 120) * this.viewDuration * 0.25;
        this.clampView();
        this.draw();
      }
    }

    updateCursor(x) {
      if (!this.el.canvas) return;
      const hit = this.hitTest(x);
      const cursor = hit === 'start' || hit === 'end' ? 'ew-resize'
        : hit === 'region' ? 'grab'
        : 'crosshair';
      if (this.el.canvas.style.cursor !== cursor) this.el.canvas.style.cursor = cursor;
    }

    // ------------------------------------------------------------------
    // 键盘快捷键（Aegisub 风格打轴）
    // ------------------------------------------------------------------

    onKeyDown(event) {
      if (!this.isVisible()) return;
      const target = event.target;
      if (target && target.closest && target.closest('input, textarea, select, [contenteditable="true"]')) return;
      // 空格/回车留给聚焦的原生控件（按钮等），避免劫持浏览器默认行为
      const nativeControl = target && target.closest && target.closest('button, a[href], select, summary');
      if (nativeControl && (event.key === ' ' || event.key === 'Enter')) return;
      const audio = this.el.audio;
      const active = this.activeEvent;
      const currentTime = () => Number(audio?.currentTime) || 0;
      const nudge = event.altKey ? 0.001 : NUDGE_STEP;
      let handled = true;

      switch (event.key) {
        case ' ':
          if (audio && audio.src) {
            if (audio.paused) this.playActive();
            else audio.pause();
          }
          break;
        case 's':
        case 'S':
          this.playActive();
          break;
        case 'r':
        case 'R':
          if (active && audio && audio.src) {
            this.seekTo(Number(active.start));
            this.playStopAt = Number(active.end);
            this.play();
          }
          break;
        case '[':
          if (active) this.afterTimingChange(this.setStartAt(currentTime()));
          break;
        case ']':
          if (active) this.afterTimingChange(this.setEndAt(currentTime()));
          break;
        case 'ArrowLeft':
        case 'ArrowRight': {
          const sign = event.key === 'ArrowLeft' ? -1 : 1;
          if (!active) { this.nudgeView(-sign); break; }
          if (event.ctrlKey || event.metaKey) {
            this.afterTimingChange(this.setTiming(Number(active.start) + sign * nudge, Number(active.end) + sign * nudge));
          } else if (event.shiftKey) {
            this.afterTimingChange(this.setTiming(active.start, Number(active.end) + sign * nudge));
          } else {
            this.afterTimingChange(this.setTiming(Number(active.start) + sign * nudge, active.end));
          }
          break;
        }
        case 'ArrowUp':
          this.stepEvent(-1);
          break;
        case 'ArrowDown':
          this.stepEvent(1);
          break;
        case 'Home':
          if (active && audio && audio.src) this.seekTo(Number(active.start));
          break;
        case 'End':
          if (active && audio && audio.src) this.seekTo(Math.max(0, Number(active.end) - 0.05));
          break;
        case 'Enter':
          if (active) {
            this.flushSave().then(() => this.stepEvent(1));
          }
          break;
        case '+':
        case '=':
          this.setZoom(this.zoom * 1.25, active ? Number(active.start) : this.viewCenter());
          break;
        case '-':
          this.setZoom(this.zoom / 1.25, active ? Number(active.start) : this.viewCenter());
          break;
        default:
          handled = false;
      }
      if (handled) event.preventDefault();
    }

    afterTimingChange(changed) {
      if (!changed) return;
      const event = this.activeEvent;
      if (!event) return;
      document.dispatchEvent(new CustomEvent('waveform:timing-changed', {
        detail: { index: event.index, start: event.start, end: event.end, name: this.name }
      }));
      this.scheduleSave();
    }

    nudgeView(direction) {
      this.viewStart += direction * this.viewDuration * 0.25;
      this.clampView();
      this.draw();
    }

    stepEvent(delta) {
      if (!this.events.length) return;
      const positions = this.events.map((e) => Number(e.index));
      const current = positions.indexOf(Number(this.activeIndex));
      const next = Math.max(0, Math.min(positions.length - 1, (current < 0 ? 0 : current) + delta));
      this.setActiveEvent(this.events[next].index, { source: 'waveform-keyboard' });
    }

    isVisible() {
      return !!(this.el.canvas && this.el.canvas.offsetParent !== null && this.events.length);
    }

    // ------------------------------------------------------------------
    // 播放控制
    // ------------------------------------------------------------------

    seekTo(time) {
      if (!this.el.audio || !this.el.audio.src) return;
      const max = Number(this.el.audio.duration) || this.duration || time;
      this.el.audio.currentTime = Math.max(0, Math.min(max, time));
      this.updateTime();
    }

    play() {
      const audio = this.el.audio;
      if (!audio || !audio.src) return;
      if (this.audioContext?.state === 'suspended') this.audioContext.resume();
      audio.play().catch(() => this.setStatus('播放不可用'));
    }

    playActive() {
      const active = this.activeEvent;
      const audio = this.el.audio;
      if (!active || !audio || !audio.src) { this.togglePlay(); return; }
      if (!audio.paused && this.playStopAt != null) { audio.pause(); return; }
      this.seekTo(Number(active.start));
      this.playStopAt = Number(active.end);
      this.play();
    }

    togglePlay() {
      this.init();
      const audio = this.el.audio;
      if (!audio || !audio.src) return;
      if (audio.paused) {
        this.playStopAt = null;
        this.play();
      } else {
        audio.pause();
      }
    }

    stop() {
      this.playStopAt = null;
      if (this.el.audio) {
        this.el.audio.pause();
        this.el.audio.currentTime = 0;
      }
      this.draw();
    }

    followPlayingView() {
      const audio = this.el.audio;
      if (!audio || audio.paused || this.dragging || !this.duration) return;
      const current = Number(audio.currentTime) || 0;
      if (this.playStopAt != null && current >= this.playStopAt) {
        audio.pause();
        this.playStopAt = null;
        return;
      }
      const ratio = (current - this.viewStart) / this.viewDuration;
      if (ratio > 0.78 || ratio < 0) {
        this.viewStart = current - this.viewDuration * 0.3;
        this.clampView();
      }
    }

    resetView() {
      this.zoom = 1;
      this.viewStart = 0;
      if (this.el.zoom) this.el.zoom.value = '1';
      if (this.el.audio) this.el.audio.currentTime = 0;
      this.draw();
    }

    toggleHelp() {
      const panel = this.el.helpPanel;
      if (panel) panel.hidden = !panel.hidden;
    }

    // ------------------------------------------------------------------
    // 播放头与状态
    // ------------------------------------------------------------------

    updatePlayhead() {
      const playhead = this.el.playhead;
      if (!playhead) return;
      const width = this.lastView.width || this.el.canvas?.clientWidth || 1;
      const position = ((Number(this.el.audio?.currentTime) || 0) - this.lastView.start) / this.lastView.duration * width;
      playhead.style.left = Math.max(0, Math.min(width, position)) + 'px';
      playhead.style.display = position < 0 || position > width ? 'none' : '';
    }

    updateTime() {
      const current = Number(this.el.audio?.currentTime) || 0;
      const label = this.el.time;
      if (label) label.textContent = this.formatTime(current) + ' / ' + this.formatTime(this.duration);
    }

    setPlayLabel(playing) {
      const button = this.el.play;
      if (button) { button.textContent = playing ? 'Ⅱ' : '▶'; button.title = playing ? '暂停' : '播放'; }
    }

    highlightEvent() {
      const current = Number(this.el.audio?.currentTime) || 0;
      const event = this.events.find((item) => current >= Number(item.start || 0) && current <= Number(item.end || 0));
      document.querySelectorAll('#subtitle-tbody tr[data-index], #review-tbody tr[data-review-index]').forEach((row) => {
        const index = Number(row.dataset.index || row.dataset.reviewIndex);
        row.classList.toggle('waveform-active-row', !!event && Number(event.index) === index);
      });
    }

    setStatus(text) { if (this.el.status) this.el.status.textContent = text; }
    hideEmpty() { if (this.el.empty) this.el.empty.hidden = true; }
    showEmpty(text) { if (this.el.empty) { this.el.empty.textContent = text; this.el.empty.hidden = false; } }
    formatTime(value) {
      const total = Math.max(0, Number(value) || 0);
      const mins = Math.floor(total / 60);
      const secs = (total % 60).toFixed(3).padStart(6, '0');
      return String(mins).padStart(2, '0') + ':' + secs;
    }
  }

  window.WaveformView = WaveformView;
  window.createWaveformView = (options) => new WaveformView(options);

  // 主面板单例：保持既有 WaveformUI 全局 API 兼容。
  const mainView = new WaveformView({
    name: 'main',
    canvasId: 'waveform-canvas',
    audioId: 'audio-player',
    playheadId: 'waveform-playhead',
    statusId: 'waveform-status',
    emptyId: 'waveform-empty',
    timeId: 'waveform-time',
    zoomId: 'waveform-zoom',
    playId: 'waveform-play',
    stopId: 'waveform-stop',
    resetId: 'waveform-reset',
    helpId: 'waveform-help',
    helpPanelId: 'waveform-help-panel',
  });

  window.WaveformUI = {
    view: mainView,
    init: () => mainView.init(),
    setTask: (taskId, result, events) => mainView.setTask(taskId, result, events),
    setLocalFile: (file) => mainView.setLocalFile(file),
    setEvents: (events) => mainView.setEvents(events),
    clear: (showEmpty) => mainView.clear(showEmpty),
    setActiveEvent: (index, options) => mainView.setActiveEvent(index, options),
    activeEventIndex: () => mainView.activeIndex,
    commitTiming: () => mainView.flushSave(),
    get onTimingChange() { return mainView.onTimingChange; },
    set onTimingChange(fn) { mainView.onTimingChange = fn; },
  };
  document.addEventListener('DOMContentLoaded', () => mainView.init());
})();
