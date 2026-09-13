(function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  // 增量 SHA-256：crypto.subtle 不支持流式，整文件 arrayBuffer() 会让 GB 级片源
  // 撑爆内存/冻结页面；大文件分块喂入此实现（正确性有 node:crypto 对照验证）
  function createIncrementalSha256() {
    const K = new Uint32Array([
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
    let h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a,
        h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
    const block = new Uint8Array(64);
    const w = new Uint32Array(64);
    let pending = 0, totalBytes = 0;

    function rotr(x, n) { return ((x >>> n) | (x << (32 - n))) >>> 0; }

    function compress(chunk) {
      for (let i = 0; i < 16; i++) {
        w[i] = ((chunk[i * 4] << 24) | (chunk[i * 4 + 1] << 16) | (chunk[i * 4 + 2] << 8) | chunk[i * 4 + 3]) >>> 0;
      }
      for (let i = 16; i < 64; i++) {
        const s0 = (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)) >>> 0;
        const s1 = (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)) >>> 0;
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
      }
      let a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
      for (let i = 0; i < 64; i++) {
        const S1 = (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) >>> 0;
        const ch = ((e & f) ^ (~e & g)) >>> 0;
        const t1 = (hh + S1 + ch + K[i] + w[i]) >>> 0;
        const S0 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) >>> 0;
        const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
        const t2 = (S0 + maj) >>> 0;
        hh=g; g=f; f=e; e=(d + t1) >>> 0; d=c; c=b; b=a; a=(t1 + t2) >>> 0;
      }
      h0=(h0+a)>>>0; h1=(h1+b)>>>0; h2=(h2+c)>>>0; h3=(h3+d)>>>0;
      h4=(h4+e)>>>0; h5=(h5+f)>>>0; h6=(h6+g)>>>0; h7=(h7+hh)>>>0;
    }

    return {
      update(data) {
        totalBytes += data.length;
        let offset = 0;
        if (pending > 0) {
          const take = Math.min(64 - pending, data.length);
          block.set(data.subarray(0, take), pending);
          pending += take; offset = take;
          if (pending === 64) { compress(block); pending = 0; }
        }
        while (offset + 64 <= data.length) {
          compress(data.subarray(offset, offset + 64));
          offset += 64;
        }
        if (offset < data.length) {
          block.set(data.subarray(offset), pending);
          pending += data.length - offset;
        }
      },
      hex() {
        const hi = Math.floor(totalBytes / 0x20000000);
        const lo = ((totalBytes % 0x20000000) * 8) >>> 0;
        const tail = [0x80];
        while ((pending + tail.length) % 64 !== 56) tail.push(0);
        tail.push((hi>>>24)&255,(hi>>>16)&255,(hi>>>8)&255,hi&255,
                  (lo>>>24)&255,(lo>>>16)&255,(lo>>>8)&255,lo&255);
        this.update(new Uint8Array(tail)); // totalBytes 增长无碍：消息长度已定格于上方
        return [h0,h1,h2,h3,h4,h5,h6,h7].map((x) => x.toString(16).padStart(8, '0')).join('');
      },
    };
  }

  // 上传学习窗口（四场景定案 D20/D23/D26）：8613 处理台的 V2 external-correction 入口。
  // 来源任务两路绑定：任务历史手选（主）/ 拖入本地片源 sha256 字节级匹配（辅，
  // 重编码/抽轨不命中时提示回退手选）。仅上传修正字幕即可学习：
  // 基线引用服务端已存任务产物（/api/feedback/learn 的 task_id 分支），无需上传音频。
  const UploadLearn = {
    initialized: false,
    tasks: [],
    selectedTaskId: null,
    subtitleFile: null,
    matching: false,
    running: false,

    init() {
      if (this.initialized) return;
      this.initialized = true;
      this.wireDropZone('upload-learn-drop', 'upload-learn-drop-input', (file) => this.matchByFile(file));
      this.wireDropZone('upload-learn-sub-zone', 'upload-learn-sub-input', (file) => this.setSubtitle(file));
      document.getElementById('btn-upload-learn-preview')?.addEventListener('click', () => this.submit(true));
      document.getElementById('btn-upload-learn-run')?.addEventListener('click', () => this.submit(false));
    },

    // 点击选择 + 拖拽落放共用一个隐藏 file input
    wireDropZone(zoneId, inputId, onFile) {
      const zone = document.getElementById(zoneId);
      const input = document.getElementById(inputId);
      if (!zone || !input) return;
      zone.addEventListener('click', () => input.click());
      ['dragover', 'dragenter'].forEach((name) => zone.addEventListener(name, (event) => {
        event.preventDefault();
        zone.classList.add('drag-over');
      }));
      ['dragleave', 'drop'].forEach((name) => zone.addEventListener(name, () => zone.classList.remove('drag-over')));
      zone.addEventListener('drop', (event) => {
        event.preventDefault();
        const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) onFile(file);
      });
      input.addEventListener('change', (event) => {
        if (event.target.files && event.target.files[0]) onFile(event.target.files[0]);
        event.target.value = '';
      });
    },

    async refresh() {
      this.init();
      const target = document.getElementById('upload-learn-task-list');
      if (!target) return;
      try {
        // 学习来源只看成功任务（completed / degraded_completed 都带完整事件基线）
        const data = await VocalSubtitleApi.getHistory(50, 0);
        this.tasks = (data.items || []).filter((item) => item.status === 'completed' || item.status === 'degraded_completed');
        this.renderTasks();
      } catch (error) {
        target.innerHTML = '<div class="workspace-empty">任务历史不可用</div>';
      }
    },

    renderTasks() {
      const target = document.getElementById('upload-learn-task-list');
      if (!target) return;
      if (!this.tasks.length) {
        target.innerHTML = '<div class="workspace-empty">暂无已完成的任务历史</div>';
        return;
      }
      target.innerHTML = this.tasks.map((item, idx) => {
        const date = item.created_at ? item.created_at.slice(0, 16).replace('T', ' ') : '';
        const selected = item.id === this.selectedTaskId;
        const style = selected ? 'cursor:pointer;border-color:var(--accent);background:rgba(88,166,255,0.08);' : 'cursor:pointer;';
        return '<div class="history-item" data-idx="' + idx + '" style="' + style + '">' +
          '<span class="h-status ok">✓</span>' +
          '<span class="h-name" title="' + esc(item.input_file_name) + '">' + esc(item.input_file_name) + '</span>' +
          '<span class="h-meta">' + esc(date) + '</span>' +
          '</div>';
      }).join('');
      target.querySelectorAll('.history-item[data-idx]').forEach((el) => {
        el.addEventListener('click', () => {
          const item = this.tasks[parseInt(el.dataset.idx)];
          if (item) this.selectTask(item.id);
        });
      });
    },

    selectTask(taskId) {
      this.selectedTaskId = taskId;
      this.renderTasks();
      const item = this.tasks.find((task) => task.id === taskId);
      toast('已选择来源任务：' + (item ? item.input_file_name : taskId), 'info');
    },

    // 拖入本地片源 → Web Crypto sha256（与后端 input_file_hash 同口径）→ 哈希查询自动选中
    async matchByFile(file) {
      if (this.matching) return;
      this.matching = true;
      const nameEl = document.getElementById('upload-learn-drop-name');
      if (nameEl) nameEl.textContent = '正在计算 ' + file.name + ' 的 sha256...';
      try {
        const digest = await this.sha256Hex(file);
        if (nameEl) nameEl.textContent = file.name + '（' + digest.slice(0, 16) + '…）';
        const data = await VocalSubtitleApi.getHistoryByHash(digest);
        if (data.items && data.items.length) {
          this.selectTask(data.items[0].id);
          toast('已按内容哈希匹配到任务：' + data.items[0].input_file_name, 'success');
        } else {
          // D26：仅字节级同文件命中；重编码/抽轨/不同片源不命中 → 明确提示回退手选
          toast('未命中任务历史（仅字节级相同的文件可匹配，重编码/抽轨不命中），请从列表手动选择来源任务', 'error');
        }
      } catch (error) {
        if (nameEl) nameEl.textContent = '';
        toast('哈希匹配失败: ' + this.detailOf(error), 'error');
      } finally {
        this.matching = false;
      }
    },

    async sha256Hex(file) {
      // 小文件走原生 one-shot（快）；大文件分块增量，避免整文件 arrayBuffer 撑爆内存
      if (file.size <= 64 * 1024 * 1024) {
        const buffer = await file.arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', buffer);
        return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
      }
      const CHUNK = 8 * 1024 * 1024;
      const state = createIncrementalSha256();
      for (let offset = 0; offset < file.size; offset += CHUNK) {
        const buffer = await file.slice(offset, Math.min(offset + CHUNK, file.size)).arrayBuffer();
        state.update(new Uint8Array(buffer));
      }
      return state.hex();
    },

    setSubtitle(file) {
      if (!/\.(srt|ass)$/i.test(file.name)) {
        toast('仅支持 .srt / .ass 修正字幕文件', 'error');
        return;
      }
      this.subtitleFile = file;
      const nameEl = document.getElementById('upload-learn-sub-name');
      if (nameEl) nameEl.textContent = file.name + '（' + (file.size / 1024).toFixed(1) + ' KB）';
      document.getElementById('upload-learn-sub-zone')?.classList.add('has-file');
    },

    async submit(dryRun) {
      if (this.running) return;
      const errorBox = document.getElementById('upload-learn-error');
      const showError = (msg) => { if (errorBox) { errorBox.textContent = msg; errorBox.hidden = false; } };
      if (errorBox) errorBox.hidden = true;
      if (!this.selectedTaskId) { showError('请先选择来源任务（历史手选，或拖入本地片源自动匹配）'); return; }
      if (!this.subtitleFile) { showError('请先选择修正字幕文件（.srt / .ass）'); return; }
      this.running = true;
      const buttons = ['btn-upload-learn-preview', 'btn-upload-learn-run']
        .map((id) => document.getElementById(id));
      buttons.forEach((btn) => { if (btn) btn.disabled = true; });
      try {
        const fd = new FormData();
        fd.append('task_id', this.selectedTaskId);
        fd.append('scenario', 'external-correction');
        fd.append('dry_run', dryRun ? 'true' : 'false');
        fd.append('reference', this.subtitleFile);
        const data = await VocalSubtitleApi.feedbackLearn(fd);
        if (data.status !== 'ok') { showError(data.message || '学习失败'); return; }
        this.renderReport(data, dryRun);
        toast(dryRun ? '对齐预览完成（未写入学习档案）' : '学习完成：' + (data.message || ''), 'success');
      } catch (error) {
        // 404（任务不存在）/ 410（会话产物已清理）等：展示后端 detail（含改走音频上传的提示）
        showError(this.detailOf(error));
      } finally {
        this.running = false;
        buttons.forEach((btn) => { if (btn) btn.disabled = false; });
      }
    },

    detailOf(error) {
      const text = error && error.message ? error.message : String(error);
      try {
        const parsed = JSON.parse(text);
        return parsed.detail || text;
      } catch (e) { return text; }
    },

    // 对齐报告：覆盖率 / 配对数 / 时间偏移 / 重构行 / 参数调整（D20/D26 报告字段）
    renderReport(data, dryRun) {
      const target = document.getElementById('upload-learn-report');
      if (!target) return;
      const pct = (value) => Math.round(Number(value || 0) * 100) + '%';
      const rows = [
        ['对齐覆盖率', pct(data.alignment_coverage)],
        ['配对数', data.total_pairs],
        ['时间偏移', data.time_shifts_count],
        ['合并/拆分', data.merge_actions_count],
        ['文本修改', data.text_edits_count],
        ['重构行（不参与时间轴学习）', data.reconstructed_lines],
        ['自动版多出行', data.inserted_lines],
        ['基线来源', data.baseline_source === 'task_history' ? '任务历史（未重跑管线）' : (data.baseline_source || '—')],
      ];
      let html = '<dl class="detail-list">' + rows.map((row) =>
        '<div><dt>' + esc(row[0]) + '</dt><dd>' + esc(row[1]) + '</dd></div>').join('') + '</dl>';
      if (data.coverage_warning) {
        html = '<div class="workspace-error" style="text-align:left;padding:8px 12px;">⚠ ' + esc(data.coverage_warning) + '</div>' + html;
      }
      const entries = Object.entries(data.param_adjustments || {});
      if (entries.length) {
        html += '<div class="section-label" style="margin-top:10px;">参数调整建议（' + entries.length + '）</div>';
        html += entries.map((entry) => {
          const adj = entry[1] || {};
          const value = (adj.direction || '—') + ' · 置信度 ' + (adj.confidence !== undefined ? Number(adj.confidence).toFixed(2) : '—');
          return '<div class="feedback-metric-row"><span class="fm-label" style="font-family:var(--font-mono);font-size:0.72rem;">' + esc(entry[0]) + '</span><span class="fm-value">' + esc(value) + '</span></div>' +
            '<div style="font-size:0.72rem;color:var(--text-tertiary);">' + esc(adj.reason || '') + '</div>';
        }).join('');
      }
      if (data.message) {
        html += '<div style="font-size:0.74rem;color:var(--text-tertiary);margin-top:8px;">' + esc(data.message) + '</div>';
      }
      target.innerHTML = html;
    },
  };
  window.UploadLearn = UploadLearn;
  document.addEventListener('DOMContentLoaded', () => UploadLearn.init());
})();
