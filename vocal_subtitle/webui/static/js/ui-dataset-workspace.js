(function() {
  'use strict';
  const esc = (value) => (window.App && App.ui && App.ui.escapeHtml) ? App.ui.escapeHtml(String(value ?? '')) : String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  // 数据集工作区（四场景定案 D31/D32/D33/D35）：8613 处理台的 dataset-v1 导出套壳。
  // 导出逻辑在服务端与 CLI `feedback export-dataset` 共用同一服务层，此处只做图形入口：
  // 统计预览（accepted-only + 场景过滤，与导出同一收集口径，预览数字 = 导出条目数）
  // → 许可必选（D33，缺失时服务端 400 门禁拒绝）→ 导出 → 结果路径与告警展示。
  // 8631 不提供导出（治理操作归处理台）；git 推送由用户手动完成。
  const SCENARIOS = [
    ['inline-review', '工作台内修正 (V1)'],
    ['external-correction', '外部工具修正上传 (V2)'],
    ['existing-subtitle', '存量字幕学习 (V3)'],
    ['from-scratch-timing', '从零打轴 (V4)'],
  ];
  const LICENSES = ['CC-BY-4.0', 'CC0-1.0', 'CC-BY-SA-4.0', 'CC-BY-NC-4.0'];

  const DatasetWorkspace = {
    initialized: false,
    defaultOutDir: '',
    running: false,

    init() {
      if (this.initialized) return;
      this.initialized = true;
      document.getElementById('btn-dataset-export')?.addEventListener('click', () => this.exportDataset());
      // 过滤条件变化后自动刷新统计预览（无需手动刷新按钮）
      document.getElementById('dataset-scenario-filter')?.addEventListener('change', () => this.loadPreview());
      document.getElementById('dataset-name')?.addEventListener('change', () => this.loadPreview());
      document.getElementById('dataset-out-dir')?.addEventListener('change', () => this.loadPreview());
      this.renderScenarioFilter();
      this.renderLicenseOptions();
    },

    renderScenarioFilter() {
      const wrap = document.getElementById('dataset-scenario-filter');
      if (!wrap || wrap.children.length) return;
      wrap.innerHTML = SCENARIOS.map(([value, label]) =>
        '<label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:0.8rem;color:var(--text-primary);">' +
        '<input type="checkbox" value="' + esc(value) + '" style="width:14px;height:14px;accent-color:var(--accent);">' +
        '<span><span style="font-family:var(--font-mono);font-size:0.72rem;">' + esc(value) + '</span> · ' + esc(label) + '</span></label>'
      ).join('');
    },

    renderLicenseOptions() {
      const select = document.getElementById('dataset-license');
      if (!select || select.options.length > 1) return;
      select.innerHTML = '<option value="">— 请选择许可（必选） —</option>' +
        LICENSES.map((name) => '<option value="' + esc(name) + '">' + esc(name) + '</option>').join('');
    },

    selectedScenarios() {
      return Array.from(document.querySelectorAll('#dataset-scenario-filter input[type="checkbox"]:checked'))
        .map((box) => box.value);
    },

    outDir() {
      return (document.getElementById('dataset-out-dir')?.value || '').trim();
    },

    datasetName() {
      return (document.getElementById('dataset-name')?.value || '').trim() || 'subtitle-feedback';
    },

    async refresh() {
      this.init();
      await this.loadPreview();
    },

    // 统计预览：与导出走同一个服务层收集口径（accepted-only + 场景过滤 +
    // 追加式已导出剔除），因此预览的"本次将导出"数字 = 导出返回的 exported_count
    async loadPreview() {
      const target = document.getElementById('dataset-preview-stats');
      if (!target) return;
      try {
        const data = await VocalSubtitleApi.datasetPreview(this.selectedScenarios(), this.outDir(), this.datasetName());
        if (data.default_out_dir) {
          this.defaultOutDir = data.default_out_dir;
          const input = document.getElementById('dataset-out-dir');
          if (input && !input.placeholder) input.placeholder = data.default_out_dir;
        }
        this.renderPreview(data);
      } catch (error) {
        target.innerHTML = '<div class="workspace-empty">统计预览不可用: ' + esc(this.detailOf(error)) + '</div>';
      }
    },

    renderPreview(data) {
      const target = document.getElementById('dataset-preview-stats');
      if (!target) return;
      const skipped = (data.skipped_missing_text || []).length;
      const filterText = (data.scenarios_filter || []).length ? data.scenarios_filter.join('、') : '全部场景';
      const rows = [
        ['过滤器', filterText],
        ['accepted 总数（过滤后）', data.accepted_total],
        ['本次将导出', data.exportable_count],
        ['已在此目录导出过', data.already_exported],
        ['缺字幕全文（跳过）', skipped],
        ['数据集累计样本', data.existing_dataset_samples],
        ['字幕总时长', (Number(data.total_duration_seconds || 0)).toFixed(1) + ' 秒'],
      ];
      let html = '<dl class="detail-list">' + rows.map((row) =>
        '<div><dt>' + esc(row[0]) + '</dt><dd>' + esc(row[1]) + '</dd></div>').join('') + '</dl>';
      html += this.distributionBlock('按场景', data.by_scenario) +
        this.distributionBlock('按语言', data.by_language);
      if (!Number(data.exportable_count)) {
        const reason = Number(data.accepted_total)
          ? '没有新增可导出的样本（此前已全部导出，或缺少字幕全文）'
          : '暂无可导出的样本：请先在"反馈档案"的审核队列中接受样本';
        html = '<div class="workspace-empty" style="text-align:left;">' + esc(reason) + '</div>' + html;
      }
      target.innerHTML = html;
    },

    distributionBlock(title, values) {
      const entries = Object.entries(values || {});
      const text = entries.length
        ? entries.map(([key, count]) => esc(key) + ' ' + esc(count) + ' 条').join('，')
        : '无';
      return '<div class="section-label" style="margin-top:10px;">' + esc(title) + '</div>' +
        '<div style="font-size:0.76rem;color:var(--text-secondary);">' + text + '</div>';
    },

    async exportDataset() {
      if (this.running) return;
      const errorBox = document.getElementById('dataset-error');
      const showError = (msg) => { if (errorBox) { errorBox.textContent = msg; errorBox.hidden = false; } };
      if (errorBox) errorBox.hidden = true;
      const license = document.getElementById('dataset-license')?.value || '';
      if (!license) { showError('请先选择数据集许可（必选，将记入 README 数据集卡片）'); return; }
      this.running = true;
      const buttons = ['btn-dataset-export'].map((id) => document.getElementById(id));
      buttons.forEach((btn) => { if (btn) btn.disabled = true; });
      try {
        const payload = {
          license: license,
          scenarios: this.selectedScenarios(),
          dataset_name: this.datasetName(),
          bundle_audio: !!document.getElementById('dataset-bundle-audio')?.checked,
        };
        const outDir = this.outDir();
        if (outDir) payload.out_dir = outDir;
        const data = await VocalSubtitleApi.datasetExport(payload);
        if (data.status !== 'ok') { showError(data.message || '导出失败'); return; }
        this.renderResult(data);
        toast(data.message || '导出完成', 'success');
        // 导出后刷新预览：同参数下"本次将导出"应归零（追加式已导出剔除），
        // 与导出结果 exported_count 呼应，验证预览 = 导出
        await this.loadPreview();
      } catch (error) {
        // 400（缺许可/空库/未知场景）等：展示服务端 detail（许可门禁语义来自服务层）
        showError(this.detailOf(error));
      } finally {
        this.running = false;
        buttons.forEach((btn) => { if (btn) btn.disabled = false; });
      }
    },

    renderResult(data) {
      const target = document.getElementById('dataset-export-result');
      if (!target) return;
      const rows = [
        ['结果目录', data.out_dir],
        ['许可', data.license],
        ['本次新增', data.exported_count + ' 条'],
        ['数据集累计', (data.total_samples || 0) + ' 条'],
        ['过滤器', (data.scenarios || []).length ? data.scenarios.join('、') : '全部场景'],
      ];
      let html = '<div class="section-label">导出结果</div><dl class="detail-list">' + rows.map((row) =>
        '<div><dt>' + esc(row[0]) + '</dt><dd>' + esc(row[1]) + '</dd></div>').join('') + '</dl>';
      if ((data.shards || []).length) {
        html += '<div class="section-label" style="margin-top:8px;">新增分片</div>' +
          data.shards.map((shard) => '<div style="font-size:0.72rem;font-family:var(--font-mono);color:var(--text-secondary);">' + esc(shard) + '</div>').join('');
      } else {
        html += '<div style="font-size:0.74rem;color:var(--text-tertiary);margin-top:6px;">无新增样本，未写入新分片（追加式幂等）</div>';
      }
      html += this.alertBlock(data);
      html += '<div style="font-size:0.72rem;color:var(--text-tertiary);margin-top:8px;">目录为 git-ready，推送请手动执行；建议以 git tag 标记发行版本</div>';
      target.innerHTML = html;
    },

    // 告警展示：缺字幕全文跳过 / 打包音频缺失 / bundle-audio 的 git 警示
    alertBlock(data) {
      let html = '';
      if ((data.skipped_missing_text || []).length) {
        html += '<div class="workspace-error" style="text-align:left;padding:8px 12px;margin-top:8px;">⚠ ' +
          esc(data.skipped_missing_text.length) + ' 个样本缺少字幕全文（旧版样本库入库），已跳过: ' +
          esc(data.skipped_missing_text.join('、')) + '</div>';
      }
      if ((data.missing_audio || []).length) {
        html += '<div class="workspace-error" style="text-align:left;padding:8px 12px;margin-top:8px;">⚠ ' +
          esc(data.missing_audio.length) + ' 个音频引用找不到实体文件（引用照写）: ' +
          esc(data.missing_audio.join('、')) + '</div>';
      }
      if (document.getElementById('dataset-bundle-audio')?.checked) {
        html += '<div class="workspace-error" style="text-align:left;padding:8px 12px;margin-top:8px;">⚠ audio/ 目录含原始音频实体，不建议推送到 git 远程</div>';
      }
      return html;
    },

    detailOf(error) {
      const text = error && error.message ? error.message : String(error);
      try {
        const parsed = JSON.parse(text);
        return parsed.detail || text;
      } catch (e) { return text; }
    },
  };
  window.DatasetWorkspace = DatasetWorkspace;
  document.addEventListener('DOMContentLoaded', () => DatasetWorkspace.init());
})();
