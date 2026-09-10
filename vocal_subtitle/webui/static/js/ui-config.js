(function() {
'use strict';
const part = {
    init() {
      // 加载持久化设置
      var saved = App.loadSettings();
      if (saved.profile) App.state.selectedProfile = saved.profile;
      if (saved.llm_base_url) App.ui._llmState.base_url = saved.llm_base_url;
      if (saved.llm_api_key) App.ui._llmState.api_key = saved.llm_api_key;
      // Restore skip-separation toggle
      if (saved._skip_separation) {
        App.state.skipSeparation = true;
        var skipCb = document.getElementById('skip-separation-cb');
        if (skipCb) skipCb.checked = true;
      }

      App.ui.initUploadZone();
      App.ui.initProfileCards();
      App.ui.initDeviceInfo();
      App.refreshHistory();
      App.refreshCacheInfo();

      // 全局设置变更监听 — 任何表单项变化时自动保存
      var optionsPanel = document.getElementById('options-panel');
      if (optionsPanel) {
        optionsPanel.addEventListener('change', function() { App.saveSettings(); });
        optionsPanel.addEventListener('input', function(e) {
          if (e.target.type === 'range') {
            clearTimeout(App._saveTimeout);
            App._saveTimeout = setTimeout(function() { App.saveSettings(); }, 300);
          } else {
            App.saveSettings();
          }
        });
      }
      // Skip-separation toggle listener
      var skipSepCb = document.getElementById('skip-separation-cb');
      if (skipSepCb) {
        skipSepCb.addEventListener('change', function() { App.saveSettings(); });
      }

    },

    // Upload Zone
    initUploadZone() {
      const zone = $('#upload-zone');
      const input = $('#file-input');

      zone.addEventListener('click', () => input.click());

      zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
      zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
      zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) App.ui.setFile(files[0]);
      });

      input.addEventListener('change', () => {
        if (input.files.length > 0) App.ui.setFile(input.files[0]);
      });
    },

    setFile(file) {
      App.state.selectedFile = file;
      App.ui.clearPipelineError();
      const zone = $('#upload-zone');
      zone.classList.add('has-file');
      $('#upload-filename').textContent = '✓ ' + file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';

      const railFile = document.getElementById('process-rail-file');
      const railStatus = document.getElementById('process-rail-status');
      if (railFile) railFile.textContent = file.name;
      if (railStatus) {
        railStatus.textContent = '待处理';
        railStatus.dataset.status = 'ready';
      }

      // Enable run button
      const btn = $('#btn-run');
      btn.disabled = false;
      $('#btn-run-text').textContent = '开始处理';
    },

    // Profile Cards
    async initProfileCards() {
      try {
        const profiles = await App.api.getProfiles();
        const container = $('#profile-cards');
        container.innerHTML = '';

        profiles.forEach((p, i) => {
          const card = document.createElement('div');
          card.className = 'profile-card' + (p.name === App.state.selectedProfile ? ' active' : '');
          card.innerHTML = `
            <div class="pc-icon">${['🎵','🎙️','📚','📺','🎸'][i] || '🎧'}</div>
            <div class="pc-info">
              <div class="pc-name">${p.name === 'default' ? '默认' : p.name === 'podcast' ? '播客/访谈' : p.name === 'education' ? '教学/演讲' : p.name === 'variety_show' ? '综艺/直播' : p.name === 'music_live' ? '音乐现场' : p.name}</div>
              <div class="pc-desc">${p.description}</div>
            </div>
            <div class="pc-badge">${(p.config_summary && p.config_summary.separation_engine) || '—'}</div>
          `;

          card.addEventListener('click', () => App.ui.selectProfile(p.name));
          container.appendChild(card);
        });

        // Load default profile config
        await App.ui.loadProfileConfig(App.state.selectedProfile);
      } catch (ex) {
        console.error('Failed to load profiles:', ex);
        toast('加载场景模板失败', 'error');
      }
    },

    async selectProfile(name) {
      App.state.selectedProfile = name;
      $$('.profile-card').forEach(c => c.classList.remove('active'));
      const cards = $$('.profile-card');
      const idx = ['default','podcast','education','variety_show','music_live'].indexOf(name);
      if (idx >= 0 && cards[idx]) cards[idx].classList.add('active');
      App.saveSettings();
      await App.ui.loadProfileConfig(name);
    },

    async loadProfileConfig(name) {
      try {
        const data = await App.api.getProfileConfig(name);
        App.state.profileConfig = data.config;
        App.ui.renderOptions(data.config);
        // 刷新设备提示
        setTimeout(function() { App.ui.updateDeviceHint(); }, 50);
      } catch (ex) {
        console.error('Failed to load config:', ex);
        toast('加载配置失败', 'error');
      }
    },

    renderOptions(config) {
      const asrEngine = config.asr_engine || 'auto';
      const genericASRModels = ['large-v3', 'medium', 'small', 'tiny'];
      const configuredASRModel = config.asr_model || 'large-v3';
      const funasrDefaultModel = 'iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch';
      const initialFunASRModel = asrEngine === 'funasr' && genericASRModels.indexOf(configuredASRModel) >= 0
        ? funasrDefaultModel : configuredASRModel;
      const asrModelOptions = asrEngine === 'funasr'
        ? '<option value="' + App.ui._escapeAttr(initialFunASRModel) + '" selected>FunASR Paraformer（中文）</option>'
        : '<option value="large-v3" ' + (configuredASRModel === 'large-v3' ? 'selected' : '') + '>large-v3</option>' +
          '<option value="medium" ' + (configuredASRModel === 'medium' ? 'selected' : '') + '>medium</option>' +
          '<option value="small" ' + (configuredASRModel === 'small' ? 'selected' : '') + '>small</option>' +
          '<option value="tiny" ' + (configuredASRModel === 'tiny' ? 'selected' : '') + '>tiny</option>';

      // Separation options
      $('#opts-separation').innerHTML = `
        <div class="option-row"><label>引擎</label><select data-key="separator">
          <option value="uvr" ${config.separator==='uvr'?'selected':''}>UVR (BS-RoFormer, 推荐)</option>
          <option value="openunmix" ${config.separator==='openunmix'?'selected':''}>Open-Unmix (品质)</option>
          <option value="spleeter" ${config.separator==='spleeter'?'selected':''}>Spleeter (旧, 仅 Py<3.12)</option>
        </select></div>
        <div class="option-row"><label>UVR 模型</label><input data-key="uvr_model" value="${config.uvr_model||''}" placeholder="model_bs_roformer..."></div>
      `;

      // VAD options
      $('#opts-vad').innerHTML = `
        <div class="option-row"><label>引擎</label><select data-key="vad_engine">
          <option value="silero" ${config.vad_engine==='silero'?'selected':''}>Silero VAD</option>
          <option value="ten" ${config.vad_engine==='ten'?'selected':''}>TEN VAD</option>
          <option value="webrtc" ${config.vad_engine==='webrtc'?'selected':''}>WebRTC VAD</option>
        </select></div>
        <div class="option-row"><label>阈值</label><input type="range" data-key="vad_threshold" min="0.1" max="0.9" step="0.05" value="${config.vad_threshold||0.5}"><span class="value-display">${config.vad_threshold||0.5}</span></div>
      `;

      // ASR options
      $('#opts-asr').innerHTML = `
        <div class="option-row"><label>引擎</label><select data-key="asr_engine">
          <option value="auto" ${asrEngine==='auto'?'selected':''}>自动（中文优先）</option>
          <option value="faster-whisper" ${config.asr_engine==='faster-whisper'?'selected':''}>faster-whisper</option>
         <option value="whisper-cpp" ${config.asr_engine==='whisper-cpp'?'selected':''}>whisper.cpp</option>
         <option value="funasr" ${config.asr_engine==='funasr'?'selected':''}>FunASR (中文优化)</option>
          <option value="qwen" ${config.asr_engine==='qwen'?'selected':''}>Qwen3-ASR</option>
        </select></div>
        <div class="option-row"><label>主引擎</label><select data-key="primary_engine">
          <option value="auto" ${config.primary_engine==='auto'?'selected':''}>自动按语言</option>
          <option value="funasr" ${config.primary_engine==='funasr'?'selected':''}>FunASR</option>
          <option value="qwen" ${config.primary_engine==='qwen'?'selected':''}>Qwen3-ASR</option>
          <option value="faster-whisper" ${config.primary_engine==='faster-whisper'?'selected':''}>Whisper</option>
        </select></div>
        <div class="option-row"><label>副引擎</label><select data-key="secondary_engine">
          <option value="auto" ${config.secondary_engine==='auto'?'selected':''}>自动按语言</option>
          <option value="qwen" ${config.secondary_engine==='qwen'?'selected':''}>Qwen3-ASR</option>
          <option value="funasr" ${config.secondary_engine==='funasr'?'selected':''}>FunASR</option>
          <option value="faster-whisper" ${config.secondary_engine==='faster-whisper'?'selected':''}>Whisper</option>
        </select></div>
        <div class="option-row"><label>复核策略</label><select data-key="engine_pair_policy">
          <option value="risk_only" ${config.engine_pair_policy==='risk_only'?'selected':''}>risk_only</option>
          <option value="full_quality" ${config.engine_pair_policy==='full_quality'?'selected':''}>full_quality</option>
        </select></div>
        <div class="option-row"><label>模型</label><select data-key="asr_model">${asrModelOptions}</select><span class="device-hint" id="funasr-status" style="display:none;"></span></div>
        <div class="option-row"><label>设备</label><select data-key="device" onchange="App.ui.onDeviceChange()">
          <option value="auto" ${(!config.asr_device||config.asr_device==='auto')?'selected':''}>自动 (跟随系统)</option>
          <option value="cuda" ${config.asr_device==='cuda'?'selected':''}>CUDA (GPU)</option>
          <option value="cpu" ${config.asr_device==='cpu'?'selected':''}>CPU</option>
        </select><span class="device-hint" id="device-hint"></span></div>
        <div class="option-row"><label>语言</label><select data-key="language">
          <option value="" ${!config.language?'selected':''}>自动检测 (Auto)</option>
          <option value="zh" ${config.language==='zh'?'selected':''}>中文 (Chinese)</option>
          <option value="en" ${config.language==='en'?'selected':''}>English</option>
          <option value="ja" ${config.language==='ja'?'selected':''}>日本語 (Japanese)</option>
          <option value="ko" ${config.language==='ko'?'selected':''}>한국어 (Korean)</option>
          <option value="fr" ${config.language==='fr'?'selected':''}>Français (French)</option>
          <option value="de" ${config.language==='de'?'selected':''}>Deutsch (German)</option>
          <option value="es" ${config.language==='es'?'selected':''}>Español (Spanish)</option>
          <option value="pt" ${config.language==='pt'?'selected':''}>Português (Portuguese)</option>
          <option value="ru" ${config.language==='ru'?'selected':''}>Русский (Russian)</option>
          <option value="ar" ${config.language==='ar'?'selected':''}>العربية (Arabic)</option>
        </select></div>
      `;

      const asrEngineSelect = $('#opts-asr [data-key="asr_engine"]');
      if (asrEngineSelect) {
        asrEngineSelect.addEventListener('change', function() {
          App.ui.syncASREngineOptions(this.value, true);
          App.saveSettings();
        });
      }

      // Subtitle options
      $('#opts-subtitle').innerHTML = `
        <div class="option-row"><label>最小时长</label><input type="range" data-key="subtitle_min_duration" min="0.3" max="2" step="0.1" value="${config.subtitle_min_duration||0.8}"><span class="value-display">${config.subtitle_min_duration||0.8}s</span></div>
        <div class="option-row"><label>最大时长</label><input type="range" data-key="subtitle_max_duration" min="2" max="10" step="0.5" value="${config.subtitle_max_duration||5.0}"><span class="value-display">${config.subtitle_max_duration||5.0}s</span></div>
      `;

      // Diarization options
      $('#opts-diarization').innerHTML = `
        <div class="option-row"><label>启用说话人分离</label><input type="checkbox" data-key="diarization_enabled" ${config.diarization_enabled?'checked':''}></div>
        <div class="option-row"><label>融合模式</label><select data-key="speaker_fusion">
          <option value="auto" ${(config.speaker_fusion||'auto')==='auto'?'selected':''}>自动</option>
          <option value="embedding" ${config.speaker_fusion==='embedding'?'selected':''}>仅声纹嵌入</option>
          <option value="dual" ${config.speaker_fusion==='dual'?'selected':''}>线路四：双路融合</option>
        </select></div>
        <div class="option-row"><label>全局模型</label><select data-key="global_diarization_model">
          <option value="auto" ${(config.global_diarization_model||'auto')==='auto'?'selected':''}>自动</option>
          <option value="none" ${config.global_diarization_model==='none'?'selected':''}>禁用</option>
          <option value="community-1" ${config.global_diarization_model==='community-1'?'selected':''}>Community-1</option>
          <option value="diarization-3.1" ${config.global_diarization_model==='diarization-3.1'?'selected':''}>Diarization 3.1</option>
        </select></div>
        <div class="option-row"><label>处理范围</label><select data-key="speaker_diarization_scope">
          <option value="hierarchical" ${(config.speaker_diarization_scope||'hierarchical')==='hierarchical'?'selected':''}>线路四：全局 + 片段精修</option>
          <option value="global" ${config.speaker_diarization_scope==='global'?'selected':''}>仅全局 turns</option>
        </select></div>
        <div class="option-row"><label>已知说话人数</label><input type="number" data-key="expected_speakers" value="${config.expected_speakers||''}" min="1" max="20" placeholder="自动" style="max-width:80px;"></div>
        <div class="option-row"><label>局部换人检测</label><select data-key="local_speaker_refinement">
          <option value="embedding" ${(config.local_speaker_refinement||'embedding')==='embedding'?'selected':''}>ECAPA 检测</option>
          <option value="full" ${config.local_speaker_refinement==='full'?'selected':''}>全局模型复核</option>
          <option value="off" ${config.local_speaker_refinement==='off'?'selected':''}>关闭</option>
        </select></div>
        <div class="option-row"><label>局部上下文 (秒)</label><input type="number" data-key="diarization_local_context" value="${config.diarization_local_context||0.6}" min="0.1" max="2" step="0.1" style="max-width:80px;"></div>
        <div class="option-row"><label>换人置信度</label><input type="range" data-key="diarization_min_change_confidence" min="0.3" max="1" step="0.05" value="${config.diarization_min_change_confidence||0.7}"><span class="value-display">${config.diarization_min_change_confidence||0.7}</span></div>
        <div class="option-row"><label>聚类阈值</label><input type="range" data-key="diarization_distance_threshold" min="0.1" max="1.0" step="0.05" value="${config.diarization_distance_threshold||0.5}"><span class="value-display">${config.diarization_distance_threshold||0.5}</span></div>
        <div class="option-row"><label>最少说话人数</label><input type="number" data-key="diarization_min_speakers" value="${config.diarization_min_speakers||1}" min="1" max="10" style="max-width:60px;"></div>
        <div class="option-row"><label>最多说话人数</label><input type="number" data-key="diarization_max_speakers" value="${config.diarization_max_speakers||10}" min="1" max="20" style="max-width:60px;"></div>
        <div class="option-row"><label>启用角色标注(LLM)</label><input type="checkbox" data-key="speaker_role_enabled" ${config.speaker_role_enabled?'checked':''}></div>
        <div class="option-row-full"><label>场景提示 (可选)</label><input data-key="speaker_role_context_hint" value="${config.speaker_role_context_hint||''}" placeholder="例如: podcast interview, lecture, meeting"></div>
      `;

      // Speaker Embedding options
      App.ui.renderSpeakerEmbeddingOptions(config);
      App.ui.renderSpeakerModelOptions();

      // LLM options
      App.ui.renderLLMOptions(config);

      // Macro Chunking options (方案〇)
      App.ui.renderMacroChunkOptions(config);

      // Merge Decision options (方案五)
      App.ui.renderMergeDecisionOptions(config);

      // Boundary Refinement options (方案四)
      App.ui.renderBoundaryRefineOptions(config);

      // Acoustic Validation options (方案七)
      App.ui.renderAcousticValidationOptions(config);

      // Persistence options
      App.ui.renderPersistenceOptions(config);

      // Bind range input display updates
      $$('.option-row input[type=range]').forEach(el => {
        el.addEventListener('input', () => {
          const display = el.parentElement.querySelector('.value-display');
          if (display) display.textContent = el.value + (el.dataset.key.includes('duration') ? 's' : '');
        });
      });

      // 应用持久化设置（用户保存的覆盖值优先于模板默认值）
      var saved = App.loadSettings();
      for (var key in saved) {
        if (key === 'profile') continue;
        var el = document.querySelector('#options-panel [data-key="' + key + '"]');
        if (!el) continue;
        if (el.type === 'checkbox') {
          el.checked = saved[key];
        } else if (el.type === 'range') {
          el.value = saved[key];
          var display = el.parentElement.querySelector('.value-display');
          if (display) {
            var suffix = key.includes('duration') ? 's' : '';
            display.textContent = saved[key] + suffix;
          }
        } else {
          el.value = saved[key];
        }
      }
      App.ui.syncASREngineOptions(
        document.querySelector('#opts-asr [data-key="asr_engine"]')?.value || asrEngine,
        true,
      );
    },

    syncASREngineOptions(engine, prepare) {
      var modelSelect = document.querySelector('#opts-asr [data-key="asr_model"]');
      var status = document.getElementById('funasr-status');
      if (!modelSelect) return;
      var funasrModel = 'iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch';
      var current = modelSelect.value;
      if (engine === 'funasr') {
        if (['large-v3', 'large-v2', 'medium', 'small', 'tiny', ''].indexOf(current) >= 0) current = funasrModel;
        modelSelect.innerHTML = '<option value="' + App.ui._escapeAttr(current) + '" selected>FunASR Paraformer（中文）</option>';
        if (status) {
          status.style.display = 'inline';
          status.textContent = prepare ? '正在检查 FunASR...' : 'FunASR';
        }
        if (prepare) App.ui.prepareFunASR(current);
      } else {
        var selected = ['large-v3', 'medium', 'small', 'tiny'].indexOf(current) >= 0 ? current : 'large-v3';
        modelSelect.innerHTML =
          '<option value="large-v3" ' + (selected === 'large-v3' ? 'selected' : '') + '>large-v3</option>' +
          '<option value="medium" ' + (selected === 'medium' ? 'selected' : '') + '>medium</option>' +
          '<option value="small" ' + (selected === 'small' ? 'selected' : '') + '>small</option>' +
          '<option value="tiny" ' + (selected === 'tiny' ? 'selected' : '') + '>tiny</option>';
        if (status) {
          status.style.display = engine === 'auto' ? 'inline' : 'none';
          status.textContent = engine === 'auto'
            ? '自动检测：纯中文走 FunASR，其他情况走 faster-whisper'
            : '';
        }
      }
    },

    async prepareFunASR(model) {
      var status = document.getElementById('funasr-status');
      App.state.funasrPreparing = true;
      var setStatus = function(text, className) {
        if (status) { status.textContent = text; status.className = 'device-hint ' + (className || ''); }
      };
      try {
        var local = await App.api.getFunASRStatus(model);
        if (local.ready) {
          setStatus('✓ FunASR 本地模型已就绪', 'gpu');
          return local;
        }
        setStatus(local.package_installed ? '正在检查/下载本地模型...' : '正在安装 FunASR 依赖...');
        var ready = await App.api.prepareFunASR(model);
        setStatus('✓ FunASR 已就绪', 'gpu');
        return ready;
      } catch (ex) {
        var message = ex && ex.message ? ex.message : String(ex);
        try { message = JSON.parse(message).detail || message; } catch (_) {}
        setStatus('✗ FunASR 准备失败: ' + message, 'cpu');
        toast('FunASR 准备失败: ' + message, 'error');
        throw ex;
      } finally {
        App.state.funasrPreparing = false;
      }
    },

    // ---- LLM 配置面板 ----
    _llmState: { base_url: '', api_key: '', models: [] },

    async renderSpeakerModelOptions() {
      const target = $('#opts-speaker-models');
      if (!target) return;
      target.innerHTML = '<div class="option-row-full" style="color:var(--text-tertiary);">正在读取模型状态...</div>';
      try {
        const response = await fetch('/api/speaker-models');
        const payload = await response.json();
        const models = payload.models || [];
        target.innerHTML = models.map(model => `
          <div class="option-row-full" style="border-bottom:1px solid var(--border);padding:8px 0;">
            <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;">
              <span><strong>${model.name}</strong><br><small style="color:var(--text-tertiary);">${model.kind} · ${model.model_ref}</small></span>
              <span style="display:flex;gap:6px;flex-shrink:0;">
                <button class="btn-secondary" type="button" data-model-id="${model.model_id}" data-action="download" ${model.cached?'disabled':''}>${model.cached?'已缓存':'下载'}</button>
                <button class="btn-secondary" type="button" data-model-id="${model.model_id}" data-action="check">检查缓存</button>
              </span>
            </div>
            <div style="font-size:0.72rem;color:var(--text-tertiary);margin-top:4px;">${model.license}${model.requires_token?' · 需要 HF Token':''}</div>
          </div>`).join('');
        target.querySelectorAll('button[data-model-id]').forEach(button => {
          button.addEventListener('click', () => {
            if (button.dataset.action === 'check') {
              this.checkSpeakerModelCache(button.dataset.modelId, button);
            } else {
              this.downloadSpeakerModel(button.dataset.modelId, button);
            }
          });
        });
      } catch (error) {
        target.innerHTML = '<div class="option-row-full" style="color:var(--accent-red);">模型状态读取失败</div>';
      }
    },

    async downloadSpeakerModel(modelId, button) {
      const token = document.querySelector('[data-key="speaker_embedding_hf_token"]');
      const form = new FormData();
      if (token && token.value && token.value !== '***') form.append('token', token.value);
      button.disabled = true;
      button.textContent = '下载中...';
      try {
        const response = await fetch('/api/speaker-models/' + encodeURIComponent(modelId) + '/download', {method:'POST', body:form});
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || 'download failed');
        }
        toast('模型下载完成', 'success');
      } catch (error) {
        toast('模型下载失败：' + error.message, 'error');
      } finally {
        this.renderSpeakerModelOptions();
      }
    },

    async checkSpeakerModelCache(modelId, button) {
      const originalText = button.textContent;
      button.disabled = true;
      button.textContent = '检查中...';
      try {
        const response = await fetch('/api/speaker-models/' + encodeURIComponent(modelId) + '/status');
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || 'cache check failed');
        if (payload.cached) {
          toast('模型缓存完整，可直接使用', 'success');
        } else {
          toast('未检测到完整模型缓存，请下载模型', 'error');
        }
      } catch (error) {
        toast('缓存检查失败：' + error.message, 'error');
      } finally {
        button.disabled = false;
        button.textContent = originalText;
        this.renderSpeakerModelOptions();
      }
    },

    renderSpeakerEmbeddingOptions(config) {
      const enabled = config.speaker_embedding_enabled !== false;  // 默认 true
      const modelRef = config.speaker_embedding_model_ref || 'speechbrain/spkrec-ecapa-voxceleb';
      const hasToken = config.speaker_embedding_hf_token && config.speaker_embedding_hf_token !== '***';
      const tokenMasked = config.speaker_embedding_hf_token || '';
      const isPyannote = modelRef === 'pyannote/embedding';

      const modelOptions = [
        { value: 'speechbrain/spkrec-ecapa-voxceleb', label: 'speechbrain/ecapa (192维, Apache 2.0, 默认)' },
        { value: 'pyannote/embedding', label: 'pyannote/embedding (512维, 需签署协议)' },
      ];

      const modelSelectOptions = modelOptions
        .map(o => `<option value="${o.value}" ${modelRef===o.value?'selected':''}>${o.label}</option>`)
        .join('');

      $('#opts-speaker-embedding').innerHTML = `
        <div class="option-row">
          <label>启用嵌入模型</label>
          <input type="checkbox" data-key="speaker_embedding_enabled" ${enabled?'checked':''}>
        </div>
        <div class="option-row-full option-field-wrapper">
          <label>模型<span class="tip-icon" title="选择说话人嵌入模型">?</span></label>
          <select data-key="speaker_embedding_model_ref" id="speaker-embedding-model-select">${modelSelectOptions}</select>
          <div class="option-field-tip">
            <div class="tip-title">📦 模型选择</div>
            <p><strong>speechbrain/ecapa</strong> — 192维，Apache 2.0 协议。<br>无需签署额外协议，开箱即用。</p>
            <p style="margin-top:4px;"><strong>pyannote/embedding</strong> — 512维 ECAPA-TDNN，精度最高。<br>需先在 huggingface.co 接受模型使用协议。</p>
          </div>
        </div>
        <div class="option-row-full option-field-wrapper" id="speaker-hf-token-row">
          <label>HF Token（pyannote 模型）<span class="tip-icon" title="HuggingFace API Token">?</span></label>
          <input data-key="speaker_embedding_hf_token" name="speaker-embedding-token" type="text" class="credential-mask" value="${tokenMasked}" placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
                 autocomplete="off" spellcheck="false" autocapitalize="off" autocorrect="off" data-form-type="other" data-lpignore="true"
                 style="${hasToken?'border-color:var(--accent-green);':''}">
          <div class="option-field-tip">
            <div class="tip-title">🔑 如何获取 HF Token</div>
            <ol class="tip-steps">
              <li>打开 <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a></li>
              <li>登录或注册 HuggingFace 账号</li>
              <li>点击「Create new token」→ 选择 <code>Read</code> 类型</li>
              <li>复制生成的 token (格式: <code>hf_xxxx...</code>)</li>
              <li>粘贴到此处</li>
            </ol>
            <p style="margin-top:4px;">SpeechBrain ECAPA 不需要 Token；pyannote/embedding、Community-1、Diarization 3.1 下载时使用此 Token。Token 由后端加密存储，页面只保留脱敏标记。</p>
          </div>
        </div>
        <div class="option-row-full" style="font-size:0.75rem;color:var(--text-tertiary);padding:6px 0;" id="speaker-embedding-hint">
          ${isPyannote
            ? '⚠️ pyannote 模型需签署协议。详见 <a href="speaker-embedding-guide.html" target="_blank" style="color:var(--accent);text-decoration:underline;">📖 配置指南</a>'
            : (enabled ? '✅ 首次运行将自动下载模型 (~80MB)，请确保网络通畅' : '💡 启用嵌入模型可显著提升说话人分离精度')}
        </div>
      `;

      // 监听模型切换：保持共享 Token 输入框可见，只更新提示信息
      const selectEl = document.getElementById('speaker-embedding-model-select');
      if (selectEl) {
        selectEl.addEventListener('change', function() {
          const isPy = this.value === 'pyannote/embedding';
          const tokenRow = document.getElementById('speaker-hf-token-row');
          const hint = document.getElementById('speaker-embedding-hint');
          if (tokenRow) tokenRow.style.display = '';
          if (hint) {
            hint.innerHTML = isPy
              ? '⚠️ pyannote 模型需签署协议。详见 <a href="speaker-embedding-guide.html" target="_blank" style="color:var(--accent);text-decoration:underline;">📖 配置指南</a>'
              : '✅ SpeechBrain ECAPA 无需 Token；全局 pyannote 模型下载时会使用上方 Token';
          }
        });
      }
    },

    renderLLMOptions(config) {
      const llmEnabled = config.llm_enabled || false;
      const llmModel = config.llm_model || 'deepseek-v4-pro';
      const savedUrl = localStorage.getItem('vocal_llm_url') || 'https://api.deepseek.com';
      const savedKey = localStorage.getItem('vocal_llm_key') || '';
      const savedModel = localStorage.getItem('vocal_llm_model') || llmModel;

      App.ui._llmState.base_url = savedUrl;
      App.ui._llmState.api_key = savedKey;

      // Provider presets（与后端 LLM_PROVIDERS 同步，截至 2026-06）
      const providers = [
        { id: 'deepseek',   name: 'DeepSeek（深度求索）',   url: 'https://api.deepseek.com' },
        { id: 'openai',     name: 'OpenAI',                url: 'https://api.openai.com' },
        { id: 'anthropic',  name: 'Anthropic (Claude)',    url: 'https://api.anthropic.com' },
        { id: 'google',     name: 'Google (Gemini)',        url: 'https://generativelanguage.googleapis.com/v1beta/openai' },
        { id: 'zhipu',      name: '智谱 AI (GLM)',         url: 'https://open.bigmodel.cn/api/paas/v4' },
        { id: 'dashscope',  name: '阿里百炼 (Qwen)',        url: 'https://dashscope.aliyuncs.com/compatible-mode' },
        { id: 'hunyuan',    name: '腾讯混元 (Hunyuan)',      url: 'https://api.hunyuan.cloud.tencent.com/v1' },
        { id: 'moonshot',   name: '月之暗面 (Kimi)',         url: 'https://api.moonshot.cn' },
        { id: 'minimax',    name: 'MiniMax',               url: 'https://api.minimax.chat/v1' },
        { id: 'siliconflow',name: '硅基流动 (SiliconFlow)',  url: 'https://api.siliconflow.cn' },
        { id: 'ollama',     name: 'Ollama（本地）',          url: 'http://localhost:11434' },
      ];

      const activeProvider = providers.find(p => savedUrl.startsWith(p.url)) || providers[0];
      const hasKey = !!savedKey;
      const keyMasked = hasKey ? savedKey.slice(0,4) + '****' + savedKey.slice(-4) : '';

      // Build model select options from state, or use saved model as default
      const modelOptions = App.ui._llmState.models.length > 0
        ? App.ui._llmState.models.map(m => `<option value="${App.ui._escapeAttr(m.id)}" ${m.id === savedModel ? 'selected' : ''}>${App.ui._escapeHtml(m.id)}</option>`).join('')
        : `<option value="${App.ui._escapeAttr(savedModel)}" selected>${App.ui._escapeHtml(savedModel)}</option>`;

      let html = `
        <div class="option-row"><label>启用</label><input type="checkbox" data-key="llm_enabled" ${llmEnabled?'checked':''} onchange="App.ui._onLLMToggle(this)"></div>

        <div class="option-row-full"><label>模型供应商</label></div>
        <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;" id="llm-provider-list">
      `;

      providers.forEach((p, i) => {
        const active = p.id === activeProvider.id;
        const colors = ['#58a6ff','#3fb950','#a371f7','#db61a2','#d29922','#f85149','#79c0ff','#56d364','#bc8cff','#e0719c','#f0883e'];
        html += `
          <div class="llm-provider-card ${active?'active':''}" data-provider="${p.id}" data-url="${p.url}" onclick="App.ui._onProviderSelect(this)">
            <span class="pc-dot" style="background:${colors[i]}"></span>${p.name}
          </div>`;
      });

      html += `</div>

        <div class="option-row-full">
          <label>LLM API 地址</label>
          <div style="display:flex;align-items:center;gap:6px;">
            <input id="llm-base-url" placeholder="https://api.deepseek.com" oninput="App.ui._onURLChange(this)" value="${App.ui._escapeAttr(savedUrl)}">
            <span class="llm-status ${savedUrl?'ok':'pending'}" id="llm-url-status">${savedUrl?'✓':'未配置'}</span>
          </div>
          <div class="llm-hint">OpenAI 兼容接口地址，选择供应商后自动填充</div>
        </div>

        <div class="option-row-full" style="margin-top:8px;">
          <label>LLM API 密钥（Ollama 本地可留空）</label>
          <div style="display:flex;align-items:center;gap:6px;">
            <input id="llm-api-key" name="llm-api-key" type="text" class="credential-mask" autocomplete="off" spellcheck="false" autocapitalize="off" autocorrect="off" data-form-type="other" data-lpignore="true" placeholder="sk-..." oninput="App.ui._onKeyChange(this)" style="flex:1;" value="${App.ui._escapeAttr(savedKey)}">
            <span class="llm-status ${hasKey?'ok':'pending'}" id="llm-key-status">${hasKey?'✓ '+keyMasked:'未配置'}</span>
          </div>
          <div class="llm-hint">输入密钥后点击"获取模型"自动拉取可用模型列表</div>
        </div>

        <div style="margin-top:8px;">
          <button class="btn-fetch-models" id="btn-fetch-models" onclick="App.ui.handleFetchModels()">
            🔄 获取模型列表
          </button>
          <span id="fetch-status" style="font-size:0.7rem;color:var(--text-secondary);margin-left:8px;"></span>
        </div>

        <div class="option-row-full" style="margin-top:8px;">
          <label>LLM 模型</label>
          <div style="display:flex;align-items:center;gap:6px;">
            <select id="llm-model-select" data-key="llm_model" class="llm-model-select" onchange="App.ui._onModelSelectChange(this)" style="flex:1;">
              ${modelOptions}
            </select>
            <span class="llm-status ${savedModel?'ok':'pending'}" id="llm-model-status">${savedModel?'✓':'未选择'}</span>
          </div>
          <div class="llm-hint">点击"获取模型列表"后从下拉列表选择</div>
        </div>
      `;

      $('#opts-llm').innerHTML = html;

      // 显式同步密钥值，兼容浏览器自动填充，并修正初始状态徽章。
      const keyInput = $('#llm-api-key');
      if (keyInput) {
        keyInput.value = savedKey;
        // 绑定 change 事件作为 input 的补充（密码管理器自动填充可能不触发 input）
        keyInput.addEventListener('change', function() {
          App.ui._onKeyChange(this);
        });
        // 立即同步状态徽章（防御 innerHTML 中状态与 localStorage 不一致）
        App.ui._syncKeyStatus();
      }

      const urlInput = $('#llm-base-url');
      if (urlInput && savedUrl) {
        urlInput.value = savedUrl;
      }
    },

    _escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str || '';
      return div.innerHTML;
    },

    _escapeAttr(str) {
      return (str || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&#39;');
    },

    _populateModelList(models) {
      App.ui._llmState.models = models;
      const sel = $('#llm-model-select');
      if (!sel) return;
      const currentVal = sel.value;
      sel.innerHTML = models.map(m => `<option value="${App.ui._escapeAttr(m.id)}">${App.ui._escapeHtml(m.id)}</option>`).join('');
      // Restore previous selection if still in list
      if (models.some(m => m.id === currentVal)) {
        sel.value = currentVal;
      } else if (models.length > 0) {
        // Prefer chat model
        const chatModel = models.find(m => m.id.includes('chat')) || models[0];
        sel.value = chatModel.id;
      }
      // Trigger change to save
      App.ui._onModelSelectChange(sel);
    },

    _onLLMToggle(cb) {
      localStorage.setItem('vocal_llm_enabled', cb.checked ? '1' : '0');
      App.saveSettings();
    },

    _onProviderSelect(el) {
      const url = el.dataset.url;
      const urlEl = $('#llm-base-url');
      if (urlEl) urlEl.value = url;
      App.ui._llmState.base_url = url;
      localStorage.setItem('vocal_llm_url', url);
      App.saveSettings();
      const statusEl = $('#llm-url-status');
      if (statusEl) { statusEl.textContent = '✓'; statusEl.className = 'llm-status ok'; }

      // Update active state on all provider cards
      document.querySelectorAll('#llm-provider-list .llm-provider-card').forEach(c => c.classList.remove('active'));
      el.classList.add('active');

      // Clear model list when switching provider
      App.ui._llmState.models = [];
      const sel = $('#llm-model-select');
      const fetchStatus = $('#fetch-status');
      if (sel) sel.innerHTML = '<option value="">请先获取模型列表</option>';
      if (fetchStatus) { fetchStatus.textContent = ''; fetchStatus.style.color = ''; }
    },

    _onURLChange(input) {
      const url = input.value.trim();
      App.ui._llmState.base_url = url;
      localStorage.setItem('vocal_llm_url', url);
      App.saveSettings();
      const statusEl = $('#llm-url-status');
      if (statusEl) {
        statusEl.textContent = url ? '✓' : '未配置';
        statusEl.className = url ? 'llm-status ok' : 'llm-status pending';
      }
    },

    _onKeyChange(input) {
      const key = input.value.trim();
      App.ui._llmState.api_key = key;
      localStorage.setItem('vocal_llm_key', key);
      App.ui._syncKeyStatus();
      App.saveSettings();
    },

    // 同步密钥状态徽章（供 renderLLMOptions 后调用及 _onKeyChange 复用）
    _syncKeyStatus() {
      const key = localStorage.getItem('vocal_llm_key') || '';
      const statusEl = $('#llm-key-status');
      if (statusEl) {
        if (key) {
          statusEl.textContent = '✓ ' + key.slice(0,4) + '****' + key.slice(-4);
          statusEl.className = 'llm-status ok';
        } else {
          statusEl.textContent = '未配置';
          statusEl.className = 'llm-status pending';
        }
      }
    },

    _onModelSelectChange(sel) {
      const model = (sel && sel.value) ? sel.value.trim() : '';
      if (model) {
        localStorage.setItem('vocal_llm_model', model);
        const statusEl = $('#llm-model-status');
        if (statusEl) { statusEl.textContent = '✓'; statusEl.className = 'llm-status ok'; }
      }
      App.saveSettings();
    },

    // ---- 方案〇 宏观切块选项 ----
    renderMacroChunkOptions(config) {
      const mc = config.macro_chunking || {};
      const saved = App.loadSettings();
      const enabled = (saved.macro_chunk_enabled !== undefined) ? saved.macro_chunk_enabled : (mc.enabled !== false);
      const targetDur = saved.macro_chunk_target_duration || mc.target_chunk_duration || 60;
      const maxDur = saved.macro_chunk_max_duration || mc.max_chunk_duration || 180;
      $('#opts-macro-chunk').innerHTML =
        '<div class="option-row"><label>启用 (长音频自动切块)</label><input type="checkbox" data-key="macro_chunk_enabled" ' + (enabled ? 'checked' : '') + '></div>' +
        '<div class="option-help">音频 > 3 分钟自动在长静音处切分为独立大块，隔离误差、支持并行</div>' +
        '<div class="option-row"><label>目标块时长</label><input type="range" data-key="macro_chunk_target_duration" min="20" max="120" step="10" value="' + targetDur + '"><span class="value-display">' + targetDur + 's</span></div>' +
        '<div class="option-row"><label>最大块时长</label><input type="range" data-key="macro_chunk_max_duration" min="60" max="300" step="30" value="' + maxDur + '"><span class="value-display">' + maxDur + 's</span></div>';
    },

    // ---- 方案五 合并决策选项 ----
    renderMergeDecisionOptions(config) {
      const md = config.merge_decision || {};
      const saved = App.loadSettings();
      const fastGap = saved.merge_fast_gap !== undefined ? saved.merge_fast_gap : (md.fast_merge_max_gap || 0.20);
      const llmMin = saved.merge_llm_min_gap !== undefined ? saved.merge_llm_min_gap : (md.llm_decision_min_gap || 0.20);
      const llmMax = saved.merge_llm_max_gap !== undefined ? saved.merge_llm_max_gap : (md.llm_decision_max_gap || 1.20);
      const hardGap = saved.merge_hard_split_gap !== undefined ? saved.merge_hard_split_gap : (md.hard_split_min_gap || 1.20);
      $('#opts-merge-decision').innerHTML =
        '<div class="option-help">Fast-Slow Path 分流: 间隔 < 快合并阈值 → 规则合并, 在中间 → LLM 裁决, > 强制不合并</div>' +
        '<div class="option-row"><label>快路径合并阈值</label><input type="range" data-key="merge_fast_gap" min="0.05" max="0.5" step="0.05" value="' + fastGap + '"><span class="value-display">' + fastGap.toFixed(2) + 's</span></div>' +
        '<div class="option-row"><label>LLM 裁决下限</label><input type="range" data-key="merge_llm_min_gap" min="0.1" max="0.6" step="0.05" value="' + llmMin + '"><span class="value-display">' + llmMin.toFixed(2) + 's</span></div>' +
        '<div class="option-row"><label>LLM 裁决上限</label><input type="range" data-key="merge_llm_max_gap" min="0.6" max="2.0" step="0.1" value="' + llmMax + '"><span class="value-display">' + llmMax.toFixed(2) + 's</span></div>' +
        '<div class="option-row"><label>强制不合并阈值</label><input type="range" data-key="merge_hard_split_gap" min="0.8" max="3.0" step="0.1" value="' + hardGap + '"><span class="value-display">' + hardGap.toFixed(2) + 's</span></div>';
    },

    // ---- 方案四 边界精修选项 ----
    renderBoundaryRefineOptions(config) {
      const br = config.boundary_refinement || {};
      const saved = App.loadSettings();
      const enabled = (saved.boundary_refine_enabled !== undefined) ? saved.boundary_refine_enabled : (br.enabled !== false);
      const maxShrink = saved.boundary_refine_max_shrink || br.max_shrink_ms || 200;
      $('#opts-boundary-refine').innerHTML =
        '<div class="option-row"><label>启用 ASR 边界精修</label><input type="checkbox" data-key="boundary_refine_enabled" ' + (enabled ? 'checked' : '') + '></div>' +
        '<div class="option-help">用 ASR 词级时间戳回修语音段边界，三帧能量斜率校验保护辅音 Attack/Release</div>' +
        '<div class="option-row"><label>最大收缩量</label><input type="range" data-key="boundary_refine_max_shrink" min="50" max="500" step="50" value="' + maxShrink + '"><span class="value-display">' + maxShrink + 'ms</span></div>';
    },

    // ---- 方案七 声学校验选项 ----
    renderAcousticValidationOptions(config) {
      const av = config.acoustic_validation || {};
      const saved = App.loadSettings();
      const enabled = (saved.acoustic_enabled !== undefined) ? saved.acoustic_enabled : (av.enabled !== false);
      const maxSnap = saved.acoustic_max_snap || av.max_snap_distance || 0.15;
      const genReport = (saved.acoustic_generate_report !== undefined) ? saved.acoustic_generate_report : (av.generate_report !== false);
      const skeletonMode = (saved.acoustic_skeleton_mode !== undefined) ? saved.acoustic_skeleton_mode : (av.skeleton_mode !== false && config.acoustic_skeleton_mode !== false);
      $('#opts-acoustic-val').innerHTML =
        '<div class="option-row"><label>启用声学校验</label><input type="checkbox" data-key="acoustic_enabled" ' + (enabled ? 'checked' : '') + '></div>' +
        '<div class="option-help">ffmpeg 声学标尺兜底校验，消除字幕切尾（模式 A 错误），输出诊断报告</div>' +
        '<div class="option-row"><label>🧩 骨架分段模式</label><input type="checkbox" data-key="acoustic_skeleton_mode" ' + (skeletonMode ? 'checked' : '') + '></div>' +
        '<div class="option-help"><b>推荐开启。</b>按 ffmpeg 声学骨架逐段独立处理再拼接，消除跨段时间戳漂移。每个骨架段是物理隔离的连续语音，段间不会互相干扰。健康度可从 ~77% 提升到 ~100%。</div>' +
        '<div class="option-row"><label>最大吸附距离</label><input type="range" data-key="acoustic_max_snap" min="0.05" max="0.3" step="0.01" value="' + maxSnap + '"><span class="value-display">' + maxSnap.toFixed(2) + 's</span></div>' +
        '<div class="option-row"><label>输出诊断报告</label><input type="checkbox" data-key="acoustic_generate_report" ' + (genReport ? 'checked' : '') + '></div>';
    },

    // 持久化文件设置
    renderPersistenceOptions(config) {
      var saved = App.loadSettings();
      var types = [
        {key: 'persist_asr_subtitle', label: 'ASR 字幕 (.srt)', def: true},
        {key: 'persist_llm_subtitle', label: 'LLM 优化字幕 (.srt)', def: true},
        {key: 'persist_final_srt', label: '最终版 SRT 字幕', def: true},
        {key: 'persist_final_ass', label: '最终版 ASS 字幕', def: false},
        {key: 'persist_vocals', label: '人声音频 (.wav)', def: true},
        {key: 'persist_accompaniment', label: '背景声音频 (.wav)', def: false},
      ];
      var html = '<div class="option-help" style="margin-bottom:6px;">选择处理完成后自动保留的文件类型，过期后自动清理</div>';
      for (var i = 0; i < types.length; i++) {
        var t = types[i];
        var checked = (saved[t.key] !== undefined) ? saved[t.key] : t.def;
        html += '<div class="option-row"><label>' + t.label + '</label><input type="checkbox" data-key="' + t.key + '" ' + (checked ? 'checked' : '') + '></div>';
      }
      var ttlSub = (saved.ttl_subtitle_days !== undefined) ? saved.ttl_subtitle_days : 90;
      var ttlAud = (saved.ttl_audio_days !== undefined) ? saved.ttl_audio_days : 30;
      html += '<div class="option-row"><label>字幕保留天数</label><input type="number" data-key="ttl_subtitle_days" value="' + ttlSub + '" min="1" max="365" style="max-width:60px;"></div>';
      html += '<div class="option-row"><label>音频保留天数</label><input type="number" data-key="ttl_audio_days" value="' + ttlAud + '" min="1" max="90" style="max-width:60px;"></div>';
      html += '<div class="option-help">处理完成后自动持久化选中文件。字幕较小默认保留90天，音频较大默认保留30天。</div>';
      $('#opts-persistence').innerHTML = html;
    },

    async handleFetchModels() {
      // Read from DOM directly (most current), fall back to state
      const urlEl = $('#llm-base-url');
      const keyEl = $('#llm-api-key');
      if (!urlEl) { toast('API 地址输入框未找到', 'error'); return; }
      const baseUrl = urlEl.value.trim() || App.ui._llmState.base_url;
      const apiKey = (keyEl && keyEl.value.trim()) || App.ui._llmState.api_key || '';

      if (!baseUrl) {
        toast('请先配置 API 地址', 'error');
        return;
      }

      // Sync state
      App.ui._llmState.base_url = baseUrl;
      localStorage.setItem('vocal_llm_url', baseUrl);
      if (apiKey) { App.ui._llmState.api_key = apiKey; localStorage.setItem('vocal_llm_key', apiKey); }

      const btn = $('#btn-fetch-models');
      const status = $('#fetch-status');

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-sm"></span>获取中...';
      if (status) { status.textContent = ''; status.style.color = ''; }

      try {
        const resp = await fetch('/api/llm/models', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
        });

        if (!resp.ok) {
          let errMsg = 'HTTP ' + resp.status;
          try { const err = await resp.json(); errMsg = err.detail || errMsg; } catch(e) {}
          throw new Error(errMsg);
        }

        const data = await resp.json();
        const models = data.models || [];
        const source = data.source || 'unknown';

        if (models.length === 0) {
          toast('未获取到模型列表，请检查 API 地址和密钥是否正确', 'error');
          if (status) { status.textContent = '✗ 无模型返回'; status.style.color = 'var(--accent-red)'; }
          return;
        }

        App.ui._populateModelList(models);
        const sourceLabel = source === 'preset' ? '(预设) ' : '';
        if (status) { status.textContent = '✓ ' + sourceLabel + '获取到 ' + models.length + ' 个模型'; status.style.color = 'var(--accent-green)'; }

        toast('成功获取 ' + models.length + ' 个可用模型' + (source === 'preset' ? '（预设）' : ''), 'success');
      } catch (ex) {
        toast('获取模型失败: ' + ex.message, 'error');
        if (status) { status.textContent = '✗ ' + ex.message; status.style.color = 'var(--accent-red)'; }
      } finally {
        btn.disabled = false;
        btn.innerHTML = '🔄 获取模型列表';
      }
    },

    // Device Info
    async initDeviceInfo() {
      try {
        const info = await App.api.getDeviceInfo();
        const dot = $('#device-dot');
        const label = $('#device-label');

        if (info.device_type === 'cuda') {
          dot.className = 'dot ok';
          const name = (info.device_names && info.device_names[0]) ? info.device_names[0].split(' ').slice(0,2).join(' ') : 'GPU';
          const mem = (info.memory_mb && info.memory_mb[0]) ? (info.memory_mb[0]/1024).toFixed(0) + 'GB' : '';
          label.textContent = name + ' ' + mem + ' ✓';
        } else if (info.device_type === 'mps') {
          dot.className = 'dot ok';
          label.textContent = 'Apple Silicon ✓';
        } else {
          dot.className = 'dot warn';
          label.textContent = 'CPU 模式';
        }
      } catch (ex) {
        $('#device-label').textContent = '设备检测失败';
      }
      // 刷新设备提示（如果 ASR 设备设置为 "auto"）
      App.ui.updateDeviceHint();
    },

    // 设备下拉框变更处理
};
window.VocalSubtitleUi = Object.assign(window.VocalSubtitleUi || {}, part);
})();
