// ---- Application State ----
const App = {
  state: {
    selectedFile: null,
    selectedProfile: 'default',
    profileConfig: null,
    outputFormat: 'srt',
    skipSeparation: false,
    taskId: null,
    isRunning: false,
    subtitleEvents: [],
    historyItems: [],
    subtitleViewMode: 'auto',  // 'auto' | 'compare' | 'simple'
    subtitleFinalFormat: 'srt',  // 'srt' | 'ass' — 最终版本字幕格式
    funasrPreparing: false,
    selectedSubtitleIndexes: new Set(),
    selectionAnchorIndex: null,
    rowClickTimer: null,
    ws: null,
    _audioPlayer: null,          // 当前音频播放器实例
    _audioTaskId: null,          // 音频播放器加载的任务 ID
  },

  // ---- Settings Persistence ----
  _SETTINGS_KEYS: [
    'profile', 'separator', 'uvr_model', 'vad_engine', 'vad_threshold',
    'vad_ffmpeg_enabled', 'vad_ffmpeg_noise_db',
    'asr_engine', 'asr_model', 'asr_device', 'language',
    'subtitle_min_duration', 'subtitle_max_duration',
    'diarization_enabled', 'diarization_distance_threshold', 'diarization_min_speakers', 'diarization_max_speakers',
    'speaker_fusion', 'global_diarization_model', 'speaker_diarization_scope',
    'local_speaker_refinement', 'expected_speakers', 'diarization_local_context',
    'diarization_min_local_segment', 'diarization_min_change_confidence',
    'speaker_role_enabled', 'speaker_role_context_hint',
    'speaker_embedding_enabled', 'speaker_embedding_model_ref',
    'llm_enabled', 'llm_model', 'llm_base_url', 'llm_api_key',
    'macro_chunk_enabled', 'macro_chunk_target_duration', 'macro_chunk_max_duration',
    'merge_fast_gap', 'merge_llm_min_gap', 'merge_llm_max_gap', 'merge_hard_split_gap',
    'boundary_refine_enabled', 'boundary_refine_max_shrink',
    'acoustic_enabled', 'acoustic_max_snap', 'acoustic_generate_report',
    'acoustic_skeleton_mode',
    // Persistence settings
    'persist_asr_subtitle', 'persist_llm_subtitle', 'persist_final_ass',
    'persist_final_srt', 'persist_vocals', 'persist_accompaniment',
    'ttl_subtitle_days', 'ttl_audio_days',
  ],
  _SETTINGS_STORAGE_KEY: 'vocal_settings',

  saveSettings() {
    var self = this;
    var settings = {};
    self._SETTINGS_KEYS.forEach(function(key) {
      var el = document.querySelector('#options-panel [data-key="' + key + '"]');
      if (!el) {
        if (key === 'profile') {
          settings[key] = App.state.selectedProfile;
        } else if (key === 'llm_base_url') {
          settings[key] = App.ui._llmState.base_url;
        } else if (key === 'llm_api_key') {
          settings[key] = App.ui._llmState.api_key;
        } else if (key === 'llm_enabled') {
          var cb = document.querySelector('#options-panel [data-key="llm_enabled"]');
          settings[key] = cb ? cb.checked : false;
        } else if (key === 'llm_model') {
          var sel = document.querySelector('#llm-model-select');
          settings[key] = sel ? sel.value : '';
        }
        return;
      }
      if (el.type === 'checkbox') {
        settings[key] = el.checked;
      } else if (el.type === 'range') {
        settings[key] = parseFloat(el.value);
      } else {
        settings[key] = el.value;
      }
    });
    try { localStorage.setItem(self._SETTINGS_STORAGE_KEY, JSON.stringify(settings)); } catch(e) {}
    // Save skip-separation toggle state
    try {
      settings['_skip_separation'] = App.state.skipSeparation;
      localStorage.setItem(self._SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch(e) {}
  },

  loadSettings() {
    var self = this;
    var saved = null;
    try {
      var raw = localStorage.getItem(self._SETTINGS_STORAGE_KEY);
      if (raw) saved = JSON.parse(raw);
    } catch(e) { saved = null; }

    // 从旧版独立键迁移
    if (!saved) {
      saved = {};
      var oldUrl = localStorage.getItem('vocal_llm_url');
      var oldKey = localStorage.getItem('vocal_llm_key');
      var oldModel = localStorage.getItem('vocal_llm_model');
      var oldEnabled = localStorage.getItem('vocal_llm_enabled');
      if (oldUrl) saved['llm_base_url'] = oldUrl;
      if (oldKey) saved['llm_api_key'] = oldKey;
      if (oldModel) saved['llm_model'] = oldModel;
      if (oldEnabled) saved['llm_enabled'] = (oldEnabled === '1');
      if (Object.keys(saved).length > 0) {
        try {
          localStorage.setItem(self._SETTINGS_STORAGE_KEY, JSON.stringify(saved));
          localStorage.removeItem('vocal_llm_url');
          localStorage.removeItem('vocal_llm_key');
          localStorage.removeItem('vocal_llm_model');
          localStorage.removeItem('vocal_llm_enabled');
        } catch(e) {}
      }
    }
    return saved || {};
  },
