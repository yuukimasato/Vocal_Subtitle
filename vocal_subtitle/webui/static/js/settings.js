(function() {
  'use strict';

  const SETTINGS_KEYS = [
    'profile', 'separator', 'uvr_model', 'vad_engine', 'vad_threshold',
    'vad_ffmpeg_enabled', 'vad_ffmpeg_noise_db', 'asr_engine', 'asr_model',
    'asr_device', 'language', 'subtitle_min_duration', 'subtitle_max_duration',
    'diarization_enabled', 'diarization_distance_threshold',
    'diarization_min_speakers', 'diarization_max_speakers', 'speaker_fusion',
    'global_diarization_model', 'speaker_diarization_scope',
    'local_speaker_refinement', 'expected_speakers', 'diarization_local_context',
    'diarization_min_local_segment', 'diarization_min_change_confidence',
    'speaker_role_enabled', 'speaker_role_context_hint',
    'speaker_embedding_enabled', 'speaker_embedding_model_ref', 'llm_enabled',
    'llm_model', 'llm_base_url', 'llm_api_key', 'macro_chunk_enabled',
    'macro_chunk_target_duration', 'macro_chunk_max_duration', 'merge_fast_gap',
    'merge_llm_min_gap', 'merge_llm_max_gap', 'merge_hard_split_gap',
    'boundary_refine_enabled', 'boundary_refine_max_shrink', 'acoustic_enabled',
    'acoustic_max_snap', 'acoustic_generate_report', 'acoustic_skeleton_mode',
    'persist_asr_subtitle', 'persist_llm_subtitle', 'persist_final_ass',
    'persist_final_srt', 'persist_vocals', 'persist_accompaniment',
    'ttl_subtitle_days', 'ttl_audio_days'
  ];

  const KEY_MAP = {
    separator: 'separator', uvr_model: 'uvr_model', vad_engine: 'vad_engine',
    vad_threshold: 'vad_threshold', vad_ffmpeg_enabled: 'ffmpeg_enabled',
    vad_ffmpeg_noise_db: 'ffmpeg_noise_db', asr_engine: 'asr_engine',
    asr_model: 'asr_model', language: 'language', device: 'device',
    diarization_enabled: 'diarization', speaker_fusion: 'speaker_fusion',
    global_diarization_model: 'global_diarization_model',
    speaker_diarization_scope: 'speaker_diarization_scope',
    local_speaker_refinement: 'local_speaker_refinement', expected_speakers: 'expected_speakers',
    diarization_local_context: 'diarization_local_context',
    diarization_min_local_segment: 'diarization_min_local_segment',
    diarization_min_change_confidence: 'diarization_min_change_confidence',
    diarization_distance_threshold: 'diarization_distance_threshold',
    diarization_min_speakers: 'diarization_min_speakers',
    diarization_max_speakers: 'diarization_max_speakers',
    speaker_role_enabled: 'speaker_role', speaker_role_context_hint: 'speaker_role_context_hint',
    speaker_embedding_enabled: 'speaker_embedding',
    speaker_embedding_model_ref: 'speaker_embedding_model', llm_enabled: 'llm_optimize',
    llm_model: 'llm_model', macro_chunk_enabled: 'macro_chunking.enabled',
    macro_chunk_target_duration: 'macro_chunking.target_chunk_duration',
    macro_chunk_max_duration: 'macro_chunking.max_chunk_duration',
    merge_fast_gap: 'merge_decision.fast_merge_max_gap',
    merge_llm_min_gap: 'merge_decision.llm_decision_min_gap',
    merge_llm_max_gap: 'merge_decision.llm_decision_max_gap',
    merge_hard_split_gap: 'merge_decision.hard_split_min_gap',
    boundary_refine_enabled: 'boundary_refinement.enabled',
    boundary_refine_max_shrink: 'boundary_refinement.max_shrink_ms',
    acoustic_enabled: 'acoustic_validation.enabled',
    acoustic_max_snap: 'acoustic_validation.max_snap_distance',
    acoustic_generate_report: 'acoustic_validation.generate_report',
    acoustic_skeleton_mode: 'skeleton_mode'
  };

  function valueOf(element) {
    if (element.type === 'checkbox') return element.checked;
    if (element.type === 'range' || element.type === 'number') {
      return element.value === '' ? '' : parseFloat(element.value);
    }
    return element.value;
  }

  window.VocalSubtitleSettings = {
    save: function(app) {
      const settings = {};
      SETTINGS_KEYS.forEach(function(key) {
        const element = document.querySelector('#options-panel [data-key="' + key + '"]');
        if (element) settings[key] = valueOf(element);
        else if (key === 'profile') settings[key] = app.state.selectedProfile;
        else if (key === 'llm_base_url') settings[key] = app.ui._llmState.base_url;
        else if (key === 'llm_api_key') settings[key] = app.ui._llmState.api_key;
        else if (key === 'llm_model') {
          const select = document.querySelector('#llm-model-select');
          settings[key] = select ? select.value : '';
        }
      });
      settings._skip_separation = app.state.skipSeparation;
      try { localStorage.setItem('vocal_settings', JSON.stringify(settings)); } catch (_) {}
    },

    load: function() {
      let saved = null;
      try {
        const raw = localStorage.getItem('vocal_settings');
        if (raw) saved = JSON.parse(raw);
      } catch (_) { saved = null; }
      if (!saved) {
        saved = {};
        const migrations = {
          llm_base_url: 'vocal_llm_url', llm_api_key: 'vocal_llm_key',
          llm_model: 'vocal_llm_model'
        };
        Object.keys(migrations).forEach(function(key) {
          const value = localStorage.getItem(migrations[key]);
          if (value) saved[key] = value;
        });
        const enabled = localStorage.getItem('vocal_llm_enabled');
        if (enabled) saved.llm_enabled = enabled === '1';
        if (Object.keys(saved).length) {
          try {
            localStorage.setItem('vocal_settings', JSON.stringify(saved));
            Object.keys(migrations).forEach(function(key) { localStorage.removeItem(migrations[key]); });
            localStorage.removeItem('vocal_llm_enabled');
          } catch (_) {}
        }
      }
      return saved || {};
    },

    collectOverrides: function() {
      const overrides = {};
      document.querySelectorAll('#options-panel [data-key]').forEach(function(element) {
        const value = valueOf(element);
        if (value !== '' && value !== null && value !== undefined) {
          overrides[KEY_MAP[element.dataset.key] || element.dataset.key] = value;
        }
      });
      const url = localStorage.getItem('vocal_llm_url');
      const key = localStorage.getItem('vocal_llm_key');
      if (url) overrides.llm_base_url = url;
      if (key) overrides.llm_api_key = key;
      return overrides;
    }
  };
})();
