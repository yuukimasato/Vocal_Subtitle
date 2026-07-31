(function() {
  'use strict';

  window.VocalSubtitleState = {
    create: function() {
      return {
        selectedFile: null,
        selectedProfile: 'default',
        profileConfig: null,
        outputFormat: 'srt',
        skipSeparation: false,
        taskId: null,
        isRunning: false,
        subtitleEvents: [],
        historyItems: [],
        subtitleViewMode: 'auto',
        subtitleFinalFormat: 'srt',
        funasrPreparing: false,
        selectedSubtitleIndexes: new Set(),
        selectionAnchorIndex: null,
        rowClickTimer: null,
        ws: null,
        _audioPlayer: null,
        _audioTaskId: null
      };
    }
  };
})();
