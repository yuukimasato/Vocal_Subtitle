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
        historyItems: [],
        funasrPreparing: false,
        ws: null
      };
    }
  };
})();
