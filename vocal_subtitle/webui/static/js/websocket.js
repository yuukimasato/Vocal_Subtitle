(function() {
'use strict';
window.VocalSubtitleWs = {
connect(taskId) {
      if (App.state.ws) {
        App.state.ws._intentionalClose = true;
        App.state.ws.close();
      }
      App.ws.stopTaskPolling();
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(protocol + '//' + location.host + '/ws/tasks/' + taskId);
      App.state.ws = ws;

      ws.onmessage = function(e) {
        try {
          const msg = JSON.parse(e.data);
          App.ws.handleMessage(msg);
        } catch(ex) { console.error('WS parse error:', ex); }
      };

      ws.onclose = function() {
        if (!ws._intentionalClose && App.state.isRunning) {
          // The task runs in a server thread. Recover the terminal result via
          // REST when the progress socket disappears instead of losing a
          // completed task or showing a false failure immediately.
          App.ws.startTaskPolling(taskId);
        }
      };

      ws.onerror = function() {
        console.error('WebSocket error');
      };

      // Heartbeat
      App.state._wsHeartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: 'ping'}));
      }, 25000);
    },

    disconnect() {
      if (App.state._wsHeartbeat) { clearInterval(App.state._wsHeartbeat); }
      App.ws.stopTaskPolling();
      if (App.state.ws) {
        App.state.ws._intentionalClose = true;
        App.state.ws.close();
        App.state.ws = null;
      }
    },

    startTaskPolling(taskId) {
      App.ws.stopTaskPolling();
      let attempts = 0;
      App.state._taskPollTimer = setInterval(async function() {
        if (!App.state.isRunning || App.state.taskId !== taskId) {
          App.ws.stopTaskPolling();
          return;
        }
        attempts += 1;
        try {
          const status = await App.api.getTaskStatus(taskId);
          if (status.status === 'completed' && status.result) {
            App.ws.stopTaskPolling();
            App.ui.pipelineComplete(status.result);
          } else if (status.status === 'failed') {
            App.ws.stopTaskPolling();
            App.ui.pipelineError(status.error || '处理失败');
          }
        } catch (ex) {
          // Keep polling briefly through transient REST failures. After the
          // timeout, expose the connection issue rather than hanging forever.
          if (attempts >= 15) {
            App.ws.stopTaskPolling();
            App.ui.pipelineError('实时连接已断开，且无法获取任务状态');
          }
        }
      }, 1000);
    },

    stopTaskPolling() {
      if (App.state._taskPollTimer) {
        clearInterval(App.state._taskPollTimer);
        App.state._taskPollTimer = null;
      }
    },

    handleMessage(msg) {
      switch (msg.type) {
        case 'stage_start':
          App.ui.stageStart(msg.stage, msg.total, msg.description);
          break;
        case 'progress':
          App.ui.stageProgress(msg.stage, msg.current, msg.total, msg.extra);
          break;
        case 'stage_finish':
          App.ui.stageFinish(msg.stage, msg.elapsed_seconds);
          break;
        case 'complete':
          App.ui.pipelineComplete(msg.result);
          break;
        case 'error':
          App.ui.pipelineError(msg.message);
          break;
      }
    }
};
})();

