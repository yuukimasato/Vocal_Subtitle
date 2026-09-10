(function() {
  'use strict';

  // 双界面收敛（2026-09-10）：仅保留 处理 / 反馈档案 两个工作区；
  // 历史与质量内联进处理工作区，审核与编辑走 subtitle-editor（8631）。
  const allowed = new Set(['process', 'feedback']);
  const Workspace = {
    current: 'process',
    initialized: false,

    init() {
      if (this.initialized) return;
      this.initialized = true;
      document.querySelectorAll('[data-workspace]').forEach((button) => {
        button.addEventListener('click', () => this.switchTo(button.dataset.workspace));
      });
      window.addEventListener('hashchange', () => this.switchTo(location.hash.slice(1), false));
      this.switchTo(location.hash.slice(1) || 'process', false);
    },

    switchTo(name, updateHash = true) {
      name = allowed.has(name) ? name : 'process';
      this.current = name;
      if (updateHash && location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name);
      document.body.classList.toggle('workspace-focused', name !== 'process');
      const processWorkspace = document.getElementById('process-workspace');
      if (processWorkspace) processWorkspace.hidden = name !== 'process';
      const progress = document.getElementById('progress-panel');
      const results = document.getElementById('results-area');
      if (progress) progress.hidden = name !== 'process';
      if (results) results.hidden = name !== 'process';
      document.querySelectorAll('.workspace-view').forEach((view) => {
        view.hidden = view.id !== 'workspace-' + name;
      });
      document.querySelectorAll('[data-workspace]').forEach((button) => {
        const active = button.dataset.workspace === name;
        button.classList.toggle('active', active);
        button.setAttribute('aria-current', active ? 'page' : 'false');
      });
      if (name === 'feedback' && window.FeedbackWorkspace) FeedbackWorkspace.refresh();
    },

    setTask(taskId) {
      if (!taskId) return;
      if (window.App && App.state) App.state.taskId = taskId;
      document.dispatchEvent(new CustomEvent('workspace:task-change', { detail: { taskId } }));
    },

    taskId() {
      return window.App && App.state ? App.state.taskId : null;
    }
  };

  window.VocalSubtitleWorkspace = Workspace;
  window.Workspace = Workspace;
  document.addEventListener('DOMContentLoaded', () => Workspace.init());
})();
