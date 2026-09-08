// 编辑草稿：按字幕文件名（退化为媒体名）自动保存到 localStorage，静默降级。
const PREFIX = 'vstEditor.draft.';
const SAVE_DELAY = 800;
// 草稿结构版本：字段布局变化时递增，load 遇到旧版本草稿直接忽略
const SCHEMA_VERSION = 1;

export function createDraftStore(store) {
  let timer = null;

  function saveNow() {
    clearTimeout(timer);
    timer = null;
    const s = store.state;
    if (!s.dirty) return true;
    try {
      const key = draftKey(s.subtitleName || s.mediaName);
      if (!key) return true;
      localStorage.setItem(
        PREFIX + key,
        JSON.stringify({
          version: SCHEMA_VERSION,
          cues: s.cues,
          doc: s.subDoc,
          format: s.subtitleFormat,
          mediaName: s.mediaName,
          savedAt: Date.now(),
        }),
      );
      store.patch({ savedAt: Date.now() });
      return true;
    } catch (err) {
      // 配额不足或不可用：不能静默，控制台告警 + 状态条显示失败（保存状态仍以 savedAt 为准）
      console.warn('[draft] 草稿保存失败（存储配额不足或不可用？）:', err);
      store.patch({ draftSaveFailedAt: Date.now() });
      return false;
    }
  }

  store.on('cues', () => {
    if (!store.state.dirty) return;
    clearTimeout(timer);
    timer = setTimeout(saveNow, SAVE_DELAY);
  });

  return {
    saveNow,
    lookup(subtitleName, mediaName) {
      for (const name of [subtitleName, mediaName]) {
        const key = draftKey(name);
        if (!key) continue;
        try {
          const raw = localStorage.getItem(PREFIX + key);
          if (!raw) continue;
          const data = JSON.parse(raw);
          // 版本不符（旧结构）的草稿直接忽略
          if (data?.version !== SCHEMA_VERSION) continue;
          return data;
        } catch {
          return null;
        }
      }
      return null;
    },
    // 用户「放弃草稿」或导出成功后删除草稿，避免反复弹恢复确认
    discard(subtitleName, mediaName) {
      for (const name of [subtitleName, mediaName]) {
        const key = draftKey(name);
        if (!key) continue;
        try {
          localStorage.removeItem(PREFIX + key);
        } catch {
          // localStorage 不可用：忽略
        }
      }
    },
  };
}

// 草稿键用完整文件名（含扩展名），避免同名不同格式的字幕互相命中
function draftKey(filename) {
  const name = String(filename ?? '').trim().toLowerCase();
  return name || '';
}
