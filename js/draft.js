// 编辑草稿：按字幕文件名（退化为媒体名）自动保存到 localStorage，静默降级。
const PREFIX = 'vstEditor.draft.';

export function createDraftStore(store) {
  let timer = null;

  store.on('cues', () => {
    const s = store.state;
    if (!s.dirty) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      try {
        const key = draftKey(s.subtitleName || s.mediaName);
        if (!key) return;
        localStorage.setItem(
          PREFIX + key,
          JSON.stringify({
            cues: s.cues,
            doc: s.subDoc,
            format: s.subtitleFormat,
            mediaName: s.mediaName,
            savedAt: Date.now(),
          }),
        );
      } catch {
        // 配额不足或不可用：草稿静默降级
      }
    }, 800);
  });

  return {
    lookup(subtitleName, mediaName) {
      for (const name of [subtitleName, mediaName]) {
        const key = draftKey(name);
        if (!key) continue;
        try {
          const raw = localStorage.getItem(PREFIX + key);
          if (raw) return JSON.parse(raw);
        } catch {
          return null;
        }
      }
      return null;
    },
  };
}

// 草稿键用完整文件名（含扩展名），避免同名不同格式的字幕互相命中
function draftKey(filename) {
  const name = String(filename ?? '').trim().toLowerCase();
  return name || '';
}
