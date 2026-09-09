// 草稿命名空间与自动保存开关：人机同屏时 agent 会话不得覆盖人类草稿（docs/开发文档 §4.5）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { createStore } from '../js/state.js';
import { createDraftStore } from '../js/draft.js';

// 极简 localStorage 桩（draft.js 直接读全局 localStorage）
function installStorage() {
  const map = new Map();
  globalThis.localStorage = {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  };
  return map;
}

function loadedStore(name = 'x.srt') {
  const store = createStore();
  store.patch({ subtitleName: name, mediaName: 'm.mkv', dirty: true });
  return store;
}

test('默认（人类）会话：键无命名空间，自动保存', async () => {
  const map = installStorage();
  const store = loadedStore();
  const draft = createDraftStore(store);
  draft.saveNow();
  assert.ok(map.has('vstEditor.draft.x.srt'));
  assert.equal(draft.lookup('x.srt', '').version, 1);
});

test('agent 会话：独立命名空间 + 默认不写草稿', async () => {
  const map = installStorage();
  const human = loadedStore('x.srt');
  const humanDraft = createDraftStore(human);
  humanDraft.saveNow(); // 人未导出的工作现场先落盘

  const agentStore = loadedStore('x.srt');
  agentStore.patch({ cues: [{ id: 'z', start: 9, end: 10, text: 'agent 改动' }] });
  const agentDraft = createDraftStore(agentStore, { namespace: 'agent', autoSave: false });

  // 模拟 agent 会话中 cues 变更触发的防抖自动保存：不写任何键
  agentStore.emit('cues');
  await new Promise((r) => setTimeout(r, 900)); // 越过 800ms 防抖窗口
  assert.equal(map.has('vstEditor.draft.agent:x.srt'), false);

  // 显式 saveNow(true) 才写，且写进 agent 命名空间
  agentDraft.saveNow(true);
  assert.ok(map.has('vstEditor.draft.agent:x.srt'));

  // 互不覆盖：人类键内容仍是人类的，agent lookup 也只看自己的命名空间
  const humanRaw = JSON.parse(map.get('vstEditor.draft.x.srt'));
  assert.equal(humanRaw.cues.length, 0); // 人类 store 的 cues 未被 agent 污染
  assert.equal(agentDraft.lookup('x.srt', '').cues[0].text, 'agent 改动');
  assert.equal(humanDraft.lookup('x.srt', '').cues.length, 0);
});

test('agent 会话：saveNow() 不带 force 时不写（beforeunload 路径）', () => {
  const map = installStorage();
  const store = loadedStore();
  const draft = createDraftStore(store, { namespace: 'agent', autoSave: false });
  draft.saveNow();
  assert.equal(map.has('vstEditor.draft.agent:x.srt'), false);
});
