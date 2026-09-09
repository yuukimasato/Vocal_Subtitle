// window.agent 契约测试：字段路径冻结（baseline JSON）+ 行为单测（假 deps，无 DOM）。
// baseline 变更流程：有意改动契约时同步更新 .smoke/agent-contract.baseline.json 并在说明里写明原因。
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createStore } from '../js/state.js';
import { createActions } from '../js/actions.js';
import { createAgentApi, AGENT_API_VERSION } from '../js/agent-api.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// ---------- 假装配（store/actions 用真件，waveform/player 用假件） ----------

function fakeWaveform({ rate = 4000, seconds = 1 } = {}) {
  const samples = new Float32Array(rate * seconds);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin((i / rate) * 440 * Math.PI * 2) * 0.5;
  return {
    getPeaksData(t0 = 0, t1 = null) {
      const start = Math.max(0, t0);
      const end = Math.min(t1 ?? seconds, seconds);
      const si = Math.floor(start * rate);
      const ei = Math.max(si, Math.floor(end * rate));
      return { rate, duration: seconds, start, end, samples: samples.slice(si, ei) };
    },
  };
}

function setup() {
  const store = createStore();
  const actions = createActions(store);
  actions.loadSubtitle(
    [
      { id: 'a', start: 1, end: 3, text: '第一句' },
      { id: 'b', start: 4, end: 6, text: '第二句' },
    ],
    { name: 'x.srt', format: 'srt', doc: null },
  );
  const deps = {
    store,
    actions,
    waveform: fakeWaveform(),
    player: { getCurrentTime: () => 2.5 },
    loadMedia: async () => 'media',
    loadSubs: async () => 'subs',
    saveDraft: () => 'saved',
    canvasFactory: (w, h) => ({
      width: w,
      height: h,
      getContext: () => ({
        fillStyle: '',
        strokeStyle: '',
        font: '',
        fillRect() {},
        beginPath() {},
        moveTo() {},
        lineTo() {},
        stroke() {},
        fillText() {},
      }),
      toDataURL: () => 'data:image/png;base64,stub',
    }),
  };
  return { store, actions, deps, api: createAgentApi(deps) };
}

// ---------- 字段路径冻结 ----------

function walkPaths(value, prefix = '', out = []) {
  if (Array.isArray(value)) {
    out.push(prefix + '[]');
    value.forEach((v) => walkPaths(v, `${prefix}[].`, out));
    return out;
  }
  if (value && typeof value === 'object') {
    out.push(prefix);
    for (const [k, v] of Object.entries(value)) walkPaths(v, prefix ? `${prefix}.${k}` : k, out);
    return out;
  }
  out.push(prefix);
  return out;
}

test('getSnapshot 字段路径与 baseline 冻结一致', () => {
  const { api } = setup();
  const snapshot = api.getSnapshot();
  const paths = [...new Set(walkPaths(snapshot))].sort();
  const baselinePath = join(ROOT, '.smoke', 'agent-contract.baseline.json');
  const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
  assert.equal(baseline.apiVersion, AGENT_API_VERSION);
  assert.deepEqual(
    paths,
    baseline.snapshotPaths,
    '契约字段路径漂移：若为有意变更，请同步更新 .smoke/agent-contract.baseline.json 并说明',
  );
  assert.deepEqual(
    Object.keys(api).sort(),
    baseline.apiSurface.sort(),
    'window.api 表面漂移：若为有意变更，请同步更新 baseline',
  );
});

// ---------- 行为 ----------

test('getSnapshot 数值正确', () => {
  const { api } = setup();
  const s = api.getSnapshot();
  assert.equal(s.version, 1);
  assert.equal(s.media.currentTime, 2.5);
  assert.equal(s.subtitle.count, 2);
  assert.equal(s.subtitle.format, 'srt');
  assert.deepEqual(s.cues.map((c) => [c.index, c.start, c.end]), [[1, 1, 3], [2, 4, 6]]);
  assert.equal(s.history.canUndo, false);
});

test('getPeaks：原生率直取与超限抽取', () => {
  const { api } = setup();
  const full = api.getPeaks();
  assert.equal(full.rate, 4000);
  assert.equal(full.samples.length, 4000); // 1s @4kHz
  const ranged = api.getPeaks(0.5);
  assert.equal(ranged.start, 0.5);
  assert.equal(ranged.samples.length, 2000);
  const decimated = api.getPeaks(0, null, { maxPoints: 100 });
  assert.equal(decimated.samples.length, 100);
  assert.equal(decimated.rate, 100); // 4000 / k(=40)：每秒 100 点
  // 抽取取 |峰值|：不小于任意原始绝对值
  const raw = api.getPeaks().samples;
  const dec = decimated.samples;
  assert.equal(Math.max(...dec) <= Math.max(...raw.map(Math.abs)), true);
});

test('renderWaveform：dataURL 与标定元数据', () => {
  const { api } = setup();
  const r = api.renderWaveform({ t0: 0, t1: 1, pxPerSec: 100 });
  assert.match(r.dataUrl, /^data:image\/png;base64,/);
  assert.equal(r.t0, 0);
  assert.equal(r.t1, 1);
  assert.match(r.calibration, /t0=0\.000s/);
  assert.match(r.calibration, /pxPerSec=100/);
});

test('act：白名单、转发与 coalesceKey 透传', async () => {
  const { api, actions, store } = setup();
  assert.equal((await api.act('loadSubtitle', [])).ok, false); // 不在白名单
  assert.equal((await api.act('nope', [])).ok, false);

  const r = await api.act('updateCueTimes', ['a', 1.2, 3.2]);
  assert.equal(r.ok, true);
  assert.equal(store.state.cues.find((c) => c.id === 'a').start, 1.2);
  assert.equal(actions.canUndo(), true);

  // 连续 50 次同 key 写操作：撤销栈深度保持 1（coalesce 生效）
  const depthBefore = actions.history.past.length;
  for (let i = 0; i < 50; i++) {
    const cur = store.state.cues.find((c) => c.id === 'a');
    await api.act('updateCueTimes', ['a', cur.start + 0.001, cur.end], { coalesceKey: 'agent-batch' });
  }
  assert.equal(actions.history.past.length, depthBefore + 1);
});

test('loadMedia/loadSubs/saveDraft 转发 deps', async () => {
  const { api } = setup();
  assert.equal(await api.loadMedia('x'), 'media');
  assert.equal(await api.loadSubs('x'), 'subs');
  assert.equal(api.saveDraft(), 'saved');
});
