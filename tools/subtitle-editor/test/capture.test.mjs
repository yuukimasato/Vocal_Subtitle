import test from 'node:test';
import assert from 'node:assert/strict';
import { createPeakCollector, isCaptureSupported } from '../js/audio/capture.js';

// 波形兜底采集（js/audio/capture.js）纯数据部分：均值降采样累计器。
// DOM 采集主体无法在 node 环境运行，仅验证可导入（纯函数不触碰 window）与
// 降采样数学本身。

test('capture 模块可在无 DOM 环境安全导入（isCaptureSupported 返回 false）', () => {
  assert.equal(typeof createPeakCollector, 'function');
  assert.equal(isCaptureSupported(), false);
});

test('factor=1：跨块 push 等价于透传拼接', () => {
  const c = createPeakCollector({ sampleRate: 4000, targetRate: 4000 });
  c.push(Float32Array.from([0.1, -0.2, 0.3]));
  c.push(Float32Array.from([-0.4]));
  assert.equal(c.factor, 1);
  const out = c.finish();
  assert.equal(out.length, 4);
  [[0, 0.1], [1, -0.2], [2, 0.3], [3, -0.4]].forEach(([i, v]) => assert.ok(Math.abs(out[i] - v) < 1e-6));
  assert.ok(Math.abs(c.maxAbs() - 0.4) < 1e-6);
});

test('factor=4：均值降采样跨块连续，尾部不足一组丢弃', () => {
  const c = createPeakCollector({ sampleRate: 48000, targetRate: 12000 });
  assert.equal(c.factor, 4);
  // 9 个采样：前 8 个分成完整两组，第 9 个是尾部残余
  const samples = [4, -4, 4, -4, 1, 2, 3, 2, 999];
  c.push(Float32Array.from(samples.slice(0, 6)));
  c.push(Float32Array.from(samples.slice(6)));
  const out = c.finish();
  assert.equal(out.length, 2);
  assert.equal(out[0], 0); // (4-4+4-4)/4
  assert.equal(out[1], 2); // (1+2+3+2)/4
  assert.equal(c.maxAbs(), 999); // 尾部残余也计入振幅统计
});

test('maxAbs 用于静音检测：全零输入 silent 判定为真', () => {
  const c = createPeakCollector({ sampleRate: 8000, targetRate: 4000 });
  c.push(new Float32Array(1000));
  assert.equal(c.maxAbs(), 0);
  assert.ok(c.finish().length > 0);
});

test('跨块边界：一组采样横跨两次 push 时均值仍正确', () => {
  const c = createPeakCollector({ sampleRate: 8000, targetRate: 2000 });
  assert.equal(c.factor, 4);
  c.push(Float32Array.from([1, 2]));
  c.push(Float32Array.from([3, 4]));
  c.push(Float32Array.from([5]));
  const out = c.finish();
  assert.equal(out.length, 1);
  assert.equal(out[0], 2.5); // (1+2+3+4)/4，5 为尾部丢弃
});

test('大数据量：块边界（65536）滚动与总长正确', () => {
  const c = createPeakCollector({ sampleRate: 48000, targetRate: 48000 });
  const total = 65536 * 2 + 17; // 恰好跨两个满块
  const step = 4096;
  for (let off = 0; off < total; off += step) {
    const n = Math.min(step, total - off);
    const buf = new Float32Array(n).fill(0.5);
    c.push(buf);
  }
  const out = c.finish();
  assert.equal(out.length, total);
  assert.equal(out[total - 1], 0.5);
});
