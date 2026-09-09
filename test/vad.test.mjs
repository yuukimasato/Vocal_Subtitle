// Tier 1–4 测试：WAV 直读、ffmpeg 解码（可用时）、内置 VAD 精度、provider 契约与降级、
// peaks 归一、check --media 的 boundaryInSilence。
// fixture 为测试内程序化合成的 16kHz 单声道 WAV（不提交二进制）；
// 每段语音/静音的起止以常量表写死，VAD 输出与之比对（验收：误差 ≤ 50ms，开发文档 §7.1）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseWav, encodeWav, WavError } from '../agent/wav.js';
import { hasFfmpeg, decodePcm } from '../agent/ffmpeg.js';
import { peaksFromPcm } from '../agent/peaks.js';
import { EnergyVad } from '../agent/vad.js';
import { validateResponse, runProvider, ProviderError } from '../agent/vadProvider.js';
import { checkCues } from '../agent/cueOps.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CLI = join(ROOT, 'agent', 'cli.mjs');
const RATE = 16000;

// 已知结构（秒）：静音 0–1、语音 1–3、静音 3–4、语音 4–6.5、静音 6.5–8、
// 语音 8–11、静音 11–12.5、语音 12.5–15、静音 15–16、语音 16–19、静音 19–20
const SPEECH = [[1, 3], [4, 6.5], [8, 11], [12.5, 15], [16, 19]];
const DURATION = 20;
const SILENCE_AMP = 0.001; // 低噪声底（非数字零，噪声分位才有意义）
const TONE_AMP = 0.4;

function synthFixture() {
  const samples = new Float32Array(RATE * DURATION);
  const speechAt = new Set();
  for (let i = 0; i < samples.length; i++) {
    const t = i / RATE;
    const inSpeech = SPEECH.some(([s, e]) => t >= s && t < e);
    if (inSpeech) {
      speechAt.add(i);
      // 叠两个频率让波形有起伏
      samples[i] = TONE_AMP * (0.6 * Math.sin(2 * Math.PI * 440 * t) + 0.4 * Math.sin(2 * Math.PI * 880 * t));
    } else {
      // 确定性伪噪声（避免依赖 Math.random 的可复现性讨论）
      samples[i] = SILENCE_AMP * Math.sin(2 * Math.PI * 37 * t) * Math.sin(2 * Math.PI * 1.7 * t);
    }
  }
  return samples;
}

let dir;
test.before(() => {
  dir = mkdtempSync(join(tmpdir(), 'st-vad-'));
  const wav = encodeWav(synthFixture(), RATE);
  writeFileSync(join(dir, 'vad-fixture.wav'), Buffer.from(wav));
});
test.after(() => rmSync(dir, { recursive: true, force: true }));

function expectSegments(segments, tolerance = 0.05) {
  assert.equal(segments.length, SPEECH.length, `段数不符：${JSON.stringify(segments)}`);
  segments.forEach((seg, i) => {
    const [wantS, wantE] = SPEECH[i];
    assert.ok(Math.abs(seg.start - wantS) <= tolerance, `段${i} start 偏差 ${Math.abs(seg.start - wantS)}s`);
    assert.ok(Math.abs(seg.end - wantE) <= tolerance, `段${i} end 偏差 ${Math.abs(seg.end - wantE)}s`);
    assert.ok(seg.confidence > 0);
  });
}

// ---------- WAV 直读 ----------

test('WAV 编码-解码往返与下混', () => {
  const stereo = new Float32Array([...Array(100)].map((_, i) => (i % 10) / 20));
  const wav = encodeWav(stereo, RATE);
  const parsed = parseWav(wav);
  assert.equal(parsed.sampleRate, RATE);
  assert.equal(parsed.channels, 1);
  assert.ok(Math.abs(parsed.duration - stereo.length / RATE) < 1e-6);
  // 16-bit 量化误差
  for (let i = 0; i < stereo.length; i++) assert.ok(Math.abs(parsed.samples[i] - stereo[i]) < 1e-4);
});

test('非 WAV 输入报 WavError（Tier 1 降级路径的前提）', () => {
  assert.throws(() => parseWav(Buffer.from('not a wav file at all')), WavError);
});

// ---------- 内置 VAD 精度（核心验收） ----------

test('内置 VAD 在已知 fixture 上误差 ≤ 50ms', () => {
  const result = new EnergyVad().detect(synthFixture(), RATE);
  assert.equal(result.engine, 'energy-vad');
  expectSegments(result.segments);
});

test('16kHz 之外的采样率同样工作（4kHz 波形数据输入）', () => {
  // 从 16kHz fixture 直采到 4kHz（模拟 getPeaks 的 4kHz 数据喂 VAD）
  const src = synthFixture();
  const down = new Float32Array(Math.floor(src.length / 4));
  for (let i = 0; i < down.length; i++) down[i] = src[i * 4];
  const result = new EnergyVad().detect(down, 4000);
  expectSegments(result.segments, 0.06); // 帧粒度变粗，放宽到 60ms
});

test('纯静音输入 → 零段', () => {
  const result = new EnergyVad().detect(new Float32Array(RATE * 3), RATE);
  assert.deepEqual(result.segments, []);
});

// ---------- peaks 归一 ----------

test('peaks：16kHz PCM → 4kHz 包络，maxAbs 反映振幅', () => {
  const env = peaksFromPcm(synthFixture(), RATE);
  assert.ok(Math.abs(env.rate - 4000) <= 1); // 16000/4
  assert.ok(Math.abs(env.duration - DURATION) < 0.01);
  assert.ok(env.maxAbs > 0.3 && env.maxAbs <= TONE_AMP + 0.01); // 语音振幅被捕获
  assert.equal(env.peaks.length, Math.floor(synthFixture().length / 4));
});

// ---------- ffmpeg（可用时验证 Tier 2；缺失时跳过） ----------

test('ffmpeg 解码 WAV → 16kHz PCM（Tier 2）', { skip: hasFfmpeg() ? false : '无 ffmpeg' }, async () => {
  const { rate, samples } = await decodePcm(join(dir, 'vad-fixture.wav'), { rate: RATE });
  assert.equal(rate, RATE);
  assert.ok(Math.abs(samples.length / RATE - DURATION) < 0.1);
  const result = new EnergyVad().detect(samples, rate);
  expectSegments(result.segments);
});

// ---------- provider 契约 ----------

test('provider 响应校验：宽容取行、schema 匹配、非法段报错', () => {
  const good = validateResponse('noise line\n{"schema_version":"vad-provider-response-1","engine":"x","segments":[{"start":1,"end":2,"confidence":0.9}]}\n');
  assert.equal(good.engine, 'x');
  assert.deepEqual(good.segments, [{ start: 1, end: 2, confidence: 0.9 }]);
  assert.throws(() => validateResponse('{"schema_version":"vad-provider-response-1","segments":"nope"}'), ProviderError);
  assert.throws(() => validateResponse('{"segments":[{"start":"a","end":2}]}'), ProviderError);
});

test('provider 端到端：合法响应被采纳', async () => {
  const script = `node -e 'let d="";process.stdin.on("data",c=>d+=c).on("end",()=>{JSON.parse(d);console.log(JSON.stringify({schema_version:"vad-provider-response-1",engine:"fake",segments:[{start:1,end:3}]}))})'`;
  const result = await runProvider(script, '/tmp/x.mkv');
  assert.equal(result.engine, 'fake');
  assert.deepEqual(result.segments, [{ start: 1, end: 3 }]);
});

test('provider 非法 JSON → ProviderError（CLI 层会降级内置）', async () => {
  const script = `node -e 'process.stdout.write("not json")'`;
  await assert.rejects(() => runProvider(script, '/tmp/x.mkv'), ProviderError);
});

test('provider 不存在 → ProviderError（同上）', async () => {
  await assert.rejects(() => runProvider('definitely-not-a-command-xyz', '/tmp/x.mkv'), ProviderError);
});

// ---------- CLI 端到端 ----------

function runCli(args) {
  const r = spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8', timeout: 60000 });
  let json = null;
  try {
    json = JSON.parse(r.stdout);
  } catch {
    /* --help 等 */
  }
  return { status: r.status, json, stdout: r.stdout };
}

test('CLI vad：WAV 直读 + 段落与常量表一致', () => {
  const r = runCli(['vad', join(dir, 'vad-fixture.wav')]);
  assert.equal(r.status, 0);
  assert.equal(r.json.engine, 'energy-vad');
  assert.equal(r.json.source, 'wav');
  assert.equal(r.json.params.note, '打轴辅助的边界候选器，不是 ASR 分段器');
  expectSegments(r.json.segments);
});

test('CLI peaks：输出 4kHz 包络切片', () => {
  const r = runCli(['peaks', join(dir, 'vad-fixture.wav'), '--t0', '1', '--t1', '2']);
  assert.equal(r.status, 0);
  assert.equal(r.json.source, 'wav');
  assert.ok(Math.abs(r.json.rate - 4000) <= 1);
  assert.equal(r.json.t0, 1);
  assert.ok(r.json.count >= 3990 && r.json.count <= 4010);
  assert.ok(r.json.peaks.every((v) => v >= -1 && v <= 1)); // 均值包络带符号，maxAbs 才是绝对峰值
});

test('CLI vad --provider：合法 provider 走 provider；非法自动降级并记录 degraded', () => {
  const wavPath = join(dir, 'vad-fixture.wav');
  const good = `node -e 'let d="";process.stdin.on("data",c=>d+=c).on("end",()=>{JSON.parse(d);console.log(JSON.stringify({schema_version:"vad-provider-response-1",engine:"fake",segments:[{start:1,end:3}]}))})'`;
  const okRun = runCli(['vad', wavPath, '--provider', good]);
  assert.equal(okRun.json.source, 'provider');
  assert.equal(okRun.json.engine, 'fake');
  assert.deepEqual(okRun.json.segments, [{ start: 1, end: 3 }]);

  const bad = `node -e 'process.stdout.write("garbage")'`;
  const degraded = runCli(['vad', wavPath, '--provider', bad]);
  assert.equal(degraded.status, 0, '降级不是错误');
  assert.equal(degraded.json.source, 'builtin');
  assert.match(degraded.json.degraded.reason, /garbage|响应/);
  expectSegments(degraded.json.segments); // 降级结果仍是正确的内置输出
});

test('CLI info --media：WAV 直读头部', () => {
  const r = runCli(['info', join(dir, 'vad-fixture.wav'), '--media']);
  assert.equal(r.status, 0);
  assert.equal(r.json.source, 'wav');
  assert.ok(Math.abs(r.json.duration - DURATION) < 0.01);
});

test('CLI check --media：移入静音区的边界被标记 boundaryInSilence', () => {
  // 正确对齐的行：不应有 boundaryInSilence
  const goodSrt = writeCueFile('good.srt', [[1.0, 3.0], [4.05, 6.4]]);
  const goodRun = runCli(['check', goodSrt, '--media', join(dir, 'vad-fixture.wav')]);
  assert.equal(goodRun.status, 0);
  assert.equal(goodRun.json.problems.filter((p) => p.kind === 'boundaryInSilence').length, 0);

  // 人为把边界移进静音区深处：start 3.7（语音 4.0 才开始）、end 7.2（语音 8.0 才开始）
  const badSrt = writeCueFile('bad.srt', [[3.7, 5.0], [7.2, 9.0]]);
  const badRun = runCli(['check', badSrt, '--media', join(dir, 'vad-fixture.wav'), '--strict']);
  assert.equal(badRun.status, 3);
  const flagged = badRun.json.problems.filter((p) => p.kind === 'boundaryInSilence');
  assert.ok(flagged.some((p) => p.index === 1 && p.edge === 'start'), JSON.stringify(flagged));
  assert.ok(flagged.some((p) => p.index === 2 && p.edge === 'start'), JSON.stringify(flagged));
});

test('checkCues 纯函数层：boundaryInSilence 不受无 media 调用影响', () => {
  const r = checkCues([{ start: 1, end: 2, text: 'a', id: 'x' }]);
  assert.equal(r.clean, true);
});

function writeCueFile(name, ranges) {
  const body = ranges.map(([s, e], i) => {
    const fmt = (t) => {
      const h = Math.floor(t / 3600);
      const m = Math.floor((t % 3600) / 60);
      const sec = (t % 60).toFixed(3).padStart(6, '0');
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${sec}`;
    };
    return `${i + 1}\n${fmt(s)} --> ${fmt(e)}\n第${i + 1}句`;
  }).join('\n\n');
  const p = join(dir, name);
  writeFileSync(p, body + '\n');
  return p;
}

// 保证 readFileSync 的引用被使用（fixture 写读自检）
test('fixture 文件可回读', () => {
  const buf = readFileSync(join(dir, 'vad-fixture.wav'));
  const parsed = parseWav(buf);
  assert.equal(parsed.sampleRate, RATE);
});
