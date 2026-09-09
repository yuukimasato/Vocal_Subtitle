// CLI 数据面测试：子进程端到端（JSON 形状/退出码）+ 与 js/actions.js 的行为一致性。
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createStore } from '../js/state.js';
import { createActions } from '../js/actions.js';
import { parseSubtitle } from '../js/format/index.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CLI = join(ROOT, 'agent', 'cli.mjs');

let dir;
test.before(() => {
  dir = mkdtempSync(join(tmpdir(), 'st-cli-'));
});
test.after(() => rmSync(dir, { recursive: true, force: true }));

function write(name, text) {
  const p = join(dir, name);
  writeFileSync(p, text);
  return p;
}

function runCli(args) {
  const r = spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8' });
  let json = null;
  try {
    json = JSON.parse(r.stdout);
  } catch {
    /* 非 JSON 输出（如 --help） */
  }
  return { status: r.status, stdout: r.stdout, stderr: r.stderr, json };
}

const SAMPLE = `1
00:00:01,000 --> 00:00:03,000
第一句

2
00:00:03,500 --> 00:00:06,000
第二句

3
00:00:30,000 --> 00:00:32,000
第三句
`;

test('info：JSON 概要与 text 模式', () => {
  const p = write('info.srt', SAMPLE);
  const r = runCli(['info', p]);
  assert.equal(r.status, 0);
  assert.equal(r.json.ok, true);
  assert.equal(r.json.format, 'srt');
  assert.equal(r.json.count, 3);
  assert.equal(r.json.speechDuration, 6.5);
  assert.equal(r.json.duration.max, 2.5);

  const t = runCli(['info', p, '--format', 'text']);
  assert.equal(t.status, 0);
  assert.match(t.stdout, /行数: 3/);
});

test('cues get：排序序号与时长', () => {
  const p = write('get.srt', SAMPLE);
  const { json, status } = runCli(['cues', p, 'get']);
  assert.equal(status, 0);
  assert.equal(json.count, 3);
  assert.deepEqual(
    json.cues.map((c) => [c.index, c.start, c.end]),
    [[1, 1, 3], [2, 3.5, 6], [3, 30, 32]],
  );
  assert.equal(json.cues[0].duration, 2);
});

test('cues set：批量改时间、非法条目跳过、-o 落盘', () => {
  const p = write('set.srt', SAMPLE);
  const plan = JSON.stringify([
    { index: 1, start: 1.5, end: 3.5 },
    { index: 2, start: Number.NaN, end: 5 }, // 非法：跳过
    { index: 99, start: 0, end: 1 }, // 不存在：无效果
  ]);
  const out = join(dir, 'set-out.srt');
  const r = runCli(['cues', p, 'set', '--plan', plan, '-o', out]);
  assert.equal(r.status, 0);
  assert.equal(r.json.changed, 1);
  assert.equal(r.json.outputIsFile, true);
  const text = readFileSync(out, 'utf8');
  assert.match(text, /00:00:01,500 --> 00:00:03,500/);
  assert.match(text, /00:00:03,500 --> 00:00:06,000/); // 第 2 行未被非法条目波及
});

test('cues shift / split / merge / renumber 端到端', () => {
  const p = write('ops.srt', SAMPLE);
  const shifted = runCli(['cues', p, 'shift', '-0.5']);
  assert.equal(shifted.status, 0);
  assert.match(shifted.json.output, /00:00:00,500 --> 00:00:02,500/);
  // 左移钳位：把第一行整体移到负区后 start 钳 0、时长不短于 MIN_LEN
  const clipped = runCli(['cues', p, 'shift', '-5']);
  assert.match(clipped.json.output, /00:00:00,000 --> 00:00:00,050/);

  const split = runCli(['cues', p, 'split', '1', '2']);
  assert.equal(split.json.count, 4);
  assert.match(split.json.output, /00:00:01,000 --> 00:00:02,000/);
  assert.match(split.json.output, /00:00:02,000 --> 00:00:03,000/);

  const merged = runCli(['cues', p, 'merge', '1']);
  assert.equal(merged.json.count, 2);
  assert.match(merged.json.output, /00:00:01,000 --> 00:00:06,000/);
  assert.match(merged.json.output, /第一句\n第二句/);

  const vtt = write('ops.vtt', 'WEBVTT\n\n00:01.000 --> 00:03.000\n甲\n\n00:03.500 --> 00:06.000\n乙\n');
  const renum = runCli(['cues', vtt, 'renumber']);
  assert.equal(renum.status, 0);
  assert.match(renum.json.output, /^WEBVTT\n\n1\n/);
});

test('convert：srt → vtt → ass 往返', () => {
  const p = write('conv.srt', SAMPLE);
  const vtt = runCli(['convert', p, '--to', 'vtt']);
  assert.equal(vtt.status, 0);
  assert.match(vtt.json.output, /^WEBVTT/);

  const vttFile = write('conv.vtt', vtt.json.output);
  const ass = runCli(['convert', vttFile, '--to', 'ass']);
  assert.match(ass.json.output, /\[Script Info\]/);
  assert.match(ass.json.output, /Dialogue:/);

  const assFile = write('conv.ass', ass.json.output);
  const back = runCli(['convert', assFile, '--to', 'srt']);
  const parsed = parseSubtitle(back.json.output, { filename: 'back.srt' });
  assert.equal(parsed.cues.length, 3);
});

test('check：退出码 0/3 与 --strict', () => {
  const bad = write('bad.srt', [
    '1', '00:00:01,000 --> 00:00:01,020', '过短', // tooShort（error）
    '', '2', '00:00:01,000 --> 00:00:03,000', '重叠', // overlaps（error）
    '', '3', '00:00:30,000 --> 00:00:32,000', '空隙', // gaps（warn，>10s 默认阈值）
    '',
  ].join('\n'));
  const r = runCli(['check', bad]);
  assert.equal(r.status, 3);
  assert.equal(r.json.summary.errors, 2);
  assert.equal(r.json.summary.warnings, 1);
  assert.equal(r.json.clean, false);

  const good = write('good.srt', SAMPLE);
  const okRun = runCli(['check', good]);
  assert.equal(okRun.status, 0); // 默认阈值下 24s 空隙只警不挂
  const strict = runCli(['check', good, '--max-gap', '5', '--strict']);
  assert.equal(strict.status, 3);
  const noGapCheck = runCli(['check', good, '--max-gap', '-1']);
  assert.equal(noGapCheck.status, 0);
  assert.equal(noGapCheck.json.problems.length, 0);
});

test('退出码 2：文件缺失 / 无法识别格式 / 未知命令', () => {
  assert.equal(runCli(['info', join(dir, 'nope.srt')]).status, 2);
  assert.equal(runCli(['info', write('plain.txt', '这不是字幕\n')]).status, 2);
  assert.equal(runCli(['frobnicate']).status, 2);
  assert.equal(runCli(['cues', write('x.srt', SAMPLE), 'explode']).status, 2);
});

// ---------- 与 actions.js 的行为一致性（开发文档 §7.2 第 2 步要求） ----------

function actionsOnSample() {
  const store = createStore();
  const actions = createActions(store);
  actions.loadSubtitle(
    [
      { id: 'a', start: 1, end: 3, text: '第一句' },
      { id: 'b', start: 3.5, end: 6, text: '第二句' },
      { id: 'c', start: 30, end: 32, text: '第三句' },
    ],
    { name: 'x.srt', format: 'srt', doc: null },
  );
  return { store, actions };
}

const strip = (cues) => cues.map(({ id, meta, ...rest }) => rest);

test('set 与 actions.updateCueTimesBulk 一致', () => {
  const { store, actions } = actionsOnSample();
  actions.updateCueTimesBulk([
    { id: 'b', start: 4, end: 7 },
    { id: 'a', start: 2, end: 3.6 },
  ]);

  const p = write('cons.srt', SAMPLE);
  const { json } = runCli(['cues', p, 'set', '--plan', JSON.stringify([
    { index: 2, start: 4, end: 7 },
    { index: 1, start: 2, end: 3.6 },
  ])]);
  const cliCues = parseSubtitle(json.output, { filename: 'cons.srt' }).cues;
  assert.deepEqual(strip(cliCues), strip(store.state.cues));
});

test('split 与 actions.splitCue 一致', () => {
  const { store, actions } = actionsOnSample();
  actions.splitCue('a', 2.2);

  const p = write('cons-split.srt', SAMPLE);
  const { json } = runCli(['cues', p, 'split', '1', '2.2']);
  const cliCues = parseSubtitle(json.output, { filename: 'cons-split.srt' }).cues;
  assert.deepEqual(strip(cliCues), strip(store.state.cues));
});

test('split 越界回退中点与 actions.splitCue 一致', () => {
  const { store, actions } = actionsOnSample();
  actions.splitCue('a', 99); // 越界 → 中点 2

  const p = write('cons-split2.srt', SAMPLE);
  const { json } = runCli(['cues', p, 'split', '1', '99']);
  const cliCues = parseSubtitle(json.output, { filename: 'cons-split2.srt' }).cues;
  assert.deepEqual(strip(cliCues), strip(store.state.cues));
});

test('merge 与 actions.mergeWithNext 一致', () => {
  const { store, actions } = actionsOnSample();
  actions.mergeWithNext('a');

  const p = write('cons-merge.srt', SAMPLE);
  const { json } = runCli(['cues', p, 'merge', '1']);
  const cliCues = parseSubtitle(json.output, { filename: 'cons-merge.srt' }).cues;
  assert.deepEqual(strip(cliCues), strip(store.state.cues));
});
