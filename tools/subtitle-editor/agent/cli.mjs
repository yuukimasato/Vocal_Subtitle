#!/usr/bin/env node
// 字幕打轴工作台 —— headless 数据面 CLI（Tier 0：仅 Node 标准库 + 本目录纯逻辑模块）。
// 约定：子命令一律 JSON 输出（--format text 为人读模式）；退出码 0 正常 / 2 输入错误 / 3 校验发现问题。
// 变换语义与 js/actions.js 一致（见 agent/cueOps.js）；能力分层见 ../AGENTS.md。
import { readFileSync, writeFileSync, openSync, readSync, closeSync } from 'node:fs';
import { formatSrtTime } from '../js/format/time.js';
import { ParseError } from '../js/format/index.js';
import { parseWav, WavError } from './wav.js';
import { decodePcm, mediaInfo as probeMedia, hasFfmpeg, FfmpegError } from './ffmpeg.js';
import { peaksFromPcm } from './peaks.js';
import { EnergyVad } from './vad.js';
import { runProvider, ProviderError } from './vadProvider.js';
import {
  loadSubtitle,
  detectFormatOrThrow,
  saveSubtitle,
  shiftCues,
  setCueTimes,
  splitCue,
  mergeCueWithNext,
  renumberCues,
  checkCues,
  subtitleInfo,
  MIN_LEN,
} from './cueOps.js';

const USAGE = `字幕打轴工作台 CLI（headless 数据面）

用法: cli.mjs <命令> [参数] [选项]

命令:
  info <file>                          字幕概要：格式、行数、时长分布
  info <media> --media                 媒体概要：时长、音轨/视频轨（WAV 直读 / ffprobe）
  cues <file> get                      列出全部行（index 为 1 起排序后序号）
  cues <file> set --plan <json|@f|->   批量改时间 [{index,start,end}]
  cues <file> shift <seconds>          整体平移（start 钳 ≥0）
  cues <file> split <index> [at]       在 at 秒处拆行（缺省取中点；越界回退中点）
  cues <file> merge <index>            与排序后的下一行合并
  cues <file> renumber                 重编号（VTT 标识行；SRT 导出时隐式重编）
  convert <file> --to srt|vtt|ass      格式互转
  check <file> [--media <m>] [--strict] 结构校验；--media 加边界-静音关系检查
  peaks <media> [--rate 4000] [--t0 S --t1 S]  波形峰值包络（WAV 直读 / ffmpeg→4kHz）
  vad <media> [--provider "<cmd>"]     语音段候选（内置能量 VAD；--provider 走外部契约，
                                       失败自动降级内置并在 JSON 记录 degraded）

选项:
  -o, --output <file>  变换结果写入文件（"-" 表 stdout）；缺省时 JSON 附 output 字段
  --format <json|text> 输出格式（默认 json）
  --pretty             JSON 缩进美化
  --min-len <sec>      check：最短行时长阈值（默认 ${MIN_LEN}）
  --max-gap <sec>      check：相邻行空隙提示阈值（默认 10；负值关闭）
  --strict             check：警告也计为失败
  -h, --help           本帮助

退出码: 0 正常；2 输入/解析错误（含缺 ffmpeg）；3 check 发现问题。

内置 VAD 是打轴辅助的边界候选器（engine: "energy-vad"），不是 ASR 分段器；
干净素材边界精度 ±10–30ms，音乐/强噪声素材不保证。
外部 provider 契约见 docs/vad-provider-contract.md。`;

// ---------- 参数解析（极简，够用即可） ----------

const state = { format: 'json', pretty: false, output: null };

function parseArgs(argv) {
  const positional = [];
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-o' || a === '--output') state.output = argv[++i];
    else if (a === '--format') state.format = argv[++i];
    else if (a === '--pretty') state.pretty = true;
    else if (a === '--to') flags.to = argv[++i];
    else if (a === '--plan') flags.plan = argv[++i];
    else if (a === '--media') { if (i + 1 < argv.length && !argv[i + 1].startsWith('-')) flags.media = argv[++i]; else flags.mediaFlag = true; }
    else if (a === '--provider') flags.provider = argv[++i];
    else if (a === '--rate') flags.rate = Number(argv[++i]);
    else if (a === '--t0') flags.t0 = Number(argv[++i]);
    else if (a === '--t1') flags.t1 = Number(argv[++i]);
    else if (a === '--min-len') flags.minLen = Number(argv[++i]);
    else if (a === '--max-gap') flags.maxGap = Number(argv[++i]);
    else if (a === '--strict') flags.strict = true;
    else if (a === '-h' || a === '--help') flags.help = true;
    else if (a.startsWith('-') && a !== '-' && !/^-?\d/.test(a)) throw new ParseError(`未知选项：${a}`); // 负数是位置参数
    else positional.push(a);
  }
  return { positional, flags };
}

// ---------- 输出 ----------

function emit(payload, exitCode = 0) {
  if (state.format === 'text') process.stdout.write(toText(payload));
  else process.stdout.write(JSON.stringify(payload, null, state.pretty ? 2 : 0) + '\n');
  process.exitCode = exitCode;
}

function fail(code2, code, message) {
  emit({ ok: false, error: { code, message } }, code2);
}

// 变换类命令的产物：优先写 --output，否则附在 JSON 的 output 字段（便于管道串联不落盘）
function emitTransform(format, cues, doc, command, extra = {}) {
  const text = saveSubtitle(format, cues, doc);
  const base = { ok: true, command, format, count: cues.length, ...extra };
  if (state.output && state.output !== '-') {
    writeFileSync(state.output, text);
    emit({ ...base, output: state.output, outputIsFile: true });
  } else {
    emit({ ...base, output: text, outputIsFile: false });
  }
}

// ---------- 人读模式 ----------

function toText(p) {
  const lines = [];
  if (p.ok === false) {
    lines.push(`错误 [${p.error.code}] ${p.error.message}`);
  } else if (p.command === 'info') {
    lines.push(`格式: ${p.format}`);
    lines.push(`行数: ${p.count}   总语音: ${p.speechDuration}s`);
    lines.push(`范围: ${fmtT(p.firstStart)} → ${fmtT(p.lastEnd)}`);
    lines.push(`时长: min ${p.duration.min}s / 中位 ${p.duration.median}s / max ${p.duration.max}s`);
  } else if (p.command === 'cues' && p.op === 'get') {
    for (const c of p.cues) {
      lines.push(`#${String(c.index).padStart(4)}  ${fmtT(c.start)} --> ${fmtT(c.end)}  ${c.text.replace(/\n/g, '\\n')}`);
    }
  } else if (p.command === 'check') {
    lines.push(`行数 ${p.summary.cues}，错误 ${p.summary.errors}，警告 ${p.summary.warnings}`);
    for (const pr of p.problems) lines.push(`[${pr.severity}] ${pr.kind} #${pr.index}: ${pr.detail}`);
    lines.push(p.clean ? '通过' : '未通过');
  } else if (p.command === 'vad') {
    lines.push(`engine ${p.engine}（${p.source}）${p.degraded ? `  降级：${p.degraded.reason}` : ''}`);
    for (const s of p.segments) lines.push(`${fmtT(s.start)} --> ${fmtT(s.end)}  置信 ${s.confidence ?? '-'}`);
  } else if (p.command === 'peaks') {
    lines.push(`rate ${p.rate}Hz  时长 ${p.duration}s  峰值 ${p.maxAbs}  点数 ${p.count}`);
  } else {
    lines.push(`ok ${p.command}  行数 ${p.count}${p.outputIsFile ? `  已写入 ${p.output}` : ''}`);
    if (p.outputIsFile === false) lines.push(p.output);
  }
  return lines.join('\n') + '\n';
}

function fmtT(t) {
  return t == null ? '-' : formatSrtTime(t);
}

// ---------- 输入 ----------

function readInput(path) {
  return path === '-' ? readFileSync(0, 'utf8') : readFileSync(path, 'utf8');
}

function loadFile(path) {
  const text = readInput(path);
  const filename = path === '-' ? '' : path;
  const format = detectFormatOrThrow(text, filename);
  const { cues, doc } = loadSubtitle(text, { filename, format });
  return { format, cues, doc };
}

function parsePlan(spec) {
  if (spec == null) throw new ParseError('set 需要 --plan <json|@file|->');
  const text = spec.startsWith('@')
    ? readFileSync(spec.slice(1), 'utf8')
    : spec === '-'
      ? readFileSync(0, 'utf8')
      : spec;
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new ParseError(`--plan 不是合法 JSON：${e.message}`);
  }
}

function sortForOutput(cues) {
  return [...cues].sort((a, b) => a.start - b.start || a.end - b.end);
}

function round(v) {
  return Math.round(v * 1000) / 1000;
}

// ---------- 媒体载入（Tier 1–2） ----------

// 只嗅探前 12 字节判断 RIFF/WAV，避免为大文件整读
function isWavFile(path) {
  const fd = openSync(path, 'r');
  try {
    const b = Buffer.alloc(12);
    const n = readSync(fd, b, 0, 12, 0);
    return n === 12 && b.toString('latin1', 0, 4) === 'RIFF';
  } finally {
    closeSync(fd);
  }
}

// WAV → 直读（Tier 1）；其他媒体 → ffmpeg 解 16kHz mono（Tier 2，缺失时 FfmpegError → 退出码 2）
async function loadMediaPcm(path, { rate = 16000 } = {}) {
  if (isWavFile(path)) {
    const parsed = parseWav(readFileSync(path));
    return { source: 'wav', sampleRate: parsed.sampleRate, samples: parsed.samples, duration: parsed.duration, channels: parsed.channels };
  }
  if (hasFfmpeg()) {
    // 先探后解：无音轨的媒体给出明确错误，而不是 ffmpeg 的 exit 码
    const info = probeMedia(path);
    if (!info.audio) {
      throw new FfmpegError(`媒体没有音轨（${info.video ? `视频 ${info.video.codec}` : '无流'}）：VAD/峰值需要音频输入`);
    }
  }
  const { rate: r, samples } = await decodePcm(path, { rate });
  return { source: 'ffmpeg', sampleRate: r, samples, duration: samples.length / r, channels: 1 };
}

async function run(argv) {
  const { positional, flags } = parseArgs(argv);
  const [cmd, ...rest] = positional;
  if (flags.help || !cmd) {
    process.stdout.write(USAGE + '\n');
    return;
  }

  if (cmd === 'info') {
    const file = rest[0];
    if (!file) return fail(2, 'badArgs', '用法: cli.mjs info <file> [--media]');
    if (flags.mediaFlag || flags.media) {
      // 媒体概要：WAV 走头部直读，其他走 ffprobe
      if (isWavFile(file)) {
        const parsed = parseWav(readFileSync(file));
        return emit({
          ok: true, command: 'info', media: true, source: 'wav',
          duration: round(parsed.duration), audio: { codec: `pcm_${parsed.channels}ch` }, video: null,
        });
      }
      return emit({ ok: true, command: 'info', media: true, source: 'ffprobe', ...probeMedia(file) });
    }
    const { format, cues } = loadFile(file);
    emit({ ok: true, command: 'info', format, ...subtitleInfo(cues) });
    return;
  }

  if (cmd === 'cues') {
    const [file, op] = rest;
    if (!file || !op) return fail(2, 'badArgs', '用法: cli.mjs cues <file> <get|set|shift|split|merge|renumber> […]');
    const { format, cues, doc } = loadFile(file);
    switch (op) {
      case 'get':
        emit({
          ok: true, command: 'cues', op, format, count: cues.length,
          cues: sortForOutput(cues).map((c, i) => ({
            index: i + 1,
            start: c.start,
            end: c.end,
            duration: round(c.end - c.start),
            text: c.text,
          })),
        });
        return;
      case 'set': {
        const plan = parsePlan(flags.plan);
        const { cues: next, changed } = setCueTimes(cues, plan);
        emitTransform(format, next, doc, 'cues.set', { changed });
        return;
      }
      case 'shift': {
        const delta = Number(rest[2]);
        if (!Number.isFinite(delta)) return fail(2, 'badArgs', '用法: cli.mjs cues <file> shift <seconds>');
        emitTransform(format, shiftCues(cues, delta), doc, 'cues.shift');
        return;
      }
      case 'split': {
        const index = Number(rest[2]);
        if (!Number.isInteger(index) || index < 1) return fail(2, 'badArgs', '用法: cli.mjs cues <file> split <index> [at]');
        const at = rest[3] != null ? Number(rest[3]) : NaN;
        if (rest[3] != null && !Number.isFinite(at)) return fail(2, 'badArgs', `at 不是数字：${rest[3]}`);
        const { cues: next } = splitCue(cues, index, at, { format, doc });
        emitTransform(format, next, doc, 'cues.split');
        return;
      }
      case 'merge': {
        const index = Number(rest[2]);
        if (!Number.isInteger(index) || index < 1) return fail(2, 'badArgs', '用法: cli.mjs cues <file> merge <index>');
        emitTransform(format, mergeCueWithNext(cues, index), doc, 'cues.merge');
        return;
      }
      case 'renumber':
        emitTransform(format, renumberCues(cues, format), doc, 'cues.renumber');
        return;
      default:
        return fail(2, 'badArgs', `未知 cues 操作：${op}（可用 get/set/shift/split/merge/renumber）`);
    }
  }

  if (cmd === 'convert') {
    const [file] = rest;
    if (!file || !flags.to) return fail(2, 'badArgs', '用法: cli.mjs convert <file> --to srt|vtt|ass');
    const to = String(flags.to).toLowerCase();
    if (!['srt', 'vtt', 'ass'].includes(to)) return fail(2, 'badArgs', `目标格式不支持：${to}`);
    const { cues, doc } = loadFile(file);
    emitTransform(to, cues, doc, 'convert', { to });
    return;
  }

  if (cmd === 'check') {
    const [file] = rest;
    if (!file) return fail(2, 'badArgs', '用法: cli.mjs check <file> [--media <m>] [--strict] [--max-gap <sec>]');
    const { cues } = loadFile(file);
    const result = checkCues(cues, {
      minLen: Number.isFinite(flags.minLen) ? flags.minLen : MIN_LEN,
      maxGap: Number.isFinite(flags.maxGap) ? (flags.maxGap < 0 ? null : flags.maxGap) : 10,
    });
    // --media：边界-静音关系检查（声学事实观察，词汇表见开发文档 §4.1）
    let mediaMeta = null;
    if (flags.media) {
      const pcm = await loadMediaPcm(flags.media);
      const vad = new EnergyVad().detect(pcm.samples, pcm.sampleRate);
      mediaMeta = { media: flags.media, engine: vad.engine, vadSegments: vad.segments.length, source: pcm.source };
      result.problems.push(...boundaryInSilence(sortForOutput(cues), vad.segments));
      result.summary.warnings = result.problems.filter((p) => p.severity === 'warn').length;
      result.summary.errors = result.problems.filter((p) => p.severity === 'error').length;
      result.clean = result.summary.errors === 0;
    }
    const failed = flags.strict ? result.problems.length > 0 : result.summary.errors > 0;
    emit({ ok: true, command: 'check', ...mediaMeta, ...result }, failed ? 3 : 0);
    return;
  }

  if (cmd === 'peaks') {
    const [file] = rest;
    if (!file) return fail(2, 'badArgs', '用法: cli.mjs peaks <media> [--rate 4000] [--t0 S --t1 S]');
    const pcm = await loadMediaPcm(file);
    const env = peaksFromPcm(pcm.samples, pcm.sampleRate, { targetRate: Number.isFinite(flags.rate) && flags.rate > 0 ? flags.rate : undefined });
    // 范围裁剪（--t0/--t1），默认全量（注意：长媒体的全量 JSON 很大）
    let peaks = env.peaks;
    let { rate } = env;
    if (Number.isFinite(flags.t0) || Number.isFinite(flags.t1)) {
      const t0 = Math.max(0, Number.isFinite(flags.t0) ? flags.t0 : 0);
      const t1 = Math.min(env.duration, Number.isFinite(flags.t1) ? flags.t1 : env.duration);
      const si = Math.max(0, Math.floor(t0 * rate));
      const ei = Math.min(peaks.length, Math.ceil(t1 * rate));
      peaks = peaks.slice(si, ei);
    }
    return emit({
      ok: true, command: 'peaks', source: pcm.source, rate,
      duration: round(env.duration), maxAbs: round(env.maxAbs * 1000) / 1000,
      count: peaks.length,
      t0: Number.isFinite(flags.t0) ? flags.t0 : 0,
      peaks: Array.from(peaks),
    });
  }

  if (cmd === 'vad') {
    const [file] = rest;
    if (!file) return fail(2, 'badArgs', '用法: cli.mjs vad <media> [--provider "<cmd>"]');
    const opts = {
      minSpeechMs: 150,
      minSilenceMs: 400,
    };
    // provider 路径：媒体路径原样交给 provider（它自带解码）；失败自动降级内置
    if (flags.provider) {
      try {
        const result = await runProvider(flags.provider, file, opts);
        return emit({ ok: true, command: 'vad', source: 'provider', engine: result.engine, duration: null, segments: result.segments, note: '打轴辅助的边界候选器，不是 ASR 分段器' });
      } catch (err) {
        if (!(err instanceof ProviderError)) throw err;
        const pcm = await loadMediaPcm(file);
        const fallback = new EnergyVad(opts).detect(pcm.samples, pcm.sampleRate);
        return emit({
          ok: true, command: 'vad', source: 'builtin', engine: fallback.engine,
          degraded: { reason: err.message },
          duration: round(pcm.duration), segments: fallback.segments, params: fallback.params,
        }, 0);
      }
    }
    const pcm = await loadMediaPcm(file);
    const result = new EnergyVad(opts).detect(pcm.samples, pcm.sampleRate);
    return emit({
      ok: true, command: 'vad', source: pcm.source, engine: result.engine,
      duration: round(pcm.duration), noiseFloor: result.noiseFloor, threshold: result.threshold,
      segments: result.segments, params: result.params,
    });
  }

  fail(2, 'badArgs', `未知命令：${cmd}（--help 查看用法）`);
}

// 边界-静音检查：行边界深入 VAD 静音区（距两侧语音都超过 margin）→ 事实性警告。
// 常规 lead-in/lead-out（约 0.2–0.3s）不算深入；返回值带最近语音边距供吸附方向判断。
function boundaryInSilence(cues, segments, { margin = 0.2 } = {}) {
  const problems = [];
  if (!segments.length) return problems;
  const gapOf = (t) => {
    for (let i = 0; i <= segments.length; i++) {
      const gapStart = i === 0 ? 0 : segments[i - 1].end;
      const gapEnd = i === segments.length ? Infinity : segments[i].start;
      if (t >= gapStart && t < gapEnd) {
        const next = segments[i] ?? null;
        const prev = segments[i - 1] ?? null;
        return {
          gapStart: round(gapStart),
          gapEnd: next ? round(next.start) : null,
          toNextSpeech: next ? round(next.start - t) : null,
          toPrevSpeech: prev ? round(t - prev.end) : null,
        };
      }
    }
    return null;
  };
  const deep = (gap) =>
    gap.toNextSpeech == null || gap.toPrevSpeech == null
      ? true // 语音开始前/结束后的区域：只要落在里面就算
      : Math.min(gap.toNextSpeech, gap.toPrevSpeech) > margin;
  cues.forEach((cue, i) => {
    for (const edge of ['start', 'end']) {
      const gap = gapOf(cue[edge]);
      if (gap && deep(gap)) {
        problems.push({
          severity: 'warn',
          kind: 'boundaryInSilence',
          index: i + 1,
          edge,
          time: round(cue[edge]),
          ...gap,
          detail: `第 ${i + 1} 行的${edge === 'start' ? '开始' : '结束'}边界深入静音区`,
        });
      }
    }
  });
  return problems;
}

// ---------- 入口 ----------

// 管道消费方提前关闭（head/grep）时安静退出，不当成崩溃
process.stdout.on('error', (err) => {
  if (err?.code === 'EPIPE') process.exit(0);
  throw err;
});

try {
  await run(process.argv.slice(2));
} catch (err) {
  if (err instanceof ParseError) fail(2, 'parseError', err.message);
  else if (err instanceof WavError) fail(2, 'badInput', err.message);
  else if (err instanceof FfmpegError) fail(2, 'missingDependency', err.message);
  else if (err instanceof ProviderError) fail(2, 'providerError', err.message);
  else if (err?.code === 'ENOENT') fail(2, 'fileNotFound', `${err.path ?? ''}: ${err.message}`);
  else if (err?.code === 'EISDIR') fail(2, 'badInput', `${err.path ?? ''}: 是目录`);
  else fail(2, 'internal', err?.stack ?? String(err));
}
