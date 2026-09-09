#!/usr/bin/env node
// MCP 包装（后置可选层）：把 headless CLI 暴露为 MCP tools，供 ZCode/claude code 等即插即用。
// 只是薄包装——全部能力都在 agent/cli.mjs 与 agent/harness.mjs；仅 Node 标准库。
// 协议：MCP over stdio（JSON-RPC 2.0，换行分帧）。挂载示例（ZCode 配置）：
//   { "command": "node", "args": ["<工具目录>/agent/mcp-server.mjs"] }
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CLI = join(ROOT, 'agent', 'cli.mjs');
const HARNESS = join(ROOT, 'agent', 'harness.mjs');

const TOOLS = [
  {
    name: 'subtitle_info',
    description: '字幕概要（格式/行数/时长分布）；传 media:true 时返回媒体概要（时长/音轨）',
    inputSchema: {
      type: 'object', required: ['file'],
      properties: { file: { type: 'string' }, media: { type: 'boolean', description: '按媒体文件读取（WAV 直读 / ffprobe）' } },
    },
  },
  {
    name: 'subtitle_cues',
    description: '字幕行读写变换。op: get | set(--plan) | shift(--delta 秒) | split(--index [--at 秒]) | merge(--index) | renumber',
    inputSchema: {
      type: 'object', required: ['file', 'op'],
      properties: {
        file: { type: 'string' }, op: { type: 'string', enum: ['get', 'set', 'shift', 'split', 'merge', 'renumber'] },
        plan: { type: 'string', description: 'set：JSON 数组字符串 [{"index":1,"start":1.0,"end":3.0}]' },
        delta: { type: 'number', description: 'shift：秒，可为负' },
        index: { type: 'number', description: 'split/merge：1 起行号' },
        at: { type: 'number', description: 'split：切分时刻（秒）' },
        output: { type: 'string', description: '结果写入的文件路径；缺省附在返回 JSON 里' },
      },
    },
  },
  {
    name: 'subtitle_convert',
    description: '字幕格式互转（srt/vtt/ass）',
    inputSchema: {
      type: 'object', required: ['file', 'to'],
      properties: { file: { type: 'string' }, to: { type: 'string', enum: ['srt', 'vtt', 'ass'] }, output: { type: 'string' } },
    },
  },
  {
    name: 'subtitle_check',
    description: '结构校验（overlaps/gaps/tooShort/endBeforeStart）+ 可选边界-静音检查（boundaryInSilence）。退出语义：问题为 error 或 strict 下有警告 → isError',
    inputSchema: {
      type: 'object', required: ['file'],
      properties: { file: { type: 'string' }, media: { type: 'string', description: '媒体路径：启用声学边界检查' }, strict: { type: 'boolean' }, maxGap: { type: 'number' }, minLen: { type: 'number' } },
    },
  },
  {
    name: 'subtitle_peaks',
    description: '波形峰值包络（4kHz，与页面波形同源）。长媒体请用 t0/t1 取片段',
    inputSchema: {
      type: 'object', required: ['file'],
      properties: { file: { type: 'string' }, t0: { type: 'number' }, t1: { type: 'number' }, rate: { type: 'number' } },
    },
  },
  {
    name: 'subtitle_vad',
    description: '语音段候选（内置能量 VAD；provider 为可选外部命令，失败自动降级）。注意：边界候选器，不是 ASR 分段器',
    inputSchema: {
      type: 'object', required: ['file'],
      properties: { file: { type: 'string' }, provider: { type: 'string', description: '外部 provider 命令行（契约见 docs/vad-provider-contract.md）' } },
    },
  },
  {
    name: 'editor_eval',
    description: '在真实页面上求值 JS 表达式（Tier 5，需 Chrome 与已打开的编辑器页面；window.agent 为推荐入口）',
    inputSchema: {
      type: 'object', required: ['expression'],
      properties: { expression: { type: 'string' }, port: { type: 'number', description: 'CDP 端口，默认 9222' }, target: { type: 'string', description: 'URL 子串匹配' } },
    },
  },
  {
    name: 'editor_shot',
    description: '真实页面截图（Tier 5）',
    inputSchema: {
      type: 'object', required: ['out'],
      properties: { out: { type: 'string' }, port: { type: 'number' }, target: { type: 'string' } },
    },
  },
];

function runTool(name, args = {}) {
  const cliArgs = [];
  switch (name) {
    case 'subtitle_info':
      cliArgs.push('info', args.file, ...(args.media ? ['--media'] : []));
      break;
    case 'subtitle_cues': {
      cliArgs.push('cues', args.file, args.op ?? 'get');
      if (args.op === 'set' && args.plan != null) cliArgs.push('--plan', String(args.plan));
      if (args.op === 'shift') cliArgs.push(String(args.delta ?? 0));
      if (args.op === 'split' || args.op === 'merge') cliArgs.push(String(args.index ?? 1));
      if (args.op === 'split' && args.at != null) cliArgs.push(String(args.at));
      break;
    }
    case 'subtitle_convert':
      cliArgs.push('convert', args.file, '--to', args.to ?? 'srt');
      break;
    case 'subtitle_check':
      cliArgs.push('check', args.file, ...(args.media ? ['--media', args.media] : []), ...(args.strict ? ['--strict'] : []), ...(args.maxGap != null ? ['--max-gap', String(args.maxGap)] : []), ...(args.minLen != null ? ['--min-len', String(args.minLen)] : []));
      break;
    case 'subtitle_peaks':
      cliArgs.push('peaks', args.file, ...(args.t0 != null ? ['--t0', String(args.t0)] : []), ...(args.t1 != null ? ['--t1', String(args.t1)] : []), ...(args.rate != null ? ['--rate', String(args.rate)] : []));
      break;
    case 'subtitle_vad':
      cliArgs.push('vad', args.file, ...(args.provider ? ['--provider', args.provider] : []));
      break;
    case 'editor_eval':
      cliArgs.push('eval', String(args.expression ?? ''), ...(args.port != null ? ['--port', String(args.port)] : []), ...(args.target ? ['--target', args.target] : []));
      return exec(HARNESS, cliArgs);
    case 'editor_shot':
      cliArgs.push('shot', '--out', String(args.out ?? ''), ...(args.port != null ? ['--port', String(args.port)] : []), ...(args.target ? ['--target', args.target] : []));
      return exec(HARNESS, cliArgs);
    default:
      return { json: { ok: false, error: `未知工具：${name}` }, code: 2 };
  }
  return exec(CLI, cliArgs);
}

function exec(bin, args) {
  const r = spawnSync(process.execPath, [bin, ...args], { encoding: 'utf8', timeout: 180000, maxBuffer: 512 * 1024 * 1024 });
  return { code: r.status ?? 2, json: r.stdout, error: r.stderr };
}

function reply(id, result) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n');
}

function replyError(id, code, message) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } }) + '\n');
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let idx;
  while ((idx = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (line) handleMessage(line);
  }
});
process.stdin.on('end', () => process.exit(0));

function handleMessage(line) {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return; // 非请求行忽略（容错日志输出）
  }
  const { id, method, params } = msg;
  try {
    switch (method) {
      case 'initialize':
        reply(id, {
          protocolVersion: params?.protocolVersion ?? '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'subtitle-timing-editor', version: '1.0.0' },
        });
        return;
      case 'notifications/initialized':
        return; // 通知不回
      case 'ping':
        reply(id, {});
        return;
      case 'tools/list':
        reply(id, { tools: TOOLS });
        return;
      case 'tools/call': {
        const { name, arguments: args } = params ?? {};
        const { code, json, error } = runTool(name, args);
        const failed = code !== 0; // check 的退出码 3（发现问题）也算 isError
        reply(id, {
          content: [{ type: 'text', text: json || error || '' }],
          isError: failed,
        });
        return;
      }
      default:
        if (id != null) replyError(id, -32601, `方法不存在：${method}`);
    }
  } catch (err) {
    if (id != null) replyError(id, -32603, err?.message ?? String(err));
  }
}
