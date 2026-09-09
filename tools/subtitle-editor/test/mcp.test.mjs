// MCP server 协议往返测试：initialize → tools/list → tools/call（真实子进程 stdio）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const SERVER = join(ROOT, 'agent', 'mcp-server.mjs');

let dir;
test.before(() => {
  dir = mkdtempSync(join(tmpdir(), 'st-mcp-'));
  writeFileSync(
    join(dir, 'sample.srt'),
    '1\n00:00:01,000 --> 00:00:03,000\n第一句\n\n2\n00:00:03,500 --> 00:00:06,000\n第二句\n',
  );
});
test.after(() => rmSync(dir, { recursive: true, force: true }));

function talk(requests) {
  const input = requests.map((r) => JSON.stringify(r)).join('\n') + '\n';
  const r = spawnSync(process.execPath, [SERVER], { input, encoding: 'utf8', timeout: 60000 });
  const responses = r.stdout.split('\n').filter(Boolean).map((l) => JSON.parse(l));
  return { responses, stderr: r.stderr, status: r.status };
}

test('initialize / tools/list 协议握手', () => {
  const { responses } = talk([
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2024-11-05' } },
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    { jsonrpc: '2.0', id: 2, method: 'tools/list' },
  ]);
  assert.equal(responses.length, 2); // 通知不回复
  assert.equal(responses[0].id, 1);
  assert.equal(responses[0].result.serverInfo.name, 'subtitle-timing-editor');
  assert.ok(responses[0].result.capabilities.tools);
  const names = responses[1].result.tools.map((t) => t.name);
  assert.deepEqual(names, [
    'subtitle_info', 'subtitle_cues', 'subtitle_convert', 'subtitle_check',
    'subtitle_peaks', 'subtitle_vad', 'editor_eval', 'editor_shot',
  ]);
  // 每个工具都有 inputSchema
  for (const t of responses[1].result.tools) assert.equal(t.inputSchema.type, 'object');
});

test('tools/call：subtitle_info / subtitle_check / subtitle_cues.roundtrip', () => {
  const file = join(dir, 'sample.srt');
  const { responses } = talk([
    { jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'subtitle_info', arguments: { file } } },
    { jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'subtitle_check', arguments: { file, maxGap: 2 } } },
    { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'subtitle_cues', arguments: { file, op: 'shift', delta: 1 } } },
  ]);
  const info = JSON.parse(responses[0].result.content[0].text);
  assert.equal(responses[0].result.isError, false);
  assert.equal(info.count, 2);

  const check = JSON.parse(responses[1].result.content[0].text);
  assert.equal(check.command, 'check');
  assert.equal(check.summary.errors, 0);
  assert.equal(responses[1].result.isError, false);

  const cues = JSON.parse(responses[2].result.content[0].text);
  assert.match(cues.output, /00:00:02,000 --> 00:00:04,000/);
});

test('tools/call：不存在的文件 → isError', () => {
  const { responses } = talk([
    { jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'subtitle_info', arguments: { file: join(dir, 'nope.srt') } } },
  ]);
  assert.equal(responses[0].result.isError, true);
});

test('未知方法返回 -32601', () => {
  const { responses } = talk([
    { jsonrpc: '2.0', id: 9, method: 'resources/list' },
  ]);
  assert.equal(responses[0].error.code, -32601);
});
