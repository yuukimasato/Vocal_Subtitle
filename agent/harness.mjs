#!/usr/bin/env node
// 通用 CDP harness（Tier 5）：经 Chrome DevTools Protocol 驱动真实页面。
// 刻意不绑定任何产品语义（window.agent 只是被 eval 的目标表达式）——同一份
// harness 可驱动本编辑器与任何 Web 前端。仅 Node 标准库；仅连 127.0.0.1。
//
// 用法:
//   harness.mjs open  --url URL [--port 9222] [--chrome <path>] [--headless] [--profile DIR]
//                     无浏览器时启动（默认有头，人可同屏监督）；已有浏览器则直接开新标签
//   harness.mjs list                [--port 9222]                列出页面目标
//   harness.mjs eval  "<expr>"      [--port 9222] [--target SUB] [--timeout MS]
//                                   表达式可 async；returnByValue + awaitPromise
//   harness.mjs shot  --out p.png   [--port 9222] [--target SUB]
//   harness.mjs close               [--port 9222]                关闭浏览器（含启动的进程）
//
// 输出一律 JSON；退出码 0 正常 / 2 失败。--target 为 URL 子串匹配（缺省取第一个 page）。
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const USAGE = `通用 CDP harness（不绑定产品语义；仅 127.0.0.1）

用法: harness.mjs <open|list|eval|shot|close> [参数]

  open  --url URL [--port 9222] [--chrome <path>] [--headless] [--profile DIR]
  list              [--port 9222]
  eval  "<expr>"    [--port 9222] [--target <url子串>] [--timeout <ms>]
  shot  --out p.png [--port 9222] [--target <url子串>]
  close             [--port 9222]

输出: JSON 一行；退出码 0 正常 / 2 失败。
安全: Chrome 调试端口默认只监听回环地址；harness 只连 127.0.0.1。`;

const state = { port: 9222, timeout: 15000 };

function parseArgs(argv) {
  const positional = [];
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const take = () => flags[a.replace(/^--?/, '')] = argv[++i];
    if (a === '--url' || a === '--port' || a === '--chrome' || a === '--profile' || a === '--target' || a === '--out' || a === '--timeout') take();
    else if (a === '--headless') flags.headless = true;
    else if (a === '-h' || a === '--help') flags.help = true;
    else if (a.startsWith('-')) throw new Error(`未知选项：${a}`);
    else positional.push(a);
  }
  return { positional, flags };
}

function emit(payload, exitCode = 0) {
  process.stdout.write(JSON.stringify(payload) + '\n');
  process.exitCode = exitCode;
}

function fail(message) {
  emit({ ok: false, error: message }, 2);
}

// ---------- CDP 连接 ----------

async function httpJson(path, method = 'GET') {
  const res = await fetch(`http://127.0.0.1:${state.port}${path}`, {
    method,
    signal: AbortSignal.timeout(state.timeout), // 端口被无关服务占用时不无限挂起
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}`);
  return res.json();
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.seq = 0;
    this.pending = new Map();
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message} (${msg.error.code})`));
        else resolve(msg.result);
      }
    });
  }

  static connect(wsUrl, timeout = state.timeout) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      const timer = setTimeout(() => {
        ws.close();
        reject(new Error(`WebSocket 连接超时：${wsUrl}`));
      }, timeout);
      ws.addEventListener('open', () => {
        clearTimeout(timer);
        resolve(new CDP(ws));
      });
      ws.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error(`WebSocket 连接失败：${wsUrl}`));
      });
    });
  }

  send(method, params = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP 调用超时：${method}`));
      }, state.timeout);
      this.pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.ws.close();
  }
}

async function findTarget(match) {
  const targets = await httpJson('/json/list');
  const pages = targets.filter((t) => t.type === 'page');
  const target = match
    ? pages.find((t) => (t.url + ' ' + t.title).includes(match))
    : pages[0];
  if (!target) throw new Error(match ? `找不到匹配目标：${match}` : '没有打开的页面目标');
  return target;
}

async function withPage(match, fn) {
  const target = await findTarget(match);
  const cdp = await CDP.connect(target.webSocketDebuggerUrl);
  try {
    return await fn(cdp, target);
  } finally {
    cdp.close();
  }
}

// ---------- 子命令 ----------

function findChrome(explicit) {
  if (explicit) return explicit;
  for (const name of ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']) {
    const r = spawnSync('sh', ['-c', 'command -v "$1"', '-', name], { encoding: 'utf8' });
    const path = r.stdout?.trim();
    if (r.status === 0 && path) return path;
  }
  return null;
}

async function cmdOpen(flags) {
  if (!flags.url) return fail('open 需要 --url');
  const port = state.port;
  // 已有调试端点则只开新标签；否则启动 Chrome
  let already = false;
  try {
    await httpJson('/json/version');
    already = true;
  } catch {
    /* 未在运行 */
  }
  if (!already) {
    const chrome = findChrome(flags.chrome);
    if (!chrome && !flags.chrome) {
      return fail('找不到 Chrome（可 --chrome <路径> 指定；或先用 --port 连接已运行的浏览器）');
    }
    const profile = flags.profile ?? mkdtempSync(join(tmpdir(), 'cdp-harness-'));
    const args = [
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-features=Translate',
    ];
    if (flags.headless) args.push('--headless');
    args.push(flags.url);
    const child = spawn(chrome, args, { stdio: 'ignore', detached: true });
    child.unref();
    // 等调试端点就绪
    const deadline = Date.now() + state.timeout;
    while (Date.now() < deadline) {
      try {
        await httpJson('/json/version');
        break;
      } catch {
        await new Promise((r) => setTimeout(r, 200));
      }
    }
  }
  // 开目标标签
  const target = await httpJson(`/json/new?${encodeURIComponent(flags.url)}`, 'PUT').catch(() => null);
  emit({
    ok: true,
    command: 'open',
    reused: already,
    port,
    target: target ? { id: target.id, url: target.url } : null,
    hint: 'eval/shot/close 用相同 --port 连接',
  });
}

async function cmdList() {
  const targets = await httpJson('/json/list');
  emit({
    ok: true,
    command: 'list',
    targets: targets
      .filter((t) => t.type === 'page')
      .map((t) => ({ id: t.id, title: t.title, url: t.url })),
  });
}

async function cmdEval(flags, positional) {
  const expr = positional[1];
  if (expr == null) return fail('eval 需要一个表达式参数');
  const result = await withPage(flags.target, async (cdp) => {
    const r = await cdp.send('Runtime.evaluate', {
      expression: expr,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails) {
      return { ok: false, error: r.exceptionDetails.exception?.description ?? r.exceptionDetails.text };
    }
    return { ok: true, type: r.result.type, value: r.result.value ?? null, description: r.result.description ?? null };
  });
  emit({ command: 'eval', ...result }, result.ok ? 0 : 2);
}

async function cmdShot(flags) {
  if (!flags.out) return fail('shot 需要 --out <file>');
  const result = await withPage(flags.target, async (cdp) => {
    const r = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    writeFileSync(flags.out, Buffer.from(r.data, 'base64'));
    return { ok: true };
  });
  emit({ command: 'shot', ...result, out: flags.out }, result.ok ? 0 : 2);
}

async function cmdClose() {
  // Browser.close 走浏览器级 target（/json/version 的 webSocketDebuggerUrl）
  const version = await httpJson('/json/version');
  const wsUrl = version.webSocketDebuggerUrl;
  if (!wsUrl) return fail('浏览器未暴露浏览器级调试端点');
  const cdp = await CDP.connect(wsUrl);
  try {
    await cdp.send('Browser.close');
  } finally {
    cdp.close();
  }
  emit({ ok: true, command: 'close' });
}

// ---------- 入口 ----------

async function run() {
  const { positional, flags } = parseArgs(process.argv.slice(2));
  if (flags.help || !positional.length) {
    process.stdout.write(USAGE + '\n');
    return;
  }
  if (flags.port) state.port = Number(flags.port);
  if (flags.timeout) state.timeout = Number(flags.timeout);
  const [cmd] = positional;
  try {
    if (cmd === 'open') await cmdOpen(flags);
    else if (cmd === 'list') await cmdList();
    else if (cmd === 'eval') await cmdEval(flags, positional);
    else if (cmd === 'shot') await cmdShot(flags);
    else if (cmd === 'close') await cmdClose();
    else fail(`未知命令：${cmd}`);
  } catch (err) {
    fail(err?.message ?? String(err));
  }
}

run();
