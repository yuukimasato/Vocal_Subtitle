// 独立开源防回归（docs/开发文档 §6.2）：产品文件中不得出现上层项目绑定——
// 品牌名、绝对路径、越出工具目录的 import。把"不绑定"从口号变成 node --test 可执行断言。
// 白名单（允许作为"集成对象"说明性提及）：README.md、docs/**、LICENSE、vendor/**（第三方原样产物）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, relative } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

function walk(dir, out = []) {
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return out; // 目录不存在（如 agent/ 尚未建立）时跳过
  }
  for (const name of entries) {
    const p = resolve(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

function productFiles() {
  const files = [];
  for (const dir of ['js', 'agent', 'css']) files.push(...walk(resolve(ROOT, dir)));
  for (const f of [
    'index.html',
    'debug-jassub.html',
    'build-standalone.mjs',
    'package.json',
    'subtitle-editor-standalone.html', // 单文件版是提交的构建产物：一起扫，改名后忘记重建即红
  ]) files.push(resolve(ROOT, f));
  return files;
}

const FORBIDDEN = [
  /vocal[\s_-]?subtitle/i, // 上层项目品牌（含 vocal_subtitle / Vocal Subtitle / Vocal-Subtitle 等变体）
  /\/home\//, // 开发机绝对路径
];

test('产品文件不含上层项目品牌或绝对路径', () => {
  const offenders = [];
  for (const f of productFiles()) {
    const text = readFileSync(f, 'utf8');
    for (const re of FORBIDDEN) {
      if (re.test(text)) offenders.push(`${relative(ROOT, f)}: /${re.source}/`);
    }
  }
  assert.deepEqual(offenders, [], `发现绑定残留：\n${offenders.join('\n')}`);
});

test('相对 import 不越出工具目录，import 不使用绝对路径', () => {
  const offenders = [];
  for (const f of productFiles()) {
    if (!/\.(js|mjs)$/.test(f)) continue;
    const text = readFileSync(f, 'utf8');
    const re = /(?:^|[\s(=])(?:import|from)\s*['"]([^'"]+)['"]/g;
    let m;
    while ((m = re.exec(text))) {
      const spec = m[1];
      if (spec.startsWith('.')) {
        const target = resolve(dirname(f), spec);
        const rel = relative(ROOT, target);
        if (rel.startsWith('..')) offenders.push(`${relative(ROOT, f)}: ${spec}`);
      } else if (spec.startsWith('/') || /^[a-zA-Z]:[\\/]/.test(spec)) {
        offenders.push(`${relative(ROOT, f)}: ${spec}`);
      }
    }
  }
  assert.deepEqual(offenders, [], `发现越界 import：\n${offenders.join('\n')}`);
});
