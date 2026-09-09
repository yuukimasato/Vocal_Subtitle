#!/usr/bin/env node
// 构建单文件版：把全部 JS/CSS 与 JASSUB 的 worker/wasm/字体 内联进一个 HTML。
// 产物 subtitle-editor-standalone.html 双击即可使用（file:// 无需 HTTP 服务）。
//
// 用法：node build-standalone.mjs
// 依赖：npx esbuild@0.25.12（仅构建期使用，产物已提交，日常使用无需构建）。
// 注意：模块版（index.html）仍是开发形态；两者功能一致。单文件版把 JASSUB 的
// worker/wasm/字体注入为 data: URL——file:// 下 blob URL 无法创建模块 worker，
// 换 data: URL 后 ASS 样式预览在双击打开的页面里同样可用；初始化失败仍回退文本预览。

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ESBUILD_VERSION = '0.25.12';

function bundle() {
  const dir = mkdtempSync(join(tmpdir(), 'vst-build-'));
  try {
    const out = join(dir, 'bundle.mjs');
    execFileSync('npx', [
      '-y', `esbuild@${ESBUILD_VERSION}`,
      join(ROOT, 'js/main.js'),
      '--bundle', '--format=esm', '--minify',
      `--outfile=${out}`,
    ], { cwd: ROOT, stdio: 'inherit' });
    return readFileSync(out, 'utf8');
  } finally {
    rmSync(dir, { recursive: true, force: true }); // try/finally 清理，避免每次构建泄漏临时目录
  }
}

// 产物头部的构建标记：日期 + 短提交号（不在 git 仓库/无 git 时只写日期）
function buildMarker() {
  const date = new Date().toISOString().slice(0, 10);
  let rev = '';
  try {
    rev = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  } catch {
    // git 不可用：忽略
  }
  return `<!-- built from source @ ${date}${rev ? ` (${rev})` : ''} -->\n`;
}

// 内联脚本里不允许出现字面 </script>，统一转义（字符串/模板/正则中 \/ 语义不变）
function escapeScript(code) {
  return code.replace(/<\/script/gi, '<\\/script');
}

function b64Literal(filePath) {
  return readFileSync(filePath).toString('base64');
}

function vendorAssetScript() {
  const b64 = (p) => b64Literal(join(ROOT, 'vendor', p));
  return `<script>
window.__vstEmbedded = {
  jassubWorker: { b64: ${JSON.stringify(b64('jassub-worker.js'))}, mime: 'text/javascript' },
  jassubWasm: { b64: ${JSON.stringify(b64('jassub-worker.wasm'))}, mime: 'application/wasm' },
  jassubFont: { b64: ${JSON.stringify(b64('jassub-default.woff2'))}, mime: 'font/woff2' },
  jassubCjkFont: { b64: ${JSON.stringify(b64('noto-sans-sc-subset.woff2'))}, mime: 'font/woff2' }
};
// 必须用 data: URL 而不是 blob: URL：file:// 下 Chrome 拒绝以 blob URL 创建模块
// worker（"cannot be accessed from origin 'null'"），且 worker 内部也 fetch 不到页面的
// blob URL；data: URL 两者皆可，ASS 样式预览才能在双击打开的 file:// 页面里工作。
window.VstEditorVendorAssets = {};
for (const [key, item] of Object.entries(window.__vstEmbedded)) {
  window.VstEditorVendorAssets[key] = 'data:' + item.mime + ';base64,' + item.b64;
}
</script>`;
}

const html = readFileSync(join(ROOT, 'index.html'), 'utf8');
const css = readFileSync(join(ROOT, 'css/editor.css'), 'utf8');
const js = bundle();

// 用替换函数避免 replacement 里的 $'/$& 等特殊模式被 String.replace 解释
const out = html
  .replace(/<title>[^<]*<\/title>/, () => '<title>字幕打轴工作台 · 单文件版</title>')
  .replace('<link rel="stylesheet" href="css/editor.css">', () => `<style>\n${css}\n</style>`)
  .replace(
    '<script type="module" src="js/main.js"></script>',
    () => `${vendorAssetScript()}\n<script type="module">\n${escapeScript(js)}\n</script>`,
  );

if (out.includes('js/main.js') || out.includes('css/editor.css')) {
  console.error('构建失败：index.html 的资源引用未能全部替换');
  process.exit(1);
}

const target = join(ROOT, 'subtitle-editor-standalone.html');
writeFileSync(target, buildMarker() + out);
console.log(`已生成 ${target}（${(out.length / 1024 / 1024).toFixed(2)} MB）`);
