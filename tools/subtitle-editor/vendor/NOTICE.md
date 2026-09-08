# 第三方组件声明（vendor）

本目录内的第三方代码按各自许可原样分发（`jassub.esm.js` 与 `jassub-worker.js` 仅为上游
dist/源码与其运行时依赖的 esbuild 打包产物，用于消除裸模块导入）。`jassub-worker.js` 含
两处本地补丁：① 构造失败时向主线程透传真实错误，避免 abslink 报出误导性的
"Unserializable return value"；② emscripten 对 `import.meta.url` 的相对 URL 解析在
blob worker（standalone 内嵌）下会抛 "Invalid URL"，为 wasm 实例化与 pthread worker
spawn 增加了回退。`jassub/` 目录为官方 npm 包 dist 的本地化副本（含相对依赖的 worker
入口与源码映射），作为参考与再打包素材，运行时不加载。

| 文件 | 来源包与版本 | 许可 | 声明文件 |
|---|---|---|---|
| `wavesurfer.esm.js` | wavesurfer.js 7.12.11 | BSD-3-Clause | `wavesurfer-LICENSE` |
| `artplayer.mjs` | artplayer 5.4.0 | MIT（© Harvey Zhao / zhw2590582） | `artplayer-LICENSE` |
| `jassub.esm.js`、`jassub-worker.js`、`jassub-worker.wasm`、`jassub-default.woff2`、`jassub/` | jassub 2.5.14 | JS 为 MIT；WASM 内含 libass/freetype/fribidi/harfbuzz 等编译库，许可栈见下 | `jassub-LICENSE` |

JASSUB 的 WASM 产物许可栈（引自其 package.json）：
`LGPL-2.1-or-later AND (FTL OR GPL-2.0-or-later) AND MIT AND MIT-Modern-Variant AND ISC AND NTP AND Zlib AND BSL-1.0`。
以上均为商用友好许可；再分发时请保留本声明与对应 LICENSE 文件。

各组件均在运行时从本目录加载，无 CDN、无网络依赖。
