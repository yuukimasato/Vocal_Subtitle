# 第三方组件声明（vendor）

本目录内的第三方代码按各自许可原样分发，未做逻辑修改（`jassub.esm.js` 仅为上游
`dist/jassub.js` 与其运行时依赖的 esbuild 打包产物，用于消除裸模块导入）。

| 文件 | 来源包与版本 | 许可 | 声明文件 |
|---|---|---|---|
| `wavesurfer.esm.js`、`regions.esm.js` | wavesurfer.js 7.12.11 | BSD-3-Clause | `wavesurfer-LICENSE` |
| `artplayer.mjs` | artplayer 5.4.0 | MIT（© Harvey Zhao / zhw2590582） | `artplayer-LICENSE` |
| `jassub.esm.js`、`jassub-worker.js`、`jassub-worker.wasm`、`jassub-default.woff2` | jassub 2.5.14 | JS 为 MIT；WASM 内含 libass/freetype/fribidi/harfbuzz 等编译库，许可栈见下 | `jassub-LICENSE` |

JASSUB 的 WASM 产物许可栈（引自其 package.json）：
`LGPL-2.1-or-later AND (FTL OR GPL-2.0-or-later) AND MIT AND MIT-Modern-Variant AND ISC AND NTP AND Zlib AND BSL-1.0`。
以上均为商用友好许可；再分发时请保留本声明与对应 LICENSE 文件。

各组件均在运行时从本目录加载，无 CDN、无网络依赖。
