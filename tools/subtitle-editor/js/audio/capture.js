// 波形兜底解码：wavesurfer 的 fetch + decodeAudioData 覆盖不了「浏览器能播、
// WebAudio 解不了」的媒体——MKV/WebM 容器、AC-3/DTS/ALAC 等音轨、file:// 单
// 文件版下的 fetch 限制、超大文件等。本模块改走「加速采集」：一条隐藏的
// <video> 挂同一 blob URL 独立解码，经 MediaElementAudioSourceNode 把音轨以
// 倍速静默快进一遍，边采边下混/降采样成单声道峰谷数据，最后经
// wavesurfer.load(url, channelData, duration) 渲染（提供 channelData 时
// wavesurfer 跳过自身的 fetch+decode）。
// 采到的数据只用于画波形（wavesurfer 出声走主播放器的媒体元素），因此
// 默认 4kHz 单声道足够；约 57MB/小时。
// 注意：绝不能对主播放器的 <video> 建 MediaElementSourceNode——那会永久改道
// 它的音频输出。这里始终用独立隐藏元素，用完即毁。

export const TARGET_RATE = 4000; // 降采样目标率：波形渲染足够，兼顾高倍缩放的细节
const RATES = [8, 4, 2, 1]; // 倍速阶梯：设置失败或产出停滞时逐级回落
const STALL_CHECK_SEC = 2; // 每隔多少墙钟秒核对一次采集进度
const NO_AUDIO_TIMEOUT_SEC = 6; // 播放这么久仍无任何采样 → 判定采集不可用

export function isCaptureSupported() {
  return (
    typeof window !== 'undefined' &&
    typeof document !== 'undefined' &&
    typeof document.createElement === 'function' &&
    Boolean(window.AudioContext || window.webkitAudioContext)
  );
}

function abortError() {
  return new DOMException('波形采集已取消', 'AbortError');
}

// 纯数据部分（可单测）：跨块连续的均值降采样累计器。
// 均值（而非峰值）相当于低通抗混叠，降采样后的波形形状不失真；
// maxAbs 在原始采样上统计，用于识别「无音轨/静音轨」。
export function createPeakCollector({ sampleRate, targetRate = TARGET_RATE }) {
  const factor = Math.max(1, Math.floor(sampleRate / targetRate));
  const BLOCK = 65536;
  const chunks = [];
  let cur = new Float32Array(BLOCK);
  let curFill = 0;
  let filled = 0;
  let acc = 0;
  let n = 0;
  let maxAbs = 0;

  const pushValue = (v) => {
    if (curFill === BLOCK) {
      chunks.push(cur);
      cur = new Float32Array(BLOCK);
      curFill = 0;
    }
    cur[curFill++] = v;
    filled++;
  };

  return {
    factor,
    maxAbs: () => maxAbs,
    // mono：一段单声道采样（各次 push 之间按采样序连续）
    push(mono) {
      for (let i = 0; i < mono.length; i++) {
        const x = mono[i];
        const a = x < 0 ? -x : x;
        if (a > maxAbs) maxAbs = a;
        acc += x;
        if (++n === factor) {
          pushValue(acc / factor);
          acc = 0;
          n = 0;
        }
      }
    },
    // 不足 factor 的尾部丢弃（时长误差 < 1/factor 秒，渲染无感）
    finish() {
      const out = new Float32Array(filled);
      let off = 0;
      for (const c of chunks) {
        out.set(c, off);
        off += c.length;
      }
      out.set(cur.subarray(0, curFill), off);
      return out;
    },
  };
}

// 采集主体。signal 用于取消（AbortController.signal）；onProgress(fraction) 节流回调。
// 成功 resolve { channels: [Float32Array], silent }；取消 reject AbortError。
export async function captureAudioPeaks({ url, duration, signal, targetRate = TARGET_RATE, onProgress }) {
  if (!isCaptureSupported()) throw new Error('此浏览器不支持音频采集');
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('媒体时长未知，无法采集波形');
  if (signal?.aborted) throw abortError();

  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });

  const AC = window.AudioContext || window.webkitAudioContext;
  const el = document.createElement('video');
  el.preload = 'auto';
  el.src = url; // 与主播放器共享的 blob URL，各持独立解码器实例
  el.volume = 1;
  el.muted = false; // muted 会连带 WebAudio 图静音：保持打开、靠零增益节点不出声
  const trySetRate = (r) => {
    try {
      el.playbackRate = r;
      return true;
    } catch {
      return false; // 个别实现拒绝超范围倍速：由看门狗回落
    }
  };
  trySetRate(RATES[0]);
  try {
    if ('preservesPitch' in el) el.preservesPitch = false;
    if ('webkitPreservesPitch' in el) el.webkitPreservesPitch = false;
  } catch {
    // 旧实现只读：变速音调变化不影响峰谷数据，忽略
  }

  const ctx = new AC();
  const srcNode = ctx.createMediaElementSource(el);
  const proc = ctx.createScriptProcessor(4096, 2, 2);
  const mute = ctx.createGain();
  mute.gain.value = 0; // 监听静音：采集过程不发声
  srcNode.connect(proc);
  proc.connect(mute);
  mute.connect(ctx.destination);

  const collector = createPeakCollector({ sampleRate: ctx.sampleRate, targetRate });
  let scratch = new Float32Array(4096);
  let audioSec = 0; // 已采集的原始速率音频秒数
  let done = false;
  let timer = 0;
  let rateIdx = 0;
  let lastAudioSec = 0;
  let lastCheckWall = performance.now();
  let lastFlowWall = performance.now();
  let lastProgressWall = 0;

  const finish = () => settle(true, { channels: [collector.finish()], silent: collector.maxAbs() < 1e-5 });
  const fail = (err) => settle(false, err);

  function cleanup() {
    clearInterval(timer);
    proc.onaudioprocess = null;
    el.removeEventListener('ended', finish);
    el.removeEventListener('error', onError);
    signal?.removeEventListener('abort', onAbort);
    for (const node of [srcNode, proc, mute]) {
      try {
        node.disconnect();
      } catch {
        // 已断开/上下文已关：忽略
      }
    }
    try {
      el.pause();
    } catch {
      // 忽略
    }
    el.removeAttribute('src');
    try {
      el.load(); // 释放媒体元素解码器
    } catch {
      // 忽略
    }
    ctx.close().catch(() => {});
  }

  function settle(ok, value) {
    if (done) return;
    done = true;
    cleanup();
    ok ? resolve(value) : reject(value);
  }

  function onError() {
    fail(new Error('媒体元素解码错误（该编码可能不受支持）'));
  }
  const onAbort = () => fail(abortError());

  proc.onaudioprocess = (e) => {
    if (done) return;
    const inp = e.inputBuffer;
    const frames = inp.length;
    if (scratch.length < frames) scratch = new Float32Array(frames);
    // 声道下混为均值 mono（ScriptProcessor 显式双声道，>2 声道输入已被下混）
    const c0 = inp.getChannelData(0);
    const c1 = inp.numberOfChannels > 1 ? inp.getChannelData(1) : null;
    for (let i = 0; i < frames; i++) scratch[i] = c1 ? (c0[i] + c1[i]) / 2 : c0[i];
    collector.push(scratch.subarray(0, frames));
    audioSec += frames / ctx.sampleRate;
  };

  el.addEventListener('ended', finish);
  el.addEventListener('error', onError);
  signal?.addEventListener('abort', onAbort, { once: true });

  // 看门狗：取消 / 无产出超时 / 产出停滞时回落倍速 / 进度回调
  timer = setInterval(() => {
    if (done) return;
    if (signal?.aborted) return fail(abortError());
    const now = performance.now();
    if (audioSec > lastAudioSec) lastFlowWall = now;
    else if ((now - lastFlowWall) / 1000 > NO_AUDIO_TIMEOUT_SEC) {
      return fail(new Error('采集不到音频数据（音轨可能为空或音频图未运行）'));
    }
    const wallDt = (now - lastCheckWall) / 1000;
    if (wallDt >= STALL_CHECK_SEC) {
      const eff = (audioSec - lastAudioSec) / wallDt; // 每墙钟秒产出的音频秒数
      lastCheckWall = now;
      lastAudioSec = audioSec;
      if (eff < RATES[rateIdx] * 0.25 && rateIdx < RATES.length - 1) {
        rateIdx++;
        trySetRate(RATES[rateIdx]);
      }
    }
    if (onProgress && now - lastProgressWall > 400) {
      lastProgressWall = now;
      onProgress(Math.min(1, audioSec / duration));
    }
  }, 500);

  try {
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch {
        // 需要用户手势：resume 被拒由下方 state 检查拦截
      }
    }
    if (ctx.state !== 'running') throw new Error('音频上下文未获准运行（请先与页面交互后重试）');
    await el.play(); // blob URL 元数据很快就绪，play 自行等待可播
    lastFlowWall = performance.now(); // 从真正开播起算无产出超时
    if (done) return promise;
  } catch (err) {
    fail(err instanceof Error ? err : new Error(String(err)));
  }
  return promise;
}
