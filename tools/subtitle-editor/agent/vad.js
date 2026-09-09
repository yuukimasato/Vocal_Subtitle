// 内置确定性 VAD（Tier 3）：能量门限 + 滞回的边界候选器。
// 定位（写进 --help 与输出）：打轴辅助的边界候选器（engine: "energy-vad"），不是 ASR 分段器。
// 干净素材精度 ±10–30ms；音乐/强噪声素材不保证。参数与打轴生态惯例一致（docs/开发文档 §4.7）。
// 纯逻辑、确定性、可单测：同一输入永远同一输出。

export const VAD_DEFAULTS = Object.freeze({
  frameMs: 20, // 短时能量窗
  hopMs: 10, // 跳长（与打轴网格 0.01s 惯例一致）
  noisePercentile: 20, // 噪声底 = 帧能量的 20 分位
  thresholdFactor: 3, // 阈值 = 噪声底 × 3
  minSpeechMs: 150, // 最短语音段（更短的候选丢弃）
  minSilenceMs: 400, // 最短静音（更短的间隙并回语音段）
});

const EPS = 1e-9; // 数字静音的保护底：噪声底为 0 时阈值不失效

export class EnergyVad {
  constructor(opts = {}) {
    this.opts = { ...VAD_DEFAULTS, ...opts };
  }

  // 输入单声道采样（任意采样率）→ {engine, segments:[{start,end,confidence}], params}
  detect(samples, rate) {
    const { frameMs, hopMs, noisePercentile, thresholdFactor, minSpeechMs, minSilenceMs } = this.opts;
    const frameLen = Math.max(1, Math.round((frameMs / 1000) * rate));
    const hopLen = Math.max(1, Math.round((hopMs / 1000) * rate));

    // 帧能量（均方）
    const energies = [];
    for (let off = 0; off + frameLen <= samples.length; off += hopLen) {
      let sum = 0;
      for (let i = off; i < off + frameLen; i++) sum += samples[i] * samples[i];
      energies.push(sum / frameLen);
    }
    if (!energies.length) {
      return { engine: 'energy-vad', segments: [], params: this.params() };
    }

    const sorted = [...energies].sort((a, b) => a - b);
    const noiseFloor = Math.max(
      EPS,
      sorted[Math.floor(((percentileToIndex(noisePercentile))) * sorted.length)] ?? EPS,
    );
    const hi = Math.max(noiseFloor * thresholdFactor, EPS * thresholdFactor);
    const lo = hi / 2; // 滞回：低于 hi/2 才退出语音

    // 滞回状态机 → 语音帧区间
    const spans = [];
    let inSpeech = false;
    let start = 0;
    for (let i = 0; i < energies.length; i++) {
      const t = (i * hopLen) / rate;
      if (!inSpeech && energies[i] > hi) {
        inSpeech = true;
        start = t;
      } else if (inSpeech && energies[i] < lo) {
        inSpeech = false;
        spans.push([start, t]);
      }
    }
    if (inSpeech) spans.push([start, samples.length / rate]);

    // 静音间隙 < minSilence 的并回；语音段 < minSpeech 的丢弃
    const merged = [];
    for (const span of spans) {
      const prev = merged[merged.length - 1];
      if (prev && span[0] - prev[1] < minSilenceMs / 1000) prev[1] = span[1];
      else merged.push([...span]);
    }
    const segments = merged
      .filter(([s, e]) => e - s >= minSpeechMs / 1000)
      .map(([s, e]) => ({
        start: round3(s),
        end: round3(e),
        confidence: segmentConfidence(energies, noiseFloor, hi, s, e, hopLen, rate),
      }));

    return {
      engine: 'energy-vad',
      segments,
      params: this.params(),
      noiseFloor,
      threshold: hi,
    };
  }

  params() {
    return { ...this.opts, note: '打轴辅助的边界候选器，不是 ASR 分段器' };
  }
}

function percentileToIndex(p) {
  return Math.min(1, Math.max(0, p / 100));
}

// confidence 由超阈裕度映射：帧能量对阈值的比值越高的帧占比越大越接近 1
function segmentConfidence(energies, noiseFloor, hi, startSec, endSec, hopLen, rate) {
  const i0 = Math.max(0, Math.floor((startSec * rate) / hopLen));
  const i1 = Math.min(energies.length, Math.ceil((endSec * rate) / hopLen));
  let sum = 0;
  let n = 0;
  for (let i = i0; i < i1; i++) {
    const ratio = energies[i] / (noiseFloor + EPS);
    sum += Math.min(1, Math.max(0, (Math.log2(ratio) - 1) / 5)); // ratio ≥4 → 起、≥32 → 满
    n += 1;
  }
  return n ? Math.round((sum / n) * 100) / 100 : 0;
}

function round3(v) {
  return Math.round(v * 1000) / 1000;
}

export function detectSegments(samples, rate, opts) {
  return new EnergyVad(opts).detect(samples, rate);
}
