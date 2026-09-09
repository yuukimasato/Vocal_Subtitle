// peaks（Tier 1–2）：PCM → 4kHz 峰值包络。所有解码路径（浏览器采集兜底 / ffmpeg / WAV 直读）
// 经同一个 createPeakCollector 归一到 ~4kHz——CLI 数值与人在屏幕上看到的波形同源。
// 注意：均值降采样是低通，适合可视化与同源对照；边界定位用 VAD 的语音区间（docs/开发文档 §3.3）。
import { createPeakCollector, TARGET_RATE } from '../js/audio/capture.js';

export { TARGET_RATE };

// samples: 单声道 Float32Array；返回 {rate, duration, peaks, maxAbs}
export function peaksFromPcm(samples, sampleRate, { targetRate = TARGET_RATE } = {}) {
  const collector = createPeakCollector({ sampleRate, targetRate });
  const BLOCK = 1 << 16;
  for (let off = 0; off < samples.length; off += BLOCK) {
    collector.push(samples.subarray(off, Math.min(samples.length, off + BLOCK)));
  }
  return {
    rate: sampleRate / collector.factor,
    duration: samples.length / sampleRate,
    peaks: collector.finish(),
    maxAbs: collector.maxAbs(),
  };
}
