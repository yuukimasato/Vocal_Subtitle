// WAV 直读（Tier 1）：纯 JS 解析 RIFF/WAVE，PCM 8/16/24/32 与 float32，多声道下混单声道。
// 另提供 encodeWav（16-bit PCM）供测试与用户合成已知结构的 fixture。
// 依据 docs/开发文档 §3.3：无 ffmpeg 时 WAV 直读是唯一的 headless 解码路径。

export class WavError extends Error {}

function findChunk(view, target) {
  const len = view.byteLength;
  let off = 12;
  while (off + 8 <= len) {
    const id = String.fromCharCode(view.getUint8(off), view.getUint8(off + 1), view.getUint8(off + 2), view.getUint8(off + 3));
    const size = view.getUint32(off + 4, true);
    if (id === target) return { off: off + 8, size: Math.min(size, len - off - 8) };
    off += 8 + size + (size % 2); // chunk 按 2 字节对齐
  }
  return null;
}

// 解析 WAV → {sampleRate, channels, duration, samples:Float32Array(单声道，已下混)}
// buffer 接受 ArrayBuffer 或 Node Buffer（Uint8Array 视图）
export function parseWav(buffer) {
  const view = buffer instanceof ArrayBuffer
    ? new DataView(buffer)
    : new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (view.byteLength < 12) throw new WavError('不是 WAV 文件（太小）');
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== 'RIFF') throw new WavError('不是 WAV 文件（缺 RIFF 头）');
  const fmt = findChunk(view, 'fmt ');
  if (!fmt) throw new WavError('WAV 缺 fmt 块');
  const audioFormat = view.getUint16(fmt.off, true);
  const channels = view.getUint16(fmt.off + 2, true);
  const sampleRate = view.getUint32(fmt.off + 4, true);
  const bitsPerSample = view.getUint16(fmt.off + 14, true);
  const isFloat = audioFormat === 3 || (audioFormat === 0xfffe && bitsPerSample === 32);
  if (audioFormat !== 1 && !isFloat) throw new WavError(`不支持的 WAV 编码（format=${audioFormat}，仅支持 PCM/float；其他编码请装 ffmpeg）`);

  const data = findChunk(view, 'data');
  if (!data) throw new WavError('WAV 缺 data 块');

  const bytesPer = bitsPerSample / 8;
  const total = Math.floor(data.size / bytesPer);
  const frames = Math.floor(total / channels);
  const out = new Float32Array(frames);
  const read = makeReader(view, isFloat, bitsPerSample);
  for (let f = 0; f < frames; f++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) acc += read(data.off + (f * channels + c) * bytesPer);
    out[f] = acc / channels;
  }
  return { sampleRate, channels, duration: frames / sampleRate, samples: out };
}

function makeReader(view, isFloat, bits) {
  if (isFloat && bits === 32) return (o) => view.getFloat32(o, true);
  if (isFloat && bits === 64) return (o) => view.getFloat64(o, true);
  switch (bits) {
    case 8: return (o) => (view.getUint8(o) - 128) / 128; // 8-bit 是无符号
    case 16: return (o) => view.getInt16(o, true) / 32768;
    case 24: {
      return (o) => {
        const b0 = view.getUint8(o);
        const b1 = view.getUint8(o + 1);
        const b2 = view.getUint8(o + 2);
        let v = (b2 << 16) | (b1 << 8) | b0;
        if (v & 0x800000) v -= 0x1000000; // 符号位
        return v / 8388608;
      };
    }
    case 32: return (o) => view.getInt32(o, true) / 2147483648;
    default: throw new WavError(`不支持的位深：${bits}`);
  }
}

// 编码 16-bit PCM WAV（单声道）。测试合成 fixture 与导出样例用。
export function encodeWav(samples, sampleRate) {
  const bytes = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) bytes.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  bytes.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  bytes.setUint32(16, 16, true);
  bytes.setUint16(20, 1, true); // PCM
  bytes.setUint16(22, 1, true); // mono
  bytes.setUint32(24, sampleRate, true);
  bytes.setUint32(28, sampleRate * 2, true);
  bytes.setUint16(32, 2, true);
  bytes.setUint16(34, 16, true);
  str(36, 'data');
  bytes.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    let v = Math.max(-1, Math.min(1, samples[i]));
    bytes.setInt16(44 + i * 2, Math.round(v * 32767), true);
  }
  return bytes.buffer;
}
