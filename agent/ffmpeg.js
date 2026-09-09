// ffmpeg 可选能力（Tier 2）："检测到即用，缺失优雅降级"——所有调用方必须准备 FfmpegError。
import { spawnSync, spawn } from 'node:child_process';

export class FfmpegError extends Error {}

let cached = null;

// 探测 ffmpeg/ffprobe（结果缓存；探测一次 ~50ms）
export function hasFfmpeg() {
  if (cached !== null) return cached;
  const probe = (cmd) =>
    spawnSync(cmd, ['-version'], { encoding: 'utf8', timeout: 5000 }).status === 0;
  cached = probe('ffmpeg') && probe('ffprobe');
  return cached;
}

export function requireFfmpeg() {
  if (!hasFfmpeg()) {
    throw new FfmpegError('未检测到 ffmpeg：任意媒体解码需要它。请安装后重试（WAV 文件无需 ffmpeg，可直读）。');
  }
}

// 解码任意媒体 → 16kHz 单声道 Float32 PCM（f32le 经 stdout 管道）
export function decodePcm(mediaPath, { rate = 16000, signal } = {}) {
  requireFfmpeg();
  return new Promise((resolve, reject) => {
    const child = spawn('ffmpeg', [
      '-v', 'error',
      '-i', mediaPath,
      '-vn', // 只要音轨
      '-ac', '1',
      '-ar', String(rate),
      '-f', 'f32le',
      'pipe:1',
    ], { signal });
    const chunks = [];
    child.stdout.on('data', (b) => chunks.push(b));
    child.on('error', reject);
    child.on('close', (code, term) => {
      if (code === 0) {
        const bytes = Buffer.concat(chunks);
        const out = new Float32Array(bytes.length / 4);
        for (let i = 0; i < out.length; i++) out[i] = bytes.readFloatLE(i * 4);
        resolve({ rate, samples: out });
      } else {
        reject(new FfmpegError(`ffmpeg 解码失败（exit=${code}${term ? ` ${term}` : ''}）`));
      }
    });
  });
}

// 媒体概要：时长、音轨/视频轨、编码。信息面命令（info --media）用。
export function mediaInfo(mediaPath) {
  requireFfmpeg();
  const r = spawnSync('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration,format_name:stream=codec_type,codec_name',
    '-of', 'json',
    mediaPath,
  ], { encoding: 'utf8', timeout: 20000 });
  if (r.status !== 0) throw new FfmpegError(`ffprobe 失败：${(r.stderr || '').trim().slice(0, 300)}`);
  const parsed = JSON.parse(r.stdout);
  const streams = parsed.streams ?? [];
  const find = (type) => streams.find((s) => s.codec_type === type) ?? null;
  return {
    format: parsed.format?.format_name ?? null,
    duration: parsed.format?.duration != null ? Number(parsed.format.duration) : null,
    audio: find('audio') ? { codec: find('audio').codec_name } : null,
    video: find('video') ? { codec: find('video').codec_name } : null,
  };
}
