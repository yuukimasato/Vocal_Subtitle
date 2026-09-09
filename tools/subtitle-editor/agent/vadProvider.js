// 外部 VAD provider（Tier 4）：契约见 docs/vad-provider-contract.md（本仓库拥有并版本化）。
// provider 是"恰好实现了契约的外部命令"，失败/非法输出一律降级回内置 VAD，绝不静默——
// 结果 JSON 中带 degraded 字段说明原因。
import { spawn } from 'node:child_process';

export const REQUEST_SCHEMA = 'vad-provider-request-1';
export const RESPONSE_SCHEMA = 'vad-provider-response-1';

export class ProviderError extends Error {}

// cmd：完整命令行字符串（经 sh -c 执行，便于 "--provider 'python run_vad.py --model x'"）。
// 成功 → {engine, segments:[{start,end,confidence?}]}；失败抛 ProviderError（调用方决定降级）。
export function runProvider(cmd, audioPath, { sampleRate = 16000, threshold = 0.5, minSpeechMs = 150, minSilenceMs = 400, timeoutMs = 120000, signal } = {}) {
  const request = JSON.stringify({
    schema_version: REQUEST_SCHEMA,
    audio_path: audioPath,
    sample_rate: sampleRate,
    threshold,
    min_speech_ms: minSpeechMs,
    min_silence_ms: minSilenceMs,
  });
  return new Promise((resolve, reject) => {
    const child = spawn('sh', ['-c', cmd], { signal });
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new ProviderError(`provider 超时（${timeoutMs}ms）`));
    }, timeoutMs);
    child.stdout.on('data', (b) => {
      stdout += b;
    });
    child.stderr.on('data', (b) => {
      stderr += b;
    });
    child.on('error', (err) => {
      clearTimeout(timer);
      reject(new ProviderError(`provider 启动失败：${err.message}`));
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new ProviderError(`provider 退出码 ${code}${stderr.trim() ? `：${stderr.trim().slice(0, 300)}` : ''}`));
        return;
      }
      try {
        resolve(validateResponse(stdout));
      } catch (err) {
        reject(err);
      }
    });
    child.stdin.on('error', () => {}); // provider 不读 stdin 时避免 EPIPE 崩溃
    child.stdin.write(request + '\n');
    child.stdin.end();
  });
}

// 单行 JSON、schema_version 匹配、segments 数组且每段为有限数值。宽容解析：
// 允许 provider 在 JSON 前后打印其他行，取包含 schema_version 的那一行。
export function validateResponse(stdout) {
  const lines = String(stdout).split('\n').map((l) => l.trim()).filter(Boolean);
  const candidates = lines.filter((l) => l.includes(RESPONSE_SCHEMA));
  let parsed = null;
  for (const line of candidates.length ? candidates : lines) {
    try {
      const obj = JSON.parse(line);
      if (obj?.schema_version === RESPONSE_SCHEMA) {
        parsed = obj;
        break;
      }
    } catch {
      /* 尝试下一行 */
    }
  }
  if (!parsed) throw new ProviderError('provider 输出中没有 vad-provider-response-1 响应行');
  if (!Array.isArray(parsed.segments)) throw new ProviderError('provider 响应缺 segments 数组');
  const segments = parsed.segments.map((s, i) => {
    if (typeof s?.start !== 'number' || typeof s?.end !== 'number' || !Number.isFinite(s.start) || !Number.isFinite(s.end)) {
      throw new ProviderError(`provider segments[${i}] 的 start/end 不是有限数值`);
    }
    return {
      start: s.start,
      end: s.end,
      ...(typeof s.confidence === 'number' ? { confidence: s.confidence } : {}),
    };
  });
  return { engine: typeof parsed.engine === 'string' ? parsed.engine : 'unknown-provider', segments };
}
