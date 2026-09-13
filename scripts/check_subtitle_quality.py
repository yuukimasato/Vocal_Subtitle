#!/usr/bin/env python3
"""字幕质量红线检查(诊断回路)

用法: python3 scripts/check_subtitle_quality.py <subtitle.ass|srt> <audio.wav>

对输出字幕断言四类症状(全部基于源音频能量门限的真实语音区间):
  A 幻听碎片: cue 与真实语音区间重叠率 < 30% (整行落在静音里)
  B 语音内截尾: 某真实语音区间末尾有一段 > 120ms 未被任何 cue 覆盖
  C 吞并静音: cue 开头之后 120ms 窗口内仍无语音能量, 且真实语音起点
    在 cue 开始后 > 150ms 处 (下一句开头吞了上一句尾部静音)
  D 说话人空缺: cue 无说话人标签

退出码: 有红色症状 = 1, 全绿 = 0。--json 输出机器可读结果。
"""
import sys, json, wave
import numpy as np

ISLAND_THRESH = 0.04
TRUNCATION_MS = 120
SWALLOW_MS = 150
COVERAGE_MIN = 0.30


def parse_cues(path):
    text = open(path, encoding="utf-8-sig").read().splitlines()
    cues = []
    if path.endswith(".ass"):
        for line in text:
            if not line.startswith("Dialogue:"):
                continue
            p = line.split(",", 9)
            t = lambda s: (lambda h, m, r: int(h) * 3600 + int(m) * 60 + float(r))(*s.strip().split(":"))
            cues.append({"start": t(p[1]), "end": t(p[2]), "speaker": p[4].strip(), "text": p[9].strip()})
    else:  # srt
        block = []
        for line in text + [""]:
            if line.strip().isdigit() or not line.strip():
                if len(block) >= 2:
                    t0, t1 = block[0].split("-->")
                    t = lambda s: (lambda h, m, r: int(h) * 3600 + int(m) * 60 + float(r.replace(",", ".")))(*s.strip().split(":"))
                    cues.append({"start": t(t0), "end": t(t1), "speaker": "", "text": " ".join(block[1:])})
                block = []
            else:
                block.append(line)
        # 提取说话人前缀 [说话人E] / 说话人E: 
        import re
        for c in cues:
            m = re.match(r"^\[?((?:说话人|Speaker)[A-Za-z0-9]+)\]?\s*[::]?\s*", c["text"])
            if m:
                c["speaker"] = m.group(1)
                c["text"] = c["text"][m.end():]
    return cues


def speech_islands(wav_path):
    w = wave.open(wav_path)
    sr = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if w.getnchannels() == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    data /= np.abs(data).max() + 1e-9
    win = int(0.02 * sr)
    m = len(data) // win
    env = np.abs(data[: m * win].reshape(m, win)).max(axis=1)
    act = env > ISLAND_THRESH
    # 音频前方静音区间供 C 检查使用
    islands, i = [], 0
    while i < m:
        if act[i]:
            j = i
            while j < m and (act[j] or (j + 8 < m and act[j + 1 : j + 8].any())):
                j += 1
            islands.append((i * 0.02, j * 0.02))
            i = j + 1
        else:
            i += 1
    return islands, data, sr


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    sub_path, wav_path = sys.argv[1], sys.argv[2]
    as_json = "--json" in sys.argv
    cues = parse_cues(sub_path)
    islands, data, sr = speech_islands(wav_path)
    duration = len(data) / sr

    red = {}

    # A 幻听碎片
    phantom = []
    for c in cues:
        cov = sum(overlap(c["start"], c["end"], a, b) for a, b in islands) / max(1e-6, c["end"] - c["start"])
        if cov < COVERAGE_MIN:
            phantom.append({"start": round(c["start"], 3), "end": round(c["end"], 3),
                            "coverage": round(cov, 2), "text": c["text"][:20]})
    red["A_phantom_fragments"] = phantom

    # B 语音内截尾: 每个语音区间尾部未被任何 cue 覆盖(union 口径)的长度
    truncated = []
    for a, b in islands:
        last_end = max(
            (c["end"] for c in cues if overlap(a, b, c["start"], c["end"]) > 0),
            default=a,
        )
        uncovered = b - min(last_end, b)
        if uncovered > TRUNCATION_MS / 1000 and last_end > a:
            truncated.append({"island": [round(a, 2), round(b, 2)],
                              "cut_at": round(min(last_end, b), 3), "lost_ms": round(uncovered * 1000)})
    red["B_truncated_tails"] = truncated

    # C 吞并静音: cue 开头后 120ms 窗口基本无能量, 且真实语音起点比 cue.start 晚 > 150ms
    swallowed = []
    win = int(0.02 * sr)
    for c in cues:
        i0 = int(c["start"] * sr)
        lead = data[i0 + int(0.02 * sr): i0 + int(0.14 * sr)]
        if lead.size == 0:
            continue
        if (np.abs(lead[: len(lead) // win * win].reshape(-1, win)).max(axis=1) > ISLAND_THRESH).mean() > 0.2:
            continue  # 开头就有语音
        onset = next((a for a, b in islands if a >= c["start"] - 0.001 and a < c["end"]), None)
        if onset is not None and (onset - c["start"]) * 1000 > SWALLOW_MS:
            swallowed.append({"start": round(c["start"], 3), "real_onset": round(onset, 3),
                              "swallowed_ms": round((onset - c["start"]) * 1000), "text": c["text"][:16]})
    red["C_swallowed_silence"] = swallowed

    # D 说话人空缺
    no_spk = [{"start": round(c["start"], 3), "text": c["text"][:16]}
              for c in cues if not c["speaker"]]
    red["D_missing_speaker"] = no_spk

    red["_summary"] = {"cues": len(cues), "islands": len(islands), "duration": round(duration, 2)}
    red["_red"] = bool(phantom or truncated or swallowed or no_spk)

    if as_json:
        print(json.dumps(red, ensure_ascii=False, indent=1))
    else:
        print(f"== {sub_path} ==")
        print(f"cues={red['_summary']['cues']} islands={red['_summary']['islands']}")
        for key, label in [("A_phantom_fragments", "幻听碎片"), ("B_truncated_tails", "语音内截尾"),
                           ("C_swallowed_silence", "吞并静音"), ("D_missing_speaker", "说话人空缺")]:
            items = red[key]
            status = "RED" if items else "green"
            print(f"[{status}] {key} {label}: {len(items)}")
            for it in items[:8]:
                print("       ", json.dumps(it, ensure_ascii=False))
    sys.exit(1 if red["_red"] else 0)


if __name__ == "__main__":
    main()
