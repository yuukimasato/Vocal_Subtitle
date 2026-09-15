"""波形打轴端到端回归（需要 Chromium + playwright 包，可选）。

用法:
    # 1. 启动 WebUI 服务器
    .venv/bin/python -m uvicorn vocal_subtitle.webui.app:create_app --factory --port 8642
    # 2. 运行（自动向任务历史注入临时任务 timtest-2026 并在结束后清理）
    .venv/bin/python scripts/e2e_waveform_timing.py

验证内容：红/蓝标记线渲染位置、左键设开始/右键设结束、拖拽边线与整体平移、
[ / ] 播放头打点、方向键微调、Enter 保存跳行、Ctrl+滚轮缩放、
审核工作区波形实例与保存、SRT 文件写回持久化。
"""

"""最终端到端回归：Aegisub 打轴全链路。"""
import os
import sys

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8642"
# 默认交给 playwright 自行解析其管理的 Chromium；特殊构建可用环境变量覆盖。
CHROME = os.environ.get("CHROME", "")
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME or None, headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(BASE)
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(1200)
    page.get_by_text("timtest.wav", exact=True).click()
    page.wait_for_timeout(2500)

    # 初始状态
    st = page.evaluate("""() => { const v = window.WaveformUI.view; const a = v.activeEvent;
        return {idx: v.activeIndex, start: a.start, end: a.end, events: v.events.length}; }""")
    check("自动激活+数据加载", st["idx"] == 1 and st["events"] == 2, str(st))

    # 标记线像素
    page.locator("#waveform-canvas").scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    mk = page.evaluate("""() => {
      const v = window.WaveformUI.view, c = v.el.canvas;
      const d = v.context.getImageData(0, 0, c.width, c.height).data;
      let red = null, blue = null;
      for (let x = 0; x < c.width && (red === null || blue === null); x++) {
        let r = 0, b = 0;
        for (let y = 0; y < c.height; y += 2) {
          const i = (y * c.width + x) * 4;
          if (Math.abs(d[i]-255)<10 && Math.abs(d[i+1]-95)<10 && Math.abs(d[i+2]-86)<10) r++;
          if (Math.abs(d[i]-89)<10 && Math.abs(d[i+1]-167)<10 && Math.abs(d[i+2]-255)<10) b++;
        }
        if (red === null && r > 10) red = x;
        if (blue === null && b > 10) blue = x;
      }
      return {red, blue, w: c.width};
    }""")
    dur = 3.0
    check(
        "红线位置=start",
        mk["red"] and abs(mk["red"] / mk["w"] * dur - st["start"]) < 0.05,
        str(mk),
    )
    check(
        "蓝线位置=end",
        mk["blue"] and abs(mk["blue"] / mk["w"] * dur - st["end"]) < 0.05,
        str(mk),
    )

    rect = page.locator("#waveform-canvas").bounding_box()
    # 左键设开始（区域内）
    page.mouse.click(
        rect["x"] + 0.4 / 3 * rect["width"], rect["y"] + rect["height"] / 2
    )
    page.wait_for_timeout(1100)
    # 右键设结束
    page.mouse.click(
        rect["x"] + 0.8 / 3 * rect["width"],
        rect["y"] + rect["height"] / 2,
        button="right",
    )
    page.wait_for_timeout(1100)
    ev = page.evaluate("""() => { const e = window.WaveformUI.view.events.find(e => e.index === 1);
        return {s: e.start, e: e.end}; }""")
    check(
        "左键0.4/右键0.8",
        abs(ev["s"] - 0.4) < 0.02 and abs(ev["e"] - 0.8) < 0.02,
        str(ev),
    )

    # 播放头 [ ]
    page.evaluate("() => window.WaveformUI.view.seekTo(0.25)")
    page.keyboard.press("[")
    page.evaluate("() => window.WaveformUI.view.seekTo(0.75)")
    page.keyboard.press("]")
    page.keyboard.press("Enter")  # 保存并跳行2
    page.wait_for_timeout(1200)
    ev = page.evaluate("""() => { const v = window.WaveformUI.view;
        const e = v.events.find(e => e.index === 1);
        return {s: e.start, e: e.end, idx: v.activeIndex}; }""")
    check(
        "[0.25 ]0.75 Enter跳行2",
        abs(ev["s"] - 0.25) < 0.015 and abs(ev["e"] - 0.75) < 0.015 and ev["idx"] == 2,
        str(ev),
    )

    # ↑ 回行1
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(300)
    idx = page.evaluate("() => window.WaveformUI.view.activeIndex")
    check("↑ 回到行1", idx == 1)

    # 拖拽整体区域平移：行1 (0.25-0.75) → 拖到 +0.3s → (0.55-1.05)
    page.wait_for_timeout(900)  # 等滚动稳定
    rect = page.locator("#waveform-canvas").bounding_box()
    cx = rect["x"] + 0.5 / 3 * rect["width"]
    cy = rect["y"] + rect["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    for t in [0.25, 0.5, 0.75, 1.0]:
        page.mouse.move(cx + (rect["width"] * 0.1) * t, cy)
        page.wait_for_timeout(40)
    page.mouse.up()
    page.wait_for_timeout(1200)
    ev = page.evaluate("""() => { const e = window.WaveformUI.view.events.find(e => e.index === 1);
        return {s: e.start, e: e.end}; }""")
    check(
        "拖拽区域整体平移 (+0.3s)",
        abs(ev["s"] - 0.55) < 0.06 and abs(ev["e"] - 1.05) < 0.06,
        str(ev),
    )

    # SRT 持久化
    srt = page.evaluate(
        "fetch('/api/tasks/timtest-2026/subtitle-file?version=clean').then(r => r.text())"
    )
    lines = [ln for ln in srt.splitlines() if "-->" in ln]
    ok = any("00:00:00,550" in ln and "00:00:01,050" in ln for ln in lines)
    check("SRT 写回 0.55-1.05", ok, str(lines))

    # 表格同步
    cells = page.evaluate("""() => Array.from(document.querySelectorAll('#subtitle-tbody tr')).map(
      tr => Array.from(tr.querySelectorAll('.col-time')).map(td => td.textContent))""")
    check(
        "表格时间同步",
        cells[0] == ["0:00.6", "0:01.1"] or cells[0] == ["0:00.5", "0:01.0"],
        str(cells),
    )

    # 审核工作区
    page.click('nav [data-workspace="review"]')
    page.wait_for_timeout(2500)
    rstate = page.evaluate("""() => { const w = window.ReviewUI.wave;
        return {events: w.events.length, idx: w.activeIndex, dur: w.duration}; }""")
    check("审核区加载", rstate["events"] == 2 and rstate["dur"] == 3, str(rstate))
    rc = page.locator("#review-waveform-canvas")
    rc.scroll_into_view_if_needed()
    page.wait_for_timeout(700)
    rrect = rc.bounding_box()
    # 左键 0.6 设开始（行1 区域外右侧? 0.6 在区域内）→ 设开始
    page.mouse.click(
        rrect["x"] + 0.6 / 3 * rrect["width"], rrect["y"] + rrect["height"] / 2
    )
    page.wait_for_timeout(1100)
    rev = page.evaluate("""() => { const e = window.ReviewUI.wave.events.find(e => e.index === 1);
        return {s: e.start}; }""")
    check("审核区左键设开始 0.6", abs(rev["s"] - 0.6) < 0.02, str(rev))
    srt2 = page.evaluate(
        "fetch('/api/tasks/timtest-2026/subtitle-file?version=clean').then(r => r.text())"
    )
    ok2 = any("00:00:00,600" in ln for ln in srt2.splitlines() if "-->" in ln)
    check("审核区保存写回 SRT", ok2)

    browser.close()

fails = [r for r in results if not r[1]]
print(f"\n=== {len(results) - len(fails)}/{len(results)} passed ===")
sys.exit(1 if fails else 0)
