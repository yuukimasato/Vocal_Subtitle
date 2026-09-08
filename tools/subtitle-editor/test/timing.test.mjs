import test from 'node:test';
import assert from 'node:assert/strict';
import { createTimingController, LEAD_IN_MS, LEAD_OUT_MS } from '../js/audio/timing.js';

// Aegisub 定时控制器（audio_timing_dialogue.cpp 语义）行为测试。
// 时间单位为秒；sensitivity/snapRange 与 px→秒 换算后的值一致（测试里直接用秒）。

function setup(cues = [
  { id: 'a', start: 10, end: 12 },
  { id: 'b', start: 14, end: 16 },
  { id: 'c', start: 18, end: 20 },
], options = {}) {
  const applied = [];
  const controller = createTimingController({
    cues: () => cues,
    activeId: () => options.activeId ?? 'a',
    selectedIds: () => options.selectedIds ?? [],
    inactiveMode: () => options.inactiveMode ?? 1,
    autoCommit: () => options.autoCommit ?? false,
    dragTiming: () => true,
    onApply: (entries, meta) => applied.push({ entries, meta }),
    onChange: () => {},
    duration: () => options.duration ?? null,
  });
  return { controller, applied, cues };
}

test('左键点远离两标记处：开始点设为点击处，拖动以结束点为锚（结束固定）', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(10.8, {}, 0.03, 0);
  assert.ok(drag);
  assert.equal(drag.mode, 'marker');
  assert.deepEqual(controller.eff('a'), { start: 10.8, end: 12 });
  // 拖动调整开始点，结束点固定不跟随
  controller.onMarkerDrag(drag, 10.3, 0);
  assert.deepEqual(controller.eff('a'), { start: 10.3, end: 12 });
  // 拖过结束点 → 交换：被拖位置成为新的结束
  controller.onMarkerDrag(drag, 12.5, 0);
  assert.deepEqual(controller.eff('a'), { start: 12, end: 12.5 });
});

test('左键点击且不拖动：只移动开始点（结束点不变）', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(9.5, {}, 0.03, 0);
  assert.ok(drag);
  assert.deepEqual(controller.eff('a'), { start: 9.5, end: 12 });
});

test('左键点在原结束点右侧：CheckMarkers 交换 → [原结束, 点击处]', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(13, {}, 0.03, 0);
  assert.ok(drag);
  // Aegisub SetPosition+CheckMarkers：开始标记被设到 13 后与原结束点 12 交换角色，
  // 行拉长为 [12, 13]，不并拢成零长选区
  assert.deepEqual(controller.eff('a'), { start: 12, end: 13 });
  // 随后拖动扩出选区 → [13, 15]
  controller.onMarkerDrag(drag, 15, 0);
  assert.deepEqual(controller.eff('a'), { start: 13, end: 15 });
});

test('多次左键扫选：结束点固定不跟随（重复扫选回归场景）', () => {
  const { controller } = setup();
  // 第一次扫选：点在原结束点右侧并拖出新范围 [13, 15]
  const d1 = controller.onLeftClick(13, {}, 0.03, 0);
  assert.deepEqual(controller.eff('a'), { start: 12, end: 13 });
  controller.onMarkerDrag(d1, 15, 0);
  assert.deepEqual(controller.eff('a'), { start: 13, end: 15 });
  // 抬起后再扫选：结束点 15 固定，只有开始点跟随拖动
  const d2 = controller.onLeftClick(14, {}, 0.03, 0);
  controller.onMarkerDrag(d2, 14.8, 0);
  assert.deepEqual(controller.eff('a'), { start: 14.8, end: 15 });
  // 第三次同样：结束点不再逐次前移
  const d3 = controller.onLeftClick(13.7, {}, 0.03, 0);
  controller.onMarkerDrag(d3, 14.2, 0);
  assert.deepEqual(controller.eff('a'), { start: 14.2, end: 15 });
});

test('左键点在标记附近：抓取较近者；点开始标记时立即移动', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(10.02, {}, 0.03, 0);
  assert.equal(drag.items[0].edge, 'left');
  assert.deepEqual(controller.eff('a'), { start: 10.02, end: 12 });
  controller.onMarkerDrag(drag, 10.5, 0);
  assert.deepEqual(controller.eff('a'), { start: 10.5, end: 12 });
});

test('左键点在结束标记附近：抓取但不立即移动，拖动时才移动', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(11.99, {}, 0.03, 0);
  assert.equal(drag.items[0].edge, 'right');
  assert.deepEqual(controller.eff('a'), { start: 10, end: 12 });
  controller.onMarkerDrag(drag, 12.5, 0);
  assert.deepEqual(controller.eff('a'), { start: 10, end: 12.5 });
});

test('右键设定结束点并抓起；点在开始点左侧时 CheckMarkers 交换 → [点击处, 原开始]', () => {
  const { controller } = setup();
  const drag = controller.onRightClick(12.8, 0);
  assert.equal(drag.items[0].edge, 'right');
  assert.deepEqual(controller.eff('a'), { start: 10, end: 12.8 });
  controller.onMarkerDrag(drag, 13.5, 0);
  assert.deepEqual(controller.eff('a'), { start: 10, end: 13.5 });

  const { controller: c2 } = setup();
  const drag2 = c2.onRightClick(8, 0); // 越过开始点 → 交换角色，行变为 [8, 10]
  assert.deepEqual(c2.eff('a'), { start: 8, end: 10 });
  // 拖拽锚点是原开始点 10（Aegisub 抓的是右标记对象）
  c2.onMarkerDrag(drag2, 9.5, 0);
  assert.deepEqual(c2.eff('a'), { start: 9.5, end: 10 });
});

test('右键把结束点拖到开始点之前 → 被拖标记成为新的开始点（交换）', () => {
  const { controller } = setup();
  const drag = controller.onRightClick(12.8, 0);
  controller.onMarkerDrag(drag, 9.2, 0);
  // Aegisub：被拖的物理标记跟随指针（9.2），另一端是原开始点 10
  assert.deepEqual(controller.eff('a'), { start: 9.2, end: 10 });
});

test('Alt+拖：活动行整体平移', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(10.5, { alt: true }, 0.03, 0);
  assert.equal(drag.mode, 'pan');
  controller.onMarkerDrag(drag, 11.0, 0);
  assert.deepEqual(controller.eff('a'), { start: 10.5, end: 12.5 });
  controller.onMarkerDrag(drag, 10.8, 0);
  assert.deepEqual(controller.eff('a'), { start: 10.3, end: 12.3 });
});

test('Ctrl+点标记：抓取所有行上共位的边界一起拖（含非活动行）', () => {
  const cues = [
    { id: 'a', start: 10, end: 12 },
    { id: 'b', start: 12, end: 16 },
  ];
  const { controller } = setup(cues, { activeId: 'a', selectedIds: [], inactiveMode: 2 });
  const drag = controller.onLeftClick(12, { ctrl: true }, 0.03, 0);
  // 点击处同时是 a 的结束与 b（非活动行）的开始：两条边界都被抓起
  assert.equal(drag.items.length, 2);
  controller.onMarkerDrag(drag, 11.2, 0);
  assert.deepEqual(controller.eff('a'), { start: 10, end: 11.2 });
  assert.deepEqual(controller.eff('b'), { start: 11.2, end: 16 });
});

test('多选行跟随活动行一起定时', () => {
  const { controller } = setup(
    [
      { id: 'a', start: 10, end: 12 },
      { id: 'b', start: 14, end: 16 },
    ],
    { activeId: 'a', selectedIds: ['a', 'b'] },
  );
  const drag = controller.onLeftClick(10.8, {}, 0.03, 0);
  controller.onMarkerDrag(drag, 11.2, 0);
  assert.deepEqual(controller.eff('a'), { start: 11.2, end: 12 });
  assert.deepEqual(controller.eff('b'), { start: 11.2, end: 16 });
});

test('吸附：拖动靠近其他行边界时整体吸上（Shift 语义由显示层换算）', () => {
  // 吸附候选受「非活动行显示模式」限制（Aegisub SnapMarkers 同款）：模式 2 含下一行
  const { controller } = setup(undefined, { inactiveMode: 2 });
  const drag = controller.onLeftClick(11.5, {}, 0.03, 0);
  controller.onMarkerDrag(drag, 14.03, 0.05); // 14.03 距 b 行开始 14 在 0.05 内
  assert.deepEqual(controller.eff('a'), { start: 12, end: 14 });
});

test('吸附目标不包含被拖动的边界自身', () => {
  const { controller } = setup();
  const drag = controller.onRightClick(11.98, 0);
  controller.onMarkerDrag(drag, 11.99, 0.05); // 若吸到自身原位置 12 会错误地跳到 12
  assert.deepEqual(controller.eff('a'), { start: 10, end: 11.99 });
});

test('远点击吸附排除活动行自身两端：结束点右侧环带点击不得吸成零长选区', () => {
  const { controller } = setup();
  // 点击处 12.05 距自身结束点 12 在吸附半径 0.05 内、且超出抓取灵敏度：
  // 若候选未排除自身两端，开始点会吸到结束点写成 [12, 12]
  const drag = controller.onLeftClick(12.05, {}, 0.03, 0.05);
  assert.ok(drag);
  assert.deepEqual(controller.eff('a'), { start: 12, end: 12.05 });
});

test('点开始标记时吸附以点击处为基准（而非标记原位置）', () => {
  const cues = [
    { id: 'z', start: 8, end: 10.06 },
    { id: 'a', start: 10, end: 12 },
  ];
  const { controller } = setup(cues, { activeId: 'a', inactiveMode: 3 });
  const drag = controller.onLeftClick(10.02, {}, 0.03, 0.05);
  assert.equal(drag.items[0].edge, 'left');
  // z 行结束点 10.06 距点击处 10.02 为 0.04（半径内），距标记原位置 10 为 0.06（半径外）
  assert.deepEqual(controller.eff('a'), { start: 10.06, end: 12 });
});

test('扫选拖过锚点后反向：跟随指针的一端不吸附到自身旧位置', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(13, {}, 0.03, 0);
  assert.deepEqual(controller.eff('a'), { start: 12, end: 13 });
  controller.onMarkerDrag(drag, 12.4, 0.05); // 拖到锚点 13 左侧：指针值落到开始端
  assert.deepEqual(controller.eff('a'), { start: 12.4, end: 13 });
  // 上一帧的指针位置 12.4 不得作为吸附目标把指针拉回去
  controller.onMarkerDrag(drag, 12.42, 0.05);
  assert.deepEqual(controller.eff('a'), { start: 12.42, end: 13 });
});

test('Alt 平移钳制：不得平移到 0 之前；duration 未知时只钳下界', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(15, { alt: true }, 0.03, 0);
  assert.equal(drag.mode, 'pan');
  // 行 [10,12]，Alt 点击在 15：拖到 2 需平移 -13，钳到 -10 → [0, 2]
  controller.onMarkerDrag(drag, 2, 0);
  assert.deepEqual(controller.eff('a'), { start: 0, end: 2 });
});

test('Alt 平移贴 0 后指针回移可继续平移', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(15, { alt: true }, 0.03, 0);
  controller.onMarkerDrag(drag, 2, 0);
  assert.deepEqual(controller.eff('a'), { start: 0, end: 2 });
  controller.onMarkerDrag(drag, 1, 0); // 继续向左：贴住 0 不再移动
  assert.deepEqual(controller.eff('a'), { start: 0, end: 2 });
  controller.onMarkerDrag(drag, 8, 0); // 回移：未生效的越界量被保留，从贴边处继续
  assert.deepEqual(controller.eff('a'), { start: 3, end: 5 });
});

test('Alt 平移钳制：duration 可用时不得越过音频末尾', () => {
  const { controller } = setup(undefined, { duration: 20 });
  const drag = controller.onLeftClick(5, { alt: true }, 0.03, 0);
  // 行 [10,12]，Alt 点击在 5：拖到 25 需平移 +20，钳到 20-12=8 → [18, 20]
  controller.onMarkerDrag(drag, 25, 0);
  assert.deepEqual(controller.eff('a'), { start: 18, end: 20 });
  controller.onMarkerDrag(drag, 23, 0); // 继续向右：贴住末尾
  assert.deepEqual(controller.eff('a'), { start: 18, end: 20 });
  controller.onMarkerDrag(drag, 12, 0); // 回移继续平移
  assert.deepEqual(controller.eff('a'), { start: 17, end: 19 });
});

test('自动提交关闭：改动留在 pending，提交返回条目并清空', () => {
  const { controller, applied } = setup();
  const drag = controller.onLeftClick(10.8, {}, 0.03, 0);
  controller.onMarkerDrag(drag, 11.2, 0);
  assert.ok(controller.hasPending());
  assert.equal(applied.length, 0);
  const entries = controller.commitPending();
  assert.deepEqual(entries.map((e) => e.id), ['a']);
  assert.equal(entries[0].start, 11.2);
  assert.equal(entries[0].end, 12);
  assert.ok(!controller.hasPending());
});

test('自动提交开启：每次改动直接应用并带 auto 标记', () => {
  const { controller, applied } = setup(undefined, { autoCommit: true });
  const drag = controller.onLeftClick(10.8, {}, 0.03, 0);
  controller.onMarkerDrag(drag, 11.0, 0);
  assert.ok(!controller.hasPending());
  // 点击（设开始点）与拖动（以结束点为锚调开始点）各应用一次，与 Aegisub 的 SetMarkers 一致
  assert.equal(applied.length, 2);
  assert.equal(applied[0].meta.auto, true);
  assert.deepEqual(applied[0].entries[0], { id: 'a', start: 10.8, end: 12 });
  assert.deepEqual(applied[1].entries[0], { id: 'a', start: 11, end: 12 });
});

test('切换行 revert 丢弃未提交改动', () => {
  const { controller } = setup();
  const drag = controller.onLeftClick(10.8, {}, 0.03, 0);
  controller.onMarkerDrag(drag, 11.2, 0);
  assert.ok(controller.revert());
  assert.deepEqual(controller.eff('a'), { start: 10, end: 12 });
  assert.ok(!controller.hasPending());
});

test('前置/延后：开始 -200ms、结束 +300ms', () => {
  const { controller } = setup();
  controller.addLeadIn();
  assert.equal(controller.eff('a').start, 10 - LEAD_IN_MS / 1000);
  controller.addLeadOut();
  assert.equal(controller.eff('a').end, 12 + LEAD_OUT_MS / 1000);
  const entries = controller.commitPending();
  assert.equal(entries.length, 1);
});

test('加长不许短于开始点；缩短不许越过开始点', () => {
  const { controller } = setup();
  controller.modifyLength(-1000); // 大幅缩短 → 钳到开始点
  const eff = controller.eff('a');
  assert.equal(eff.end, eff.start);
  controller.modifyStart(1000); // 开始点后移 → 钳到结束点
  const eff2 = controller.eff('a');
  assert.equal(eff2.start, eff2.end);
});

test('isNearbyMarker：Alt 任意处可抓；否则需在灵敏度内', () => {
  const { controller } = setup();
  assert.ok(controller.isNearbyMarker(11, 0.03, false) === false);
  assert.ok(controller.isNearbyMarker(10.01, 0.03, false));
  assert.ok(controller.isNearbyMarker(11, 0.03, true));
});

test('非活动行模式 1（仅上一行）与 3（全部）', () => {
  const view1 = setup().controller.lineView();
  assert.deepEqual(view1.inactive.map((r) => r.id), []);
  assert.equal(view1.active.id, 'a');

  const view3 = setup(undefined, { activeId: 'b', inactiveMode: 3 }).controller.lineView();
  assert.deepEqual(view3.inactive.map((r) => r.id), ['a', 'c']);
});

test('getPrimaryRange 返回生效时间（含 pending）', () => {
  const { controller } = setup();
  assert.deepEqual(controller.getPrimaryRange(), { start: 10, end: 12 });
  const drag = controller.onLeftClick(10.8, {}, 0.03, 0);
  controller.onMarkerDrag(drag, 11.2, 0);
  assert.deepEqual(controller.getPrimaryRange(), { start: 11.2, end: 12 });
});
