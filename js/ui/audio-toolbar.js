// Aegisub 音频盒工具栏（default_toolbar.json 的 audio 顺序）+ 右侧滑条竖栏。
// 按钮按下不抢焦点，保持音频盒的「Audio」键位上下文。
import { INACTIVE_MODE } from '../audio/timing.js';

const PX_PER_SEC_MIN = 5;
const PX_PER_SEC_MAX = 1000;

export function createAudioToolbar({ store, actions, player, waveform, commands, toolbarEl, sideEl }) {
  const state = () => store.state.audioOptions;

  const buttons = [];
  const toggleButtons = new Map();

  function mkButton(html, title, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'abtn';
    b.innerHTML = html;
    b.title = title;
    b.setAttribute('aria-label', title);
    b.addEventListener('pointerdown', (e) => e.preventDefault()); // 不抢音频盒焦点
    b.addEventListener('click', onClick);
    toolbarEl.appendChild(b);
    buttons.push(b);
    return b;
  }

  function mkSep() {
    const s = document.createElement('span');
    s.className = 'abtn-sep';
    toolbarEl.appendChild(s);
  }

  const run = (name) => () => commands.run(name);
  const TOOLTIPS = {
    autoCommit: '自动提交：拖拽立即写入字幕行（连续改动合并为一步撤销）',
    autoNext: '自动下一行：提交后转至下一行',
    autoScroll: '自动滚动：换行时滚动视图；鼠标释放在显示边缘时滚屏',
    snap: '吸附行边界：拖动时吸附到其他行边界（按住 Shift 临时反转）',
  };
  const toggle = (key, name) => mkButton(
    `<span class="abtn-box"></span>${name}`,
    TOOLTIPS[key] ?? name,
    () => store.patch({ audioOptions: { ...state(), [key]: !state()[key] } }),
  );

  // 顺序对齐 Aegisub audio 工具栏
  mkButton('◀ 行', '上一行（← / Z）', run('time/prev'));
  mkButton('行 ▶', '下一行（→ / X）', run('time/next'));
  mkButton('▶ 选区', '播放选区（S；音频盒内 Space；小键盘 5）', run('play/selection'));
  mkButton('▶ 行', '播放当前行（R）', run('play/line'));
  mkButton('⏹', '停止（H；小键盘 8）', run('stop'));
  mkSep();
  mkButton('|◀ 500', '播放选区开始前 500ms（Q；小键盘 1）', run('play/selection/before'));
  mkButton('500▶|', '播放选区结束后 500ms（W；小键盘 3）', run('play/selection/after'));
  mkButton('首 500', '播放选区前 500ms（E）', run('play/selection/begin'));
  mkButton('尾 500', '播放选区后 500ms（D）', run('play/selection/end'));
  mkButton('▶→ 尾', '从选区开始播放到结尾（T）', run('play/to_end'));
  mkSep();
  mkButton('前置 −200', '开始点提前 200ms（C）', run('time/lead/in'));
  mkButton('延后 +300', '结束点延后 300ms（V）', run('time/lead/out'));
  mkSep();
  mkButton('<b>✓ 提交</b>', '提交定时改动；自动转至下一行开启时前进（G；Enter；小键盘 Enter）', run('commit'));
  mkButton('⌖ 定位', '滚动视图到当前选区', run('go_to'));
  mkSep();
  toggleButtons.set('autoCommit', toggle('autoCommit', '自动提交'));
  toggleButtons.set('autoNext', toggle('autoNext', '自动下一行'));
  toggleButtons.set('autoScroll', toggle('autoScroll', '自动滚动'));
  toggleButtons.set('snap', toggle('snap', '吸附行边界'));
  mkSep();

  // 非活动行显示模式（Aegisub Audio/Inactive Lines Display Mode）
  const inactiveSel = document.createElement('select');
  inactiveSel.className = 'abtn ab-sel';
  inactiveSel.title = '非活动行显示范围';
  inactiveSel.setAttribute('aria-label', '非活动行显示范围');
  inactiveSel.addEventListener('pointerdown', (e) => e.stopPropagation());
  [
    [INACTIVE_MODE.NONE, '非活动：无'],
    [INACTIVE_MODE.PREV, '非活动：上一行'],
    [INACTIVE_MODE.PREV_NEXT, '非活动：前后一行'],
    [INACTIVE_MODE.ALL, '非活动：全部'],
  ].forEach(([value, label]) => {
    const opt = document.createElement('option');
    opt.value = String(value);
    opt.textContent = label;
    inactiveSel.appendChild(opt);
  });
  inactiveSel.addEventListener('change', () => {
    store.patch({ audioOptions: { ...state(), inactiveMode: Number(inactiveSel.value) } });
  });
  toolbarEl.appendChild(inactiveSel);

  // ---------- 右侧滑条竖栏（横向缩放 / 纵向振幅 / 音量，Aegisub 同款布局） ----------
  function mkSlider(title, onInput) {
    const wrap = document.createElement('div');
    wrap.className = 'ab-slider';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '100';
    input.step = '1';
    input.value = '50';
    input.title = title;
    input.setAttribute('aria-label', title);
    input.addEventListener('pointerdown', (e) => e.stopPropagation());
    input.addEventListener('input', () => onInput(Number(input.value)));
    wrap.appendChild(input);
    sideEl.appendChild(wrap);
    return input;
  }

  const ampFromSlider = (v) => Math.pow(Math.min(100, Math.max(1, v)) / 50, 3);

  // 横向缩放滑条：0-100 对数映射到 5-1000 px/秒（与 waveform.setZoom 的钳位一致）
  const zoomFromSlider = (v) => PX_PER_SEC_MIN * Math.pow(PX_PER_SEC_MAX / PX_PER_SEC_MIN, v / 100);
  const sliderFromZoom = (px) => {
    if (!(px > 0)) return 0;
    return Math.round((Math.log(px / PX_PER_SEC_MIN) / Math.log(PX_PER_SEC_MAX / PX_PER_SEC_MIN)) * 100);
  };
  const hZoom = mkSlider('横向缩放（波形）', (v) => waveform.setZoom(zoomFromSlider(v)));
  hZoom.value = String(Math.min(100, Math.max(0, sliderFromZoom(waveform.getPxPerSec?.() ?? 0))));

  // 波形侧缩放（滚轮/ready 初始缩放）回写滑条。程序赋值不触发 input，不会形成回路
  store.on('zoom', () => {
    hZoom.value = String(Math.min(100, Math.max(0, sliderFromZoom(waveform.getPxPerSec?.() ?? 0))));
  });

  // 音量应用（音量滑条与 vZoom 联动共用）；无媒体时仅记录显示值
  function setVolume(v) {
    const art = player.video();
    if (art) art.volume = Math.min(1, ampFromSlider(v));
  }

  const vZoom = mkSlider('纵向缩放（振幅）', (v) => {
    store.patch({ audioOptions: { ...state(), vZoom: v } });
    applyAmplitude();
    if (state().vLink) {
      volume.value = String(v);
      setVolume(v); // 联动必须真正生效：联动开启时音量滑条被 disabled，只改显示则音量不变
    }
  });
  vZoom.value = '50';

  const volume = mkSlider('音量（开启联动时跟随纵向缩放）', setVolume);

  const linkBtn = document.createElement('button');
  linkBtn.type = 'button';
  linkBtn.className = 'abtn ab-link';
  linkBtn.innerHTML = '<span class="abtn-box"></span>⛓';
  linkBtn.title = '纵向缩放与音量联动（Aegisub Audio/Link，默认开）';
  linkBtn.setAttribute('aria-label', linkBtn.title);
  linkBtn.addEventListener('pointerdown', (e) => e.preventDefault());
  linkBtn.addEventListener('click', () => {
    store.patch({ audioOptions: { ...state(), vLink: !state().vLink } });
  });
  sideEl.appendChild(linkBtn);

  function applyAmplitude() {
    const v = state().vZoom ?? 50;
    waveform.setAmplitudeScale(ampFromSlider(v));
  }

  function syncOptions() {
    const o = state();
    toggleButtons.forEach((btn, key) => btn.classList.toggle('active', Boolean(o[key])));
    linkBtn.classList.toggle('active', Boolean(o.vLink));
    volume.disabled = Boolean(o.vLink);
    if (o.vLink) {
      const v = o.vZoom ?? 50;
      volume.value = String(v);
      setVolume(v); // 联动打开（含初始即开）瞬间按当前 vZoom 应用一次音量
    }
    if (String(o.inactiveMode) !== inactiveSel.value) inactiveSel.value = String(o.inactiveMode);
  }
  store.on('audioOptions', syncOptions);
  syncOptions();

  return {};
}
