// 轻量自绘右键菜单：Escape / 点击外部 / 滚动 / 失焦时关闭。
// 菜单项在每次打开时通过 getItems() 现算，便于动态启用态。
export function createContextMenu() {
  let el = null;

  function close() {
    if (!el) return;
    el.remove();
    el = null;
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('pointerdown', onOutside, true);
    window.removeEventListener('wheel', close, { capture: true, passive: true });
    window.removeEventListener('resize', close);
    window.removeEventListener('blur', close);
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      event.stopPropagation();
      close();
    }
  }

  function onOutside(event) {
    if (el && !el.contains(event.target)) close();
  }

  function open(x, y, items) {
    close();
    el = document.createElement('div');
    el.className = 'ctx-menu';
    el.setAttribute('role', 'menu');
    items().forEach((item) => {
      if (item.type === 'separator') {
        const sep = document.createElement('div');
        sep.className = 'ctx-sep';
        el.appendChild(sep);
        return;
      }
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ctx-item${item.danger ? ' ctx-danger' : ''}`;
      btn.disabled = item.enabled === false;
      btn.setAttribute('role', 'menuitem');
      const label = document.createElement('span');
      label.className = 'ctx-label';
      label.textContent = item.label;
      btn.appendChild(label);
      if (item.shortcut) {
        const key = document.createElement('kbd');
        key.textContent = item.shortcut;
        btn.appendChild(key);
      }
      btn.addEventListener('click', () => {
        close();
        item.onClick?.();
      });
      el.appendChild(btn);
    });
    document.body.appendChild(el);
    const rect = el.getBoundingClientRect();
    el.style.left = `${Math.max(4, Math.min(x, window.innerWidth - rect.width - 8))}px`;
    el.style.top = `${Math.max(4, Math.min(y, window.innerHeight - rect.height - 8))}px`;
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('pointerdown', onOutside, true);
    window.addEventListener('wheel', close, { capture: true, passive: true });
    window.addEventListener('resize', close);
    window.addEventListener('blur', close);
  }

  // shouldOpen(event) 返回 false 时不拦截，放行浏览器原生菜单（如文本输入框内）
  function attach(target, getItems, shouldOpen) {
    target.addEventListener('contextmenu', (event) => {
      if (shouldOpen && !shouldOpen(event)) return;
      event.preventDefault();
      open(event.clientX, event.clientY, getItems);
    });
  }

  return { attach, open, close };
}
