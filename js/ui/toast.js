// 轻量提示条。toast(message, type) → DOM 元素，3.5s 自动消失；点击立即关闭；
// 同屏最多 MAX_TOASTS 条，超出移除最旧（一次 blur 可能叠多条错误提示）。
let container = null;

const MAX_TOASTS = 5;

export function showToast(message, type = 'info') {
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const item = document.createElement('div');
  item.className = `toast toast-${type}`;
  item.setAttribute('role', 'status');
  item.textContent = message;
  item.title = '点击关闭';
  container.appendChild(item);
  while (container.children.length > MAX_TOASTS) {
    container.firstElementChild.remove();
  }
  const dismiss = () => {
    if (item.classList.contains('toast-out')) return;
    item.classList.add('toast-out');
    setTimeout(() => item.remove(), 300);
  };
  const timer = setTimeout(dismiss, 3500);
  item.addEventListener('click', () => {
    clearTimeout(timer);
    dismiss();
  });
  return item;
}
