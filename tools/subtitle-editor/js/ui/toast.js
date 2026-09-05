// 轻量提示条。toast(message, type) → DOM 元素，3.5s 自动消失。
let container = null;

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
  container.appendChild(item);
  setTimeout(() => {
    item.classList.add('toast-out');
    setTimeout(() => item.remove(), 300);
  }, 3500);
  return item;
}
