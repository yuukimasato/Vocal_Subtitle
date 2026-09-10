#!/usr/bin/env bash
# Vocal_Subtitle 双界面启动器（双界面收敛方案，2026-09-10）：
#   8613 = 处理台（管线/历史/质量/反馈档案），8631 = 编辑台（打轴/校对/学习回传）
# 行为：编辑台静态服务没起就拉起并随会话托管；处理台没起就启动（自动打开浏览器）；
#       都已在跑则直接打开处理台页面。日志在启动终端内可见。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDITOR_DIR="$ROOT/tools/subtitle-editor"
BACKEND_PORT=8613
EDITOR_SERVE_PORT=8631

port_up() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

# 选择处理台入口：优先生产 venv 控制台脚本，回退开发 venv
if [ -x "$ROOT/.venv-production/bin/vocal-subtitle-gui" ]; then
  GUI_CMD=("$ROOT/.venv-production/bin/vocal-subtitle-gui")
elif [ -x "$ROOT/.venv/bin/python" ]; then
  GUI_CMD=("$ROOT/.venv/bin/python" "$ROOT/main_gui.py")
else
  echo "未找到 vocal-subtitle-gui：请先运行 install.sh 或创建 .venv-production" >&2
  exit 1
fi

# 8631 编辑台：纯静态服务，已在跑则复用
if ! port_up "$EDITOR_SERVE_PORT"; then
  echo "启动编辑台静态服务：http://127.0.0.1:${EDITOR_SERVE_PORT}/ （$EDITOR_DIR）"
  python3 -m http.server "$EDITOR_SERVE_PORT" --directory "$EDITOR_DIR" &
  HTTP_PID=$!
fi

# 8613 处理台：已在跑则直接打开页面，否则前台启动（gui 自带浏览器拉起）
if port_up "$BACKEND_PORT"; then
  echo "处理台已在运行：http://127.0.0.1:${BACKEND_PORT}/"
  xdg-open "http://127.0.0.1:${BACKEND_PORT}/" >/dev/null 2>&1 || true
else
  echo "启动处理台：http://127.0.0.1:${BACKEND_PORT}/"
  exec "${GUI_CMD[@]}" --host 127.0.0.1 --port "$BACKEND_PORT"
fi
