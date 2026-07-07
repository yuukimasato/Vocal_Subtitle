#!/bin/bash
# ============================================================
# Vocal Subtitle — 依赖安装脚本
# ============================================================
# 一键安装项目所需的系统和 Python 依赖。
#
# Usage:
#   bash scripts/install_deps.sh
#   bash scripts/install_deps.sh --gpu    # 包含 GPU 支持
#   bash scripts/install_deps.sh --all    # 全量安装
#   bash scripts/install_deps.sh --webui  # CLI + Web GUI
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Vocal Subtitle 依赖安装 ==="
echo ""

# ------------------------------------------------------------------
# 检测操作系统
# ------------------------------------------------------------------
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        *)          echo "unknown";;
    esac
}

OS=$(detect_os)
echo "[1/4] 检测操作系统: $OS"

# ------------------------------------------------------------------
# 安装系统依赖
# ------------------------------------------------------------------
echo "[2/4] 安装系统依赖..."

case "$OS" in
    linux)
        echo "  → 检测包管理器..."
        if command -v apt &> /dev/null; then
            echo "  → 使用 apt 安装 ffmpeg..."
            sudo apt update
            sudo apt install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            echo "  → 使用 dnf 安装 ffmpeg..."
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            echo "  → 使用 pacman 安装 ffmpeg..."
            sudo pacman -S --noconfirm ffmpeg
        else
            echo "  ⚠ 未检测到支持的包管理器，请手动安装 ffmpeg"
        fi
        ;;
    macos)
        if command -v brew &> /dev/null; then
            echo "  → 使用 Homebrew 安装 ffmpeg..."
            brew install ffmpeg
        else
            echo "  ⚠ 未检测到 Homebrew，请手动安装 ffmpeg"
        fi
        ;;
    *)
        echo "  ⚠ 未知操作系统，请手动安装 ffmpeg"
        ;;
esac

# ------------------------------------------------------------------
# 创建虚拟环境
# ------------------------------------------------------------------
echo "[3/4] 创建 Python 虚拟环境..."

cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  → 虚拟环境已创建: venv/"
else
    echo "  → 虚拟环境已存在: venv/"
fi

source venv/bin/activate
echo "  → 已激活虚拟环境"

# ------------------------------------------------------------------
# 安装 Python 依赖
# ------------------------------------------------------------------
echo "[4/4] 安装 Python 依赖..."

INSTALL_MODE="${1:---base}"

case "$INSTALL_MODE" in
    --gpu)
        echo "  → 安装模式: 基础 + GPU 支持"
        pip install -e ".[gpu,spleeter,faster-whisper]"
        ;;
    --all)
        echo "  → 安装模式: 全量安装"
        pip install -e ".[all]"
        ;;
    --webui|--gui)
        echo "  → 安装模式: CLI + Web GUI"
        pip install -e ".[faster-whisper,webui]"
        ;;
    *)
        echo "  → 安装模式: 基础安装"
        pip install -e "."
        ;;
esac

echo ""
echo "=== 安装完成 ==="
echo ""
echo "激活虚拟环境:"
echo "  source venv/bin/activate"
echo ""
echo "验证安装:"
echo "  vocal-subtitle --help"
echo "  vocal-subtitle info"
echo ""
echo "快速开始:"
echo "  vocal-subtitle run input.mp3 -o output.srt"
echo ""
echo "Web GUI (如已安装):"
echo "  vocal-subtitle-gui"
