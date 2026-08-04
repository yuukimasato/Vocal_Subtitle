#!/usr/bin/env bash
# ============================================================
# Vocal Subtitle — 一键安装脚本
# ============================================================
# 自动检测系统环境，安装系统依赖、Python 依赖、预下载模型。
#
# Usage:
#   bash install.sh                 # 基础安装 (自动检测 GPU)
#   bash install.sh --cpu            # 强制 CPU 模式
#   bash install.sh --llm           # 含 LLM 优化支持
#   bash install.sh --uvr           # 含 UVR 分离引擎 (默认推荐，BS-RoFormer)
#   bash install.sh --spleeter      # 含 Spleeter 分离引擎 (旧，仅 Python < 3.12)
#   bash install.sh --webui         # 含 Web GUI
#   bash install.sh --all           # 全量安装
#   bash install.sh --dev           # 开发环境
#   bash install.sh --gui           # CLI + Web GUI 一键部署
#   bash install.sh --local-nlp     # 本地 NLP 语义合并 (CPU, 无需 GPU)
#   bash install.sh --download-models  # 安装 + 预下载模型
#   bash install.sh --help          # 查看帮助
# ============================================================

set -euo pipefail

# ------------------------------------------------------------------
# 颜色输出
# ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "\n${CYAN}==>${NC} ${CYAN}$*${NC}"; }

# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------
banner() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}     ${CYAN}Vocal Subtitle — 人声分离 + 字幕生成工具${NC}       ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}              一键安装脚本 v0.3.0                    ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ------------------------------------------------------------------
# 帮助
# ------------------------------------------------------------------
show_help() {
    echo "Usage: bash install.sh [OPTIONS]"
    echo ""
    echo "安装选项:"
    echo "  (无参数)        基础安装 (自动检测 GPU, 智能选择 VAD + 加速方案)"
    echo "  --cpu           强制 CPU 模式，使用 CPU 版 PyTorch + WebRTC VAD"
    echo "  --no-torch      跳过 PyTorch，仅用 WebRTC VAD (最轻量, ~100MB)"
    echo "  --llm           基础 + LLM 字幕优化 (openai, tenacity, json-repair)"
    echo "  --uvr           基础 + UVR 分离引擎 (audio-separator, 默认推荐)"
    echo "  --spleeter      基础 + Spleeter 分离引擎 (已停维，仅 Python < 3.12)"
    echo "  --openunmix     基础 + Open-Unmix 分离引擎 (openunmix, soundfile)"
    echo "  --webrtcvad     显式指定 WebRTC VAD (覆盖 GPU 自动选择)"
    echo "  --local-nlp     基础 + 本地 NLP 语义合并 (sentence-transformers, CPU)"
    echo "  --webui         基础 + Web GUI 图形界面 (fastapi, uvicorn)"
    echo "  --diarization   基础 + 说话人分离 (librosa, scikit-learn, scipy)"
    echo "  --gui           一键部署 CLI + Web GUI (同 --webui)"
    echo "  --all           全量安装 (包含以上全部)"
    echo "  --dev           开发环境 (基础 + 测试 + Lint 工具)"
    echo ""
    echo "GPU 自动适配 (默认):"
    echo "  检测到 NVIDIA GPU    → Silero VAD (GPU) + CTranslate2 (CUDA) + CUDA 版 PyTorch"
    echo "  检测到 Apple Silicon → Silero VAD (MPS) + CTranslate2"
    echo "  无 GPU               → WebRTC VAD (纯 CPU, 轻量, 免装 torch)"
    echo "  --cpu                → CPU 版 PyTorch + Silero VAD (保留 torch 生态)"
    echo "  --no-torch           → WebRTC VAD, 完全跳过 PyTorch (~100MB)"
    echo "  --webrtcvad          → 显式覆盖 GPU 自动选择, 强制 WebRTC VAD"
    echo ""
    echo "GPU 参数:"
    echo "  --cuda VER      指定 CUDA 版本 (如 --cuda 12.4)"
    echo ""
    echo "模型下载:"
    echo "  --download-models        安装完成后预下载常用模型"
    echo "  --download-asr MODEL      指定 ASR 模型 (large-v3 / medium / small)"
    echo "  --download-separator ENG  指定分离引擎 (uvr / spleeter)"
    echo ""
    echo "其他选项:"
    echo "  --no-venv      跳过虚拟环境创建，直接安装到当前 Python"
    echo "  --venv NAME    指定虚拟环境名称 (默认: venv)"
    echo "  --no-model-dl  跳过模型下载"
    echo "  --yes, -y      非交互模式，自动确认所有提示"
    echo "  --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash install.sh                                    # 基础安装 (自动检测 GPU)"
    echo "  bash install.sh --cpu                              # 强制 CPU 模式"
    echo "  bash install.sh --cuda 12.4                        # 指定 CUDA 版本"
    echo "  bash install.sh --all                              # 全量安装 (自动检测 GPU)"
    echo "  bash install.sh --all --download-models            # 全量 + 模型预下载"
    echo "  bash install.sh --gui                              # CLI + 浏览器界面"
    echo "  bash install.sh --dev --venv myenv                 # 开发环境 + 自定义 venv"
    echo "  bash install.sh --no-torch                         # 保留现有 PyTorch"
    exit 0
}

# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------
INSTALL_MODE="base"
CREATE_VENV=true
VENV_NAME="venv"
DOWNLOAD_MODELS=false
DOWNLOAD_ASR_MODEL=""
DOWNLOAD_SEPARATOR_ENG=""
SKIP_MODEL_DL=false
GPU_MODE=true   # 默认自动检测 GPU，传 --cpu 可强制跳过
CPU_EXPLICIT=false  # 区分「自动检测无 GPU」和「--cpu 显式指定」
CUDA_VERSION=""
SKIP_TORCH=false
YES_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_MODE=true
            INSTALL_MODE="${INSTALL_MODE},gpu"
            shift ;;
        --cpu)
            GPU_MODE=false
            CPU_EXPLICIT=true
            shift ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2 ;;
        --no-torch)
            SKIP_TORCH=true
            shift ;;
        --llm)
            INSTALL_MODE="${INSTALL_MODE},llm"
            shift ;;
        --uvr)
            INSTALL_MODE="${INSTALL_MODE},uvr"
            shift ;;
        --spleeter)
            INSTALL_MODE="${INSTALL_MODE},spleeter"
            shift ;;
        --openunmix)
            INSTALL_MODE="${INSTALL_MODE},openunmix"
            shift ;;
        --webrtcvad)
            INSTALL_MODE="${INSTALL_MODE},webrtcvad"
            shift ;;
        --local-nlp)
            INSTALL_MODE="${INSTALL_MODE},local-nlp"
            shift ;;
        --webui|--gui)
            INSTALL_MODE="${INSTALL_MODE},webui"
            shift ;;
        --diarization)
            INSTALL_MODE="${INSTALL_MODE},diarization"
            shift ;;
        --all)
            INSTALL_MODE="all"
            shift ;;
        --dev)
            INSTALL_MODE="dev"
            shift ;;
        --no-venv)
            CREATE_VENV=false
            shift ;;
        --venv)
            VENV_NAME="$2"
            shift 2 ;;
        --download-models)
            DOWNLOAD_MODELS=true
            shift ;;
        --download-asr)
            DOWNLOAD_ASR_MODEL="$2"
            DOWNLOAD_MODELS=true
            shift 2 ;;
        --download-separator)
            DOWNLOAD_SEPARATOR_ENG="$2"
            DOWNLOAD_MODELS=true
            shift 2 ;;
        --no-model-dl)
            SKIP_MODEL_DL=true
            shift ;;
        --yes|-y)
            YES_MODE=true
            shift ;;
        --help|-h)
            show_help ;;
        *)
            error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# 检测操作系统
# ------------------------------------------------------------------
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

OS=$(detect_os)

# 检测 Linux 发行版
detect_linux_distro() {
    if [ "$OS" != "linux" ]; then
        echo "unknown"
        return
    fi
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "${ID:-unknown}"
    else
        echo "unknown"
    fi
}

LINUX_DISTRO=$(detect_linux_distro)

# 检测系统架构
detect_arch() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) echo "$arch" ;;
    esac
}

ARCH=$(detect_arch)

# ------------------------------------------------------------------
# Step 1: 环境检测
# ------------------------------------------------------------------
banner

step "Step 1/6 — 检测系统环境"
info "操作系统: $OS ($ARCH)"
if [ "$OS" = "linux" ]; then
    info "Linux 发行版: $LINUX_DISTRO"
fi
info "Python: $(python3 --version 2>/dev/null || echo '未找到')"
info "Shell: $SHELL"

# 检查 Python 版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    error "需要 Python >= 3.10，当前版本: $PYTHON_VERSION"
    error "请先安装 Python 3.10+ 后再运行此脚本"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt install python3.12 python3.12-venv python3-pip"
    echo ""
    echo "macOS:"
    echo "  brew install python@3.12"
    exit 1
fi
ok "Python $PYTHON_VERSION"

# ------------------------------------------------------------------
# Step 2: GPU 检测与 PyTorch 策略
# ------------------------------------------------------------------
step "Step 2/6 — GPU 检测与推理设备"

HAS_GPU=false
GPU_TYPE=""         # cuda / mps / rocm / none
GPU_NAME=""
GPU_MEM_MB=0
CUDA_VER=""
CUDA_MAJOR=""
TORCH_INDEX_URL=""  # PyTorch 安装源

# --- 2a. NVIDIA CUDA 检测 ---
detect_nvidia_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        return 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -z "$GPU_NAME" ]; then
        return 1
    fi

    # 驱动版本
    local driver_ver
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "未知")

    # 显存 (MiB)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | grep -oE '[0-9]+' || echo "0")

    # CUDA 版本 (从 nvidia-smi 获取驱动支持的最高 CUDA 版本)
    CUDA_VER=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
    local cuda_driver_ver
    cuda_driver_ver=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | grep -oE '[0-9]+\.[0-9]+' || echo "")

    # 如果有指定 CUDA 版本则用指定的
    if [ -n "$CUDA_VERSION" ]; then
        CUDA_VER="$CUDA_VERSION"
    elif [ -n "$cuda_driver_ver" ]; then
        CUDA_VER="$cuda_driver_ver"
    fi

    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)

    ok "检测到 NVIDIA GPU: $GPU_NAME"
    info "  驱动版本: $driver_ver"
    info "  CUDA 版本: $CUDA_VER"
    info "  显存: ${GPU_MEM_MB} MiB"
    info "  计算能力: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo '未知')"

    GPU_TYPE="cuda"
    HAS_GPU=true

    # 选择 PyTorch CUDA 索引
    # PyTorch 支持的 CUDA 版本: cu118, cu121, cu124, cu126
    if [ -z "$CUDA_VERSION" ]; then
        # 自动选择: 使用 nvidia-smi 报告的 CUDA 版本
        local cuda_major minor
        cuda_major=$(echo "$cuda_driver_ver" | cut -d. -f1)
        minor=$(echo "$cuda_driver_ver" | cut -d. -f2)

        if [ "$cuda_major" -ge 12 ] && [ "$minor" -ge 6 ]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
        elif [ "$cuda_major" -ge 12 ] && [ "$minor" -ge 4 ]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
        elif [ "$cuda_major" -ge 12 ]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
        elif [ "$cuda_major" -ge 11 ]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        else
            warn "CUDA 版本过旧 (${cuda_driver_ver})，将使用 CPU 版 PyTorch"
            TORCH_INDEX_URL=""
        fi
    else
        # 用户指定了 CUDA 版本
        local cu_tag
        cu_tag="cu$(echo "$CUDA_VERSION" | cut -d. -f1)$(echo "$CUDA_VERSION" | cut -d. -f2)"
        TORCH_INDEX_URL="https://download.pytorch.org/whl/${cu_tag}"
    fi

    if [ -n "$TORCH_INDEX_URL" ]; then
        info "PyTorch 安装源: $TORCH_INDEX_URL"
    fi

    # 推荐模型大小
    if [ "$GPU_MEM_MB" -ge 8000 ]; then
        info "推荐 ASR 模型: large-v3 (显存充足)"
        info "推荐计算精度: float16"
    elif [ "$GPU_MEM_MB" -ge 4000 ]; then
        info "推荐 ASR 模型: medium"
        info "推荐计算精度: int8_float16"
    else
        info "推荐 ASR 模型: small (显存受限)"
        info "推荐计算精度: int8"
    fi

    return 0
}

# --- 2b. Apple Silicon (MPS) 检测 ---
detect_apple_silicon() {
    if [ "$OS" != "macos" ]; then
        return 1
    fi

    # 检查是否为 Apple Silicon
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        local chip_name
        chip_name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        ok "检测到 Apple Silicon: $chip_name"
        info "  GPU 后端: MPS (Metal Performance Shaders)"
        info "  统一内存架构，可运行 medium/large-v3 模型"

        GPU_TYPE="mps"
        HAS_GPU=true
        GPU_NAME="$chip_name"
        GPU_MEM_MB=0  # 统一内存

        # macOS 使用 MPS 后端的 PyTorch (从标准 PyPI 安装即可)
        TORCH_INDEX_URL=""

        info "推荐 ASR 模型: medium 或 large-v3"
        info "推荐计算精度: float16"
        return 0
    fi

    return 1
}

# --- 2c. AMD ROCm 检测 ---
detect_amd_rocm() {
    if [ "$OS" != "linux" ]; then
        return 1
    fi

    # 检查 ROCm
    if command -v rocminfo &> /dev/null; then
        local rocm_ver
        rocm_ver=$(rocminfo 2>/dev/null | grep "ROCm" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "未知")

        GPU_NAME=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -1 | sed 's/.*:[[:space:]]*//' || echo "AMD GPU")
        ok "检测到 AMD GPU (ROCm): $GPU_NAME"
        info "  ROCm 版本: $rocm_ver"

        # ROCm 对应的 PyTorch 索引
        if [ -n "$rocm_ver" ]; then
            local rocm_major
            rocm_major=$(echo "$rocm_ver" | cut -d. -f1)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm${rocm_major}"
            info "  PyTorch 安装源: $TORCH_INDEX_URL"
        fi

        GPU_TYPE="rocm"
        HAS_GPU=true
        info "推荐 ASR 模型: medium 或 small"
        info "推荐计算精度: float16"
        return 0
    fi

    # 也检查 amdgpu 驱动
    if ls /dev/dri/render* &>/dev/null && (lspci 2>/dev/null | grep -qi "AMD.*VGA\|AMD.*Graphics"); then
        warn "检测到 AMD GPU，但未安装 ROCm 运行时"
        info "安装 ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        info "当前将使用 CPU 推理模式"
    fi

    return 1
}

# 执行 GPU 检测
if [ "$GPU_MODE" = true ]; then
    info "自动检测可用 GPU..."

    if detect_nvidia_gpu; then
        : # NVIDIA GPU 已检测
    elif detect_apple_silicon; then
        : # Apple Silicon 已检测
    elif detect_amd_rocm; then
        : # AMD ROCm 已检测
    else
        warn "未检测到支持的 GPU，使用 CPU 推理模式"
        GPU_MODE=false
        GPU_TYPE="cpu"
    fi
else
    info "CPU 模式 (通过 --cpu 手动指定)"
    # 静默检测，仅显示信息
    if detect_nvidia_gpu 2>/dev/null; then
        info "检测到 NVIDIA GPU ($GPU_NAME)，移除 --cpu 可启用加速"
    elif detect_apple_silicon 2>/dev/null; then
        info "检测到 Apple Silicon，移除 --cpu 可启用 MPS 加速"
    fi
    # 确保 CPU 模式
    HAS_GPU=false
    GPU_TYPE="cpu"
    TORCH_INDEX_URL=""
fi

# ------------------------------------------------------------------
# Step 3: 安装系统依赖
# ------------------------------------------------------------------
step "Step 3/6 — 安装系统依赖"

install_system_deps_linux() {
    local packages=("$@")
    if command -v apt &> /dev/null; then
        info "使用 apt 安装系统依赖..."
        sudo apt update -qq
        sudo apt install -y -qq "${packages[@]}"
    elif command -v dnf &> /dev/null; then
        info "使用 dnf 安装系统依赖..."
        sudo dnf install -y "${packages[@]}"
    elif command -v pacman &> /dev/null; then
        info "使用 pacman 安装系统依赖..."
        sudo pacman -S --noconfirm "${packages[@]}"
    elif command -v apk &> /dev/null; then
        info "使用 apk 安装系统依赖..."
        sudo apk add "${packages[@]}"
    else
        warn "未检测到支持的包管理器，请手动安装: ${packages[*]}"
        return 1
    fi
}

install_system_deps_macos() {
    if command -v brew &> /dev/null; then
        info "使用 Homebrew 安装系统依赖..."
        brew install "$@"
    else
        warn "未检测到 Homebrew，请手动安装: $*"
        warn "安装 Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        return 1
    fi
}

# 3a. ffmpeg
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    ok "ffmpeg 已安装 ($FFMPEG_VER)"
else
    warn "ffmpeg 未安装，正在尝试自动安装..."
    case "$OS" in
        linux)
            install_system_deps_linux ffmpeg && ok "ffmpeg 安装完成" || warn "ffmpeg 自动安装失败"
            ;;
        macos)
            install_system_deps_macos ffmpeg && ok "ffmpeg 安装完成" || warn "ffmpeg 自动安装失败"
            ;;
        *)
            warn "未知操作系统，请手动安装 ffmpeg (>= 4.4)"
            ;;
    esac
fi

# 3b. Python 开发工具 (Linux)
if [ "$OS" = "linux" ]; then
    info "检查 Python 开发工具..."
    HAS_VENV=true
    HAS_PIP=true
    python3 -m venv --help &>/dev/null || HAS_VENV=false
    python3 -m pip --version &>/dev/null || HAS_PIP=false

    if [ "$HAS_VENV" = false ] || [ "$HAS_PIP" = false ]; then
        PIP_PKG="python3-pip"
        VENV_PKG="python3-venv"

        if [ "$LINUX_DISTRO" = "ubuntu" ] || [ "$LINUX_DISTRO" = "debian" ]; then
            PYTHON_DOT_VER="python${PYTHON_VERSION}"
            VENV_PKG="${PYTHON_DOT_VER}-venv"
        fi

        INSTALL_PKGS=""
        [ "$HAS_PIP" = false ] && INSTALL_PKGS="$INSTALL_PKGS $PIP_PKG"
        [ "$HAS_VENV" = false ] && INSTALL_PKGS="$INSTALL_PKGS $VENV_PKG"

        if [ -n "$INSTALL_PKGS" ]; then
            # shellcheck disable=SC2086
            install_system_deps_linux $INSTALL_PKGS && ok "Python 开发工具安装完成" || warn "Python 开发工具安装失败"
        fi
    else
        ok "Python 开发工具 (venv, pip) 已就绪"
    fi
fi

# 3c. libsndfile (Open-Unmix + soundfile 依赖)
_has_libsndfile() {
    # 多种方式检测 libsndfile 是否已安装
    # 方式1: ldconfig 缓存
    if ldconfig -p 2>/dev/null | grep -q libsndfile; then
        return 0
    fi
    # 方式2: 直接查找 .so 文件
    if find /usr/lib /usr/local/lib -name 'libsndfile.so*' 2>/dev/null | grep -q libsndfile; then
        return 0
    fi
    # 方式3: dpkg (Debian/Ubuntu)
    if command -v dpkg &>/dev/null && dpkg -l libsndfile1 2>/dev/null | grep -q '^ii'; then
        return 0
    fi
    # 方式4: rpm (Fedora/RHEL)
    if command -v rpm &>/dev/null && rpm -q libsndfile 2>/dev/null; then
        return 0
    fi
    return 1
}

case "$OS" in
    linux)
        if _has_libsndfile; then
            ok "libsndfile 已安装"
        else
            info "安装 libsndfile (音频处理依赖)..."
            install_system_deps_linux libsndfile1 2>/dev/null || \
            install_system_deps_linux libsndfile 2>/dev/null || \
            warn "libsndfile 安装失败，Open-Unmix 引擎可能不可用；可手动执行: sudo apt install libsndfile1"
        fi
        ;;
    macos)
        if brew list libsndfile &>/dev/null 2>&1; then
            ok "libsndfile 已安装"
        else
            info "安装 libsndfile..."
            install_system_deps_macos libsndfile && ok "libsndfile 安装完成" || warn "libsndfile 安装失败"
        fi
        ;;
esac

# 3d. GPU 相关系统库 (仅 GPU 模式)
if [ "$HAS_GPU" = true ] && [ "$GPU_TYPE" = "cuda" ]; then
    info "检查 CUDA 相关系统库..."
    if ldconfig -p 2>/dev/null | grep -q libcudnn; then
        ok "cuDNN 已安装"
    else
        info "未检测到 cuDNN (非必需，faster-whisper 使用 CTranslate2 不需要 cuDNN)"
    fi
fi

# ------------------------------------------------------------------
# Step 4: 创建虚拟环境
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

if [ "$CREATE_VENV" = true ]; then
    step "Step 4/6 — 创建 Python 虚拟环境"

    cd "$PROJECT_DIR"

    if [ -d "$VENV_NAME" ]; then
        info "虚拟环境已存在: $VENV_NAME/"
        if [ "$YES_MODE" = true ]; then
            info "非交互模式，使用现有虚拟环境"
        else
            read -p "  是否重新创建? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$VENV_NAME"
                info "已删除旧的虚拟环境"
            else
                info "使用现有虚拟环境"
            fi
        fi
    fi

    if [ ! -d "$VENV_NAME" ]; then
        python3 -m venv "$VENV_NAME"
        ok "虚拟环境已创建: $VENV_NAME/"
    fi

    # 激活虚拟环境
    source "$VENV_NAME/bin/activate"
    ok "已激活虚拟环境"

    # 升级 pip + setuptools + wheel
    info "升级 pip/setuptools/wheel..."
    pip install --upgrade pip setuptools wheel -q 2>/dev/null || true
    ok "pip 已升级 ($(pip --version | awk '{print $2}'))"
fi

# ------------------------------------------------------------------
# Step 5: 安装 Python 依赖
# ------------------------------------------------------------------
step "Step 5/6 — 安装 Python 依赖"

cd "$PROJECT_DIR"

# 安全的 pip install 包装: 过滤冗余输出，但不吞没错误
_safe_pip_install() {
    local desc="$1"    # 安装描述 (用于日志)
    shift
    local pip_status=0
    "$@" 2>&1 | grep -v "^Requirement already satisfied" || { pip_status="${PIPESTATUS[0]}"; true; }
    if [ "$pip_status" -ne 0 ]; then
        error "$desc 失败 (exit code: $pip_status)，请检查上述错误信息"
        exit 1
    fi
}

install_pip_deps() {
    local extras="$1"
    info "安装模式: $extras"
    _safe_pip_install "pip install -e .[$extras]" pip install -e ".[$extras]"
}

# --- GPU 模式: 先安装 CUDA 版 PyTorch ---
if [ "$HAS_GPU" = true ] && [ "$SKIP_TORCH" = false ]; then
    if [ -n "$TORCH_INDEX_URL" ] && [ "$GPU_TYPE" = "cuda" ]; then
        info "安装 CUDA 版 PyTorch (${TORCH_INDEX_URL})..."
        _safe_pip_install "CUDA PyTorch" pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
        ok "CUDA 版 PyTorch 安装完成"
    elif [ "$GPU_TYPE" = "rocm" ] && [ -n "$TORCH_INDEX_URL" ]; then
        info "安装 ROCm 版 PyTorch (${TORCH_INDEX_URL})..."
        _safe_pip_install "ROCm PyTorch" pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
        ok "ROCm 版 PyTorch 安装完成"
    elif [ "$GPU_TYPE" = "mps" ]; then
        info "Apple Silicon: 使用标准 PyTorch (MPS 后端已内置)"
        _safe_pip_install "MPS PyTorch" pip install torch torchaudio
        ok "PyTorch (MPS) 安装完成"
    fi
elif [ "$SKIP_TORCH" = true ]; then
    info "跳过 PyTorch 安装 (--no-torch)"
elif [ "$CPU_EXPLICIT" = true ]; then
    # --cpu 显式: 安装 CPU 版 PyTorch + Silero VAD
    info "安装 CPU 版 PyTorch (--cpu 模式)..."
    _safe_pip_install "CPU PyTorch" pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ok "CPU 版 PyTorch 安装完成"
else
    # 自动检测无 GPU: 跳过 PyTorch (使用 webrtcvad, 最轻量)
    info "无 GPU 可用, 跳过 PyTorch 安装 (将使用 WebRTC VAD, ~100MB)"
fi

# --- 安装 vocal-subtitle ---
# 根据 GPU 检测结果自动选择最优 VAD + 加速方案:
#   GPU 可用 → silero-vad (GPU 加速) + gpu (CTranslate2)
#   CPU/--no-torch → webrtcvad (纯 CPU, 轻量, 无需 torch)
_build_base_extras() {
    # 根据 GPU/CPU/torch 可用性自动选择最优方案:
    #   GPU + torch   → silero-vad (GPU) + gpu (CTranslate2 CUDA)    [最强]
    #   CPU + torch   → silero-vad (CPU PyTorch, --cpu 模式)         [中等]
    #   无 torch       → webrtcvad (纯 CPU, 轻量, 默认/--no-torch)    [最轻]
    local extras="faster-whisper,funasr"
    if [ "$SKIP_TORCH" = true ]; then
        extras="${extras},webrtcvad"
        info "VAD: WebRTC VAD (纯 CPU, 轻量, --no-torch)" >&2
    elif [ "$HAS_GPU" = true ]; then
        extras="${extras},silero-vad,gpu"
        info "VAD: Silero VAD (GPU 加速) + CTranslate2 (CUDA)" >&2
    elif [ "$CPU_EXPLICIT" = true ]; then
        # --cpu 显式: 有 CPU 版 PyTorch → 用 silero-vad
        extras="${extras},silero-vad"
        info "VAD: Silero VAD (CPU, --cpu 模式)" >&2
    else
        # 自动检测无 GPU → 不装 torch, 用 webrtcvad 最轻
        extras="${extras},webrtcvad"
        info "VAD: WebRTC VAD (纯 CPU, 轻量, 无 GPU)" >&2
    fi
    echo "$extras"
}

case "$INSTALL_MODE" in
    base)
        BASE_EXTRAS=$(_build_base_extras)
        install_pip_deps "$BASE_EXTRAS"
        ;;
    all)
        # 全量安装 (GPU 模式用 silero-vad, CPU 用 webrtcvad)
        if [ "$HAS_GPU" = true ] && [ "$SKIP_TORCH" = false ]; then
            install_pip_deps "all"
        else
            install_pip_deps "gpu,llm,local-nlp,uvr,spleeter,openunmix,webrtcvad,faster-whisper,funasr,webui,diarization"
        fi
        ;;
    dev)
        BASE_EXTRAS=$(_build_base_extras)
        install_pip_deps "dev,${BASE_EXTRAS}"
        ;;
    *)
        # 组合模式: 自动补充 GPU 相关 extras
        # 始终包含 faster-whisper (ASR 核心) 作为基础
        EXTRAS="faster-whisper,funasr"
        # GPU 加速: 检测到 GPU 且未显式指定 webrtcvad → 自动启用 silero-vad
        if [ "$HAS_GPU" = true ] && [ "$SKIP_TORCH" = false ]; then
            if [[ "$INSTALL_MODE" != *webrtcvad* ]]; then
                EXTRAS="${EXTRAS},silero-vad,gpu"
                info "GPU 模式: 自动启用 Silero VAD + CTranslate2 加速"
            fi
        fi
        [[ "$INSTALL_MODE" == *gpu* ]] && EXTRAS="${EXTRAS},gpu"
        [[ "$INSTALL_MODE" == *llm* ]] && EXTRAS="${EXTRAS},llm"
        [[ "$INSTALL_MODE" == *uvr* ]] && EXTRAS="${EXTRAS},uvr"
        [[ "$INSTALL_MODE" == *spleeter* ]] && EXTRAS="${EXTRAS},spleeter"
        [[ "$INSTALL_MODE" == *openunmix* ]] && EXTRAS="${EXTRAS},openunmix"
        [[ "$INSTALL_MODE" == *webrtcvad* ]] && EXTRAS="${EXTRAS},webrtcvad"
        [[ "$INSTALL_MODE" == *local-nlp* ]] && EXTRAS="${EXTRAS},local-nlp"
        [[ "$INSTALL_MODE" == *webui* ]] && EXTRAS="${EXTRAS},webui"
        [[ "$INSTALL_MODE" == *diarization* ]] && EXTRAS="${EXTRAS},diarization"
        install_pip_deps "$EXTRAS"
        ;;
esac

ok "Python 依赖安装完成"

# ------------------------------------------------------------------
# Step 6: 验证安装
# ------------------------------------------------------------------
step "Step 6/6 — 验证安装"

VERIFY_OK=true

# 6a. 包安装验证
PIP_LIST=$(pip list 2>/dev/null | grep vocal-subtitle || true)
if [ -n "$PIP_LIST" ]; then
    ok "vocal-subtitle 已成功安装"
else
    error "vocal-subtitle 安装验证失败，请检查错误日志"
    VERIFY_OK=false
fi

# 6b. CLI 可用性验证
if command -v vocal-subtitle &> /dev/null; then
    CLI_VER=$(vocal-subtitle --version 2>&1 || echo "0.1.0")
    ok "CLI 命令可用: vocal-subtitle ($CLI_VER)"
else
    warn "CLI 命令不在 PATH 中，确认虚拟环境已激活"
fi

# 6c. GUI 可用性验证
if [[ "$INSTALL_MODE" == *webui* ]] || [[ "$INSTALL_MODE" == "all" ]]; then
    if python3 -c "from vocal_subtitle.webui.app import create_app" 2>/dev/null; then
        ok "Web GUI 模块可用: vocal-subtitle-gui"
    else
        warn "Web GUI 模块验证失败，请检查 fastapi/uvicorn 安装"
    fi
fi

# 6d. 关键模块导入验证
info "验证关键模块导入..."
python3 -c "
import sys
errors = []
modules = [
    ('numpy', 'numpy'),

    ('pysubs2', 'pysubs2'),
    ('yaml', 'pyyaml'),
    ('click', 'click'),
    ('tqdm', 'tqdm'),
    ('structlog', 'structlog'),
    ('diskcache', 'diskcache'),
    ('vocal_subtitle', 'vocal_subtitle'),
]
for mod, pkg in modules:
    try:
        __import__(mod)
        print(f'  ✓ {pkg}')
    except ImportError as e:
        errors.append(pkg)
        print(f'  ✗ {pkg} — 未安装 ({e})')

# Optional modules
optional = [
    ('faster_whisper', 'faster-whisper', 'ASR 引擎'),
    ('spleeter', 'spleeter', '分离引擎'),
    ('audio_separator', 'audio-separator', 'UVR 分离引擎'),
    ('openunmix', 'openunmix', 'Open-Unmix 分离引擎'),
    ('soundfile', 'soundfile', 'Open-Unmix 音频 I/O'),
    ('openai', 'openai', 'LLM 客户端'),
    ('tenacity', 'tenacity', 'LLM 重试'),
    ('json_repair', 'json-repair', 'JSON 修复'),
    ('fastapi', 'fastapi', 'Web GUI'),
    ('uvicorn', 'uvicorn', 'Web 服务器'),
    ('sentence_transformers', 'sentence-transformers', '本地 NLP 语义合并'),
    ('GPUtil', 'GPUtil', 'GPU 显存监控'),
    ('webrtcvad', 'webrtcvad', 'WebRTC VAD'),
    ('librosa', 'librosa', '说话人分离'),
    ('sklearn', 'scikit-learn', '说话人聚类'),
]
for mod, pkg, desc in optional:
    try:
        __import__(mod)
        print(f'  ✓ {pkg} ({desc})')
    except ImportError:
        print(f'  - {pkg} ({desc}) — 可选，未安装')

if errors:
    print(f'\n错误: {len(errors)} 个核心模块缺失')
    sys.exit(1)
print('\n所有核心模块验证通过 ✓')
" || VERIFY_OK=false

# 6e. GPU 可用性验证
if [ "$HAS_GPU" = true ]; then
    info "验证 GPU 推理可用性..."
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ CUDA 可用 (PyTorch {torch.__version__})')
    print(f'    设备数: {torch.cuda.device_count()}')
    print(f'    当前设备: {torch.cuda.get_device_name(0)}')
    print(f'    显存: {torch.cuda.get_device_properties(0).total_memory // (1024*1024)} MiB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  ✓ MPS 可用 (PyTorch {torch.__version__})')
    print('    Apple Silicon GPU 后端已就绪')
else:
    print(f'  ⚠ GPU 不可用 (PyTorch {torch.__version__})')
    print('    请检查 CUDA/ROCm/MPS 安装')
" || warn "GPU 验证失败，请检查 PyTorch 安装"
elif command -v nvidia-smi &> /dev/null; then
    warn "检测到 NVIDIA GPU 但未安装 CUDA 版 PyTorch"
    info "使用 --gpu 重新安装以启用 GPU 加速"
fi

# 6f. ffmpeg 可用性
if command -v ffmpeg &> /dev/null; then
    ok "ffmpeg 可用"
else
    warn "ffmpeg 不可用，音频格式转换功能受限"
fi

# 显示关键依赖版本
echo ""
info "已安装的关键依赖:"
pip list 2>/dev/null | grep -iE "numpy|torch|faster-whisper|ctranslate2|pydub|pysubs2|click|spleeter|audio-separator|openunmix|soundfile|openai|sentence-transformers|diskcache|structlog|GPUtil|tqdm|PyYAML|fastapi|uvicorn|librosa|scikit-learn|scipy|webrtcvad|websockets" | while read -r line; do
    echo "  - $line"
done || true

# ------------------------------------------------------------------
# 模型预下载 (可选)
# ------------------------------------------------------------------
if [ "$DOWNLOAD_MODELS" = true ] && [ "$SKIP_MODEL_DL" = false ]; then
    step "额外: 预下载模型文件"

    MODEL_CACHE_DIR="${HOME}/.cache/vocal-subtitle"
    mkdir -p "$MODEL_CACHE_DIR"

    download_faster_whisper_model() {
        local model_name="${1:-large-v3}"
        info "下载 faster-whisper 模型: $model_name"

        python3 -c "
try:
    from faster_whisper import WhisperModel
    print(f'  正在下载 {model_name} 模型...')
    model = WhisperModel('$model_name', device='cpu', compute_type='int8',
                         download_root='$MODEL_CACHE_DIR/faster-whisper')
    print(f'  模型 $model_name 下载完成')
except ImportError:
    print('  faster-whisper 未安装，跳过模型下载')
except Exception as e:
    print(f'  模型下载失败: {e}')
        " 2>&1 || warn "faster-whisper 模型 $model_name 下载失败"
    }

    download_spleeter_model() {
        info "下载 Spleeter 模型 (2stems)..."

        python3 -c "
try:
    from spleeter.separator import Separator
    print('  正在下载 Spleeter 2stems 模型...')
    model = Separator('spleeter:2stems')
    print('  Spleeter 模型下载完成')
except ImportError:
    print('  spleeter 未安装，跳过模型下载')
except Exception as e:
    print(f'  模型下载失败: {e}')
        " 2>&1 || warn "Spleeter 模型下载失败"
    }

    download_uvr_model() {
        local model_name="${1:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}"
        local model_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/${model_name}"
        local model_dir="${PROJECT_DIR}/cache/models"
        local model_path="${model_dir}/${model_name}"

        info "下载 UVR 模型: $model_name"

        if [ -f "$model_path" ]; then
            local size
            size=$(du -h "$model_path" 2>/dev/null | cut -f1)
            ok "UVR 模型已存在: $model_path ($size)"
            return 0
        fi

        mkdir -p "$model_dir"

        if command -v curl &> /dev/null; then
            info "从 GitHub 下载 UVR 模型 (约 610MB)..."
            if curl -# -L -o "$model_path" "$model_url"; then
                ok "UVR 模型下载完成: $model_path"
            else
                warn "UVR 模型下载失败，请手动下载到: $model_path"
                warn "下载地址: $model_url"
                rm -f "$model_path"
                return 1
            fi
        elif command -v wget &> /dev/null; then
            info "从 GitHub 下载 UVR 模型 (约 610MB)..."
            if wget -q --show-progress -O "$model_path" "$model_url"; then
                ok "UVR 模型下载完成: $model_path"
            else
                warn "UVR 模型下载失败，请手动下载到: $model_path"
                warn "下载地址: $model_url"
                rm -f "$model_path"
                return 1
            fi
        else
            warn "未找到 curl 或 wget，无法下载 UVR 模型"
            info "请手动下载: $model_url → $model_path"
            return 1
        fi
    }

    download_silero_vad_model() {
        info "下载 Silero VAD 模型..."

        python3 -c "
try:
    import torch
    print('  正在下载 Silero VAD 模型...')
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
    )
    print('  Silero VAD 模型下载完成')
except ImportError:
    print('  torch 未安装，跳过模型下载')
except Exception as e:
    print(f'  模型下载失败: {e}')
        " 2>&1 || warn "Silero VAD 模型下载失败"
    }

    if [ -n "$DOWNLOAD_ASR_MODEL" ]; then
        download_faster_whisper_model "$DOWNLOAD_ASR_MODEL"
    elif [ "$DOWNLOAD_MODELS" = true ]; then
        if python3 -c "import faster_whisper" 2>/dev/null; then
            download_faster_whisper_model "large-v3"
        fi
        if python3 -c "import audio_separator" 2>/dev/null; then
            download_uvr_model
        fi
        if python3 -c "import spleeter" 2>/dev/null; then
            download_spleeter_model
        fi
        if python3 -c "import torch" 2>/dev/null; then
            download_silero_vad_model
        fi
    fi

    # 单独指定分离引擎模型下载 (--download-separator uvr|spleeter)
    if [ -n "$DOWNLOAD_SEPARATOR_ENG" ]; then
        case "$DOWNLOAD_SEPARATOR_ENG" in
            uvr)
                download_uvr_model
                ;;
            spleeter)
                download_spleeter_model
                ;;
            *)
                warn "未知分离引擎: $DOWNLOAD_SEPARATOR_ENG (支持: uvr, spleeter)"
                ;;
        esac
    fi

    info "模型缓存目录: $MODEL_CACHE_DIR"
    ls -lh "$MODEL_CACHE_DIR" 2>/dev/null || info "(目录为空，模型将在首次使用时自动下载)"
else
    info "模型将在首次运行时自动下载到 ~/.cache/"
fi

# ------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                 ${CYAN}✓ 安装完成!${NC}                       ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$CREATE_VENV" = true ]; then
    echo -e "激活虚拟环境:"
    echo -e "  ${CYAN}source $VENV_NAME/bin/activate${NC}"
    echo ""
fi

echo -e "快速开始:"
echo -e "  ${CYAN}vocal-subtitle --help${NC}"
echo -e "  ${CYAN}vocal-subtitle profiles${NC}"
echo -e "  ${CYAN}vocal-subtitle info${NC}"
echo -e "  ${CYAN}vocal-subtitle run input.mp3 -o output.srt${NC}"
echo -e "  ${CYAN}vocal-subtitle run input.mp3 --profile podcast --language zh${NC}"
echo ""

if [[ "$INSTALL_MODE" == *webui* ]] || [[ "$INSTALL_MODE" == "all" ]]; then
    echo -e "${YELLOW}Web GUI 图形界面:${NC}"
    echo -e "  ${CYAN}vocal-subtitle-gui${NC}              # 启动 Web GUI (自动打开浏览器)"
    echo -e "  ${CYAN}vocal-subtitle-gui --no-browser${NC}  # 仅启动服务，不打开浏览器"
    echo -e "  ${CYAN}python main_gui.py${NC}              # 等价入口"
    echo ""
fi

echo -e "批量处理:"
echo -e "  ${CYAN}vocal-subtitle batch ./inputs/ -o ./outputs/${NC}"
echo ""

echo -e "Python API:"
echo -e "  ${CYAN}python -c \"from vocal_subtitle import Pipeline; print('OK')\"${NC}"
echo ""

echo -e "运行测试:"
echo -e "  ${CYAN}pytest tests/ -v${NC}"
echo ""

if [ "$HAS_GPU" = true ]; then
    echo -e "${YELLOW}GPU 加速:${NC}"
    echo -e "  ${CYAN}vocal-subtitle run input.mp3 --profile podcast --device cuda${NC}"
else
    echo -e "${YELLOW}提示:${NC} 当前为 CPU 推理模式，较慢。"
    echo -e "  - 移除 --cpu 参数可自动检测并启用 GPU 加速"
fi

echo ""
echo -e "模型管理:"
echo -e "  ${CYAN}vocal-subtitle download-models --all${NC}   # 预下载所有模型"
echo -e "  ${CYAN}vocal-subtitle download-models --asr-model large-v3${NC}"
echo ""

echo -e "更多信息: ${CYAN}README.md${NC} | ${CYAN}docs/${NC}"
echo ""
