#!/usr/bin/env bash
# ============================================================================
# Vocal Subtitle — 说话人嵌入模型下载脚本
# ============================================================================
# 下载 speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN) 到本地缓存。
# 模型协议: Apache 2.0，无需 HuggingFace token。
#
# 用法:
#   bash scripts/download_speaker_model.sh
#   bash scripts/download_speaker_model.sh --mirror         # 使用 hf-mirror.com 镜像
#   bash scripts/download_speaker_model.sh --dir /custom/path
#   bash scripts/download_speaker_model.sh --force           # 强制重新下载
#
# 模型文件说明 (hyperparams.yaml 中 Pretrainer 引用的 5 个文件):
#   - hyperparams.yaml        模型架构 & 参数映射  (~2KB,  必需)
#   - embedding_model.ckpt    ECAPA-TDNN 权重       (~80MB, LFS, 必需)
#   - classifier.ckpt         分类器权重             (~5MB,  LFS, 必需)
#   - mean_var_norm_emb.ckpt  嵌入层归一化参数       (~2KB,  必需)
#   - label_encoder.txt       说话人标签映射         (~128KB,必需)
#   - label_encoder.ckpt  →   label_encoder.txt      (symlink, SpeechBrain 兼容)
# ============================================================================

set -euo pipefail

# ---- 配置 ----
MODEL_REPO="speechbrain/spkrec-ecapa-voxceleb"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_CACHE_DIR="$PROJECT_DIR/cache/speaker_models"
CACHE_DIR="$DEFAULT_CACHE_DIR"
USE_MIRROR=false
FORCE=false
QUIET=false

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---- 辅助函数 ----
info()  { $QUIET || echo -e "${CYAN}[..]${NC} $*"; }
ok()    { $QUIET || echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*" >&2; }
error() { echo -e "${RED}[✗]${NC} $*" >&2; }

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror)
            USE_MIRROR=true
            shift
            ;;
        --dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --quiet|-q)
            QUIET=true
            shift
            ;;
        -h|--help)
            echo "用法: bash scripts/download_speaker_model.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --mirror      使用 hf-mirror.com 镜像加速下载"
            echo "  --dir PATH    指定模型存放目录 (默认: cache/speaker_models/)"
            echo "  --force       强制重新下载已存在的文件"
            echo "  -q, --quiet   安静模式，仅输出错误"
            echo "  -h, --help    显示此帮助"
            exit 0
            ;;
        *)
            error "未知参数: $1"
            exit 1
            ;;
    esac
done

TARGET_DIR="$CACHE_DIR/${MODEL_REPO//\//_}"

# ---- 选择下载源 ----
if $USE_MIRROR; then
    HF_BASE="https://hf-mirror.com/$MODEL_REPO/resolve/main"
else
    HF_BASE="https://huggingface.co/$MODEL_REPO/resolve/main"
fi

# ---- 必要文件清单 ----
# 格式: "文件名|最小字节数|sha256(可选)"
# 这 5 个文件对应 hyperparams.yaml 中 Pretrainer.loadables 的 paths 字段。
declare -a FILE_LIST=(
    "hyperparams.yaml|100"
    "embedding_model.ckpt|1000000"
    "classifier.ckpt|1000000"
    "mean_var_norm_emb.ckpt|1000"
    "label_encoder.txt|10000"
)

# ---- 检测下载工具 ----
TOOL=""
if command -v wget &>/dev/null; then
    TOOL="wget"
elif command -v curl &>/dev/null; then
    TOOL="curl"
else
    error "需要 wget 或 curl 才能下载模型。请安装其中之一:"
    error "  sudo apt install wget"
    exit 1
fi

# ---- Banner ----
if ! $QUIET; then
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Vocal Subtitle — 说话人嵌入模型下载                      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  模型: ${YELLOW}$MODEL_REPO${NC}"
    echo -e "  协议: ${GREEN}Apache 2.0 (无需 token)${NC}"
    echo -e "  维度: ${GREEN}192${NC}"
    echo -e "  目标: $TARGET_DIR"
    if $USE_MIRROR; then
        echo -e "  源:   ${YELLOW}hf-mirror.com (镜像)${NC}"
    else
        echo -e "  源:   huggingface.co"
    fi
    if $FORCE; then
        echo -e "  模式: ${YELLOW}强制重新下载${NC}"
    fi
    echo ""
fi

# ---- 创建目标目录 ----
mkdir -p "$TARGET_DIR"

# ---- 下载单个文件 ----
download_file() {
    local filename="$1"
    local min_size="$2"
    local url="${HF_BASE}/${filename}"
    local dest="$TARGET_DIR/$filename"
    local tmp_dest="${dest}.tmp"

    # 检查是否已存在有效文件
    if ! $FORCE && [[ -f "$dest" ]]; then
        local existing_size
        existing_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        if [[ "$existing_size" -ge "$min_size" ]]; then
            local size_kb=$((existing_size / 1024))
            if [[ $size_kb -gt 1024 ]]; then
                local size_mb=$((size_kb / 1024))
                ok "${filename} (已存在, ${size_mb}MB)"
            else
                ok "${filename} (已存在, ${size_kb}KB)"
            fi
            return 0
        else
            warn "${filename} 已存在但大小异常 (${existing_size}B < ${min_size}B)，重新下载..."
        fi
    fi

    info "下载 ${filename} ..."

    # 清理残留临时文件
    rm -f "$tmp_dest"

    local download_ok=true

    if [[ "$TOOL" == "wget" ]]; then
        # wget: -c 断点续传, -q 安静模式, --show-progress 进度条
        # 先下载到 .tmp，校验通过后再 mv（原子化操作）
        #
        # 注意：set -o pipefail 下 pipe 的退出码是最后一个命令的。
        # 这里不使用 pipe，直接用 -o 输出到文件，然后检查 wget 返回值。
        if ! wget -cq --show-progress --timeout=30 -O "$tmp_dest" "$url"; then
            download_ok=false
        fi
    else
        # curl: -C - 断点续传, -L 跟随重定向, -# 进度条
        if ! curl -C - -L --progress-bar --connect-timeout 30 -o "$tmp_dest" "$url"; then
            download_ok=false
        fi
    fi

    if ! $download_ok; then
        error "${filename} 下载失败"
        rm -f "$tmp_dest"
        return 1
    fi

    # 校验文件大小（防止下载到 LFS 指针或 HTML 错误页面）
    local actual_size
    actual_size=$(stat -c%s "$tmp_dest" 2>/dev/null || echo 0)

    if [[ "$actual_size" -lt "$min_size" ]]; then
        error "${filename} 下载内容异常 (size=${actual_size}B, 预期 >${min_size}B)"
        rm -f "$tmp_dest"

        # 诊断建议
        if [[ "$actual_size" -lt 200 ]] && file "$tmp_dest" 2>/dev/null | grep -qi "text\|HTML"; then
            warn "  可能是 LFS 指针文件或 HTML 错误页，尝试以下方案:"
            warn "    1. 使用 --mirror 镜像下载"
            warn "    2. 安装 git-lfs: sudo apt install git-lfs && git lfs install"
        fi
        return 1
    fi

    # 原子化移动
    mv "$tmp_dest" "$dest"

    local size_kb=$((actual_size / 1024))
    if [[ $size_kb -gt 1024 ]]; then
        local size_mb=$((size_kb / 1024))
        ok "${filename} (${size_mb}MB)"
    else
        ok "${filename} (${size_kb}KB)"
    fi
    return 0
}

# ---- 下载所有文件 ----
failed=0
if ! $QUIET; then
    echo -e "${CYAN}下载模型文件 (${#FILE_LIST[@]} 个)...${NC}"
    echo ""
fi

for entry in "${FILE_LIST[@]}"; do
    IFS='|' read -r fname minsize _ <<< "$entry"
    if ! download_file "$fname" "$minsize"; then
        failed=$((failed + 1))
    fi
done

# ---- 创建 symlink (SpeechBrain Pretrainer 兼容) ----
# SpeechBrain 的 Pretrainer 内部通过 loadables 名称推断扩展名，
# label_encoder 的 CategoricalEncoder 保存为 .txt，但 Pretrainer
# 会按 .ckpt 查找。创建此 symlink 确保兼容。
if ! $QUIET; then
    echo ""
    echo -e "${CYAN}创建兼容符号链接...${NC}"
    echo ""
fi

SYMLINK_SRC="$TARGET_DIR/label_encoder.txt"
SYMLINK_DST="$TARGET_DIR/label_encoder.ckpt"

if [[ -f "$SYMLINK_SRC" ]]; then
    # 如果目标已存在且不是 symlink，先备份
    if [[ -e "$SYMLINK_DST" ]] && [[ ! -L "$SYMLINK_DST" ]]; then
        warn "label_encoder.ckpt 已存在且非符号链接，跳过创建"
    else
        ln -sf "$SYMLINK_SRC" "$SYMLINK_DST"
        ok "label_encoder.ckpt → label_encoder.txt"
    fi
else
    warn "label_encoder.txt 缺失，跳过符号链接创建"
fi

# ---- post-check: 验证关键文件到位 ----
verify_model_dir() {
    local dir="$1"
    local missing=()

    # SpeechBrain EncoderClassifier.from_hparams() 所需的全部文件
    local required=(
        "hyperparams.yaml"
        "embedding_model.ckpt"
        "classifier.ckpt"
        "mean_var_norm_emb.ckpt"
        "label_encoder.txt"
    )

    for f in "${required[@]}"; do
        if [[ ! -f "$dir/$f" ]]; then
            missing+=("$f")
        elif [[ $(stat -c%s "$dir/$f" 2>/dev/null || echo 0) -lt 50 ]]; then
            missing+=("$f (文件过小，可能损坏)")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        return 1
    fi
    return 0
}

# ---- 结果 ----
if ! $QUIET; then
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
fi

if [[ $failed -eq 0 ]] && verify_model_dir "$TARGET_DIR"; then
    ok "模型下载完成!"
    if ! $QUIET; then
        echo ""
        echo -e "  模型目录: $TARGET_DIR"
        echo -e "  占用空间: $(du -sh "$TARGET_DIR" 2>/dev/null | cut -f1)"
        echo ""
        echo -e "  Pipeline 将自动使用此模型进行说话人分离。"
        echo -e "  预期日志: ${GREEN}SpeechBrain model loaded (dim=192)${NC}"
    fi
else
    error "$failed 个文件下载失败"
    echo ""
    echo -e "  排查建议:"
    echo -e "    1. 检查网络连接"
    echo -e "    2. 使用 --mirror 选项尝试镜像下载"
    echo -e "    3. 手动下载: https://huggingface.co/$MODEL_REPO/tree/main"
    echo -e "       将上述文件放入: $TARGET_DIR/"
    echo -e "    4. 确保 git-lfs 已安装 (LFS 大文件需此工具):"
    echo -e "       sudo apt install git-lfs && git lfs install"
    echo -e "    5. 或使用 git clone 方式:"
    echo -e "       git clone https://huggingface.co/$MODEL_REPO \"$TARGET_DIR\""
    echo -e "       cd \"$TARGET_DIR\" && git lfs pull"
fi

if ! $QUIET; then
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
    echo ""
fi

exit $failed
