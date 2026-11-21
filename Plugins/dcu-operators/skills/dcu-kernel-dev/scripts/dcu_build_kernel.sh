#!/usr/bin/env bash
# dcu_build_kernel.sh —— 运行时探测 DCU 架构后编译 HIP kernel（不硬编码 gfx）。
#
# 对应本 skill 第 12 节纪律：DCU 有多种架构（gfx906/gfx908/gfx928/gfx90a ...），
# 编译目标必须由"在目标机上现查设备"决定，绝不在脚本里写死 --offload-arch。
#
# 用法：
#   bash dcu_build_kernel.sh <src.cpp> [额外 hipcc 参数...]
#   bash dcu_build_kernel.sh scripts/hip_gemm_tiled.cpp -o gemm
#
# 行为：
#   1. 调用 dcu_detect_arch.sh 得到 gfx（如 gfx928）
#   2. 用该 gfx 作为 --offload-arch 编译
#   3. 若探测到的是产品名（非 gfx 代号），打印提示并退出，请用户按表映射
#
# 前置：
#   - 在挂载了 -v /opt/hyhal:/opt/hyhal:ro 的 DCU 容器内运行（否则 DCU 不可见）
#   - ROCm/DTK 的 hipcc 在 PATH（或设置 HIP_PATH）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECT="$SCRIPT_DIR/dcu_detect_arch.sh"

if [ "$#" -lt 1 ]; then
    echo "用法: bash $0 <src.cpp> [hipcc 额外参数...]" >&2
    exit 1
fi

SRC="$1"; shift
[ -f "$SRC" ] || { echo "源文件不存在: $SRC" >&2; exit 1; }

# 取 hipcc
HIPCC="${HIP_PATH:+${HIP_PATH}/bin/}hipcc"
command -v "$HIPCC" >/dev/null 2>&1 || { echo "未找到 hipcc（设置 HIP_PATH 或加入 PATH）" >&2; exit 1; }

# 探测架构
ARCH_RAW="$(bash "$DETECT")" || { echo "设备探测失败"; exit 1; }
# 若是产品名（product:xxx），需映射；这里只做 gfx 代号透传
if [[ "$ARCH_RAW" == gfx* ]]; then
    ARCH="$ARCH_RAW"
elif [[ "$ARCH_RAW" == product:* ]]; then
    echo "探测到产品名 ${ARCH_RAW#product:}，请按 DCU 架构映射表确定 gfx 后手动指定：" >&2
    echo "  K100 / Z100 系列多为 gfx906；K100-AI 多为 gfx928（以物理机 rocminfo 为准）" >&2
    echo "  或重跑: DCU_ARCH=gfx928 bash $0 $SRC $*" >&2
    exit 1
else
    echo "未能确定 gfx（得到 '$ARCH_RAW'），请用 DCU_ARCH=... 显式指定" >&2
    exit 1
fi

# 允许手动覆盖（用户在已知机型时直接指定，跳过探测）
ARCH="${DCU_ARCH:-$ARCH}"
echo ">> 目标架构: $ARCH  (源: $SRC)"

# 编译：用探测到的 arch 作为 offload 目标；不写死
set -x
"$HIPCC" --offload-arch="$ARCH" "$SRC" "$@"
set +x
echo ">> 编译完成"
