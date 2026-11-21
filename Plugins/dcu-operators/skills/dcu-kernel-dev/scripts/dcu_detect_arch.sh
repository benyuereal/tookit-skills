#!/usr/bin/env bash
# dcu_detect_arch.sh —— 运行时探测海光 DCU 设备架构（gfx），不硬编码。
#
# 用途：算子编译/调优通常在远端 DCU 物理机上进行（本机无卡），
#       因此编译目标的 offload-arch 必须"用的时候现查设备"得到，
#       严禁把 gfx 写死在脚本里。
#
# 输出：向 stdout 打印检测到的 gfx 代号（如 gfx906 / gfx908 / gfx928 / gfx90a ...），
#       无设备时打印 "unknown" 并以非零退出码提示。
#
# 依赖（任选其一，按可用顺序探测）：
#   - rocminfo   （带 ROCM_PATH / LD_LIBRARY_PATH，容器内可能不可靠 → 见 memory 纪律：gfx 用物理机 rocminfo 查）
#   - hy-smi --showproductname
#   - /opt/hyhal 下设备信息
#
# 用法：
#   ARCH=$(bash dcu_detect_arch.sh) || { echo "未检测到 DCU 设备"; exit 1; }
#   echo "target arch = $ARCH"

set -uo pipefail

detect_via_rocminfo() {
    # rocminfo 输出形如：  Name:                    gfx928
    if command -v rocminfo >/dev/null 2>&1; then
        local g
        g=$(rocminfo 2>/dev/null | grep -m1 -iE '^\s*Name:\s+gfx[0-9a-f]+' | sed -E 's/.*(gfx[0-9a-f]+).*/\1/')
        [ -n "$g" ] && { echo "$g"; return 0; }
    fi
    # 备选： hipconfig 也能给
    if command -v hipconfig >/dev/null 2>&1; then
        local g
        g=$(hipconfig --offload-arch 2>/dev/null | grep -m1 -iE 'gfx[0-9a-f]+' | tr -d ' ')
        [ -n "$g" ] && { echo "$g"; return 0; }
    fi
    return 1
}

detect_via_hysmi() {
    # hy-smi --showproductname 给出产品名；再尝试从 hy-smi 设备属性拿 gfx
    if command -v hy-smi >/dev/null 2>&1; then
        # 有些版本直接能列出 arch；先试
        local g
        g=$(hy-smi --showproductname 2>/dev/null | grep -m1 -iE 'gfx[0-9a-f]+' | sed -E 's/.*(gfx[0-9a-f]+).*/\1/')
        [ -n "$g" ] && { echo "$g"; return 0; }
        # 否则只报产品名，提示人工确认（K100/K100-AI 等 → 用户查表）
        local name
        name=$(hy-smi --showproductname 2>/dev/null | grep -m1 -iE 'K100|DCU|Z100' | sed 's/Product.*Name://I' | tr -d ' \t')
        [ -n "$name" ] && { echo "product:$name"; return 0; }
    fi
    return 1
}

detect_via_kfd() {
    # 无 rocm 工具时，从 /sys 或 hyhal 设备节点推断（弱化）
    if [ -e /dev/kfd ]; then
        echo "kfd-present"; return 0
    fi
    return 1
}

main() {
    local out
    if out=$(detect_via_rocminfo); then
        echo "$out"; exit 0
    fi
    if out=$(detect_via_hysmi); then
        echo "$out"; exit 0
    fi
    if out=$(detect_via_kfd); then
        echo "$out"; exit 0
    fi
    echo "unknown" >&2
    exit 2
}

main "$@"
