#!/bin/bash
# 复现 Case 001 的全部测量并落盘到 results/。
#
# 用的库（canonical id 见 data/ops/libraries-registry.yaml）：
#   lightop 0.6.0+dtk2604.torch2110  —— lightop.FuseBiasSwiGLU（融合实现）
#   torch   2.11.0+das.opt1.dtk2604  —— 分拆对照（x+b → silu → 乘）
#
# 依赖：一张空闲卡（脚本会检查 HCU% 与 VRAM% 并警告）
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}

usage() {
  cat <<'USAGE'
用法: ./reproduce.sh [目标]
  bench   完整测量并落盘 results/（含带宽标定，约 60s）
  quick   跳过带宽标定（约 20s，但没有分母）
  check   只验证 lightop 入口存在（约 5s）
环境变量:
  PYTHON              解释器
  HIP_VISIBLE_DEVICES 指定卡（默认 0）
USAGE
}

case "${1:-bench}" in
  bench)
    "$PY" "$HERE/bench.py" --out "$HERE/results"
    ;;
  quick)
    "$PY" "$HERE/bench.py" --quick --out "$HERE/results"
    ;;
  check)
    "$PY" -I -c "import lightop; print('lightop.FuseBiasSwiGLU:', hasattr(lightop,'FuseBiasSwiGLU'))"
    ;;
  *) usage; exit 2 ;;
esac
