#!/bin/bash
# 复现 op-fused-add-rmsnorm 的测量。
#
# 用的库（librares-registry.yaml 的 canonical id）：
#   lightop 0.6.0+dtk2604.torch2110  —— lightop.RMSNormAdd（primary）
#   aiter   0.1.6+dtk2604.torch2110  —— norm 族（非融合对照）
#   in-house                          —— ref-heavy（fp32 参照，仅正确性对照）
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
PY=${PYTHON:-python3}

usage() {
  cat <<'USAGE'
用法: ./reproduce.sh <目标>
  check      正确性门（tiny 档；需要 lightop + 一张卡）
  invariants 不变量门
  bench      A/B 基准（需要 lightop + 一张空闲卡）
  trace      从 torch-profiler trace 提取 norm 预算
  all        check + invariants + bench

环境变量:
  PYTHON              解释器（默认 python3）
  HIP_VISIBLE_DEVICES 一张空闲卡
  TRACE_DECODE        trace 文件（.json / .json.gz），trace 目标用
USAGE
}

cmd=${1:-check}
case "$cmd" in
  check)
    "$PY" "$REPO/scripts/opcheck.py" correctness \
      --contract "$HERE/contract.yaml" --shape tiny --variants lightop-rmsnormadd
    ;;
  invariants)
    "$PY" "$REPO/scripts/opcheck.py" invariants \
      --contract "$HERE/contract.yaml" --variant lightop-rmsnormadd
    ;;
  bench)
    : "${HIP_VISIBLE_DEVICES:?bench 需要一张空闲卡：export HIP_VISIBLE_DEVICES=<id>}"
    "$PY" "$HERE/bench/ab.py" --json "$HERE/results/ab.json" --rounds 5
    ;;
  trace)
    : "${TRACE_DECODE:?trace 需要 TRACE_DECODE=<path/to/trace.json.gz>}"
    "$PY" "$HERE/trace_budget.py" --trace "$TRACE_DECODE"
    ;;
  all) "$0" check; "$0" invariants; "$0" bench ;;
  *) usage; exit 2 ;;
esac
