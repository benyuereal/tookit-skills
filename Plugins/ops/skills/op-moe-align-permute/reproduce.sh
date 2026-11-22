#!/bin/bash
# 复现 op-moe-align-permute 的全部测量。环境变量缺失时会明确报错，不会静默跑错口径。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
PY=${PYTHON:-python3}

usage() {
  cat <<'USAGE'
用法: ./reproduce.sh <目标>
  check     正确性门（tiny 档，adapter 自检，无需后端）
  invariants 不变量门（同上）
  bench     A/B 基准（需要 MoE/norm 后端）
  trace     从 torch-profiler trace 提取本段的子算子预算
  all       check + invariants + bench

环境变量:
  PYTHON             解释器（默认 python3）
  HIP_VISIBLE_DEVICES 一张空闲卡
  TRACE_DECODE        trace 文件（.json 或 .json.gz），trace 目标用
USAGE
}

cmd=${1:-check}
case "$cmd" in
  check)
    "$PY" "$REPO/scripts/opcheck.py" correctness \
      --contract "$HERE/contract.yaml" --shape tiny --variants ref-heavy
    ;;
  invariants)
    "$PY" "$REPO/scripts/opcheck.py" invariants \
      --contract "$HERE/contract.yaml" --variant ref-heavy
    ;;
  bench)
    : "${HIP_VISIBLE_DEVICES:?bench 需要一张空闲卡：export HIP_VISIBLE_DEVICES=<id>}"
    "$PY" "$HERE/bench/ab.py" --json "$HERE/results/ab.json" --rounds 5
    ;;
  trace)
    : "${TRACE_DECODE:?trace 需要 TRACE_DECODE=<path/to/trace.json.gz>}"
    "$PY" "$HERE/trace_budget.py" --trace "$TRACE_DECODE"
    ;;
  all)
    "$0" check; "$0" invariants; "$0" bench
    ;;
  *)
    usage; exit 2
    ;;
esac
