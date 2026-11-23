#!/bin/bash
# 复现 op-rmsnorm-fused 的全部测量。环境变量缺失时会明确报错，不会静默跑错口径。
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
  triton)
    # 一键复现 tuned 结论：fused-3pass-triton vs ref-heavy（prefill-8k）
    # 注意 --shape 必须显式给：ab.py 默认 decode-m32（M=32），与 prefill-8k 不可比。
    : "${HIP_VISIBLE_DEVICES:?triton 需要一张空闲卡：export HIP_VISIBLE_DEVICES=<id>}"
    export HIP_VISIBLE_DEVICES
    "$PY" - "$HERE" <<'PYEOF'
import sys, subprocess, json, pathlib
here = pathlib.Path(sys.argv[1])
import torch, triton
print("# torch %s / triton %s" % (torch.__version__, triton.__version__))
cmd = [sys.executable, "-I", str(here / "bench/ab.py"),
       "--shape", "prefill-8k",
       "--variants", "fused-3pass-triton", "--base", "ref-heavy",
       "--rounds", "3", "--json", str(here / "results/prefill-8k-triton.json")]
print("# " + " ".join(cmd))
rc = subprocess.run(cmd).returncode
if rc == 0:
    d = json.loads((here / "results/prefill-8k-triton.json").read_text())
    r = d["runs"][0]
    M, H = d["shape"]["params"]["M"], d["shape"]["params"]["H"]
    byts = 3 * M * H * 2
    ms = r["variants"][0]["ms"]
    print("# 冷L2 参考值：186.2 us / 1081 GB/s（util 0.63/1724）")
    print("# 本次： %.1f us / %.0f GB/s" % (ms * 1000, byts / (ms * 1e-3) / 1e9))
    print("# caveat ab.py 的 bench_fn 是连续调用（热 L2），会报 ~119 us；"
          "冷 L2 需每次换输入，见 case README")
sys.exit(rc)
PYEOF
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
