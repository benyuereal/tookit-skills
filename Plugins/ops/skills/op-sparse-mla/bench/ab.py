#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 适配器：把本卡的测量喂给 loop.py run（契约见 loop.py 的 ADAPTER_AB）。

    bench/ab.py --json out.json --rounds 3 --s 128 --topk 2048 \
                --nvalid 2000 --heads 64 --skv 32768 [--variants tilelang,flash_mla]

输出（loop.py 直接消费）：

    {"caliber": {"what","alloc","state","device"},
     "env": <env_fingerprint()>,
     "shape": {...},
     "runs": [{"skv": 32768, "base_ms": ..., "base_mad": ...,
               "variants": [{"tag","ms","mad","delta_pct","maxdiff","rel"}]}]}

**这个脚本只做「按契约编排」，不做新的测量实现**——
测量一律走 _common.bench_fn（median + MAD + CUDA/HIP event），
与 bench.py 同一口径。口径不一致是本项目最贵的一类错。

契约里最关键的一条：**base 必须在每轮被重测**。只报一次 base 的实现
会把频率漂移算成收益。这里 rounds 轮每轮都重新测 base，取各自的 median/MAD。

用法注意：`--skv` 可以给多个值，每个值产出一条 run。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import (  # noqa: E402
    bench_fn, build_inputs, dims_from_params, env_fingerprint,
    get_runner, load_shape, sm_scale_for,
)

DEFAULT_BASE = "tilelang"


def _measure(run, q, kv, idx, rounds, warmup_ms, rep_ms):
    """多轮测量，返回 (median_of_medians, mad_of_medians, per_round)。

    每轮内部先 warmup 再取 median；跨轮再取 median——
    单轮 median 只能压住轮内抖动，跨轮漂移要靠第二层中位数。
    """
    meds = []
    for _ in range(rounds):
        st = bench_fn(lambda: run(q, kv, idx), warmup_ms=warmup_ms,
                      rep_ms=rep_ms, repeats=3)
        meds.append(st["median_ms"])
    meds.sort()
    n = len(meds)
    med = meds[n // 2] if n % 2 else 0.5 * (meds[n // 2 - 1] + meds[n // 2])
    mad = sorted(abs(x - med) for x in meds)[n // 2]
    return med, mad, meds


def main(argv=None):
    p = argparse.ArgumentParser(description="op-sparse-mla A/B 适配器（喂 loop.py run）")
    p.add_argument("--json", required=True, help="输出路径")
    p.add_argument("--rounds", type=int, default=3, help="每变体测量轮数（建议 ≥5）")
    p.add_argument("--shape", default="decode-s128", help="shapes.yaml 档位名")
    p.add_argument("--variants", default="", help="逗号分隔；缺省从 candidates 取 tag")
    p.add_argument("--base", default=DEFAULT_BASE, help="基线变体名（每轮重测）")
    p.add_argument("--skv", type=int, nargs="+", default=[32768])
    p.add_argument("--s", type=int, default=128)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--nvalid", type=int, default=2000)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--tilelang-root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    a = p.parse_args(argv)

    shape = load_shape(a.shape)
    params = dict(shape.get("params") or {})
    # 命令行覆盖：让 loop.py 的 --s/--topk/--nvalid/--heads 能真正生效
    params.update({"S": a.s, "topk": a.topk, "nvalid": a.nvalid, "H": a.heads})

    d_qk, d_v, _tail = dims_from_params(params)
    scale = sm_scale_for(params)

    tags = [x.strip() for x in a.variants.split(",") if x.strip()]
    if not tags:
        print("--variants 为空：本适配器只测显式点名的变体，不猜", file=sys.stderr)
        return 2

    runners = {}

    def _runner(tag):
        if tag not in runners:
            runners[tag] = get_runner(tag, tilelang_root=a.tilelang_root,
                                      d_v=d_v, sm_scale=scale)
        return runners[tag]

    out = {
        "caliber": {
            "what": "op_call",
            "alloc": True,          # bench_fn 的计时含 wrapper 分配（与 bench.py 同口径）
            "state": "hot",         # 每轮先 warmup
            "rounds": a.rounds,
            "timer": "cuda_event",
            "note": "per-round median → cross-round median；base 每轮重测",
        },
        "env": env_fingerprint(),
        "shape": {"s": a.s, "heads": a.heads, "d": d_qk, "topk": a.topk,
                  "nvalid": a.nvalid, "seed": a.seed},
        "runs": [],
    }

    for skv in a.skv:
        p2 = dict(params, SKV=skv)
        q, kv, idx, meta = build_inputs(p2, seed=a.seed, device=a.device)
        row = {"skv": skv, "variants": []}

        try:
            base_med, base_mad, _ = _measure(_runner(a.base), q, kv, idx,
                                             a.rounds, a.warmup_ms, a.rep_ms)
        except Exception as e:                     # noqa: BLE001
            print("base=%s 失败: %s: %s" % (a.base, type(e).__name__, e), file=sys.stderr)
            return 1
        row["base_ms"] = base_med
        row["base_mad"] = base_mad

        for tag in tags:
            try:
                med, mad, _ = _measure(_runner(tag), q, kv, idx,
                                       a.rounds, a.warmup_ms, a.rep_ms)
            except Exception as e:                 # noqa: BLE001
                print("variant=%s 跳过: %s: %s" % (tag, type(e).__name__, e),
                      file=sys.stderr)
                continue
            row["variants"].append({
                "tag": tag,
                "ms": med,
                "mad": mad,
                "delta_pct": (med / base_med - 1.0) * 100.0,
                # 逐位一致性由 accuracy probe 判；这里不猜
                "maxdiff": 0.0 if tag == a.base else None,
                "rel": None,
            })
        out["runs"].append(row)
        print("# skv=%d base=%.4fms(%s) variants=%d"
              % (skv, base_med, a.base, len(row["variants"])))

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→ %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
