#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prefill 稀疏 MLA：bf16 基线 vs fp8（B_H=64）的性能 + 精度闸门。

背景（2026-11-27 prefill trace 实测，见 shapes.yaml:prefill-glm53-8k 的 focus）：
    生产 chunk 是 S=8192、bs=1；现役 `flash_mla` bf16 路径（B_H=16）跑 6.87 µs/token
    = 306 GB/s = 15.6% HBM 峰值，算力 1.8~3.5% of 279 TFLOPS。
    结构上 B_H=16 ⇒ 算术强度 32 FLOP/B（bf16 ridge 142）⇒ 算力利用率天花板 ~22%。

本脚本只回答两件事：
    ① fp8 KV 直读 + B_H=64（/workspace/fp8_mla/flashmla-fp8）在本档能快多少
    ② 它和 bf16 基线的数值差在不在容差里

口径：与 _common.bench_fn 一致（HIP event、median + MAD），单进程顺序测。
三个 tag：
    flash_mla            bf16 基线（B_H=16）
    fp8-bh64             fp8 kernel + **计入** Q 的 per-row 量化（保守口径）
    fp8-bh64-qonly       fp8 kernel，Q 量化在计时外（只衡量 kernel）

判据（rc=0 才算过）：
    ratio(kernel) = t(fp8-bh64-qonly) / t(flash_mla) <= --ratio-max（默认 0.6）
    ratio(净收益) = t(fp8-bh64)       / t(flash_mla) <= --ratio-max-net（默认 0.8）
    cos(fp8, bf16) >= --cos-min（默认 0.9999）
    max_abs(fp8 - bf16) <= --max-abs（默认 3e-4）

用法：
    python3 -I bench/prefill_fp8_gate.py --json /tmp/prefill_fp8.json
    python3 -I bench/prefill_fp8_gate.py --shape prefill-glm53-8k --rounds 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import (  # noqa: E402
    bench_fn, build_inputs, compare, dims_from_params, env_fingerprint,
    get_runner, load_shape, sm_scale_for,
)

TAGS = ("flash_mla", "fp8-bh64", "fp8-bh64-qonly")


def _time(fn, q, kv, idx, rounds, warmup_ms, rep_ms):
    """跨轮中位数 + MAD（与 ab.py 同一口径：每轮先 warmup 再取 median）。"""
    meds = []
    for _ in range(rounds):
        st = bench_fn(lambda: fn(q, kv, idx), warmup_ms=warmup_ms, rep_ms=rep_ms, repeats=3)
        meds.append(st["median_ms"])
    meds.sort()
    n = len(meds)
    med = meds[n // 2] if n % 2 else 0.5 * (meds[n // 2 - 1] + meds[n // 2])
    mad = sorted(abs(x - med) for x in meds)[n // 2]
    return med, mad, meds


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prefill-glm53-8k")
    p.add_argument("--json", default=None)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratio-max", type=float, default=0.60)
    p.add_argument("--ratio-max-net", type=float, default=0.80)
    p.add_argument("--cos-min", type=float, default=0.9999)
    p.add_argument("--max-abs", type=float, default=3e-4)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    shape = load_shape(a.shape)
    params = shape.get("params") or {}
    name = shape.get("shape") or Path(shape["_path"]).stem
    _d_qk, dv, _tail = dims_from_params(params)
    scale = sm_scale_for(params)

    q, kv, idx, meta = build_inputs(params, seed=a.seed, device=a.device)
    S, H, d_qk = q.shape
    topk = idx.shape[-1]
    kv_rows = int(kv.shape[0])
    sm_scale = float(scale)

    print("# shape=%s S=%d H=%d topk=%d SKV=%d d_qk=%d d_v=%d sm_scale=%.6f"
          % (name, S, H, topk, kv_rows, d_qk, dv, sm_scale))
    print("# flops/token = 2*H*topk*(d_qk+d_v) = %.2f MFLOP"
          % (2.0 * H * topk * (d_qk + dv) / 1e6))

    def tflops(ms, nbytes_per_elem):
        f = 2.0 * S * H * topk * (d_qk + dv)
        return f / (ms * 1e-3) / 1e12

    def gbps(ms, nbytes_per_elem):
        b = float(S) * topk * (d_qk + dv) * nbytes_per_elem
        return b / (ms * 1e-3) / 1e9

    out, runs = {}, []
    for tag in TAGS:
        try:
            fn = get_runner(tag, d_v=dv, sm_scale=sm_scale)
            y = fn(q, kv, idx)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print("# variant=%s SKIPPED: %s: %s" % (tag, type(e).__name__, e))
            runs.append({"tag": tag, "skipped": "%s: %s" % (type(e).__name__, e)})
            continue
        out[tag] = y
        med, mad, allmed = _time(fn, q, kv, idx, a.rounds, a.warmup_ms, a.rep_ms)
        nb = 1 if tag.startswith("fp8") else 2
        rec = {
            "tag": tag, "ms": med, "mad_ms": mad, "rounds_ms": allmed,
            "us_per_token": med * 1e3 / S,
            "tflops": tflops(med, nb), "gbps": gbps(med, nb),
            "bytes_per_elem": nb,
        }
        runs.append(rec)
        print("# variant=%-16s ms=%.3f mad=%.4f  %.2f µs/token  %.1f TFLOPS  %.0f GB/s"
              % (tag, med, mad, rec["us_per_token"], rec["tflops"], rec["gbps"]))

    base = out.get("flash_mla")
    checks = []
    for tag in ("fp8-bh64", "fp8-bh64-qonly"):
        if tag in out and base is not None:
            c = compare(out[tag], base)
            for r in runs:
                if r.get("tag") == tag:
                    r.update({"maxdiff": c["max_abs"], "mean_abs": c["mean_abs"],
                              "cos": c["cosine"]})
            print("# accuracy %s vs flash_mla: cos=%.8f max_abs=%.3e mean_abs=%.3e"
                  % (tag, c["cosine"], c["max_abs"], c["mean_abs"]))

    t = {r["tag"]: r.get("ms") for r in runs}
    if t.get("flash_mla") and t.get("fp8-bh64-qonly"):
        ratio = t["fp8-bh64-qonly"] / t["flash_mla"]
        checks.append(("ratio_kernel<=%.2f" % a.ratio_max, ratio, ratio <= a.ratio_max))
    if t.get("flash_mla") and t.get("fp8-bh64"):
        rn = t["fp8-bh64"] / t["flash_mla"]
        checks.append(("ratio_net<=%.2f" % a.ratio_max_net, rn, rn <= a.ratio_max_net))
    acc_ref = None
    for r in runs:
        if r.get("cos") is not None:
            acc_ref = r
            break
    if acc_ref:
        checks.append(("cos>=%.6f" % a.cos_min, acc_ref["cos"], acc_ref["cos"] >= a.cos_min))
        checks.append(("max_abs<=%.1e" % a.max_abs, acc_ref["maxdiff"],
                       acc_ref["maxdiff"] <= a.max_abs))

    ok = bool(checks) and all(c[2] for c in checks)
    for nm, val, passed in checks:
        print("# gate %-22s value=%.6g  %s" % (nm, val, "PASS" if passed else "FAIL"))

    payload = {
        "op": "sparse-mla", "shape": name, "shape_file": shape["_path"],
        "S": S, "H": H, "topk": topk, "SKV": kv_rows, "d_qk": d_qk, "d_v": dv,
        "sm_scale": sm_scale, "meta": meta,
        "env": env_fingerprint(), "runs": runs,
        "checks": [{"name": n, "value": v, "pass": bool(pp)} for n, v, pp in checks],
        "verdict": "PASS" if ok else "FAIL",
    }
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print("# json -> %s" % a.json)
    print("# VERDICT %s" % payload["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
