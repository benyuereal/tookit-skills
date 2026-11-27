#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B 方向的第一步诊断：top-k gather 的地板价到底是多少。

问题：现役 flash_mla prefill 在 prefill-glm53-8k 档实测 306 GB/s（15.6% HBM 峰值）。
两种可能，处置完全不同：
  (i)  字节模型对、kernel 抓得慢  → 该改 gather 布局/预取（合并连续行、分桶）
  (ii) 纯 gather 本身也就这么慢   → 说明有效字节远大于模型，该回头核对布局/冗余读

做法：对同一组 (q, kv, idx) 量三个数
    gather_only : kv[idx] 的纯搬运（torch index_select），只读 KV、不算注意力 → 带宽地板
    flash_mla   : 现役 bf16 kernel
    ratio       : flash_mla / gather_only；越接近 1 说明 kernel 越贴近搬运地板

判读：
    gather_only ≥ 1200 GB/s 而 flash_mla ≈ 306 GB/s ⇒ (i)，gather 有 4 倍空间
    gather_only ≈ 300 GB/s ⇒ (ii)，先核对字节模型

用法：
    python3 -I bench/gather_probe.py --shape prefill-glm53-8k
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import (  # noqa: E402
    bench_fn, build_inputs, dims_from_params, get_runner, load_shape, sm_scale_for,
)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="prefill-glm53-8k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    shape = load_shape(a.shape)
    params = shape.get("params") or {}
    _d_qk, dv, _tail = dims_from_params(params)
    scale = sm_scale_for(params)
    q, kv, idx, _meta = build_inputs(params, seed=a.seed, device=a.device)
    S, H, d_qk = q.shape
    topk = idx.shape[-1]
    SKV = int(kv.shape[0])

    flat = idx.reshape(S, topk).to(torch.long)
    flat = flat.clamp_min(0)
    kv2d = kv.reshape(SKV, d_qk)

    # gather 的落盘缓冲复用（不让分配器把 17 GB 的分配算进时间）
    gout = torch.empty((S * topk, d_qk), dtype=kv.dtype, device=kv.device)

    def gather_only():
        return torch.index_select(kv2d, 0, flat.reshape(-1), out=gout)

    base = get_runner("flash_mla", d_v=dv, sm_scale=scale)

    gb = float(S) * topk * d_qk * 2 / 1e9          # bf16 读字节
    for name, fn in (("gather_only", gather_only),
                     ("flash_mla", lambda: base(q, kv, idx))):
        st = bench_fn(fn, warmup_ms=25.0, rep_ms=100.0, repeats=3)
        ms = st["median_ms"]
        tok_us = ms * 1e3 / S
        moved = gb * 2 if name == "gather_only" else gb   # gather 还写一份 17 GB
        print("%-12s %8.3f ms  %7.2f µs/token  %7.0f GB/s  (%s 口径：%.1f GB)"
              % (name, ms, tok_us, moved / (ms * 1e-3), "读+写" if name == "gather_only" else "只读", moved))

    print("# 判读：gather_only 明显高于 flash_mla ⇒ B 方向（gather 布局/预取）有空间；"
          "两者都在 300 GB/s 附近 ⇒ 先核对字节模型与冗余读")
    return 0


if __name__ == "__main__":
    sys.exit(main())
