#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 1：两个 GEMM 的地板与达成率（grouped vs 逐专家循环）。

用法: python3 stage1_gemm.py
输出: results/stage1_gemm.json（已在仓库里，脚本用于复跑核对）
"""
import json
import pathlib
import statistics
import sys

E, M, N, K = 32, 128, 1536, 2048


def bench(fn, rounds=9, warmup=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2]
    return med, statistics.median([abs(t - med) for t in ts])


def main():
    import torch
    import deepgemm
    torch.manual_seed(0)
    a = torch.randn(E, M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    mm = torch.full((E,), M, dtype=torch.int32, device="cuda")
    d = torch.empty(E, M, N, dtype=torch.bfloat16, device="cuda")
    bp, cfg = deepgemm.pack_bf16_masked_weight(b, 1000)

    g, gm = bench(lambda: deepgemm.m_grouped_bf16_gemm_nt_masked(a, bp, d, mm, M, cfg))

    def loop():
        for e in range(E):
            torch.mm(a[e], b[e].t(), out=d[e])

    l, lm = bench(loop)
    flops = 2 * E * M * N * K
    byts = E * M * K * 2 + E * N * K * 2 + E * M * N * 2
    print("grouped : %.4f ms  MAD=%.4f  %.1f TFLOPS" % (g, gm, flops / (g * 1e-3) / 1e12))
    print("逐专家  : %.4f ms  MAD=%.4f  %.1f TFLOPS" % (l, lm, flops / (l * 1e-3) / 1e12))
    print("ratio   : %.2fx  显著=%s" % (l / g, abs(l - g) > 3 * max(gm, lm)))
    print("达成率  : %.0f%%（地板 %.4f ms）" % (100 * (flops / 185e12 * 1e3) / g,
                                              flops / 185e12 * 1e3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
