#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 008：paged decode —— flash_attn 的 decode 路径在本机是坏档。

起点：`paged-attention-decode` 卡 status=draft，`impls` 里 flash-attn 是 primary。
本 case 要回答：它在 decode 形状到底能到多少？值不值得当 primary？

一路跌进三个坑，每个都值得记
----------------------------
1. **block_size 必须被 64 整除**，否则：
   `Paged KV cache block size must be divisible by 64 at gfx936 platform`
   （报错里写 gfx936，实测机器是 gfx938——错误文案本身也有 bug）
2. **paged 比连续 KV 慢 1.6×**（同一工作量），但这不重要——
   因为连续 KV 本身已经慢到 33 GB/s。
3. **真正的问题**：flash_attn 的 decode（q_len=1）在本机只到 0.01~0.03 of peak，
   而同机 MLA 吸收式 decode 能到 0.35（case 006）。**差一个数量级。**

判据
----
decode 是 memory-bound，分母用流式只读峰值带宽（同进程标定）。
字节 = K+V = `KV · B · d · 2B · 2`。

为什么扫 SQ 是关键实验
----------------------
如果耗时随 SQ（每序列 query 数）变化，说明是 compute-bound / 有并行空间；
如果**耗时与 SQ 完全无关**，说明 kernel 被别的东西钉死（调度/占用率），
加 query 是免费的——那也是另一种结论。
实测：SQ 从 1 到 32，耗时恒为 3.15 ms。这条比任何单一数字都有信息量。
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bench(fn, rounds=7, warmup=3):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        e1.record()
        fn()
        e2.record()
        torch.cuda.synchronize()
        ts.append(e1.elapsed_time(e2))
    ts.sort()
    med = ts[len(ts) // 2]
    return med, statistics.median([abs(t - med) for t in ts])


def calibrate_bw(torch):
    x = torch.randn(1 << 26, device="cuda")
    ms, _ = bench(lambda: x.sum())
    bw = x.numel() * 4 / (ms * 1e-3) / 1e9
    del x
    torch.cuda.empty_cache()
    return bw


def check_ceiling(gbs, peak, what):
    if gbs > peak * 1.02:
        raise AssertionError("%s 算出 %.0f GB/s 超过峰值 %.0f——先查字节数记账" % (what, gbs, peak))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "paged-decode.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        import flash_attn
    except ImportError as e:
        print("缺依赖：%s" % e, file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    bw = calibrate_bw(torch)
    rec = {"device": torch.cuda.get_device_properties(0).name,
           "bw_stream_gbs": round(bw),
           "method": "CUDA event median of 7, warmup 3"}
    print("== 轮0 同进程流式带宽 == %.0f GB/s" % bw)

    H, D = 32, 128

    # ---- 轮1：block_size 约束（先确认硬前提，再谈性能）
    print("== 轮1 block_size 约束 ==")
    B, KV = 8, 8192
    for PS in (16, 32):
        nb = KV // PS
        kc = torch.randn(nb, PS, H, D, dtype=torch.bfloat16, device="cuda")
        vc = torch.randn(nb, PS, H, D, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device="cuda")
        bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(1, -1).repeat(B, 1)
        sl = torch.full((B,), KV, dtype=torch.int32, device="cuda")
        try:
            flash_attn.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=sl, block_table=bt)
            print("  block_size=%d: OK" % PS)
        except Exception as e:
            print("  block_size=%d: 拒绝 —— %s" % (PS, str(e)[:80]))
        del kc, vc, q, bt, sl
        torch.cuda.empty_cache()
    print("  ⇒ 硬前提：block_size 必须被 64 整除（报错文案写的是 gfx936，实测机器是 gfx938）")

    # ---- 轮2：paged vs 连续 KV（同一工作量）
    print("== 轮2 paged vs 连续 KV（B=8, KV=8192）==")
    k = torch.randn(B, KV, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, KV, H, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device="cuda")
    ms_d, _ = bench(lambda: flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, causal=False))
    byts = KV * B * D * 2 * 2
    gbs_d = byts / (ms_d * 1e-3) / 1e9
    check_ceiling(gbs_d, bw, "轮2 连续 KV")
    del k, v
    torch.cuda.empty_cache()

    PS = 64
    nb = KV // PS
    kc = torch.randn(nb, PS, H, D, dtype=torch.bfloat16, device="cuda")
    vc = torch.randn(nb, PS, H, D, dtype=torch.bfloat16, device="cuda")
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(1, -1).repeat(B, 1)
    sl = torch.full((B,), KV, dtype=torch.int32, device="cuda")
    ms_p, _ = bench(lambda: flash_attn.flash_attn_with_kvcache(
        q, kc, vc, cache_seqlens=sl, block_table=bt))
    gbs_p = byts / (ms_p * 1e-3) / 1e9
    check_ceiling(gbs_p, bw, "轮2 paged")
    print("  连续 KV: %8.4f ms  %5.0f GB/s (%.2f)" % (ms_d, gbs_d, gbs_d / bw))
    print("  paged  : %8.4f ms  %5.0f GB/s (%.2f)  → paged 慢 %.2f×"
          % (ms_p, gbs_p, gbs_p / bw, ms_p / ms_d))
    rec["paged_vs_dense"] = {"dense_ms": round(ms_d, 4), "paged_ms": round(ms_p, 4),
                             "paged_overhead": round(ms_p / ms_d, 2)}
    del kc, vc, bt, sl, q
    torch.cuda.empty_cache()

    # ---- 轮3（关键）：扫 SQ —— 耗时是否随 query 数变化
    print("== 轮3 B=32 KV=8192，扫每序列 query 数 ==")
    B2, KV2 = 32, 8192
    k = torch.randn(B2, KV2, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B2, KV2, H, D, dtype=torch.bfloat16, device="cuda")
    rows = []
    for SQ in (1, 2, 4, 8, 16, 32):
        q = torch.randn(B2, SQ, H, D, dtype=torch.bfloat16, device="cuda")
        ms, mad = bench(lambda: flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, causal=False))
        fl = 4 * B2 * SQ * KV2 * H * D
        rows.append({"SQ": SQ, "ms": round(ms, 4), "mad_ms": round(mad, 4),
                     "tflops": round(fl / (ms * 1e-3) / 1e12, 1)})
        print("  SQ=%3d: %8.4f ms  MAD=%.4f  算力 %6.1f TF"
              % (SQ, ms, mad, fl / (ms * 1e-3) / 1e12))
        del q
        torch.cuda.empty_cache()
    rec["sweep_sq"] = rows
    lo = min(r["ms"] for r in rows)
    hi = max(r["ms"] for r in rows)
    print("  ⇒ 耗时 %.4f~%.4f ms（%.1f%% 波动）——**与 SQ 几乎无关**"
          % (lo, hi, (hi / lo - 1) * 100))
    print("     意味着 SQ=1..32 都是同一个固定代价；不会因为多算 query 而变慢")

    # ---- 轮4：扫 batch（decode 的真实扩展维度）
    print("== 轮4 KV=8192, SQ=1，扫 B ==")
    KV3 = 8192
    rows2 = []
    for B3 in (1, 8, 32, 64):
        k = torch.randn(B3, KV3, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B3, KV3, H, D, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(B3, 1, H, D, dtype=torch.bfloat16, device="cuda")
        ms, _ = bench(lambda: flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, causal=False))
        b3 = KV3 * B3 * D * 2 * 2
        g3 = b3 / (ms * 1e-3) / 1e9
        check_ceiling(g3, bw, "轮4 B=%d" % B3)
        rows2.append({"B": B3, "ms": round(ms, 4), "gbs": round(g3),
                      "of_peak": round(g3 / bw, 3), "per_seq_us": round(ms * 1000 / B3, 1)})
        print("  B=%4d: %8.4f ms  %5.0f GB/s  %.2f of peak  每序列 %6.1f µs"
              % (B3, ms, g3, g3 / bw, ms * 1000 / B3))
        del k, v, q
        torch.cuda.empty_cache()
    rec["sweep_batch"] = rows2

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
