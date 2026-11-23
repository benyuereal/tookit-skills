#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 006：MLA 吸收式 decode —— 达成率随 KV 长度走，短 KV 是固定开销主导。

起点：`mla-absorb` 卡 status=draft。它和 `rope-mla` 卡容易被混为一谈，
但两张卡的结论口径不同：rope-mla 测的是「整 kernel 的 KV 读带宽」，
本 case 要回答的是「这个 kernel 在不同的 KV 长度下分别到什么水平」。

判据
----
decode 是 memory-bound（每步只读 KV，M=1）。所以达成率的分母用
**流式峰值带宽 1724 GB/s**，不是算力。字节数 = 有效 KV 字节
（`KV · B · d_qk · 2B`），不含 block_table 与输出。

轮次
----
  轮0 标定流式带宽（同进程）
  轮1 扫 KV 长度（2048/8192/16384/32768），看达成率趋势
  轮2 扫 batch（B=1/8/32），判「是不是靠 batch 摊固定开销」
  轮3 与纯 copy 同字节对照，判「是不是访存模式的问题」

踩坑记录（API）
---------------
`flash_mla.get_mla_metadata()` 在新版本里**不接参数**，且返回 (meta, None) 元组。
按旧文档传 `(cache_seqlens, num_heads)` 会得到 "must be of type FlashMLASchedMeta"。
`head_dim_v` 是第 5 个位置参数，与 q/k_cache 同级，不能当关键字重复传。
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
BW_PEAK = 1724.0


def check_ceiling(gbs, peak, what):
    """任何「超过标定峰值」的读数一定是记账错，不是机器变异。直接拒出结论。

    为什么做成硬失败而不是警告：本 case 第一版就算出 1666 GB/s > 峰值 1605，
    原因是轮2 的循环变量 B 残留成 32，字节数按 B=32 算、耗时却是 B=8 的。
    如果只打印不拦截，这个数就会进 README、进卡片。
    同类错误在案例 004 也犯过（varlen 多乘了 batch，算出 1.8× bf16 峰值）。
    """
    if gbs > peak * 1.02:
        raise AssertionError(
            "%s 算出 %.0f GB/s，超过标定峰值 %.0f GB/s 的 2%%——"
            "实测不可能超过峰值，先查字节数/FLOPs 记账，不要先怀疑机器"
            % (what, gbs, peak))


def bench(fn, rounds=9, warmup=4):
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


def make_kv(torch, B, KV, DQK=576, PB=64):
    nb = KV * B // PB
    kc = torch.randn(nb, PB, 1, DQK, dtype=torch.bfloat16, device="cuda")
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(1, -1).repeat(B, 1)
    sl = torch.full((B,), KV, dtype=torch.int32, device="cuda")
    return kc, bt, sl, nb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "mla-decode.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        import flash_mla
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
           "method": "CUDA event median of 9, warmup 4"}
    print("== 轮0 同进程流式带宽 == %.0f GB/s" % bw)

    H, DQK, DV = 32, 576, 512
    # ⚠️ sched_meta 绑定形状（在首次调用时被填上 config.b），**不能跨形状复用**。
    # 复用会要么 assert 挂掉，要么静默走错调度（实测达成率从 0.29 掉到 0.11）。
    # 所以下面每个形状都重新取一次。

    # ---- 轮1：扫 KV，B=8
    print("== 轮1 B=8，扫 KV ==")
    B = 8
    rows = []
    for KV in (2048, 8192, 16384, 32768):
        kc, bt, sl, nb = make_kv(torch, B, KV, DQK)
        q = torch.randn(B, 1, H, DQK, dtype=torch.bfloat16, device="cuda")
        meta = flash_mla.get_mla_metadata()[0]      # 每形状重建
        ms, mad = bench(lambda: flash_mla.flash_mla_with_kvcache(q, kc, bt, sl, DV, meta))
        byts = KV * B * DQK * 2
        gbs = byts / (ms * 1e-3) / 1e9
        check_ceiling(gbs, bw, "轮1 KV=%d" % KV)
        rows.append({"KV": KV, "B": B, "ms": round(ms, 4), "mad_ms": round(mad, 4),
                     "kv_gbs": round(gbs), "of_bw_peak": round(gbs / bw, 3)})
        print("  KV=%6d  %8.4f ms MAD=%.4f  %5.0f GB/s  %.2f of peak"
              % (KV, ms, mad, gbs, gbs / bw))
        del kc, q, bt, sl
        torch.cuda.empty_cache()
    rec["sweep_kv"] = rows

    # ---- 轮2：扫 batch，判「靠 batch 摊固定开销」
    print("== 轮2 KV=8192，扫 B ==")
    KV = 8192
    rows2 = []
    for B in (1, 8, 32):
        kc, bt, sl, nb = make_kv(torch, B, KV, DQK)
        q = torch.randn(B, 1, H, DQK, dtype=torch.bfloat16, device="cuda")
        meta = flash_mla.get_mla_metadata()[0]      # 每形状重建
        ms, mad = bench(lambda: flash_mla.flash_mla_with_kvcache(q, kc, bt, sl, DV, meta))
        byts = KV * B * DQK * 2
        gbs = byts / (ms * 1e-3) / 1e9
        check_ceiling(gbs, bw, "轮2 B=%d" % B)
        rows2.append({"B": B, "ms": round(ms, 4), "kv_gbs": round(gbs),
                      "of_bw_peak": round(gbs / bw, 3),
                      "per_seq_us": round(ms * 1000 / B, 1)})
        print("  B=%3d  %8.4f ms  %5.0f GB/s  %.2f of peak  每序列 %.1f µs"
              % (B, ms, gbs, gbs / bw, ms * 1000 / B))
        del kc, q, bt, sl
        torch.cuda.empty_cache()
    rec["sweep_batch"] = rows2

    # ---- 轮2b：固定总 KV，拆成不同 batch —— 判「摊销下降」是真降本还是假象
    # 上一轮我只看「每序列摊销 87.5→16.6」就写下「固定开销靠 batch 摊」，
    # 但那是把「摊销值下降」当成了「总代价下降」。总量对照一测就露馅。
    print("== 轮2b 固定总 KV=8192，拆不同 batch ==")
    TOT = 8192
    PB = 64
    rows2b = []
    for B2 in (1, 2, 4, 8, 16):
        KV2 = TOT // B2
        if KV2 % PB:            # page 边界不齐会改变读放大，跳过不测
            continue
        kc, bt, sl, nb = make_kv(torch, B2, KV2, DQK)
        q = torch.randn(B2, 1, H, DQK, dtype=torch.bfloat16, device="cuda")
        meta = flash_mla.get_mla_metadata()[0]
        ms, _ = bench(lambda: flash_mla.flash_mla_with_kvcache(q, kc, bt, sl, DV, meta))
        rows2b.append({"B": B2, "KV": KV2, "total_tokens": TOT, "ms": round(ms, 4),
                       "per_seq_us": round(ms * 1000 / B2, 1)})
        print("  B=%3d KV=%5d: %8.4f ms  每序列 %6.1f µs" % (B2, KV2, ms, ms * 1000 / B2))
        del kc, q, bt, sl
        torch.cuda.empty_cache()
    if len(rows2b) >= 2:
        lo = min(r["ms"] for r in rows2b)
        hi = max(r["ms"] for r in rows2b)
        print("  ⇒ 总耗时 %.4f~%.4f ms（%.0f%% 波动）——work 总量相同，总代价没降"
              % (lo, hi, (hi / lo - 1) * 100))
        print("     每序列摊销变小 ≠ 固定开销被摊薄；真实量纲是 total_kv_tokens")
    rec["fixed_total_sweep"] = rows2b

    # ---- 轮3：同字节纯 copy 对照
    # ⚠️ 显式写死 B=8：轮2 的循环变量 B 结束时是 32，直接复用会让
    #    「字节数按 B=32 算、耗时却用 B=8 的」——分子分母口径不一致，
    #    算出 1666 GB/s 超过标定峰值 1605。第一次写就踩了。
    print("== 轮3 同字节纯 copy（KV=8192, B=8）==")
    B8 = 8
    byts = KV * B8 * DQK * 2
    n = byts // 4
    src = torch.randn(n, device="cuda")
    dst = torch.empty(n, device="cuda")
    ms_c, _ = bench(lambda: dst.copy_(src))
    gbs_c = 2 * byts / (ms_c * 1e-3) / 1e9      # copy 是 R+W
    if gbs_c > BW_PEAK * 1.02:
        print("  ⚠️ copy 的 R+W 口径 %.0f 超过只读峰值 %.0f — 只读峰值不能当 copy 分母"
              % (gbs_c, BW_PEAK))
    main_ms = [r for r in rows if r["KV"] == KV][0]["ms"]
    print("  copy(R+W) %.4f ms  %.0f GB/s | MLA %.4f ms  %.0f GB/s | 比 %.2fx"
          % (ms_c, gbs_c, main_ms, byts / (main_ms * 1e-3) / 1e9,
             main_ms / ms_c * 2))
    rec["copy_control"] = {"ms": round(ms_c, 4), "gbs_rw": round(gbs_c),
                           "mla_ms": main_ms}

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
