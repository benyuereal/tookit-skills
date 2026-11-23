#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 005：bf16 GEMM 的 M 拐点 —— 同一张卡，小 M 与大 M 是两个不同的算子。

起点：`gemm-bf16` 卡 status=draft。它是所有量化档的基准，
但「基准」这个角色只有在说清**它在哪个形状受什么约束**之后才成立。

要回答的问题
------------
1. 小 M（decode）与大 M（prefill）的瓶颈是不是同一个？
2. 拐点在哪？——用什么判据定拐点，而不是拍脑袋选形状？
3. 同一形状，还有没有更快的路径（对照实现）？

判据（先定判据再测）
--------------------
用「时间 / 两个地板」谁更小：
  内存地板 = 字节 / 1724 GB/s
  算力地板 = FLOPs / 310 TF
两者比值的倒数就是各自的理论加速比。哪个地板在实测里先顶到，就是那个约束。
**不靠肉眼看 TFLOPS 数字跳动定性。**

五轮：
  轮0 同进程标定两个分母（bf16 峰值 + 流式带宽），避免跨进程漂移
  轮1 扫 M（K=N=4096），看达成率 vs M
  轮2 扫 M 时同时报「内存地板 / 算力地板」，定拐点
  轮3 换 N=14336（SwiGLU 的 up+gate 合并宽度）复测拐点是否随之移动
  轮4 换路径对照（torch.mm vs F.linear vs aiter），看同一形状谁最快
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
BW_PEAK = 1724.0          # constants.yaml:memory.stream_read_peak（实测流式只读）


def bench(fn, rounds=11, warmup=5):
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


def calibrate(torch):
    """两个分母都同进程测。峰值用大 GEMM，带宽用流式只读。"""
    a = torch.randn(6144, 6144, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(6144, 6144, dtype=torch.bfloat16, device="cuda")
    ms, _ = bench(lambda: a @ b)
    peak = 2 * 6144 ** 3 / (ms * 1e-3) / 1e12
    del a, b
    torch.cuda.empty_cache()
    x = torch.randn(1 << 26, device="cuda")
    msb, _ = bench(lambda: x.sum())
    bw = x.numel() * 4 / (msb * 1e-3) / 1e9
    del x
    torch.cuda.empty_cache()
    return peak, bw


def floors(M, N, K, peak_tf, bw_gbs):
    """返回 (内存地板 ms, 算力地板 ms)。地板 = 该约束下理论最短时间。"""
    byts = (M * K + N * K + M * N) * 2
    mem_ms = byts / (bw_gbs * 1e9) * 1e3
    cmp_ms = 2 * M * N * K / (peak_tf * 1e12) * 1e3
    return mem_ms, cmp_ms, byts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "gemm-crossover.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        print("缺依赖：%s" % e, file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    peak, bw = calibrate(torch)
    rec = {"device": torch.cuda.get_device_properties(0).name,
           "peak_bf16_tf": round(peak, 1), "bw_stream_gbs": round(bw),
           "method": "CUDA event median of 11, warmup 5"}
    print("== 轮0 同进程分母 == bf16 峰值 %.1f TF, 流式带宽 %.0f GB/s" % (peak, bw))

    # ---- 轮1/2：扫 M，同时报两个地板
    print("== 轮1/2 K=N=4096，扫 M（判据：谁的地板先顶到）==")
    print("  %6s %9s %8s %7s %7s %9s %9s %s"
          % ("M", "ms", "TFLOPS", "of peak", "GB/s", "内存地板", "算力地板", "约束"))
    rows = []
    for M in (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        N = K = 4096
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ms, mad = bench(lambda: F.linear(a, b))
        tf = 2 * M * N * K / (ms * 1e-3) / 1e12
        mem_ms, cmp_ms, byts = floors(M, N, K, peak, bw)
        # 约束判定：实测更接近哪个地板，就是哪个约束。
        # 用地板占实测的比例：占比高的那个说明「理论最短时间」已经逼近实测。
        mem_frac, cmp_frac = mem_ms / ms, cmp_ms / ms
        bound = "memory" if mem_frac > cmp_frac else "compute"
        rows.append({"M": M, "N": N, "K": K, "ms": round(ms, 4), "mad_ms": round(mad, 4),
                     "tflops": round(tf, 1), "of_peak": round(tf / peak, 3),
                     "gbs": round(byts / (ms * 1e-3) / 1e9),
                     "mem_floor_ms": round(mem_ms, 4), "cmp_floor_ms": round(cmp_ms, 4),
                     "bound": bound})
        print("  %6d %9.4f %8.1f %7.2f %7.0f %9.4f %9.4f %s"
              % (M, ms, tf, tf / peak, byts / (ms * 1e-3) / 1e9, mem_ms, cmp_ms, bound))
        del a, b
        torch.cuda.empty_cache()
    rec["sweep_m_4096"] = rows

    # ---- 轮3：换 N=14336，看拐点是否随宽度移动
    print("== 轮3 N=14336（SwiGLU up+gate 宽度），看拐点是否移动 ==")
    rows2 = []
    for M in (32, 128, 512, 2048):
        N, K = 14336, 4096
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ms, _ = bench(lambda: F.linear(a, b))
        tf = 2 * M * N * K / (ms * 1e-3) / 1e12
        mem_ms, cmp_ms, byts = floors(M, N, K, peak, bw)
        bound = "memory" if mem_ms / ms > cmp_ms / ms else "compute"
        rows2.append({"M": M, "N": N, "K": K, "ms": round(ms, 4),
                      "tflops": round(tf, 1), "of_peak": round(tf / peak, 3),
                      "gbs": round(byts / (ms * 1e-3) / 1e9), "bound": bound})
        print("  M=%5d  %8.4f ms  %6.1f TF  %.2f of peak  %5.0f GB/s  %s"
              % (M, ms, tf, tf / peak, byts / (ms * 1e-3) / 1e9, bound))
        del a, b
        torch.cuda.empty_cache()
    rec["sweep_m_14336"] = rows2

    # ---- 轮4：同一形状换路径
    print("== 轮4 M=4096 换路径对照 ==")
    M = N = K = 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    bt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")   # mm 要 [K,N]
    bnt = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # linear 要 [N,K]
    variants = {}
    ms_lin, _ = bench(lambda: F.linear(a, bnt))
    variants["F.linear"] = round(ms_lin, 4)
    ms_mm, _ = bench(lambda: torch.mm(a, bt))
    variants["torch.mm"] = round(ms_mm, 4)
    print("  F.linear %.4f ms | torch.mm %.4f ms | 差 %.1f%%"
          % (ms_lin, ms_mm, (ms_mm / ms_lin - 1) * 100))
    try:
        import aiter
        # aiter 的 gemm 入口随版本变；找得到就测，找不到就诚实标未测定
        cand = None
        for name in ("gemm", "gemm2", "linear"):
            f = getattr(aiter, name, None)
            if callable(f):
                cand = name
                break
        if cand:
            ms_ai, _ = bench(lambda: getattr(aiter, cand)(a, bnt.t()))
            variants["aiter." + cand] = round(ms_ai, 4)
            print("  aiter.%s %.4f ms（vs F.linear %.2fx）"
                  % (cand, ms_ai, ms_ai / ms_lin))
        else:
            print("  aiter: 无可用顶层 gemm 入口（未测定，不插值）")
    except Exception as e:
        print("  aiter: 不可用（%s），未测定" % str(e)[:60])
    rec["paths_4096"] = variants

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
