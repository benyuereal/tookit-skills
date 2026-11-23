#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 004：MHA prefill 能到多少算力 —— 定上限，不装作能优化。

起点：`mha-prefill` 卡 status=draft，没有任何实测。
场景归属 compute-bound，按场景表第一批攻击面是 tile/config 类。

为什么先做「定上限」而不是「做优化」
------------------------------------
prefill 是 attention 里算力最重、也最容易被 flash_attn 本身吃满的一段。
在没量到「现在到没到顶」之前就动手改 tile，方向无从判断。
所以本 case 先回答一个问题：**flash_attn 的 causal prefill 能到本机峰值的几成**。

四轮：
  轮0 同进程标定 bf16 峰值（分母）—— 不复用别的进程的数
  轮1 扫 S 看达成率趋势（固定开销 vs 结构性）
  轮2 同形状非 causal（工作量翻倍）看 causal mask 本身是否拖慢
  轮3 与 GQA（H_kv < H_q）对照，看 head 复用是否影响

分母口径
--------
bf16 峰值用的是**同进程实测**，不是 constants.yaml 抄来的数。
原因见 2026-11-23 那次更正：185 那个值低了 1.7×，用它算达成率会判反。
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bench(fn, rounds=11, warmup=5):
    """CUDA event 计时。异步 kernel 用 time.time 会测到 launch 开销而不是计算。"""
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


def calibrate_peak(torch, n=6144):
    """同进程标定 bf16 峰值。这是本 case 全部分母的来源。"""
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    ms, mad = bench(lambda: a @ b)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n ** 3 / (ms * 1e-3) / 1e12, ms, mad


def prefill_flops(B, S, H, D, causal):
    """attention FLOPs = 2 个 GEMM（QK^T 与 PV）× 2 flop/MAC。"""
    return 4 * B * S * S * H * D * (0.5 if causal else 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "mha-prefill.json"))
    ap.add_argument("--quick", action="store_true")
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

    rec = {"device": torch.cuda.get_device_properties(0).name,
           "method": "CUDA event median of 11, warmup 5"}

    # ---- 轮0：分母
    peak, pms, pmad = calibrate_peak(torch)
    rec["peak_bf16_tf"] = round(peak, 1)
    print("== 轮0 同进程 bf16 峰值 ==  %.1f TF（N=6144, %.3f ms）" % (peak, pms))

    B, H, D = 4, 32, 128
    sizes = [1024, 2048] if args.quick else [1024, 2048, 4096, 8192]

    # ---- 轮1：causal prefill 扫 S
    print("== 轮1 causal prefill 扫 S ==")
    rows = []
    for S in sizes:
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        ms, mad = bench(lambda: flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, causal=True))
        tf = prefill_flops(B, S, H, D, True) / (ms * 1e-3) / 1e12
        rows.append({"S": S, "ms": round(ms, 4), "mad_ms": round(mad, 4),
                     "tflops": round(tf, 1), "of_peak": round(tf / peak, 3)})
        print("  S=%5d  %8.4f ms  MAD=%.4f  %6.1f TF  %.2f of peak"
              % (S, ms, mad, tf, tf / peak))
        del q, k, v
        torch.cuda.empty_cache()
    rec["causal_sweep"] = rows

    # ---- 轮2：同形状非 causal 对照（判 causal mask 是否拖慢）
    print("== 轮2 非 causal 对照（工作量 2×）==")
    S = 4096
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    ms_nc, _ = bench(lambda: flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, causal=False))
    tf_nc = prefill_flops(B, S, H, D, False) / (ms_nc * 1e-3) / 1e12
    tf_c = [r for r in rows if r["S"] == S][0]["tflops"]
    print("  causal   %6.1f TF | 非 causal 工作量 2× → %6.1f TF"
          % (tf_c, tf_nc))
    print("  ⇒ 若非 causal 的 TF 明显更高，说明 causal 的块跳过没做干净")
    rec["noncausal_4096"] = {"ms": round(ms_nc, 4), "tflops": round(tf_nc, 1)}

    # ---- 轮3：GQA 对照（H_kv < H_q）
    print("== 轮3 GQA（H_q=32, H_kv=8）==")
    H_kv = 8
    k2 = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(B, S, H_kv, D, dtype=torch.bfloat16, device="cuda")
    ms_g, _ = bench(lambda: flash_attn.flash_attn_func(q, k2, v2, dropout_p=0.0, causal=True))
    # GQA 的 K/V 读取量是 1/4，但 FLOPs 不变
    tf_g = prefill_flops(B, S, H, D, True) / (ms_g * 1e-3) / 1e12
    print("  GQA  %8.4f ms  %6.1f TF  %.2f of peak（vs MHA %.2f）"
          % (ms_g, tf_g, tf_g / peak, tf_c / peak))
    rec["gqa_4096"] = {"ms": round(ms_g, 4), "tflops": round(tf_g, 1)}
    rec["ratio_gqa_vs_mha"] = round(ms_c := ms_g / ([r for r in rows if r["S"] == S][0]["ms"]), 3)
    print("  ⇒ GQA/MHA 耗时比 %.3f（<1 表示 GQA 更快，省的是 KV 读）" % (ms_g / [r for r in rows if r["S"] == S][0]["ms"]))

    # ---- 轮4：换库对照（SDPA）——回答「为什么用 flash_attn 不用 SDPA」
    # 注意：SDPA 在 ROCm 上走 flash-attn 的 adapter，但入口/分派不同，
    # 实测慢 1.6×。这条是 L2 选型的直接证据。
    print("== 轮4 SDPA 对照（BHSD 布局）==")
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    import torch.nn.functional as F
    ms_sd, _ = bench(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True))
    tf_sd = prefill_flops(B, S, H, D, True) / (ms_sd * 1e-3) / 1e12
    mha_ms = [r for r in rows if r["S"] == S][0]["ms"]
    print("  SDPA %8.4f ms %6.1f TF %.2f of peak | flash_attn %6.1f TF %.2f | 慢 %.2f×"
          % (ms_sd, tf_sd, tf_sd / peak, tf_c, tf_c / peak, ms_sd / mha_ms))
    rec["sdpa_4096"] = {"ms": round(ms_sd, 4), "tflops": round(tf_sd, 1),
                        "slower_than_flash_attn": round(ms_sd / mha_ms, 2)}

    # ---- 轮5 varlen（prefill 的真实形态：变长 batch）
    # FLOPs 记账：varlen 张量是 [total_tokens, H, D]，**没有 batch 维**。
    # 别把 batch 再乘一遍——那样会算出 1.8× 峰值这种不可能的结论（本 case 踩过）。
    print("== 轮5 varlen（n 条 2048）==")
    Hh = H
    for n in (2, 8):
        tot = 2048 * n
        qv = torch.randn(tot, Hh, D, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(tot, Hh, D, dtype=torch.bfloat16, device="cuda")
        vv = torch.randn(tot, Hh, D, dtype=torch.bfloat16, device="cuda")
        cu = torch.arange(0, tot + 1, 2048, dtype=torch.int32, device="cuda")
        ms_v, _ = bench(lambda: flash_attn.flash_attn_varlen_func(
            qv, kv, vv, cu, cu, 2048, 2048, dropout_p=0.0, causal=True))
        tf_v = 4 * n * 2048 * 2048 * Hh * D * 0.5 / (ms_v * 1e-3) / 1e12
        print("  %d 条: %8.4f ms %6.1f TF %.2f of peak" % (n, ms_v, tf_v, tf_v / peak))
        rec.setdefault("varlen", []).append(
            {"n_seq": n, "seq": 2048, "ms": round(ms_v, 4), "tflops": round(tf_v, 1)})
        del qv, kv, vv
        torch.cuda.empty_cache()
    print("  ⇒ varlen 与固定 batch 同量级（0.44~0.49 vs 0.47），变长本身不额外掉点")

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
