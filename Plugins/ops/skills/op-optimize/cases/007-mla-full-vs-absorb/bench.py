#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 007：MLA 非吸收 vs 吸收式 —— 同一库，两个形态差 4.5 倍。

起点：`mla-full` 卡 status=draft，`mla-absorb` 已 measured（decode 达峰值 0.35）。
两张卡都指向 `flash_mla`，但**形态不同**：
  吸收式（absorb）：q_len=1，每步只读 KV，memory-bound
  非吸收（full） ：q_len>1，prefill 形态，理论上是 compute-bound

本 case 要回答
--------------
1. 非吸收形态在本机能到多少？（分母：实测 bf16 峰值）
2. 为什么它比吸收式慢这么多？是算法使然，还是实现问题？
3. 「需要完整注意力权重」这个 `best_for` 理由，值不值这个代价？

等 FLOPs 对照是关键
-------------------
D=576 的 attention 没有现成的 flash_attn 对照（flash_attn 上限 D=256）。
直接比「D=576 的 TFLOPS」与「D=128 的 TFLOPS」是不公平的——
D 变大会增加每元素的 MMA 与 LDS 压力。

所以本 case 用**等 FLOPs** 构造对照：FLOPs 线性于 SQ·D，
把 flash_attn 的 SQ 从 512 放大到 2304（=512×576/128），使两边 FLOPs 都是约 309 G。
只有这个对照才干净。
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bench(fn, rounds=7, warmup=4):
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
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    ms, _ = bench(lambda: a @ b)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n ** 3 / (ms * 1e-3) / 1e12


def check_ceiling(tf, peak, what):
    """超峰值 = 记账错。见 case 006 的同名函数与 opbench.floor_status。"""
    if tf > peak * 1.02:
        raise AssertionError("%s 算出 %.1f TF 超过峰值 %.1f TF——先查 FLOPs 记账" % (what, tf, peak))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "mla-full-vs-absorb.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        import flash_mla
        import flash_attn
    except ImportError as e:
        print("缺依赖：%s" % e, file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    peak = calibrate_peak(torch)
    rec = {"device": torch.cuda.get_device_properties(0).name,
           "peak_bf16_tf": round(peak, 1),
           "method": "CUDA event median of 7, warmup 4"}
    print("== 轮0 同进程 bf16 峰值 == %.1f TF" % peak)

    H, DQK, DV, PB = 32, 576, 512, 64

    # ---- 轮1：MLA 非吸收，扫 q_len
    print("== 轮1 MLA 非吸收（flash_mla dense，q_len 变化）KV=8192 ==")
    KV = 8192
    nb = KV // PB
    kc = torch.randn(nb, PB, 1, DQK, dtype=torch.bfloat16, device="cuda")
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(1, -1)
    sl = torch.full((1,), KV, dtype=torch.int32, device="cuda")
    rows = []
    for SQ in (1, 64, 256, 512, 1024):
        q = torch.randn(1, SQ, H, DQK, dtype=torch.bfloat16, device="cuda")
        meta = flash_mla.get_mla_metadata()[0]      # 每形状重建（见 case 006）
        ms, mad = bench(lambda: flash_mla.flash_mla_with_kvcache(q, kc, bt, sl, DV, meta))
        fl = 4 * 1 * SQ * KV * H * DQK
        tf = fl / (ms * 1e-3) / 1e12
        if SQ > 1:
            check_ceiling(tf, peak, "轮1 SQ=%d" % SQ)
        rows.append({"SQ": SQ, "ms": round(ms, 4), "mad_ms": round(mad, 4),
                     "tflops": round(tf, 1), "of_peak": round(tf / peak, 3)})
        print("  SQ=%5d  %8.4f ms MAD=%.4f  %6.1f TF  %.2f of peak"
              % (SQ, ms, mad, tf, tf / peak))
        del q
        torch.cuda.empty_cache()
    rec["mla_full_sweep"] = rows

    # ---- 轮2：等 FLOPs 对照（flash_attn D=128，放大 SQ）
    print("== 轮2 等 FLOPs 对照（flash_attn D=128）==")
    SQ_mla = 512
    fl_target = 4 * 1 * SQ_mla * KV * H * DQK
    D2 = 128
    SQ_attn = int(round(SQ_mla * DQK / D2))     # FLOPs 线性于 SQ·D
    q2 = torch.randn(1, SQ_attn, H, D2, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(1, KV, H, D2, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(1, KV, H, D2, dtype=torch.bfloat16, device="cuda")
    ms_a, mad_a = bench(lambda: flash_attn.flash_attn_func(q2, k2, v2, dropout_p=0.0, causal=False))
    fl_a = 4 * 1 * SQ_attn * KV * H * D2
    tf_a = fl_a / (ms_a * 1e-3) / 1e12
    check_ceiling(tf_a, peak, "轮2 flash_attn 等FLOPs")
    mla_row = [r for r in rows if r["SQ"] == SQ_mla][0]
    print("  flash_attn SQ=%d D=128: %8.4f ms  %6.1f TF  %.2f of peak  (FLOPs %.0f G)"
          % (SQ_attn, ms_a, tf_a, tf_a / peak, fl_a / 1e9))
    print("  MLA 非吸收 SQ=%d D=576: %8.4f ms  %6.1f TF  %.2f of peak  (FLOPs %.0f G)"
          % (SQ_mla, mla_row["ms"], mla_row["tflops"], mla_row["of_peak"], fl_target / 1e9))
    print("  ⇒ 等 FLOPs 下 flash_attn 快 %.2f×" % (mla_row["ms"] / ms_a))
    rec["equal_flops"] = {"mla_ms": mla_row["ms"], "mla_tflops": mla_row["tflops"],
                          "attn_ms": round(ms_a, 4), "attn_tflops": round(tf_a, 1),
                          "speedup_attn_over_mla": round(mla_row["ms"] / ms_a, 2),
                          "sq_mla": SQ_mla, "sq_attn": SQ_attn, "d_mla": DQK, "d_attn": D2}
    del q2, k2, v2
    torch.cuda.empty_cache()

    # ---- 轮3：吸收式（q_len=1）放大到同样工作量需要多少次调用
    print("== 轮3 吸收式的等价工作量 ==")
    q1 = torch.randn(1, 1, H, DQK, dtype=torch.bfloat16, device="cuda")
    meta = flash_mla.get_mla_metadata()[0]
    ms_1, _ = bench(lambda: flash_mla.flash_mla_with_kvcache(q1, kc, bt, sl, DV, meta))
    total_absorb = ms_1 * SQ_mla
    print("  单次 q_len=1: %.4f ms；%d 个 query 需 %d 次调用 = %.2f ms"
          % (ms_1, SQ_mla, SQ_mla, total_absorb))
    print("  vs 非吸收一次做完 %.2f ms ⇒ 吸收式分步反而%s %.2f×"
          % (mla_row["ms"], "慢" if total_absorb > mla_row["ms"] else "快",
             max(total_absorb, mla_row["ms"]) / min(total_absorb, mla_row["ms"])))
    rec["absorb_equivalent"] = {"per_call_ms": round(ms_1, 4), "calls": SQ_mla,
                                "total_ms": round(total_absorb, 2)}

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
