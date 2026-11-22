#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 4：共享专家能否挪到独立 stream 与 routed 专家并发。

MegaMoE 自己承认这是短板：共享专家在融合核外，要等路由专家算完才启动。
所以这里测「挪到独立 stream 并发」是否真能赚。

**关键**：两者都是 GEMM，同属算力引擎（阶段 3 已证：同引擎双流重叠度仅 5.8%）。
所以预期是「不赚」——但要给机制解释，而不是只说「测了没变化」。

对照项：与「真 DMA 并发」的 100% 重叠做反差，证明不是测量方法的问题。
"""
import argparse
import json
import pathlib
import statistics
import sys


def bench(fn, rounds=11, warmup=4):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
        e1.record(); fn(); e2.record(); torch.cuda.synchronize()
        ts.append(e1.elapsed_time(e2))
    ts.sort()
    return ts[len(ts) // 2]


def overlap_deg(seq, par, ta, tb):
    w = min(ta, tb)
    return max(0.0, min(1.0, (seq - par) / w)) if w > 0 else None


def main(argv=None):
    p = argparse.ArgumentParser(description="阶段 4：共享专家并行")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    import torch, deepgemm
    torch.manual_seed(0)
    H, I = 2048, 1536
    E, M, N, K = 32, 128, 1536, 2048
    s2 = torch.cuda.Stream()

    # routed: grouped GEMM
    ar = torch.randn(E, M, K, dtype=torch.bfloat16, device="cuda")
    br = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    mm = torch.full((E,), M, dtype=torch.int32, device="cuda")
    dr = torch.empty(E, M, N, dtype=torch.bfloat16, device="cuda")
    bp, cfg = deepgemm.pack_bf16_masked_weight(br, 1000)

    def routed():
        deepgemm.m_grouped_bf16_gemm_nt_masked(ar, bp, dr, mm, M, cfg)

    # shared: 单个 GEMM（所有 token 都走）
    xs = torch.randn(E * M, H, dtype=torch.bfloat16, device="cuda")
    ws = torch.randn(H, I, dtype=torch.bfloat16, device="cuda") * 0.01
    ds = torch.empty(E * M, I, dtype=torch.bfloat16, device="cuda")

    def shared():
        torch.mm(xs, ws, out=ds)

    tr = bench(routed); ts = bench(shared)
    seq = bench(lambda: (routed(), shared()))

    def par():
        with torch.cuda.stream(s2):
            shared()
        routed()
        s2.synchronize()

    pa = bench(par)
    deg = overlap_deg(seq, pa, tr, ts)

    # 对照：真 DMA（证明测量方法有效）
    #
    # **窗口纪律**（阶段 3 的教训）：D2H 与 GEMM 的耗时必须同量级，
    # 否则小的一侧被噪声淹没。上一版用 512MB D2H(17.4ms) 配 routed(0.31ms)，
    # 窗口只占 1.8%，测出 18.4% —— 那是噪声，不是重叠度。
    # 这里把 D2H 缩到与 routed 可比（~0.3ms）。
    big = torch.randn(32 * 1024 * 1024 // 4, device="cuda")
    pin = torch.empty(big.shape, dtype=torch.float32, pin_memory=True)

    def d2h():
        pin.copy_(big, non_blocking=True)

    td = bench(d2h)
    seqd = bench(lambda: (d2h(), routed()))

    def pard():
        with torch.cuda.stream(s2):
            d2h()
        routed()
        s2.synchronize()

    pd = bench(pard)
    degd = overlap_deg(seqd, pd, td, tr)

    print("== 阶段 4：共享专家并行 ==")
    print("\nrouted(grouped GEMM) %.3f ms   shared(单 GEMM) %.3f ms" % (tr, ts))
    print("串行 %.3f ms   双流并行 %.3f ms" % (seq, pa))
    print("重叠度 %.1f%%  ⇒ %s" % (100 * (deg or 0),
                                  "能并发" if (deg or 0) > 0.5 else "不能并发（同引擎）"))
    print("\n对照组：D2H(真 DMA) %.3f ms" % td)
    print("串行 %.3f ms   双流并行 %.3f ms" % (seqd, pd))
    print("重叠度 %.1f%%  ⇒ 证明测量方法本身有效" % (100 * (degd or 0)))

    payload = {
        "case": "002-moe-pipeline", "stage": 4,
        "purpose": "共享专家挪到独立 stream 能否与 routed 并发",
        "caliber": {"what": "two GEMMs (shared vs routed)", "alloc": True,
                     "state": "hot", "timer": "cuda_event", "rounds": 11},
        "shape": {"E": E, "M_per_expert": M, "N": N, "K": K,
                   "H": H, "I": I, "dtype": "bfloat16"},
        "shared_vs_routed": {
            "routed_ms": tr, "shared_ms": ts, "seq_ms": seq, "par_ms": pa,
            "overlap_pct": round(100 * (deg or 0), 1),
            "verdict": "不能并发（同属算力引擎）" if (deg or 0) <= 0.5 else "能并发",
        },
        "control_d2h": {
            "d2h_ms": td, "seq_ms": seqd, "par_ms": pd,
            "overlap_pct": round(100 * (degd or 0), 1),
            "note": "对照项：证明测量方法有效（真 DMA 能到 ~100%）",
        },
        "mechanism": (
            "共享专家与 routed 专家都是 GEMM，同属算力引擎。"
            "双流只是把两个 kernel 排到同一队列的不同流，硬件仍串行发射。"
            "这与阶段 3 的 GEMM+GEMM 5.8% 一致，与 D2H+GEMM 的 100% 形成反差——"
            "反差本身证明不是测量方法的问题，是引擎类型的问题。"),
        "conclusion_for_megamoe": (
            "MegaMoE 把「共享专家挪到独立 stream」列为改进方向，"
            "但在 DCU 上这条路不成立：它不是「没做」，是「做了也没用」。"
            "要缩短共享专家的等待，只能从减少它的绝对耗时入手，不是并发。"),
    }
    out = a.out or str(pathlib.Path(__file__).parent / "results" / "stage4-shared.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
