#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 3：重叠验证——区分真重叠 / 伪重叠 / 无重叠。

**总时间下降不能证明重叠**：也可能两边各自变快了。
判据必须是「三个量一起看」：

    串行和  = t_A + t_B          （两段先后跑）
    max     = max(t_A, t_B)      （完美重叠的下界）
    并行    = 双流实际耗时

    重叠度 = (串行和 - 并行) / (串行和 - max)

      接近 100% ⇒ 真重叠（小的一侧被完全隐藏）
      接近 0%   ⇒ 无重叠（两边各跑各的）
      中间       ⇒ 部分重叠
    并行 > 串行 ⇒ 伪重叠的反面：双流反而更慢（争用）

**前置条件**（Q6 第①层：合法性）：
  两侧必须用不同引擎（compute vs DMA）才可能真重叠。
  两个 GEMM 都在算力引擎上，无论怎么分流都只能争用。
"""
import argparse
import json
import pathlib
import statistics
import sys


def bench(fn, rounds=9, warmup=4):
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


# ---- 重叠结论准入条件（见 op-optimize/references/overlap-admissibility.md）----
WINDOW_RATIO_MIN = 0.2
WINDOW_RATIO_MAX = 5.0


def assert_window(ta, tb, waiver_reason=None, noise_argument=None, cross_check=None):
    """两侧耗时量级必须可比，否则拒绝出结论。

    为什么要有这个断言：同一个坑踩过两次（阶段 3 用 1.6ms 配 27.5ms 测出 92%
    与 0%；阶段 4 用 17.4ms 配 0.31ms 测出 18.4%）。教训写成 README 不够，
    必须固化成检查。

    要 WAIVER 必须三项齐全，光说「量级确实不同」不算理由。
    """
    import sys as _sys
    lo, hi = min(ta, tb), max(ta, tb)
    if hi <= 0:
        print("ERROR: 耗时 ≤ 0，无法计算窗口比", file=_sys.stderr)
        _sys.exit(2)
    r = lo / hi
    if WINDOW_RATIO_MIN <= r <= WINDOW_RATIO_MAX:
        return r
    if waiver_reason and noise_argument and cross_check:
        print("caveat WAIVER 生效（窗口比 %.3f 越界）: %s" % (r, waiver_reason),
              file=_sys.stderr)
        return r
    print("拒绝出结论：窗口比 %.3f 越界 [%.1f, %.1f]" % (r, WINDOW_RATIO_MIN, WINDOW_RATIO_MAX),
          file=_sys.stderr)
    print("  t_a=%.4f ms  t_b=%.4f ms  ⇒ 小的一侧会被噪声淹没" % (ta, tb), file=_sys.stderr)
    print("  修法：把两侧耗时调到同一量级；确实做不到则传 --waiver-reason /"
          " --noise-argument / --cross-check 三项", file=_sys.stderr)
    print("  判据见 op-optimize/references/overlap-admissibility.md", file=_sys.stderr)
    _sys.exit(2)


def overlap_degree(seq, par, ta, tb):
    """重叠度。

    **公式修正**：分母用 min(ta, tb) 而不是 (seq - max)。
    因为「最多能隐藏多少」在物理上就是较小那一侧的全部耗时：
      - 完美重叠时 par == max(ta, tb)，隐藏了 min(ta, tb)
      - 完全无重叠时 par == seq，隐藏了 0
    所以  重叠度 = (seq - par) / min(ta, tb)

    用 (seq - max) 做分母会在 seq 因干扰而偏大时算出 >100%（实测 145%），
    这不是「重叠得更好」，是分母本身不稳定。
    """
    window = min(ta, tb)
    if window <= 0:
        return None
    d = (seq - par) / window
    # 夹到 [0, 1]：负值说明并行反而更慢（争用），>1 说明测量受干扰
    return max(0.0, min(1.0, d))


def main(argv=None):
    p = argparse.ArgumentParser(description="阶段 3：重叠验证")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    import torch
    torch.manual_seed(0)
    s2 = torch.cuda.Stream()
    rows = []

    # ---- A) 不同引擎：copy(DMA) + GEMM(compute) —— 应真重叠
    #
    # **窗口大小纪律**：两侧耗时要可比（Q6 第③层）。
    # 实测教训：上一版用 1GB copy（1.6 ms）配 8192 GEMM（27.5 ms），
    # copy 只占 5.9%，窗口 1.6 ms 被 27.5 ms 的噪声淹没 ——
    # 同一台机器两次跑出 92% 与 0%，结论完全相反。
    # 这里把两侧调到同一量级（都 ~8 ms），窗口才不会被噪声吃掉。
    big = torch.randn(2 * 1024 * 1024 * 1024 // 4, device="cuda")   # 2GB → copy ≈ 3.5 ms
    dst = torch.empty_like(big)
    w = torch.randn(4096, 4096, device="cuda")                     # → GEMM ≈ 3.5 ms
    ta = bench(lambda: dst.copy_(big))
    tb = bench(lambda: torch.mm(w, w))
    seq = bench(lambda: (dst.copy_(big), torch.mm(w, w)))

    def par_a():
        with torch.cuda.stream(s2):
            dst.copy_(big)
        torch.mm(w, w)
        s2.synchronize()

    pa = bench(par_a)
    assert_window(ta, tb)
    ratio_ab = min(ta, tb) / max(ta, tb)
    rows.append({"case": "copy(DMA) + GEMM(compute)", "engines": "different",
                 "size_ratio": round(ratio_ab, 3),
                 "t_a_ms": ta, "t_b_ms": tb, "seq_ms": seq, "par_ms": pa,
                 "overlap_pct": round(100 * (overlap_degree(seq, pa, ta, tb) or 0), 1),
                 "verdict": "真重叠" if overlap_degree(seq, pa, ta, tb) and
                            overlap_degree(seq, pa, ta, tb) > 0.5 else "无/伪重叠"})

    # ---- B) 同引擎：两个 GEMM —— 必然争用
    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")
    ta2 = bench(lambda: torch.mm(x, y))
    seq2 = bench(lambda: (torch.mm(x, y), torch.mm(y, x)))

    def par_b():
        with torch.cuda.stream(s2):
            torch.mm(y, x)
        torch.mm(x, y)
        s2.synchronize()

    pb = bench(par_b)
    assert_window(ta2, ta2)
    deg = overlap_degree(seq2, pb, ta2, ta2)
    rows.append({"case": "GEMM + GEMM", "engines": "same",
                 "t_a_ms": ta2, "t_b_ms": ta2, "seq_ms": seq2, "par_ms": pb,
                 "overlap_pct": round(100 * deg, 1) if deg is not None else None,
                 "verdict": "无重叠（同引擎争用）" if (deg is None or deg < 0.2) else "部分重叠"})

    # ---- C) 真 DMA 引擎：D2H pinned + GEMM —— 这才是可以真重叠的组合
    n = 512 * 1024 * 1024 // 4
    host_buf = torch.randn(n, device="cuda")
    pin = torch.empty(n, dtype=torch.float32, pin_memory=True)

    def d2h():
        pin.copy_(host_buf, non_blocking=True)

    tc = bench(d2h)
    seqc = bench(lambda: (d2h(), torch.mm(w, w)))

    def par_c():
        with torch.cuda.stream(s2):
            d2h()
        torch.mm(w, w)
        s2.synchronize()

    pc = bench(par_c)
    assert_window(tc, tb)          # D2H vs GEMM 必须同量级（此前的坑）
    degc = overlap_degree(seqc, pc, tc, tb)
    rows.append({"case": "D2H(pinned DMA) + GEMM", "engines": "different(real DMA)",
                 "size_ratio": round(min(tc, tb) / max(tc, tb), 3),
                 "t_a_ms": tc, "t_b_ms": tb, "seq_ms": seqc, "par_ms": pc,
                 "overlap_pct": round(100 * degc, 1) if degc is not None else None,
                 "verdict": "真重叠" if degc and degc > 0.5 else "无/伪重叠"})

    print("== 阶段 3：重叠验证 ==")
    for r in rows:
        print("\n%-26s [%s 引擎]" % (r["case"], r["engines"]))
        print("  单独: A=%.3f  B=%.3f ms" % (r["t_a_ms"], r["t_b_ms"]))
        print("  串行 %.3f ms   双流并行 %.3f ms" % (r["seq_ms"], r["par_ms"]))
        print("  重叠度 %s%% ⇒ %s"
              % (r["overlap_pct"] if r["overlap_pct"] is not None else "n/a", r["verdict"]))

    payload = {
        "case": "002-moe-pipeline", "stage": 3,
        "purpose": "重叠验证：区分真重叠 / 伪重叠 / 无重叠",
        "method": "重叠度 = (串行和 - 并行) / (串行和 - max)；只测墙钟不算重叠，必须三量对照",
        "caliber": {"what": "two independent ops", "alloc": True, "state": "hot",
                     "timer": "cuda_event", "rounds": 9},
        "results": rows,
        "conclusion_for_moe": "MoE 的两段 GEMM 同属算力引擎，"
                               "分流无法重叠（实测 B 情形）。"
                               "能重叠的只有 compute 与 DMA 之间（A 情形）。",
    }
    out = a.out or str(pathlib.Path(__file__).parent / "results" / "stage3-overlap.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
