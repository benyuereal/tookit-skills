#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2b：融合三级阶梯的拐点（可执行版 M*）。

阶段 2 的二元对比（融合 vs 拆分）做不出结论——全融合侧不可用。
改成三级阶梯：

  L0 全拆分     手工 permute → deepgemm grouped GEMM → swiglu
  L1 部分融合   lightop.fuse_groupgemm_permute_quant（融掉 topk+permute+quant）
  L2 全融合     asm_fmoe_a8（不可用，见 failure-modes.yaml）

**L1 才是可执行的答案**——「从哪一级开始换」比「能不能一步到位」有用。

判据：在哪个 M 上 L0 与 L1 交叉。
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


def main(argv=None):
    p = argparse.ArgumentParser(description="阶段 2b：融合三级阶梯拐点")
    p.add_argument("--expert-ms", default="16,32,64,128,256")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    import torch, lightop, deepgemm
    torch.manual_seed(0)
    E, H, N, K = 32, 2048, 1536, 2048
    points = [int(x) for x in a.expert_ms.split(",")]

    rows = []
    print("%-8s %12s %14s %10s %8s" % ("M/exp", "L0 全拆分", "L1 部分融合", "L1/L0", "谁快"))
    for M in points:
        tok = E * M
        aa = torch.randn(E, M, K, dtype=torch.bfloat16, device="cuda")
        bb = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
        mm = torch.full((E,), M, dtype=torch.int32, device="cuda")
        dd = torch.empty(E, M, N, dtype=torch.bfloat16, device="cuda")
        bp, cfg = deepgemm.pack_bf16_masked_weight(bb, 1000)
        tokens = torch.randn(tok, H, dtype=torch.bfloat16, device="cuda")
        rm = torch.zeros(tok, E, dtype=torch.float32, device="cuda")
        rm[torch.arange(tok), torch.arange(tok) // M] = 1.0

        def l0():
            deepgemm.m_grouped_bf16_gemm_nt_masked(aa, bp, dd, mm, M, cfg)
            return torch.nn.functional.silu(dd[:, :, : N // 2]) * dd[:, :, N // 2:]

        def l1():
            return lightop.fuse_groupgemm_permute_quant(tokens, rm)

        t0 = bench(l0); t1 = bench(l1)
        who = "L1 部分融合" if t1 < t0 else "L0 全拆分"
        rows.append({"m_per_expert": M, "l0_split_ms": t0, "l1_partial_ms": t1,
                     "ratio_l1_over_l0": round(t1 / t0, 3), "faster": who})
        print("%-8d %12.4f %14.4f %9.2fx %8s" % (M, t0, t1, t1 / t0, who))
        del aa, bb, dd, tokens, rm
        torch.cuda.empty_cache()

    # 找交叉点：ratio 跨过 1.0 的区间
    cross = None
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if (prev["ratio_l1_over_l0"] - 1) * (cur["ratio_l1_over_l0"] - 1) < 0:
            cross = (prev["m_per_expert"], cur["m_per_expert"])
            break

    print("\n交叉点在 M/exp ∈ [%s, %s]" % cross if cross else "\n未观察到交叉")
    print("⇒ M/exp 小于交叉点用 **L1 部分融合**；大于则用 **L0 全拆分**")
    print("（L2 全融合不可用，见 op-moe-w8a8/references/failure-modes.yaml）")

    payload = {
        "case": "002-moe-pipeline", "stage": "2b",
        "purpose": "融合三级阶梯的拐点（可执行版 M*）",
        "ladder": {
            "L0": "全拆分：手工 permute → deepgemm grouped → swiglu",
            "L1": "部分融合：lightop.fuse_groupgemm_permute_quant（topk+permute+quant 融）",
            "L2": "全融合：asm_fmoe_a8 —— 不可用（failure-modes 第 1 条 hard）",
        },
        "caliber": {"what": "pipeline", "alloc": True, "state": "hot",
                     "timer": "cuda_event", "rounds": 9},
        "shape": {"E": E, "H": H, "N": N, "K": K, "dtype": "bfloat16"},
        "points": rows,
        "crossover_ms_per_expert": cross,
        "conclusion": (
            "M/exp < %s 用 L1 部分融合；M/exp > %s 用 L0 全拆分。"
            % (cross[1] if cross else "?", cross[0] if cross else "?")
            if cross else "未观察到交叉点，全部区间 L0 更快"),
    }
    out = a.out or str(pathlib.Path(__file__).parent / "results" / "stage2b-ladder.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
