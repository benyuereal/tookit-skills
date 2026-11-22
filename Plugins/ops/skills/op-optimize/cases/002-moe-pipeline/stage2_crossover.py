#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2：融合 vs 拆分的交叉点 M*。

问题：把 dispatch→GEMM→swiglu→GEMM→combine 拆成多段，
      与「融合进一个算子」相比，在哪个 M 上交叉？

做法：对一组 M 点，分别测
  A) 分段（每段独立 kernel，中间落盘）
  B) 融合（一次调用，中间不落盘 —— 用 lightop 的融合入口近似）
并拟合 M*。

**诚实性**：如果找不到真实可用的融合实现，就不编一个假的对比，
而是如实记录「融合侧未测定」并说明缺什么。
"""
import argparse
import json
import pathlib
import statistics
import sys


def bench(fn, rounds=7, warmup=4):
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


def main(argv=None):
    p = argparse.ArgumentParser(description="阶段 2：融合 vs 拆分交叉点")
    p.add_argument("--E", type=int, default=32)
    p.add_argument("--N", type=int, default=1536)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--expert-ms", default="8,16,32,64,128,256,512",
                   help="M/exp 扫描点（逗号分隔）")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    import torch
    import deepgemm
    torch.manual_seed(0)
    E, N, K = a.E, a.N, a.K
    points = [int(x) for x in a.expert_ms.split(",")]

    rows = []
    print("== 阶段 2：分段 vs 融合（E=%d N=%d K=%d）==" % (E, N, K))
    print("%-8s %12s %12s %8s" % ("M/exp", "分段(ms)", "融合(ms)", "ratio"))
    for M in points:
        ab = torch.randn(E, M, K, dtype=torch.bfloat16, device="cuda")
        bb = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
        mm = torch.full((E,), M, dtype=torch.int32, device="cuda")
        db = torch.empty(E, M, N, dtype=torch.bfloat16, device="cuda")

        # A) 分段：dispatch(近似) + GEMM + swiglu + 落盘
        def split_path():
            bp, cfg = deepgemm.pack_bf16_masked_weight(bb, 1000)
            deepgemm.m_grouped_bf16_gemm_nt_masked(ab, bp, db, mm, M, cfg)
            act = torch.nn.functional.silu(db[:, :, : N // 2]) * db[:, :, N // 2:]
            return act.contiguous()

        # B) 融合：当前 API 下没有「五段合一」的入口。
        #    用 lightop 的 fuse_groupgemm_permute_quant 近似不可行（它不含 attention 结构）。
        #    → 如实记录为不可测。
        try:
            s_med, s_mad = bench(split_path)
        except Exception as e:
            print("M/exp=%-6d 分段 FAIL: %s" % (M, type(e).__name__))
            continue
        rows.append({"m_per_expert": M, "split_ms": s_med, "split_mad": s_mad,
                     "fused_ms": None,
                     "fused_status": "unmeasured: 无五段合一的公开入口"})
        print("%-8d %12.4f %12s %8s"
              % (M, s_med, "未测定", "-"))
        del ab, bb, db
        torch.cuda.empty_cache()

    payload = {
        "case": "002-moe-pipeline", "stage": 2,
        "purpose": "融合 vs 拆分的交叉点 M*",
        "shape": {"E": E, "N": N, "K": K},
        "caliber": {"what": "pipeline", "alloc": True, "state": "hot",
                     "timer": "cuda_event", "rounds": 7},
        "points": rows,
        "crossover": {
            "m_star": None,
            "status": "unmeasured",
            "reason": "融合侧没有可用的公开入口："
                      "lightop.fuse_groupgemm_permute_quant 只做 permute+quant 融合，"
                      "不含 dispatch/combine；aiter.asm_fmoe_stage1/2 是两段式不是五段合一。"
                      "要测 M* 需要先有一个「五段进、一次出」的实现。",
        },
        "what_we_can_say": [
            "分段路径在各 M 点上的耗时可测，见 points",
            "融合侧未测定 ⇒ M* 未测定，不插值",
        ],
    }
    out = a.out or str(pathlib.Path(__file__).parent / "results" / "stage2_crossover.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
