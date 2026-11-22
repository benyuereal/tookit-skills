#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 0：MoE pipeline 五段分段计时。

**为什么必须先做这一步**：MegaMoE 能成不是因为它先写了汇编，是因为它先
知道瓶颈在「四次下发 + 中间落盘」。没有段占比，后面所有优化方向都是猜。

五段（按 MoE 前向的真实结构）：
  1 dispatch      gate/softmax → topk → align_block_size → permute
  2 gate_up GEMM  每专家 [2I, H] 的 gate/up 投影
  3 swiglu        silu(gate) * up
  4 down GEMM     每专家 [H, I] 的 down 投影
  5 combine       反 permute + 按 gate 权重加权求和

纪律：
  - 计时只用 CUDA event，**不用 time.time**（异步 kernel 会被 time.time 量成 0）
  - 每段之间 synchronize，避免把前后段的耗时算进来
  - 每段单独 warmup，避免首调虚高
  - 报告 median + MAD，不报均值
"""
import argparse
import json
import pathlib
import statistics
import sys


def _bench(fn, rounds=9, warmup=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    med = ts[len(ts) // 2]
    return med, statistics.median([abs(t - med) for t in ts])


def build_stages(torch, M, H, I, E, K, device):
    """用 torch 实现搭出五段，作为「未融合」的基线（这不是性能候选，是分段占用的探针）。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    # CPU generator 只在 CPU 上造数，再搬到卡上——避免 device 不匹配
    x = torch.randn(M, H, generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    w1 = (torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(device)
    w2 = (torch.randn(E, H, I, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(device)
    logits = torch.randn(M, E, generator=g)
    topk_ids_cpu = logits.topk(K, dim=-1).indices
    gates = torch.softmax(logits.gather(1, topk_ids_cpu), dim=-1).to(torch.float32).to(device)
    topk_ids = topk_ids_cpu.to(torch.int32).to(device)

    state = {}

    def s1_dispatch():
        # gate → topk → 按专家重排（用 sort 近似 align_block_size）
        order = torch.argsort(topk_ids.view(-1))
        state["src"] = order
        state["perm"] = x.repeat_interleave(1, dim=0)[order // K]

    def s2_gate_up():
        xp = state["perm"]
        experts = topk_ids.view(-1)[state["src"]]
        # 逐专家做（未分组，最慢形态；只为量占比）
        acc = torch.empty(xp.shape[0], 2 * I, dtype=torch.bfloat16, device=device)
        for e in range(E):
            idx = (experts == e).nonzero(as_tuple=True)[0]
            if idx.numel():
                acc[idx] = xp[idx] @ w1[e].t()
        state["gu"] = acc

    def s3_swiglu():
        gu = state["gu"]
        state["act"] = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]

    def s4_down():
        xp = state["act"]
        experts = topk_ids.view(-1)[state["src"]]
        acc = torch.empty(xp.shape[0], H, dtype=torch.bfloat16, device=device)
        for e in range(E):
            idx = (experts == e).nonzero(as_tuple=True)[0]
            if idx.numel():
                acc[idx] = xp[idx] @ w2[e].t()
        state["down"] = acc

    def s5_combine():
        down = state["down"]
        inv = torch.argsort(state["src"])
        per_token = down[inv].view(M, K, H).float()
        state["out"] = (per_token * gates.unsqueeze(-1)).sum(1).to(torch.bfloat16)

    return [
        ("dispatch", s1_dispatch),
        ("gate_up_gemm", s2_gate_up),
        ("swiglu", s3_swiglu),
        ("down_gemm", s4_down),
        ("combine", s5_combine),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description="阶段 0：MoE 五段分段计时")
    p.add_argument("--M", type=int, default=4096, help="token 数（P 档大 M）")
    p.add_argument("--H", type=int, default=2048)
    p.add_argument("--I", type=int, default=1408)
    p.add_argument("--E", type=int, default=32)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    import torch
    dev = "cuda"
    stages = build_stages(torch, a.M, a.H, a.I, a.E, a.K, dev)

    rows = []
    total = 0.0
    for name, fn in stages:
        med, mad = _bench(fn, rounds=a.rounds)
        rows.append({"stage": name, "median_ms": med, "mad_ms": mad})
        total += med

    print("== 阶段 0：五段分段计时（M=%d H=%d I=%d E=%d K=%d）==" % (a.M, a.H, a.I, a.E, a.K))
    print("%-16s %10s %10s %8s" % ("stage", "median_ms", "MAD", "占比"))
    for r in rows:
        print("%-16s %10.4f %10.4f %7.1f%%"
              % (r["stage"], r["median_ms"], r["mad_ms"], 100 * r["median_ms"] / total))
    print("%-16s %10.4f %10s %7s" % ("合计", total, "-", "100%"))

    payload = {
        "case": "002-moe-pipeline",
        "stage": 0,
        "purpose": "五段占比基线，用于决定后续优化方向",
        "shape": {"M": a.M, "H": a.H, "I": a.I, "E": a.E, "K": a.K},
        "caliber": {"what": "per-stage kernel time", "alloc": False, "state": "hot",
                     "timer": "cuda_event", "rounds": a.rounds,
                     "note": "每段之间 synchronize，不含前后段"},
        "stages": rows,
        "total_ms": total,
        "not_optimized": "这是未融合基线（逐专家循环），不是性能候选；"
                          "目的是拿占比，不是拿倍数",
    }
    out = a.out or str(pathlib.Path(__file__).parent / "results" / ("stage0_%d.json" % a.M))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
