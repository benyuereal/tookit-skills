#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 009：top-k block 选择 —— 达成率判据不适用，该用「离 launch 地板多远」。

起点：`topk-block-index` 卡 status=draft。

为什么本 case 换判据
--------------------
前几个 case（rope / MLA / GEMM）都是 bandwidth 或 compute 受限，用「达成率」判。
这个算子不是：它读 B·L 个 fp32 打分、写 B·K 个 int32 索引，
L=8192/K=2048 时数据量只有几 MB——**在任何机器上都比 launch 开销小**。

所以本 case 的判据是 **「耗时 ÷ 空 kernel 耗时」**：
  ≈1  ⇒ 已经在下限，任何优化都没有意义
  >>1  ⇒ 有真实计算量，值得看

对照必须包括
------------
1. 空 kernel（`.add_(1)`）——这是 launch 开销的下限
2. `torch.topk` —— 同语义的通用实现，看专用 kernel 值不值

硬前提（实测踩到）
------------------
`page_table` 必须 **2D 且 stride(1)==1**：
  传 1D 得到 `Expected src_page_table.dim() == 2 && src_page_table.stride(1) == 1`
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bench(fn, rounds=13, warmup=6):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "topk-block-index.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        import lightop
    except ImportError as e:
        print("缺依赖：%s" % e, file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rec = {"device": torch.cuda.get_device_properties(0).name,
           "method": "CUDA event median of 13, warmup 6"}

    # ---- 轮0：launch 地板。这是本 case 的「分母」，不是带宽
    e = torch.zeros(1, device="cuda")
    t_launch, mad_l = bench(lambda: e.add_(1))
    rec["launch_floor_us"] = round(t_launch * 1000, 1)
    print("== 轮0 空 kernel（launch 地板）== %.1f µs" % (t_launch * 1000))

    # ---- 轮1：硬前提
    print("== 轮1 page_table 形状约束 ==")
    B, L, K = 8, 8192, 2048
    score = torch.randn(B, L, device="cuda", dtype=torch.float32)
    lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
    page1d = torch.arange(L, dtype=torch.int32, device="cuda")
    try:
        lightop.fast_topk_transform_fused(score, lens, page1d, cu, K)
        print("  1D page_table: 竟然通过（与预期不符）")
    except Exception as ex:
        print("  1D page_table: 拒绝 —— %s" % str(ex)[:90])
    page = page1d.view(1, -1).repeat(B, 1).contiguous()
    out = lightop.fast_topk_transform_fused(score, lens, page, cu, K)
    print("  2D contiguous: OK, out=%s dtype=%s" % (tuple(out.shape), out.dtype))
    rec["page_table_2d_required"] = True

    # ---- 轮2：扫 B / L，看耗时是否随规模变化
    print("== 轮2 扫 B / L（K=2048）==")
    rows = []
    for B2, L2 in ((1, 8192), (8, 8192), (32, 8192), (8, 32768)):
        sc = torch.randn(B2, L2, device="cuda", dtype=torch.float32)
        ln = torch.full((B2,), L2, dtype=torch.int32, device="cuda")
        pg = torch.arange(L2, dtype=torch.int32, device="cuda").view(1, -1).repeat(B2, 1).contiguous()
        cq = torch.arange(0, B2 + 1, dtype=torch.int32, device="cuda")
        ms, mad = bench(lambda: lightop.fast_topk_transform_fused(sc, ln, pg, cq, K))
        byts = B2 * L2 * 4 + B2 * K * 8
        rows.append({"B": B2, "L": L2, "K": K, "ms": round(ms, 4), "mad_ms": round(mad, 4),
                     "over_launch": round(ms / t_launch, 2),
                     "input_mb": round(byts / 1e6, 2)})
        print("  B=%3d L=%6d: %8.4f ms MAD=%.4f  输入 %.1f MB  = %.2f× launch"
              % (B2, L2, ms, mad, byts / 1e6, ms / t_launch))
        del sc, pg
        torch.cuda.empty_cache()
    rec["sweep"] = rows

    # ---- 轮3：与 torch.topk 对照（专用 kernel 值不值）
    print("== 轮3 vs torch.topk（同语义）==")
    cmp_rows = []
    for B2, L2 in ((8, 8192), (32, 8192), (8, 32768)):
        sc = torch.randn(B2, L2, device="cuda", dtype=torch.float32)
        ln = torch.full((B2,), L2, dtype=torch.int32, device="cuda")
        pg = torch.arange(L2, dtype=torch.int32, device="cuda").view(1, -1).repeat(B2, 1).contiguous()
        cq = torch.arange(0, B2 + 1, dtype=torch.int32, device="cuda")
        t_lt, m_lt = bench(lambda: lightop.fast_topk_transform_fused(sc, ln, pg, cq, K))
        t_tt, m_tt = bench(lambda: torch.topk(sc, K, dim=-1))
        sig = abs(t_tt - t_lt) > 3 * max(m_lt, m_tt)
        cmp_rows.append({"B": B2, "L": L2, "lightop_ms": round(t_lt, 4),
                         "torch_topk_ms": round(t_tt, 4),
                         "speedup": round(t_tt / t_lt, 2), "significant": bool(sig)})
        print("  B=%3d L=%6d: lightop %8.4f | torch.topk %8.4f | 快 %.2f×  %s"
              % (B2, L2, t_lt, t_tt, t_tt / t_lt, "显著" if sig else "在噪声内"))
        del sc, pg
        torch.cuda.empty_cache()
    rec["vs_torch_topk"] = cmp_rows

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
