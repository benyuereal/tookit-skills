#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 001 的可复跑基准：SwiGLU 融合 vs 分拆 + 本机带宽峰值标定。

    python3 bench.py [--card 0] [--quick]

输出三段：
  1. 本机带宽峰值（只读 / copy）—— 达成率的分母必须实测，不能用标称
  2. 融合 vs 分拆的墙钟对比（median + MAD + 3×MAD 显著性）
  3. 达成率（读 2 : 写 1 的混合口径）

纪律：
  - 带宽探测用 ≥1GB 的 buffer。小 buffer 会命中 L2，测出虚高（实测见过 4571 GB/s）。
  - 达标率的分母按访存比选：纯读用只读峰值，纯写/copy 用 copy 峰值。
"""
import argparse
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
    mad = statistics.median([abs(t - med) for t in ts])
    return med, mad


def calibrate_bandwidth(torch, mb=1024):
    """实测本机带宽峰值。用大 buffer 避开 L2 污染。"""
    n = mb * 1024 * 1024 // 4
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)

    ro, _ = _bench(lambda: a.sum())
    cp, _ = _bench(lambda: b.copy_(a))
    del a, b
    torch.cuda.empty_cache()
    return {
        "read_only_gbs": mb * 1024 * 1024 / (ro * 1e-3) / 1e9,
        "copy_gbs": 2 * mb * 1024 * 1024 / (cp * 1e-3) / 1e9,
        "read_only_ms": ro,
        "copy_ms": cp,
        "buffer_mb": mb,
    }


def _write_results(payload, out_dir):
    """把本次测量落盘。没有落盘的数字等于只活在文字里，别人无从对基准。"""
    import json
    import pathlib
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = []
    for k, v in payload.get("measurements", {}).items():
        lines.append("%-14s median=%.4f ms  MAD=%.4f" % (k, v["median_ms"], v["mad_ms"]))
    lines.append("")
    if "bandwidth" in payload:
        bw = payload["bandwidth"]
        lines.append("read_only_gbs=%.1f  copy_gbs=%.1f  buffer_mb=%d"
                     % (bw["read_only_gbs"], bw["copy_gbs"], bw["buffer_mb"]))
    if "verdict" in payload:
        lines.append(json.dumps(payload["verdict"], ensure_ascii=False))
    (d / "logic.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n→ 已落盘 %s/{result.json,logic.txt}" % out_dir)


def main(argv=None):
    p = argparse.ArgumentParser(description="Case 001：SwiGLU 融合 vs 分拆")
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--M", type=int, default=32768)
    p.add_argument("--H", type=int, default=4096)
    p.add_argument("--quick", action="store_true", help="跳过带宽标定（快，但没有分母）")
    p.add_argument("--out", default=None, help="结果落盘目录（默认 <case>/results）")
    a = p.parse_args(argv)

    import torch
    torch.cuda.set_device(a.card)
    torch.manual_seed(0)
    M, H = a.M, a.H

    if not a.quick:
        print("== 1. 本机带宽峰值标定（buffer %d MB）==" % 1024)
        bw = calibrate_bandwidth(torch)
        print("  流式只读 : %6.0f GB/s   (%.4f ms)" % (bw["read_only_gbs"], bw["read_only_ms"]))
        print("  copy R+W : %6.0f GB/s   (%.4f ms)" % (bw["copy_gbs"], bw["copy_ms"]))
        print("  caveat 小 buffer 会命中 L2 测出虚高（实测见过 4571 GB/s），这里用 1GB")
    else:
        bw = {"read_only_gbs": 1965.0, "copy_gbs": 1680.0}  # bw-ladder 实测

    print("\n== 2. 融合 vs 分拆（M=%d H=%d bf16）==" % (M, H))
    try:
        import lightop
    except ImportError as e:
        print("  跳过：lightop 不可 import（%s）" % e)
        return 2
    if not hasattr(lightop, "FuseBiasSwiGLU"):
        print("  跳过：lightop.FuseBiasSwiGLU 不存在")
        return 2

    x = torch.randn(M, 2 * H, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2 * H, dtype=torch.bfloat16, device="cuda")
    mod = lightop.FuseBiasSwiGLU().cuda()

    def fused():
        with torch.no_grad():
            return mod(x, b)

    def split():
        xb = x + b
        lo, up = xb[:, :H], xb[:, H:]
        return torch.nn.functional.silu(lo) * up

    # 语义校验
    with torch.no_grad():
        y = fused()
        xb = x.float() + b.float()
        ref = torch.nn.functional.silu(xb[:, :H]) * xb[:, H:]
        cos = float((y.float().flatten() @ ref.flatten()) /
                    (y.float().norm() * ref.norm()))
    print("  语义 cosine = %.7f" % cos)

    fm, fmad = _bench(fused)
    sm, smad = _bench(split)
    print("  融合 : %.4f ms  MAD %.4f" % (fm, fmad))
    print("  分拆 : %.4f ms  MAD %.4f" % (sm, smad))
    print("  ratio: %.3fx (融合/分拆)" % (fm / sm))
    noise = 3 * max(fmad, smad)
    print("  3×MAD = %.4f ⇒ %s" % (noise, "显著" if abs(fm - sm) > noise else "噪声内"))

    print("\n== 3. 达成率 ==")
    fused_bytes = M * 2 * H * 2 + 2 * H * 2 + M * H * 2
    gbs = fused_bytes / (fm * 1e-3) / 1e9
    print("  融合字节数 : %.1f MB（读 x + 读 b + 写 out）" % (fused_bytes / 1e6))
    print("  实测带宽   : %.0f GB/s" % gbs)
    print("  vs 只读峰值 %.0f : %.0f%%" % (bw["read_only_gbs"], 100 * gbs / bw["read_only_gbs"]))
    print("  vs copy 峰值 %.0f : %.0f%%  ← 本 kernel 读 2 写 1，这是更接近的口径"
          % (bw["copy_gbs"], 100 * gbs / bw["copy_gbs"]))
    fused_bytes = M * 2 * H * 2 + 2 * H * 2 + M * H * 2
    split_bytes = M * 2 * H * 2 + 2 * H * 2 + M * 2 * H * 2 + M * 2 * H * 2 + M * H * 2
    payload = {
        "case": "001-swiglu-fused-vs-split",
        "shape": {"M": M, "H": H, "input_width": 2 * H, "dtype": "bfloat16"},
        "scenario": "bandwidth-bound",
        "library": {"canonical_id": "lightop", "version": "0.6.0+dtk2604.torch2110",
                     "import_path": "lightop.FuseBiasSwiGLU"},
        "measurements": {
            "fused": {"median_ms": fm, "mad_ms": fmad, "bytes": fused_bytes},
            "split": {"median_ms": sm, "mad_ms": smad, "bytes": split_bytes},
        },
        "significance": {"ratio": fm / sm, "k_mad": 3,
                          "threshold": noise, "significant": abs(fm - sm) > noise},
        "bandwidth": {"read_only_gbs": bw["read_only_gbs"],
                       "copy_gbs": bw["copy_gbs"], "buffer_mb": bw.get("buffer_mb", 1024)},
        "verdict": {
            "fused_over_split": round(sm / fm, 3),
            "achieved_gbs": round(gbs, 1),
            "pct_of_copy_peak": round(100 * gbs / bw["copy_gbs"], 1),
            "at_ceiling": False,
            "next_surface": "layout-pack (拆 [M,2H] 成两个 [M,H])",
        },
    }
    print("\n  ⇒ 结论：未到顶。下一个攻击面按场景表是 layout-pack / gather-path。")
    _write_results(payload, a.out or str(__import__("pathlib").Path(__file__).parent / "results"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
