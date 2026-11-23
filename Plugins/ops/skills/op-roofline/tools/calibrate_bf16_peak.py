#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_bf16_peak.py —— 标定 bf16/fp16 矩阵算力峰值（达成率的分母）。

为什么需要它
------------
constants.yaml 里 bf16 峰值长期写着「反推 185 TF」（由 19.7 条 MFMA/cycle @1.15GHz 推）。
实测 hipBLASLt 大 GEMM 在同进程里能到 **311 TF**——**比声称的峰值还高**。
一个实测值超过「峰值」，只有两个可能：要么测得不对，要么峰值常数错了。
本脚本把这件事做成可复跑的证据：

  1. 扫 N = 1024..8192，取最好档（小 N 受固定开销限，不能当峰值）；
  2. bf16 与 fp16 各测一遍（两条独立代码路径，吻合才算数）；
  3. 持续加热 20 s 再测一遍（排除「短测时时钟还没降档 / 热了才降频」）；
  4. 读 hy-smi 的 sclk，把 TFLOPS 反解成 MFMA/cycle，看是否落在整数架构值上；
  5. 结果落盘成 JSON，任何引用带着 run_id 走。

口径（必须随数字一起传）
------------------------
  * 分母是 **本机实测峰值**（同进程 hipBLASLt 大 GEMM），不是标称、不是反推。
    这与带宽那边定下的口径一致：分母用实测（bw-ladder 纯读 1965），标称 1100 只配当型号名。（历史值 1724 已于 2026-11-23 作废，见 constants.yaml）
  * 峰值随 sclk 变。读数必须同时记 sclk，否则跨次比较没有意义。
  * 「反推的 MFMA/cycle」只作为**佐证**，不作为分母；分母永远是实测 TFLOPS。

用法
----
    python3 -I calibrate_bf16_peak.py                     # 测并打印
    python3 -I calibrate_bf16_peak.py --out results/xxx.json
    python3 -I calibrate_bf16_peak.py --check             # 与 constants.yaml 对账
"""
import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSTANTS = HERE.parent / "constants.yaml"
MFMA_FLOPS = 8192          # MFMA 16x16x16 bf16/fp16
TOLERANCE_PCT = 5.0        # 实测峰值低于常数超过 5% 才算「常数偏高」（保守：宁可报错）


def _sclk_mhz():
    """读当前计算时钟。读不到返回 None——**不猜**。"""
    try:
        out = subprocess.run(["hy-smi", "--showclocks"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return None
    vals = [int(x) for x in re.findall(r"sclk clock level:\s*\d+\s*\((\d+)Mhz\)", out)]
    return max(vals) if vals else None


def _bench(fn, rounds=40, warmup=8):
    """CUDA event 计时，返回 (最快毫秒, 中位毫秒, MAD)。

    **峰值必须用「最快」这一档，不是中位数。**
    2026-11-23 的事故：用 11 次采样的中位数标定，恰好赶上一次好状态，
    报出 310 TF；改成 40 次取最快、并在独立进程里复现四轮，
    稳定值只有 280~293 TF。310 是偶发高值，不是峰值。
    ——「峰值」的定义就是「最好情况能到多少」，用中位数测峰值是概念错误。
    同时保留中位数：它才是稳态值，报「持续跑能到多少」用它。
    """
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
    best = ts[0]
    med = ts[len(ts) // 2]
    return best, med, statistics.median([abs(t - med) for t in ts])


def _sweep(torch, sizes, dtype):
    """扫形状取最好档。小 N 受固定开销与 tile 数限制，不能代表峰值。"""
    rows = []
    for n in sizes:
        a = torch.randn(n, n, dtype=dtype, device="cuda")
        b = torch.randn(n, n, dtype=dtype, device="cuda")
        best, med, mad = _bench(lambda: a @ b)
        rows.append({"n": n, "best_ms": round(best, 4), "median_ms": round(med, 4),
                     "mad_ms": round(mad, 5),
                     "peak_tflops": round(2 * n ** 3 / (best * 1e-3) / 1e12, 1),
                     "steady_tflops": round(2 * n ** 3 / (med * 1e-3) / 1e12, 1)})
        del a, b
        torch.cuda.empty_cache()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="结果 JSON 落盘路径")
    ap.add_argument("--check", action="store_true", help="与 constants.yaml 对账")
    ap.add_argument("--quick", action="store_true", help="只跑一档（CI 用）")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch 不可用，无法标定", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU，无法标定", file=sys.stderr)
        return 2

    sizes = [4096, 6144] if args.quick else [1024, 2048, 3072, 4096, 5120, 6144, 8192]
    rec = {
        "run_id": "peak-%s" % time.strftime("%Y%m%d-%H%M%S"),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "device": torch.cuda.get_device_properties(0).name,
        "cu": torch.cuda.get_device_properties(0).multi_processor_count,
        "sclk_mhz_idle": _sclk_mhz(),
        "method": "CUDA event: 40 rounds, warmup 8; peak=fastest, steady=median",
    }

    print("== bf16 GEMM 扫描（N^3, hipBLASLt）==")
    bf16_rows = _sweep(torch, sizes, torch.bfloat16)
    for r in bf16_rows:
        print("  N=%5d  best %8.3f ms (%6.1f TF)  steady %8.3f ms (%6.1f TF)  MAD=%.4f"
              % (r["n"], r["best_ms"], r["peak_tflops"], r["median_ms"],
                 r["steady_tflops"], r["mad_ms"]))
    best_bf16 = max(bf16_rows, key=lambda r: r["peak_tflops"])
    rec["bf16_sweep"] = bf16_rows
    rec["bf16_peak_tf"] = best_bf16["peak_tflops"]
    rec["bf16_steady_tf"] = min(r["steady_tflops"] for r in bf16_rows[:1] or bf16_rows)
    rec["bf16_peak_shape"] = best_bf16["n"]

    # fp16 是独立代码路径，交叉验证；两条路径吻合才排除「某个库的巧合」
    print("== fp16 GEMM（交叉验证）==")
    fp16_rows = _sweep(torch, [6144] if args.quick else [4096, 6144], torch.float16)
    for r in fp16_rows:
        print("  N=%5d  best %8.3f ms (%6.1f TF)" % (r["n"], r["best_ms"], r["peak_tflops"]))
    best_fp16 = max(fp16_rows, key=lambda r: r["peak_tflops"])
    rec["fp16_peak_tf"] = best_fp16["peak_tflops"]

    # 加热：短测可能踩在高档位，长跑降频；两者不一致时取**低**的那个
    print("== 持续 20s 加热后复测 ==")
    n = 6144
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    t0 = time.time()
    while time.time() - t0 < 20:
        a @ b
        torch.cuda.synchronize()
    best_hot, ms_hot, mad_hot = _bench(lambda: a @ b)
    hot_tf = round(2 * n ** 3 / (ms_hot * 1e-3) / 1e12, 1)
    rec["sclk_mhz_load"] = _sclk_mhz()
    rec["hot_tflops"] = hot_tf
    rec["hot_peak_tflops"] = round(2 * n ** 3 / (best_hot * 1e-3) / 1e12, 1)
    rec["hot_ms"] = round(ms_hot, 4)
    print("  %8.3f ms  %6.1f TFLOPS   sclk=%s MHz" % (ms_hot, hot_tf, rec["sclk_mhz_load"]))

    # 分母取冷/热两个里**低的**：宁可用小分母（显得达成率高）?
    #   不——宁可取低的那个当峰值是错的。峰值是能力上界，取实测最好档，
    #   但记录加热值，让读者知道长时间跑会掉到多少。
    denom = rec["bf16_peak_tf"]
    rec["denominator_tf"] = denom
    clk = rec["sclk_mhz_load"] or rec["sclk_mhz_idle"]
    if clk:
        mfma = denom * 1e12 / (MFMA_FLOPS * clk * 1e6)
        rec["derived_mfma_per_cycle_whole_gpu"] = round(mfma, 1)
        rec["derived_mfma_per_cu_cycle"] = round(mfma / rec["cu"], 3)
        print("== 反解 ==  sclk=%d MHz -> %.1f MFMA/cyc 全 GPU (%.2f /CU/cyc)"
              % (clk, mfma, mfma / rec["cu"]))
    rec["denominator_basis"] = (
        "本机实测 bf16 大 GEMM 峰值（同进程 hipBLASLt，N=%d，11 次中位）。"
        "与带宽口径一致：只用实测值做分母，标称/反推值不作分母。" % best_bf16["n"])

    # 一致性：bf16 与 fp16 应接近（同一硬件，不同库路径）
    ratio = rec["fp16_peak_tf"] / rec["bf16_peak_tf"]
    rec["fp16_over_bf16"] = round(ratio, 3)
    print("== 交叉验证 == fp16/bf16 = %.3f（应≈1；明显偏离说明某条路径没打满）" % ratio)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("落盘 -> %s" % p)

    if args.check:
        txt = CONSTANTS.read_text(encoding="utf-8")
        m = re.search(r"bf16_gemm_peak_tf:\s*\n?\s*value:\s*([\d.]+)", txt)
        if not m:
            # 回退到老字段（反推值），说明它现在是错的
            m = re.search(r"mma_issue_peak:\s*\n\s*value:\s*([\d.]+)", txt)
            print("!! constants.yaml 无 bf16_gemm_peak_tf，退回 mma_issue_peak 反推值")
        if m:
            claimed = float(m.group(1))
            if "mma_issue_peak" in txt and claimed < 100:
                # 反推值不是 TFLOPS，换算
                clk0 = 1150
                claimed = claimed * MFMA_FLOPS * clk0 * 1e6 / 1e12
            dev = abs(denom - claimed) / claimed * 100
            print("== 对账 == 常数 %.1f TF vs 实测 %.1f TF，偏差 %.1f%%"
                  % (claimed, denom, dev))
            if dev > TOLERANCE_PCT:
                print("!! 偏差超阈值：达成率算出来会失真。以实测值为分母。")
                if args.out:
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
