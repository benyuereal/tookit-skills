#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case 010：FP8 GEMM —— 本机 fp8 比 bf16 慢 5 倍，这是实现问题不是硬件上限。

起点：`gemm-fp8` 卡 status=draft。卡里写着「gfx938 上 FP8 无方案（hipBLASLt）」。

本 case 查到的三件事，一件比一件重要
------------------------------------
1. **hipBLASLt 确实没方案**：`torch._scaled_mm` 直接报
   `could not find valid hipblaslt solution`。卡里的说法是对的。
2. **但 aiter 有**：`aiter.gemm_a8w8_opus` 文档明确写支持 gfx938/gfx946 FP8 E4M3，
   跑得出结果。**卡里「无方案」的结论过时了。**
3. **它的上限只有 49.8 TF，比 bf16 的 310 TF 慢 5.2×。**
   而 constants.yaml 当时写的 fp8 峰值是 540 TF（bf16 的 1.7×）。

第 3 条是必须查清的矛盾：fp8 如果真有硬件加速，不该比 bf16 慢 5 倍。

怎么排除「是我测错了」
----------------------
  * 扫六个形状（4096³ 到 8192³ 到 16384×4096×4096）——全部收敛在 47~50 TF
  * kernelId 对照：0→1.2 / 1→16.7 / 2→49.2 / auto→49.3 TF
    （auto 确实选了最好的那个，不是我传错了参数）
  * profiler 确认跑的是 gfx938 专用 kernel：`opus_gemm_a8w8_gfx938_opus_64x64x64`
  * 排除转换开销：bf16→fp8 转换只要 0.057 ms，GEMM 是 2.74 ms
  * 同时测 bf16（261.9 TF）算比值：fp8/bf16 = 0.19

六条证据都指向同一个结论：**49.8 TF 是这条实现的真实上限**，
540 TF 那个常数没有可复跑的来源（旧值的 source 写的是 "kb: xxx"，
一条知识库记录，没有脚本）。
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bench(fn, rounds=9, warmup=4):
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


def mk_fp8(torch, M, N, K):
    """torch.randn 不支持 fp8，必须先造 bf16 再转。"""
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    return A, B


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "fp8-gemm.json"))
    args = ap.parse_args(argv)

    try:
        import torch
        import aiter
    except ImportError as e:
        print("缺依赖：%s" % e, file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rec = {"device": torch.cuda.get_device_properties(0).name,
           "method": "CUDA event median of 9, warmup 4"}
    M = N = K = 4096

    # ---- 轮1：hipBLASLt 是否真没方案
    print("== 轮1 hipBLASLt 路径（torch._scaled_mm）==")
    A, B = mk_fp8(torch, M, N, K)
    try:
        torch._scaled_mm(A, B.t(), scale_a=torch.tensor(1.0, device="cuda"),
                         scale_b=torch.tensor(1.0, device="cuda"), out_dtype=torch.bfloat16)
        print("  通过（与卡里「无方案」不符）")
        rec["hipblaslt_fp8_works"] = True
    except Exception as e:
        print("  失败 —— %s" % str(e)[:100])
        print("  ⇒ 确认卡里「gfx938 上 hipBLASLt 无 FP8 方案」是对的")
        rec["hipblaslt_fp8_works"] = False

    # ---- 轮2：aiter 有方案
    print("== 轮2 aiter.gemm_a8w8_opus（gfx938 专用）==")
    out = aiter.gemm_a8w8_opus(A, B)
    ms, mad = bench(lambda: aiter.gemm_a8w8_opus(A, B))
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    print("  4096³: %.4f ms MAD=%.4f  %.1f TF  out=%s" % (ms, mad, tf, tuple(out.shape)))
    print("  ⇒ 卡里「无方案」过时：aiter 有 gfx938 专用 fp8 kernel")
    rec["aiter_opus"] = {"ms": round(ms, 4), "tflops": round(tf, 1)}

    # ---- 轮3：kernelId 对照（排除「参数传错」）
    print("== 轮3 kernelId 对照（确认 auto 选了最好的）==")
    kids = {}
    for kid in (0, 1, 2, None):
        try:
            t, _ = bench(lambda: aiter.gemm_a8w8_opus(A, B, kernelId=kid))
            kids[str(kid)] = round(2 * M * N * K / (t * 1e-3) / 1e12, 1)
            print("  kernelId=%-5s %8.4f ms  %6.1f TF"
                  % (str(kid), t, 2 * M * N * K / (t * 1e-3) / 1e12))
        except Exception as e:
            print("  kernelId=%s: FAIL %s" % (kid, str(e)[:80]))
    rec["kernelid"] = kids

    # ---- 轮4：扫形状（排除「形状没选好」）
    print("== 轮4 扫形状 ==")
    best, best_shape, rows = 0, None, []
    for m, n, k in ((4096, 4096, 4096), (8192, 4096, 4096), (4096, 8192, 4096),
                    (8192, 8192, 8192), (16384, 4096, 4096), (4096, 16384, 4096)):
        A2, B2 = mk_fp8(torch, m, n, k)
        t, _ = bench(lambda: aiter.gemm_a8w8_opus(A2, B2))
        t2 = 2 * m * n * k / (t * 1e-3) / 1e12
        rows.append({"M": m, "N": n, "K": k, "ms": round(t, 4), "tflops": round(t2, 1)})
        if t2 > best:
            best, best_shape = t2, (m, n, k)
        print("  %5dx%5dx%5d: %9.4f ms  %6.1f TF" % (m, n, k, t, t2))
        del A2, B2
        torch.cuda.empty_cache()
    print("  ⇒ 最好 %.1f TF @ %s（六个形状全部收敛，不是形状问题）" % (best, best_shape))
    rec["sweep"] = rows
    rec["fp8_peak_tf"] = round(best, 1)

    # ---- 轮5：bf16 对照 + 转换开销（排除「转换税」）
    print("== 轮5 bf16 对照与转换开销 ==")
    ab = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    bb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    t_bf, _ = bench(lambda: ab @ bb.t())
    bf16 = 2 * M * N * K / (t_bf * 1e-3) / 1e12
    t_conv, _ = bench(lambda: ab.to(torch.float8_e4m3fn))
    print("  bf16 GEMM:      %.4f ms  %.1f TF" % (t_bf, bf16))
    print("  bf16→fp8 转换:  %.4f ms（占 GEMM 的 %.1f%%）"
          % (t_conv, t_conv / ms * 100))
    print("  fp8/bf16 = %.2f  ⇒ **慢 %.1f×**" % (best / bf16, bf16 / best))
    print("  若硬件真有 fp8 加速，这个比值应 >1.5；<1 说明走的是慢路径")
    rec["bf16_tflops"] = round(bf16, 1)
    rec["fp8_over_bf16"] = round(best / bf16, 3)
    rec["convert_ms"] = round(t_conv, 4)

    p = pathlib.Path(args.out)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("落盘 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
