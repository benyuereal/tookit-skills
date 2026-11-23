"""复核 fp8 路径：torch._scaled_mm 在本机是否可用，以及能到多少 TF。

为什么需要它
------------
2026-11-23 gemm-fp8 卡曾写「torch._scaled_mm 报 could not find valid hipblaslt
solution」，据此推出「本机 fp8 栈不成熟 / 还有 11 倍空间」。实测两次都错：

  8192³ = 2205.7 µs = 498.5 TF（89.0% of nominal 560）
  4096³ =  300.3 µs = 457.6 TF（81.7%）
  rel_l2 vs fp32 dequant matmul = 0.0017

口径（对应 TESTING.md 的「口径三问」）
  what=kernel 单发（连续 iters 后取中位）；alloc_in_timing=false（预分配）；
  state=hot（warmup 15 次后）；timer=cuda_event。
  注意：这是**背靠背连跑**口径，与 rank_impls 的「每轮 event 隔离」不同，
  单发时间会比隔离口径偏乐观，跨口径不可直接比。

用法
    python3 -I Plugins/ops/skills/op-roofline/tools/verify_fp8_claim.py
    python3 -I ... --json results/fp8-claim.json
"""
import argparse
import json
import statistics
import sys
import time

import torch

CALIBER = {
    "what": "kernel",
    "alloc_in_timing": False,
    "state": "hot",
    "input_strategy": "reuse_same_buffer",
    "timer": "cuda_event",
    "statistic": "median",
    "warmup": 15,
    "iters": 60,
    "note": "背靠背连跑口径；与 rank_impls 的每轮 event 隔离不同，跨口径不可直接比",
}

print("torch", torch.__version__, "hip", torch.version.hip)
print("preferred_blas:", torch.backends.cuda.preferred_blas_library())

NOMINAL = 560.0
dev = "cuda"


def bench_events(fn, warmup=15, iters=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0


def run(M, N, K):
    print("\n%s" % ("=" * 72))
    print("FP8 %dx%dx%d" % (M, N, K))
    print("=" * 72)

    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b_nk = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    aq = a.to(torch.float8_e4m3fn)
    bq_t = b_nk.to(torch.float8_e4m3fn).t()
    sa = torch.ones(M, 1, device=dev, dtype=torch.float32)
    sb = torch.ones(1, N, device=dev, dtype=torch.float32)

    out = {}

    # 1. _scaled_mm 是否真的可用
    try:
        r = torch._scaled_mm(aq, bq_t, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print("  [OK] torch._scaled_mm 可运行，输出 shape=%s dtype=%s"
              % (tuple(r.shape), r.dtype))
        out["available"] = True

        # 正确性校验（对角 scale=1 时，应与 fp32 反量化 matmul 接近）
        a32 = aq.float()
        b32 = bq_t.float()
        ref = a32 @ b32
        rel = float((r.float() - ref).norm() / ref.norm())
        print("  rel_l2 vs fp32 dequant matmul = %.4f" % rel)
        out["rel_l2"] = round(rel, 4)

        # 多轮计时（排除首轮 JIT）
        meds = []
        for _ in range(5):
            meds.append(bench_events(
                lambda: torch._scaled_mm(aq, bq_t, scale_a=sa, scale_b=sb,
                                         out_dtype=torch.bfloat16),
                iters=40))
        med = statistics.median(meds)
        mad = statistics.median([abs(x - med) for x in meds])
        tf = 2 * M * N * K / (med * 1e-6) / 1e12
        print("  median %.1f us  (MAD %.1f)  = %.1f TF  (%.1f%% of 560)"
              % (med, mad, tf, tf / NOMINAL * 100))
        out.update({"us": round(med, 1), "mad": round(mad, 1),
                    "tflops": round(tf, 1), "pct": round(tf / NOMINAL * 100, 1)})
    except Exception as e:
        print("  [FAIL] torch._scaled_mm: %s: %s"
              % (type(e).__name__, str(e).splitlines()[0][:140]))
        out["available"] = False
        out["error"] = str(e).splitlines()[0][:200]

    return out


res = {}
for shape in ((8192, 8192, 8192), (4096, 4096, 4096), (8192, 4096, 4096)):
    res["%dx%dx%d" % shape] = run(*shape)

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open("/tmp/fp8_verdict.json", "w"), indent=1, ensure_ascii=False)
