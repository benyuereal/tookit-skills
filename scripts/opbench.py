#!/usr/bin/env python3
"""opbench — 算子微基准的通用计时与统计工具（标准库优先，torch 可选）。

设计原则（见 Plugins/ops/skills/op-bench/SKILL.md）：
  1. GPU 侧计时用 CUDA/HIP event，不用 time.time()（异步 queue 只量到 launch）
  2. 一律多轮取中位数 + MAD，禁止「跑两次看着差不多」
  3. 显著性用 |Δ| > k × MAD，k 默认 3
  4. 地板判定按「达成率 ≥ 阈值」判 at_floor，阈值默认 85%

用法（库）：
    from opbench import bench_fn, significant, floor_status, emit_metric
    s = bench_fn(lambda: kernel(...), warmup_ms=25, rep_ms=100, repeats=3)
    ok, ratio, why = significant(base, s)
    print(floor_status(1020, 1200))

用法（自检，不需要 DCU）：
    python3 -I scripts/opbench.py --selftest
"""

import argparse
import json
import statistics
import sys
import time

DEFAULT_K = 3.0
DEFAULT_FLOOR_THRESHOLD = 0.85


# ---------------------------------------------------------------- 统计

def summarize(samples, unit="ms"):
    """中位数 + MAD（中位数绝对偏差）。samples 为同口径的逐轮耗时。"""
    xs = [float(x) for x in samples if x is not None]
    if not xs:
        return {"n": 0, "median_" + unit: None, "mad_" + unit: None}
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs]) if len(xs) > 1 else 0.0
    return {
        "n": len(xs),
        "median_" + unit: med,
        "mad_" + unit: mad,
        "min_" + unit: min(xs),
        "max_" + unit: max(xs),
        "mean_" + unit: statistics.fmean(xs),
    }


def _as_median_mad(v, key="median_ms"):
    """接受 float 或 bench_fn 的返回 dict，统一取出 (median, mad)。"""
    if isinstance(v, dict):
        med = v.get(key)
        mad = v.get(key.replace("median_", "mad_"), 0.0) or 0.0
        return med, mad
    return float(v), 0.0


def significant(base, cand, k=DEFAULT_K, lower_is_better=True):
    """判定 cand 相对 base 的变化是否超出噪声。

    返回 (过门?, ratio, 说明)。ratio = cand/base 的耗时比：
      ratio < 1 表示 cand 更快（lower_is_better=True）。
    噪声门：|cand - base| <= k * max(mad_base, mad_cand) 视为无差异。
    """
    b, bmad = _as_median_mad(base)
    c, cmad = _as_median_mad(cand)
    if b is None or c is None:
        return False, None, "缺少数值，无法判定"
    ratio = c / b if b else float("inf")
    noise = k * max(bmad, cmad)
    delta = abs(c - b)
    if delta <= noise:
        return False, ratio, "在噪声内（|Δ|=%.6g <= %.1f*MAD=%.6g），判 INCONCLUSIVE" % (delta, k, noise)
    better = (ratio < 1.0) if lower_is_better else (ratio > 1.0)
    return better, ratio, "显著：ratio=%.4f，|Δ|=%.6g > %.1f*MAD=%.6g" % (ratio, delta, k, noise)


def floor_status(achieved, peak, threshold=DEFAULT_FLOOR_THRESHOLD):
    """达成率判定。achieved/peak 同单位（GB/s、TFLOPS 等）。"""
    if not peak:
        return "unknown", 0.0
    r = float(achieved) / float(peak)
    return ("at_floor" if r >= threshold else "headroom"), r


# ---------------------------------------------------------------- 计时

def _cuda_available():
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _time_batch(fn, iters):
    """执行 iters 次 fn 并返回每次的平均毫秒数。"""
    if _cuda_available():
        import torch
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def bench_fn(fn, warmup_ms=25.0, rep_ms=100.0, repeats=3, min_iters=3, max_iters=100000):
    """对可调用对象做多轮计时，返回 summarize() 结构（median_ms / mad_ms / ...）。

    口径：hot 态；含 fn 内部一切开销（若 fn 内含分配，则含分配）。
    """
    # 校准：先跑 min_iters 次估单次耗时
    t = _time_batch(fn, min_iters) or 1e-6
    warm_iters = max(min_iters, min(max_iters, int(warmup_ms / max(t, 1e-6))))
    _time_batch(fn, warm_iters)
    rep_iters = max(min_iters, min(max_iters, int(rep_ms / max(t, 1e-6))))
    samples = [_time_batch(fn, rep_iters) for _ in range(repeats)]
    out = summarize(samples)
    out["iters_per_round"] = rep_iters
    out["caliber"] = {"what": "callable", "alloc": "depends_on_fn", "state": "hot",
                      "timer": "cuda_event" if _cuda_available() else "wall_clock"}
    return out


# ---------------------------------------------------------------- 输出

def emit_metric(**kv):
    """输出一行可被 grep/解析的 METRIC。"""
    order = ["op", "variant", "shape", "ms", "mad", "tflops", "bw_gbs", "effective_ratio", "status"]
    parts = ["METRIC"]
    keys = [k for k in order if k in kv] + [k for k in kv if k not in order]
    for k in keys:
        v = kv[k]
        if isinstance(v, float):
            v = "%.6g" % v
        parts.append("%s=%s" % (k, v))
    line = " ".join(parts)
    print(line)
    return line


def dump_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
    return path


# ---------------------------------------------------------------- 自检

def selftest():
    ok = True

    s = summarize([1.0, 1.1, 0.9, 1.0, 1.2])
    assert s["n"] == 5 and abs(s["median_ms"] - 1.0) < 1e-9, s
    assert abs(s["mad_ms"] - 0.1) < 1e-9, s
    print("[ok] summarize  median=%.3f mad=%.3f" % (s["median_ms"], s["mad_ms"]))

    base = {"median_ms": 10.0, "mad_ms": 0.2}
    cand = {"median_ms": 8.0, "mad_ms": 0.2}
    passed, ratio, why = significant(base, cand)
    assert passed and abs(ratio - 0.8) < 1e-9, (passed, ratio, why)
    print("[ok] significant 外部显著  ratio=%.3f %s" % (ratio, why))

    near = {"median_ms": 10.1, "mad_ms": 0.2}
    passed2, ratio2, why2 = significant(base, near)
    assert not passed2, (passed2, ratio2, why2)
    print("[ok] significant 噪声内    %s" % why2)

    st, r = floor_status(1020, 1200)
    assert st == "at_floor" and abs(r - 0.85) < 1e-9, (st, r)
    st2, r2 = floor_status(600, 1200)
    assert st2 == "headroom", (st2, r2)
    print("[ok] floor_status  %.2f -> %s" % (r, st))

    t = bench_fn(lambda: sum(range(1000)), warmup_ms=5, rep_ms=20, repeats=2, min_iters=2)
    assert t["median_ms"] is not None and t["n"] == 2, t
    print("[ok] bench_fn      median=%.4f ms (timer=%s)" % (t["median_ms"], t["caliber"]["timer"]))

    line = emit_metric(op="demo", variant="v1", shape="s1", ms=1.2345, status="ok")
    assert line.startswith("METRIC ") and "ms=1.2345" in line, line
    print("[ok] emit_metric   %s" % line)

    print("\nselftest %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="opbench — 算子微基准计时与统计")
    p.add_argument("--selftest", action="store_true", help="不需要 DCU 的自检")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
