"""
DCU vLLM 线性层 (M,K,N) 形状分布探针（诊断用）。

在 patch 关闭时安装，统计真实 decode 的 (M,K,N) 调用频次，用 SIGUSR1 dump，
用于决定「该为哪些形状搜专用 config」。
依赖 vllm 的两个 triton_scaled_mm 入口（按需惰性导入，故本文件可独立 py_compile）。

用法（在 sitecustomize 或一次性脚本里）：
    from trace_linear_shapes import install_linear_shape_trace
    install_linear_shape_trace(out_path="/tmp/shape_trace.json")
    # 跑一次 decode 后：kill -SIGUSR1 <pid>  触发 dump
"""
from __future__ import annotations

import json
import os
import signal
from collections import Counter


def install_linear_shape_trace(out_path: str = "") -> None:
    import importlib

    ct_mod = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
    )
    linear_mod = importlib.import_module(
        "vllm.model_executor.kernels.linear.scaled_mm.triton"
    )

    counts: Counter[tuple[int, int, int]] = Counter()
    total = 0
    orig = ct_mod.triton_scaled_mm

    def _wrapped(inp, w, *args, **kwargs):
        nonlocal total
        try:
            counts[(int(inp.shape[0]), int(inp.shape[1]), int(w.shape[1]))] += 1
            total += 1
        except Exception:
            pass
        return orig(inp, w, *args, **kwargs)

    def _dump(signum=None, frame=None):
        del signum, frame
        rows = [
            {"m": m, "k": k, "n": n, "calls": c}
            for (m, k, n), c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        print("[DCU shape trace] total=%d rows=%d" % (total, len(rows)), flush=True)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"total": total, "rows": rows}))

    signal.signal(signal.SIGUSR1, _dump)
    ct_mod.triton_scaled_mm = _wrapped
    linear_mod.triton_scaled_mm = _wrapped
    print("[DCU shape trace] enabled; send SIGUSR1 to dump (M,K,N) counts", flush=True)
