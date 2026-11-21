#!/usr/bin/env python3
"""op-sparse-mla 微基准 + 结构消融：一个入口，四种 mode。

  single   （默认）按形状档位测各实现耗时，输出 METRIC 行 + JSON。
  workset  只改 KV 表大小（72KB → 64k 行）。表缩到能进缓存后快多少，就是「内存 vs 计算」
           的直接占比 —— 这是「计算密集还是带宽密集」的**决定性实验**。
  config   block_I / threads / num_stages 扫描（每迭代 GEMM 规模与并行度）。
  split    num_split（SplitKV）扫描：每 query 拆成几个 CTA；含 combine，端到端。

用法：

    python3 bench/bench.py --shape decode-s128 --variants tilelang,flash_mla \\
            --tilelang-root /path/to/tilelang-hygon --json bench.json
    python3 bench/bench.py --mode workset --shape decode-s128 --tilelang-root <repo>/tilelang-hygon
    python3 bench/bench.py --mode config  --shape decode-s128
    python3 bench/bench.py --mode split   --shape decode-s128 --json ablate.json

口径固定为 op_call（含 wrapper 分配）、hot 态、CUDA/HIP event 计时、3 轮取中位数。
绝对耗时只进 run 产物 / kb，不进算子卡。

判读：workset 必须出现**台阶**（耗时随表大小跨过缓存容量掉下来）才算内存受限；
config / split 两档的正向结论要能同时过精度门（split 行带 vs split=1 的 max_abs）。
「结构占比」不等于「可省耗时」，只认实测 —— 删除一段占 7.5% 的代码反而变慢 1.1% 发生过。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import (  # noqa: E402
    bench_fn, build_inputs, compare, dims_from_params, dump_json, emit_metric,
    env_fingerprint, get_runner, load_shape, make_report, runtime_artifacts,
    significant, sm_scale_for, so_snapshot, _load_tilelang_module,
)

MODES = ("single", "workset", "config", "split")


def _bench(fn):
    try:
        return bench_fn(fn, warmup_ms=25.0, rep_ms=100.0, repeats=3)["median_ms"]
    except Exception as e:  # noqa: BLE001
        print("    error: %s: %s" % (type(e).__name__, e))
        return None


def _ablation_runner(mod, d_v, tail, scale, block_I, threads, num_stages, num_split=1):
    """直接用 tilelang 的 kernel 工厂造 runner：不经过 wrapper，配置由调用方钉死。"""
    cache = {}

    def run(q, kv, idx):
        S, H, _ = q.shape
        topk = idx.shape[-1]
        key = (H, topk)
        if key not in cache:
            cache[key] = mod._get_sparse_mla_fwd_kernel(
                H, d_v, tail, topk, int(kv.shape[1]), scale,
                block_I, threads, num_stages, num_split, 0, 1, "bfloat16")
        k = cache[key]
        out = torch.empty((1, S, H, d_v), dtype=q.dtype, device=q.device)
        glse = torch.empty((1, S, num_split, H), dtype=torch.float16, device=q.device)
        partial = torch.empty((1, S, num_split, H, d_v), dtype=torch.float16, device=q.device)
        k(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0), glse, partial, out)
        return out
    return run


def _run_ablation(a):
    shape = load_shape(a.shape)
    base = dict(shape.get("params") or {})
    S, H, topk = int(base.get("S", 128)), int(base.get("H", 16)), int(base.get("topk", 2048))
    SKV = int(base.get("SKV", 65536))
    d_qk, d_v, tail = dims_from_params(base)
    scale = sm_scale_for(base)
    flops = 2 * S * H * topk * (d_qk + d_v)
    traffic = S * topk * d_qk * 2
    print("# mode=%s S=%d H=%d topk=%d d_qk=%d d_v=%d  FLOPs=%.2f GF  需读 KV=%.0f MB"
          % (a.mode, S, H, topk, d_qk, d_v, flops / 1e9, traffic / 2**20))

    mod = _load_tilelang_module(a.tilelang_root or None)
    out = {"shape": shape.get("shape"), "S": S, "H": H, "topk": topk,
           "mode": a.mode, "rows": {}}

    if a.mode == "workset":
        for skv in (64, 512, 4096, SKV):
            q, kv, idx, _ = build_inputs(dict(base, S=S, H=H, SKV=skv, topk=topk),
                                         seed=a.seed, device=a.device)
            fn = _ablation_runner(mod, d_v, tail, scale, 32, 256, 0)
            ms = _bench(lambda: fn(q, kv, idx))
            if ms:
                out["rows"]["skv=%d" % skv] = {"table_kb": skv * d_qk * 2 / 1024, "ms": ms,
                                               "tf": flops / (ms * 1e-3) / 1e12}
                print("SKV=%-7d 表 %8.1f KB  %.4f ms  %5.1f TF" % (
                    skv, skv * d_qk * 2 / 1024, ms, flops / (ms * 1e-3) / 1e12))
            del q, kv, idx
            torch.cuda.empty_cache()

    elif a.mode == "config":
        q, kv, idx, _ = build_inputs(dict(base, S=S, H=H, SKV=SKV, topk=topk),
                                     seed=a.seed, device=a.device)
        for bi in (16, 32, 64):
            for th in (128, 256, 512):
                for ns in (0, 1):
                    fn = _ablation_runner(mod, d_v, tail, scale, bi, th, ns)
                    ms = _bench(lambda: fn(q, kv, idx))
                    tag = "bI=%d,th=%d,ns=%d" % (bi, th, ns)
                    if ms is None:
                        print("%-22s FAIL（该组合编译不过或不支持）" % tag)
                        continue
                    out["rows"][tag] = {"ms": ms, "tf": flops / (ms * 1e-3) / 1e12}
                    print("%-22s %.4f ms  %5.1f TF" % (tag, ms, flops / (ms * 1e-3) / 1e12))

    else:  # split
        for (s_, h_) in ((S, H), (32, H)):
            q, kv, idx, _ = build_inputs(dict(base, S=s_, H=h_, SKV=SKV, topk=topk),
                                         seed=a.seed, device=a.device)
            ref = None
            for ns in (1, 2, 4, 8, 16):
                if topk % (ns * 32) != 0:
                    continue
                mod.get_best_config = (
                    lambda _ns: lambda batch, seq_len, q_heads: (32, 256, 0, _ns, 0, 1))(ns)
                try:
                    got = mod.tilelang_sparse_fwd(q, kv, idx, scale, d_v=d_v).reshape(s_, h_, d_v)
                except Exception as e:  # noqa: BLE001
                    print("S=%-4d num_split=%-3d FAIL %s" % (s_, ns, e))
                    continue
                if ref is None:
                    ref = got
                ms = _bench(lambda: mod.tilelang_sparse_fwd(q, kv, idx, scale, d_v=d_v))
                if ms is None:
                    continue
                tag = "S=%d,num_split=%d" % (s_, ns)
                out["rows"][tag] = {"ms": ms, "tf": flops * s_ / S / (ms * 1e-3) / 1e12,
                                    "gbps": (s_ * topk * d_qk * 2) / (ms * 1e-3) / 1e9,
                                    "max_abs_vs_split1": compare(got, ref)["max_abs"]}
                print("S=%-4d num_split=%-3d %.4f ms  %6.0f GB/s  vs split1 max_abs=%.2e" % (
                    s_, ns, ms, (s_ * topk * d_qk * 2) / (ms * 1e-3) / 1e9,
                    out["rows"][tag]["max_abs_vs_split1"]))
            del q, kv, idx
            torch.cuda.empty_cache()

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("# wrote %s" % a.json)
    return 0


def _run_single(a):
    shape = load_shape(a.shape)
    params = shape.get("params") or {}
    name = shape.get("shape") or Path(shape["_path"]).stem
    q, kv, idx, meta = build_inputs(params, seed=a.seed, device=a.device)

    d_qk, d_v, _tail = dims_from_params(params)
    scale = sm_scale_for(params)
    noms = int(meta["topk"]) * int(meta["S"]) * int(meta["H"])
    flops = 2.0 * noms * (d_qk + d_v)
    kv_bytes = noms * d_qk * 2
    print("# shape=%s d_qk=%d d_v=%d params=%s eff=%.3f sm_scale=%.6f"
          % (name, d_qk, d_v, params, meta["effective_ratio"], scale))

    results = {}
    for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            run = get_runner(v, tilelang_root=a.tilelang_root, d_v=d_v, sm_scale=scale)
            maps_before = so_snapshot()           # /proc/self/maps，抓首次调用新 dlopen 的 .so
            run(q, kv, idx)                       # 首次调用（含 JIT），不计时
            # 执行身份：当场抓「实际加载的产物」。事后猜必错（缓存下 1700+ 个 .so）
            artifacts = runtime_artifacts(v, maps_before)
            stats = bench_fn(lambda r=run: r(q, kv, idx),
                             warmup_ms=a.warmup_ms, rep_ms=a.rep_ms, repeats=a.repeats)
        except Exception as e:                     # noqa: BLE001
            print("# variant=%s SKIPPED: %s: %s" % (v, type(e).__name__, e))
            results[v] = {"status": "skipped", "reason": "%s: %s" % (type(e).__name__, e)}
            continue

        ms = stats["median_ms"]
        results[v] = make_report("sparse-mla", v, name, stats, meta,
                                 {"status": "ok", "artifacts": artifacts,
                                  "tflops": flops / (ms * 1e-3) / 1e12 if ms else None,
                                  "bw_gbs_gathered": kv_bytes / (ms * 1e-3) / 1e9 if ms else None})
        emit_metric(op="sparse-mla", variant=v, shape=name, ms=ms, mad=stats.get("mad_ms"),
                    tflops=results[v]["tflops"], bw_gbs=results[v]["bw_gbs_gathered"],
                    effective_ratio=meta["effective_ratio"], status="ok")

    ok = [v for v, r in results.items() if r.get("status") == "ok"]
    base = a.baseline or (ok[0] if ok else None)
    if base and len(ok) > 1:
        for v in ok:
            if v == base:
                continue
            passed, ratio, why = significant(results[base]["metrics"]["median_ms"],
                                             results[v]["metrics"]["median_ms"])
            results[v]["vs_baseline"] = {"baseline": base, "ratio": ratio,
                                         "significant": bool(passed), "note": why}
            print("# %s vs %s: ratio=%.4f %s" % (v, base, ratio, why))

    payload = {"op": "sparse-mla", "shape": name, "shape_file": shape["_path"],
               "params": meta, "env": env_fingerprint(), "results": results}
    if a.json:
        dump_json(a.json, payload)
        print("# wrote %s" % a.json)
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="op-sparse-mla 微基准 / 结构消融")
    p.add_argument("--mode", choices=MODES, default="single")
    p.add_argument("--shape", default=None,
                   help="形状档位名（shapes.yaml 的顶层键）或 yaml 路径；single 必填，其余缺省 decode-s128")
    p.add_argument("--variants", default="tilelang,flash_mla")
    p.add_argument("--baseline", default=None, help="算比值用的基线变体（默认取第一个成功的变体）")
    p.add_argument("--tilelang-root", default=None, help="tilelang-hygon 仓库根目录")
    p.add_argument("--device", default=None, help="如 cuda:0；默认 cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", default=None, help="结果写到此路径")
    a = p.parse_args(argv)

    if a.mode == "single":
        if not a.shape:
            p.error("--mode single 需要 --shape")
        return _run_single(a)
    a.shape = a.shape or "decode-s128"
    return _run_ablation(a)


if __name__ == "__main__":
    sys.exit(main())
