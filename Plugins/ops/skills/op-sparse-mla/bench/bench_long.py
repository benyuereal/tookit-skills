#!/usr/bin/env python3
"""跨 S × 上下文长度的站位扫描（生产口径：真随机池）。

与 `bench.py` 的分工：`bench.py` 一次一个档位（shapes.yaml 里的一个键），适合复现某个结论；
本脚本扫一个网格，适合回答「交叉点在哪」。两者都走 `_common`，口径不会分裂。

**为什么必须 pool=full**：ledger §一 的口径红线 —— `build_inputs` 默认的 window 池
在 S=32 时 working set 只有 ~4.8 MB，全在 L2，flash_mla 几乎不付内存代价。
只把池从 377 MB 缩到 2.4 MB，flash_mla 快 1.83×、tilelang 只快 1.03×
⇒ window 池会把「我们落后」测成「我们领先」（本项目真发生过，见 ledger §二 17）。
所以本脚本**不接受**换池，写死 full。

用法：
    # 默认网格：S=1/32/128/4096 × ctx=128k/320k，H=64，topk=2048
    python3 bench/bench_long.py --variants tilelang,flash_mla

    # 厂商 DSA 形状 + V4.1 Flash 生产 topk
    python3 bench/bench_long.py --h 16 --topk 512 --ctx 327680 --s 32,128

    # 带预取变体（走 _common 的 tilelang-prefetch，动态改写厂商源码）
    python3 bench/bench_long.py --variants tilelang,tilelang-prefetch,flash_mla \
            --tilelang-root /path/to/tilelang-hygon

输出：METRIC 行（可 grep）+ 一张 markdown 站位表 + 可选 --json。
比值口径：`vs:base` = 变体 ÷ 基线，**<1 表示变体快**。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    bench_fn, build_inputs, dims_from_params, dump_json, emit_metric,
    env_fingerprint, get_runner, significant, sm_scale_for,
)


def parse_ints(s):
    return [int(x) for x in str(s).replace(" ", "").split(",") if x]


def main(argv=None):
    p = argparse.ArgumentParser(description="跨 S × ctx 站位扫描（生产口径真随机池）")
    p.add_argument("--s", default="1,32,128,4096", help="逗号分隔的 S 列表")
    p.add_argument("--ctx", default="131072,327680", help="逗号分隔的 SKV 列表")
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--d-v", type=int, default=None, help="输出宽度；缺省按档位推导（=d_qk-tail）")
    p.add_argument("--tail-dim", type=int, default=None, help="rope 尾宽；DeepSeek DSA=64，GLM5.3=0")
    p.add_argument("--variants", default="tilelang,flash_mla")
    p.add_argument("--baseline", default=None, help="算比值用的基线（默认第一个成功的变体）")
    p.add_argument("--tilelang-root", default=None, help="tilelang-hygon 仓库根目录")
    p.add_argument("--device", default=None, help="如 cuda:0；默认 cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", default=None, help="结果写到此路径")
    a = p.parse_args(argv)

    Ss, ctxs = parse_ints(a.s), parse_ints(a.ctx)
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    rows = []
    payload = {"op": "sparse-mla", "kind": "grid",
               "caliber": {"what": "op_call", "alloc": True, "state": "hot",
                           "pool": "full（生产口径，真随机池）",
                           "ratio": "vs:base = 变体 ÷ 基线，<1 表示变体快"},
               "env": env_fingerprint(), "rows": rows}

    hdr = " ".join(f"{n:>14}" for n in names)
    print(f"# pool=full（生产口径真随机池） H={a.h} topk={a.topk} "
          f"warmup={a.warmup_ms}ms rep={a.rep_ms}ms repeats={a.repeats}")
    print(f"{'S':>6} {'ctx':>8} {hdr}   | {'vs:base':>8} {'读量MB':>8} {'标称GB/s':>9}")

    for ctx in ctxs:
        for S in Ss:
            params = dict(S=S, H=a.h, SKV=ctx, kv_group=1, topk=a.topk,
                          q_start_s_index=ctx, pool="full")
            if a.tail_dim is not None:
                params["tail_dim"] = a.tail_dim
            if a.d_v is not None:
                params["d_v"] = a.d_v
            try:
                q, kv, idx, meta = build_inputs(params, seed=a.seed, device=a.device)
            except SystemExit as e:          # SKV < topk 等不合法组合
                print(f"{S:>6} {ctx:>8}  skipped: {e}", file=sys.stderr)
                continue

            d_qk, d_v, _tail = dims_from_params(params)
            scale = sm_scale_for(params)
            noms = int(meta["topk"]) * S * int(meta["H"])
            flops = 2.0 * noms * (d_qk + d_v)
            kv_bytes = noms * d_qk * 2                 # gather 读量（每 token 每 head 独立取）

            ms = {}
            for n in names:
                try:
                    run = get_runner(n, tilelang_root=a.tilelang_root, d_v=d_v, sm_scale=scale)
                    run(q, kv, idx)                    # 首次调用含 JIT，不计时
                    st = bench_fn(lambda r=run: r(q, kv, idx), warmup_ms=a.warmup_ms,
                                  rep_ms=a.rep_ms, repeats=a.repeats)
                    ms[n] = {"median_ms": st["median_ms"], "mad_ms": st.get("mad_ms")}
                except Exception as e:                 # 单个变体失败不该拖垮整张表
                    print(f"# {n} @ S={S} ctx={ctx} SKIPPED: {type(e).__name__}: {e}",
                          file=sys.stderr)

            ok = [n for n in names if n in ms]
            if not ok:
                rows.append({"S": S, "SKV": ctx, "H": a.h, "topk": a.topk,
                             "d_qk": d_qk, "d_v": d_v, "status": "all_skipped"})
                print(f"{S:>6} {ctx:>8}  (全部变体失败)", flush=True)
                del q, kv, idx
                continue

            base = a.baseline if a.baseline in ok else ok[0]
            bms = ms[base]["median_ms"]
            vs = {}
            for n in ok:
                if n == base:
                    continue
                passed, ratio, why = significant(bms, ms[n]["median_ms"])
                vs[n] = {"ratio": ratio, "significant": bool(passed), "note": why}

            read_mb = kv_bytes / 2 ** 20
            gbs = kv_bytes / (bms * 1e-3) / 1e9 if bms else None
            row = {"S": S, "SKV": ctx, "H": a.h, "topk": a.topk,
                   "d_qk": d_qk, "d_v": d_v, "d_v_eff": d_v,
                   "params": {k: v for k, v in meta.items() if k != "device"},
                   "ms": ms, "baseline": base, "vs": vs,
                   "read_mb": read_mb, "base_nominal_gbs": gbs,
                   "tflops_base": flops / (bms * 1e-3) / 1e12 if bms else None}
            rows.append(row)

            cells = " ".join(f"{(ms[n]['median_ms'] if n in ms else float('nan')):>14.4f}"
                             for n in names)
            first = next(iter(vs.values()), None)
            tail = (f"{first['ratio']:>8.3f}" if first else f"{'—':>8}")
            print(f"{S:>6} {ctx:>8} {cells}   | {tail} {read_mb:>8.0f} "
                  f"{(gbs or float('nan')):>9.0f}", flush=True)
            for n in ok:
                emit_metric(op="sparse-mla", variant=n,
                            shape=f"S{S}_ctx{ctx}_H{a.h}_topk{a.topk}",
                            ms=ms[n]["median_ms"], mad=ms[n].get("mad_ms"),
                            tflops=flops / (ms[n]["median_ms"] * 1e-3) / 1e12,
                            nominal_gbs=kv_bytes / (ms[n]["median_ms"] * 1e-3) / 1e9,
                            effective_ratio=meta["effective_ratio"], status="ok")
            del q, kv, idx

    if rows:
        done = [r for r in rows if r.get("ms")]
        print()
        if done:
            print("| S | ctx | H | topk | " + " | ".join(names)
                  + f" | {done[0]['baseline']} 标称GB/s | 读量(MB) |")
            print("|---|---|---|---|" + "---|" * (len(names) + 2))
            for r in done:
                cells = " | ".join(
                    "—" if n not in r["ms"] else f"{r['ms'][n]['median_ms']:.4f}" for n in names)
                gbs = "—" if r["base_nominal_gbs"] is None else f"{r['base_nominal_gbs']:.0f}"
                print(f"| {r['S']} | {r['SKV']} | {r['H']} | {r['topk']} "
                      f"| {cells} | {gbs} | {r['read_mb']:.0f} |")
            print(f"\n口径：pool=full（真随机池，生产口径）；单位 ms；"
                  f"供比站位的比值来自 `significant()`（\\|Δ\\| > 3×MAD 才算差异）。")
            print("「标称GB/s」= 名义 gather 读量 ÷ 耗时，**不是 DRAM 带宽**：")
            print("topk 大而池小时同一行数据被反复读，L2 命中会把该值推高到远超 DRAM 峰值")
            print("（如 GLM5.3 S=32/topk2048/ctx128k 算出 ~19.5 TB/s，见 ledger §二 19）。")
            print("用它比「同一档位不同实现」可以；当绝对带宽上限引用是错的。")

    if a.json:
        dump_json(a.json, payload)
        print(f"# wrote {a.json}")
    return 0 if any(r.get("ms") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
