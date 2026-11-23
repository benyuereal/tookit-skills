#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-rmsnorm-fused A/B 适配器：把本卡测量喂给 loop.py run。

契约见 loop.py 的 ADAPTER_AB。测量一律走 opbench.bench_fn（median + MAD），
不在这里另写一套计时——口径不一致是本项目最贵的一类错。

**base 每轮重测**：只报一次 base 会把频率漂移算成收益。

    python3 bench/ab.py --json out.json --variants lightop-4stage --rounds 3
"""
import argparse
import json
import sys
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))
sys.path.insert(0, str(CARD.parents[3] / "scripts"))

from opbench import bench_fn, significant  # noqa: E402
import _common as C  # noqa: E402


def _measure(fn, inp, rounds, warmup_ms, rep_ms):
    meds = []
    for _ in range(rounds):
        st = bench_fn(lambda: fn(*inp[:-1]), warmup_ms=warmup_ms, rep_ms=rep_ms, repeats=3)
        meds.append(st["median_ms"])
    meds.sort()
    n = len(meds)
    med = meds[n // 2] if n % 2 else 0.5 * (meds[n // 2 - 1] + meds[n // 2])
    mad = sorted(abs(x - med) for x in meds)[n // 2]
    return med, mad


def main(argv=None):
    p = argparse.ArgumentParser(description="op-rmsnorm-fused A/B 适配器")
    p.add_argument("--json", default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shape", default="decode-m32")
    p.add_argument("--variants", required=True, help="逗号分隔的变体名")
    p.add_argument("--base", default="ref-heavy")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    a = p.parse_args(argv)

    shape = C.load_shape(a.shape)
    params = dict(shape.get("params") or {})
    kw = C.runner_kwargs(params)
    inp = C.build_inputs(params, seed=a.seed, device=a.device)

    row = {"skv": 0, "variants": []}
    try:
        bmed, bmad = _measure(C.get_runner(a.base, **kw), inp,
                              a.rounds, a.warmup_ms, a.rep_ms)
    except Exception as e:                       # noqa: BLE001
        print("base=%s 失败: %s: %s" % (a.base, type(e).__name__, e), file=sys.stderr)
        return 1
    row["base_ms"], row["base_mad"] = bmed, bmad

    for tag in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            med, mad = _measure(C.get_runner(tag, **kw), inp, a.rounds,
                                a.warmup_ms, a.rep_ms)
        except Exception as e:                   # noqa: BLE001
            print("variant=%s 跳过: %s: %s" % (tag, type(e).__name__, e), file=sys.stderr)
            continue
        sig, ratio, why = significant({"median_ms": bmed, "mad_ms": bmad},
                                      {"median_ms": med, "mad_ms": mad})
        row["variants"].append({"tag": tag, "ms": med, "mad": mad,
                                "delta_pct": (ratio - 1.0) * 100.0 if ratio else None,
                                "significant": sig, "why": why,
                                "maxdiff": None, "rel": None})
        print("# %s: %.4fms (%.4fx) %s" % (tag, med, ratio or 0, why))
        print("# ⚠ 口径=背靠背无 per-iter sync，偏乐观约 23%；"
              "卡片口径用端到端（183.0 µs / util 0.63）")

    # ⚠️ 测量口径必须写进 JSON，否则读的人会把「背靠背无 sync」的数当成绩。
    # 2026-11-23 三方对账（业主亲自跑，同机 prefill-8k）：
    #   ab.py 背靠背无 per-iter sync   140 µs  偏乐观 23%（kernel 间重叠）
    #   events 隔离 + 预分配 y         172.5 µs 纯 kernel
    #   C.get_runner 端到端（含分配）   183.0 µs ← **卡片采用这个口径**
    # 本文件的 _measure 走 opbench.bench_fn，是连续调用（背靠背），
    # 所以这里报的 ms 天生偏乐观；卡片/周报一律用端到端口径。
    out = {"caliber": {"what": "op_call", "alloc": False, "state": "hot",
                       "rounds": a.rounds, "timer": "hip_event"},
           "measure_mode": "back-to-back, no per-iter sync",
           "measure_note": ("偏乐观约 23%（kernel 间可重叠）。"
                            "卡片/周报口径必须用端到端（每次 empty_like）："
                            "2026-11-23 实测 183.0 µs / 1081 GB/s / util 0.63。"
                            "纯 kernel（event 隔离）172.5 µs。"),
           "shape": dict(shape, shape=a.shape),
           "runs": [row]}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("→ %s" % a.json)
    return 0 if row["variants"] else 1


if __name__ == "__main__":
    sys.exit(main())
