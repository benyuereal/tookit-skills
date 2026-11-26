#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-norm-act-swiglu A/B 适配器：把本卡测量喂给 loop.py run。

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


def _measure(fn, inp, rounds, warmup_ms, rep_ms, mode="back_to_back"):
    """mode 决定口径，也决定结论能不能 claim。

    back_to_back（默认，向后兼容）：连续调用，kernel 间可重叠 ⇒ **偏乐观**，
        measure_mode.py 判它不可 claim。只能筛方向。
    event_isolated：每次调用前后插 event + synchronize，杜绝重叠 ⇒ 可 claim。
    """
    meds = []
    if mode == "event_isolated":
        import torch
        def _timed():
            e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
            e1.record()
            fn(*inp[:-1])
            e2.record()
            torch.cuda.synchronize()
            return e1.elapsed_time(e2)
        for _ in range(5):
            _timed()
        for _ in range(rounds * 7):
            meds.append(_timed())
        meds.sort(); n = len(meds)
        med = meds[n // 2] if n % 2 else 0.5 * (meds[n // 2 - 1] + meds[n // 2])
        return med, sorted(abs(x - med) for x in meds)[n // 2]
    for _ in range(rounds):
        st = bench_fn(lambda: fn(*inp[:-1]), warmup_ms=warmup_ms, rep_ms=rep_ms, repeats=3)
        meds.append(st["median_ms"])
    meds.sort()
    n = len(meds)
    med = meds[n // 2] if n % 2 else 0.5 * (meds[n // 2 - 1] + meds[n // 2])
    mad = sorted(abs(x - med) for x in meds)[n // 2]
    return med, mad


def main(argv=None):
    p = argparse.ArgumentParser(description="op-norm-act-swiglu A/B 适配器")
    p.add_argument("--json", default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shape", default="s0")
    p.add_argument("--variants", required=True, help="逗号分隔的变体名")
    p.add_argument("--base", default="two-step")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    p.add_argument("--measure-mode", default="back_to_back", dest="measure_mode",
                   choices=["back_to_back", "event_isolated", "end_to_end"],
                   help="back_to_back 不可 claim（仅筛方向）；event_isolated 可 claim")
    p.add_argument("--skip-preflight", action="store_true",
                   help="跳过环境前置检查（**只在已手工验过环境时用**）")
    a = p.parse_args(argv)

    # **默认跑前置检查**（scripts/bench_preflight.py）。
    # 2026-11-25 的 5 小时根因是「设备绑定方式」问题 —— 纯 torch 探活查不出，
    # 必须用自定义 kernel 试。默认检查是为了不依赖「谁记得跑」。
    if not a.skip_preflight:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[4] / "scripts"))
        try:
            from bench_preflight import preflight
            if not preflight(device=getattr(a, "device", None)):
                print("✗ 前置检查未全过 —— 加 --skip-preflight 可强制继续，"
                      "但测出的数字不可信", file=sys.stderr)
                return 2
        except ImportError:
            pass   # 没装检查器时不阻塞（但会在 stderr 提一句）


    shape = C.load_shape(a.shape)
    params = dict(shape.get("params") or {})
    kw = C.runner_kwargs(params)
    inp = C.build_inputs(params, seed=a.seed, device=a.device)

    row = {"skv": 0, "variants": []}
    try:
        bmed, bmad = _measure(C.get_runner(a.base, **kw), inp,
                              a.rounds, a.warmup_ms, a.rep_ms, mode=a.measure_mode)
    except Exception as e:                       # noqa: BLE001
        print("base=%s 失败: %s: %s" % (a.base, type(e).__name__, e), file=sys.stderr)
        return 1
    row["base_ms"], row["base_mad"] = bmed, bmad

    for tag in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            med, mad = _measure(C.get_runner(tag, **kw), inp, a.rounds,
                                a.warmup_ms, a.rep_ms, mode=a.measure_mode)
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

    out = {"caliber": {"what": "op_call", "alloc": False, "state": "hot",
                       "rounds": a.rounds, "timer": "hip_event"},
"measure_mode": a.measure_mode,
           "measure_note": ("背靠背连续调用，kernel 间可能重叠 ⇒ 偏乐观。卡片/周报口径须用端到端（含每次分配）。"
                            "本卡未做三方对账，偏乐观幅度未测定。"
      "rmsnorm-fused 实测该差为 23%（140 vs 183 µs），可作量级参考。"),
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
