#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 A/B 适配器 —— 由 `scripts/op_bootstrap.py` 拷进新卡的 bench/ab.py。

契约见 `engine/loop.py` 的 ADAPTER_AB：输出
    {"shape": {...}, "runs": [{"base_ms", "base_mad", "variants": [
        {"tag", "ms", "mad", "delta_pct", "significant", "why", "maxdiff", "rel"}]}]}

**为什么要有通用版**：新建卡（op_bootstrap）以前只生成 contract/shapes/
_common/SKILL/variants，没有 `bench/ab.py`，于是 `loop.py run` 直接拒绝
（"本卡没有 A/B 适配器"）——"建卡 → 闭环"这条链在最后一环断掉。
本文件只依赖卡自己的 `_common.py`（build_inputs / runner_kwargs /
get_runner / load_shape），所以对任何算子都成立。

口径：`--measure-mode event_isolated` 才可 claim（每次调用前后插 event +
synchronize）；`back_to_back` 只用来筛方向。base **每轮重测**。
"""
import argparse
import json
import sys
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))
sys.path.insert(0, str(CARD.parents[3] / "scripts"))

import _common as C  # noqa: E402


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return 0.0, 0.0
    med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    mad = sorted(abs(x - med) for x in xs)[n // 2]
    return med, mad


def _call(fn, inp):
    """本仓库的 build_inputs 约定：返回 (…输入, meta)。meta 不是张量就不传。"""
    try:
        import torch
        if inp and isinstance(inp[-1], dict):
            return fn(*inp[:-1])
    except Exception:  # noqa: BLE001
        pass
    return fn(*inp)


def _measure(fn, inp, rounds, warmup_ms, rep_ms, mode):
    """返回 (median_ms, mad_ms)。inp 的最后一项是输出/占位，不传进 fn。"""
    if mode == "event_isolated":
        import torch

        def _timed():
            e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
            e1.record()
            _call(fn, inp)
            e2.record()
            torch.cuda.synchronize()
            return e1.elapsed_time(e2)

        for _ in range(5):          # warmup
            _timed()
        vals = [_timed() for _ in range(max(3, rounds * 7))]
        return _median(vals)

    # back_to_back：连续调用取中位；仅筛方向，不可 claim
    import torch
    for _ in range(5):
        _call(fn, inp)
    torch.cuda.synchronize()
    vals = []
    for _ in range(max(3, rounds * 7)):
        e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        e1.record()
        for _ in range(3):
            _call(fn, inp)
        e2.record()
        torch.cuda.synchronize()
        vals.append(e1.elapsed_time(e2) / 3.0)
    return _median(vals)


def main(argv=None):
    p = argparse.ArgumentParser(description="通用 A/B 适配器（bootstrap 生成）")
    p.add_argument("--json", default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shape", default="s0")
    p.add_argument("--variants", required=True, help="逗号分隔的变体名")
    p.add_argument("--base", default="ref-heavy")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=100.0)
    p.add_argument("--measure-mode", default="event_isolated", dest="measure_mode",
                   choices=["back_to_back", "event_isolated", "end_to_end"])
    p.add_argument("--skip-preflight", action="store_true",
                   dest="skip_preflight",
                   help="跳过环境前置检查（**只在已手工验过环境时用**）")
    a = p.parse_args(argv)

    # **默认跑前置检查**（scripts/bench_preflight.py）。
    # 2026-11-25 的 5 小时根因是「设备绑定方式」问题 —— 纯 torch 探活查不出，
    # 必须用自定义 kernel 试。默认检查是为了不依赖「谁记得跑」。
    # 想跳过必须显式 --skip-preflight，且那时测出的数字不可信。
    if not a.skip_preflight:
        try:
            from bench_preflight import preflight
        except ImportError:                                      # 没装检查器不阻塞
            preflight = None
        if preflight is not None and not preflight(
                device=getattr(a, "device", None)):
            print("✗ 前置检查未全过 —— 加 --skip-preflight 可强制继续，"
                  "但测出的数字不可信", file=sys.stderr)
            return 2

    shape = C.load_shape(a.shape)
    params = dict(shape.get("params") or {})
    kw = C.runner_kwargs(params) if hasattr(C, "runner_kwargs") else {}
    inp = C.build_inputs(params, seed=a.seed, device=a.device)

    row = {"skv": 0, "variants": []}
    try:
        bmed, bmad = _measure(C.get_runner(a.base, **kw), inp, a.rounds,
                              a.warmup_ms, a.rep_ms, a.measure_mode)
    except Exception as e:  # noqa: BLE001
        print("base=%s 失败: %s: %s" % (a.base, type(e).__name__, e), file=sys.stderr)
        return 1
    row["base_ms"], row["base_mad"] = bmed, bmad

    for tag in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            med, mad = _measure(C.get_runner(tag, **kw), inp, a.rounds,
                                a.warmup_ms, a.rep_ms, a.measure_mode)
        except Exception as e:  # noqa: BLE001
            print("variant=%s 跳过: %s: %s" % (tag, type(e).__name__, e),
                  file=sys.stderr)
            continue
        delta = (med / bmed - 1.0) * 100.0 if bmed else None
        # 显著性：沿用本仓库的 3×MAD 判据（与 op-bench 一致）
        sig = (bmed and mad + bmad > 0 and abs(bmed - med) > 3.0 * max(mad, bmad))
        row["variants"].append({
            "tag": tag, "ms": med, "mad": mad, "delta_pct": delta,
            "significant": bool(sig),
            "why": "Δ=%.2f%% vs base=%.4fms（3×MAD 判据）" % (delta or 0, bmed),
            "maxdiff": None, "rel": None,
        })
        print("# %s: %.4fms (%.4fx) Δ=%.2f%% sig=%s"
              % (tag, med, (med / bmed) if bmed else 0, delta or 0, sig))

    out = {"shape": dict(shape, shape=a.shape),
           "measure_mode": a.measure_mode,
           "caliber": {"what": "op_call", "alloc": False, "state": "hot",
                       "rounds": a.rounds, "timer": "hip_event"},
           "runs": [row]}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("→ %s" % a.json)
    return 0 if row["variants"] else 1


if __name__ == "__main__":
    sys.exit(main())
