#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用精度适配器（loop.py 的 ADAPTER_ACC 契约）—— 由 op_bootstrap 生成。

## 为什么必须有它

loop.py 的代码是：

    if not a.no_acc and (card / acc_adapter).exists():

**文件不存在时整块精度检查被静默跳过**，`acc_ok` 恒 False。实测 2026-11-28：
两张 bootstrap 出来的新卡（op-gated-add / op-silu-mul）都能 `loop.py run` 出数，
但记录里 `acc={} acc_ok=False`，于是永远只能给 `CALIBRATION_ONLY` /
`ACCEPTED_NO_NUMERICS`，拿不出可 claim 的结论。**能跑 ≠ 能下结论。**

## 契约

    python3 -I bench/acc_probe.py <tag> [<tag>...] --json <out.json>

输出：

    {"qscale": {"1": {"ref_absmax": <float>, "rows": [
        {"tag": "...", "max_abs": <float>, "mean_abs": <float>,
         "cosine": <float>, "max_ulp": <float>, "nonfinite": 0}, ...]}}}

`qscale` 是 sparse-mla 的历史遗留（logit 尺度）。本算子没有 logit 尺度，
只报 "1" 一档 —— 但**保留这层结构**，因为 loop.py 的消费者按它取值，
缺了这层 `acc` 会静默为空。

## 判据用 ULP 而不是固定 max_abs

bf16 只有 7 位尾数，ULP 随量级变（[0.5,1) 2^-8 / [1,2) 2^-7 / [2,4) 2^-6）。
同一个张量里既有 0.1 量级也有 3.0 量级的元素，它们能达到的精度本来就差 32 倍，
用统一绝对容差必然顾此失彼：要么放过真误差，要么冤杀正确实现。

## 这个文件是**模板**

`__CARD__` / `__OP__` 由 op_bootstrap 替换；其余逻辑与算子无关 ——
它只依赖 `_common.py` 的三个约定：`build_inputs` / `reference` / `get_runner`。
"""
import argparse
import json
import sys
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))
sys.path.insert(0, str(CARD.parents[3] / "scripts"))

import torch  # noqa: E402
import _common as C  # noqa: E402


def _ulp_worst(got, ref):
    """最差 ULP。多输出取最差那份 —— 一份合格掩盖不了另一份不合格。"""
    if isinstance(got, (tuple, list)):
        worst = 0.0
        for a, b in zip(got, ref):
            worst = max(worst, _ulp_worst(a, b))
        return worst
    af, bf = got.float(), ref.float()
    d = (af - bf).abs()
    mag = torch.maximum(af.abs(), bf.abs())
    nz = mag > 0
    if not bool(nz.any()):
        return 0.0
    # 逐元素算 ULP：先在全张量上算 exp，再只在非零处取用。
    _, exp = torch.frexp(mag.clamp_min(torch.finfo(torch.float32).tiny))
    ulp = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp - 8)
    return float((d / ulp.clamp_min(1e-30))[nz].max())


def _stats(got, ref):
    """max_abs / mean_abs / cosine，多输出取最差。"""
    if isinstance(got, (tuple, list)):
        w = {"max_abs": 0.0, "mean_abs": 0.0, "cosine": 1.0}
        for a, b in zip(got, ref):
            m = _stats(a, b)
            w["max_abs"] = max(w["max_abs"], m["max_abs"])
            w["mean_abs"] = max(w["mean_abs"], m["mean_abs"])
            w["cosine"] = min(w["cosine"], m["cosine"])
        return w
    af, bf = got.float(), ref.float()
    d = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "cosine": cos}


def _nonfinite(got):
    if isinstance(got, (tuple, list)):
        return sum(_nonfinite(x) for x in got)
    return int((~torch.isfinite(got.float())).sum())


def _call_ref(inp, params):
    """调本卡的参考实现。

    两种签名都要能吃：`reference(*inputs, params=...)`（本仓 bootstrap 卡的约定）
    和 `reference(*inputs)`（老卡的写法）。猜错签名会让整份探针直接崩，
    所以这里显式回落一次，而不是改卡。
    """
    try:
        return C.reference(*inp, params=params)
    except TypeError:
        return C.reference(*inp)


def main(argv=None):
    ap = argparse.ArgumentParser(description="__OP__ 的精度探针")
    ap.add_argument("tags", nargs="*", help="变体名（逗号分隔也接受）")
    ap.add_argument("--json", required=True, help="输出路径")
    ap.add_argument("--shape", default="s0", help="shapes.yaml 的档位名")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    # loop.py 在卡没声明 defaults.acc_args 时会退回 sparse-mla 的参数
    # （--qscale / --skv）。那些对逐元素/归约类算子没有意义，但不能因此报错 ——
    # 报错会让 acc 静默为空，回到「能跑但拿不出结论」的老问题。
    ap.add_argument("--qscale", nargs="*", default=None)
    ap.add_argument("--skv", default=None)
    a, _unknown = ap.parse_known_args(argv)

    tags = []
    for t in a.tags:
        tags.extend(x.strip() for x in str(t).split(",") if x.strip())
    if not tags:
        print("没有指定变体：本探针只测点名的变体，不猜", file=sys.stderr)
        return 2

    shape = C.load_shape(a.shape)
    params = dict(shape.get("params") or {})
    kw = C.runner_kwargs(params) if hasattr(C, "runner_kwargs") else {}
    inp = C.build_inputs(params, seed=a.seed, device=a.device)
    ref = _call_ref(inp, params)

    ref_absmax = 0.0
    for x in (ref if isinstance(ref, (tuple, list)) else (ref,)):
        ref_absmax = max(ref_absmax, float(x.float().abs().max()))

    rows = []
    for tag in tags:
        try:
            got = C.get_runner(tag, **kw)(*inp)
        except Exception as e:                  # noqa: BLE001
            # 单个变体跑不起来不是整份探针的失败：记录 nonfinite=-1 让
            # loop.py 那边能看到"这个 tag 没测成"，而不是假装它过了。
            print("variant=%s 跳过: %s: %s" % (tag, type(e).__name__, e),
                  file=sys.stderr)
            rows.append({"tag": tag, "max_abs": float("inf"), "mean_abs": float("inf"),
                         "cosine": 0.0, "max_ulp": float("inf"), "nonfinite": -1})
            continue
        st = _stats(got, ref)
        rows.append({"tag": tag, "max_abs": st["max_abs"],
                     "mean_abs": st["mean_abs"], "cosine": st["cosine"],
                     "max_ulp": _ulp_worst(got, ref),
                     "nonfinite": _nonfinite(got)})

    out = {"qscale": {"1": {"ref_absmax": ref_absmax, "rows": rows}}}
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print("# acc: %d 个变体，ref_absmax=%.4g" % (len(rows), ref_absmax))
    for r in rows:
        print("  %-14s max_abs=%.3e max_ulp=%.1f cosine=%.7f nonfinite=%d"
              % (r["tag"], r["max_abs"], r["max_ulp"], r["cosine"], r["nonfinite"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
