#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-rmsnorm-fused 的精度适配器（loop.py 的 ADAPTER_ACC 契约）。

## 为什么必须有它

在此之前本卡没有 acc_probe.py —— loop.py 的代码是
    if not a.no_acc and (card / acc_adapter).exists():
所以文件不存在时**整块精度检查被静默跳过**，acc_ok 恒 False。
全仓 0 条 acc_ok=true 记录，根因就在这里：不是实现不准，是**没人测**。

## 契约

    python3 -I bench/acc_probe.py <tag> [<tag>...] --json <out.json>

输出：
    {"qscale": {"1": {"ref_absmax": <float>, "rows": [
        {"tag": "...", "max_abs": <float>, "mean_abs": <float>,
         "cosine": <float>, "max_ulp": <float>, "nonfinite": 0}, ...]}}}

qscale 这一层是 sparse-mla 的历史遗留（logit 尺度）。rmsnorm 没有 logit，
只报 "1" 一档 —— 但**保留这层结构**，因为 loop.py 的消费者按它取值。

## 判据

用 ULP 而不是固定 max_abs：bf16 只有 7 位尾数，ULP 随量级变
（[0.5,1) 2^-8 / [1,2) 2^-7 / [2,4) 2^-6）。同一个算子输出里既有 0.1 量级
也有 3.0 量级的元素，它们的精度本来就差 32 倍，用统一绝对容差必然顾此失彼。
详见 scripts/opcheck.py 的 _bf16_ulp。
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
    # **逐元素算 ULP**：先在全张量上算 exp，再只在非零处取用。
    # 早先写成 ulp[nz] = ... 会因掩码赋值产生形状不一致。
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
    return int((~torch.isfinite(got)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description="rmsnorm-fused 精度适配器")
    ap.add_argument("tags", nargs="+",
                    help="要测的变体 tag（可用逗号分隔，与 ab.py 的 --variants 一致）")
    ap.add_argument("--shape", default="prefill-8k")
    ap.add_argument("--json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    shape = C.load_shape(a.shape)
    params = dict(shape.get("params") or {})
    kw = C.runner_kwargs(params)
    inp = C.build_inputs(params, seed=a.seed, device=a.device)
    ref = C.reference(*inp[:-1], params=params)

    # 逗号分隔要拆开 —— loop.py 传参用的是 ab.py 那套 --variants 写法。
    wanted = [t.strip() for raw in a.tags for t in str(raw).split(",") if t.strip()]
    rows = []
    # base 必须也测：它是「相对谁」的锚。只报候选会让 maxdiff 失去含义。
    for tag in ["base"] + [t for t in wanted if t != "base"]:
        name = "base" if tag == "base" else tag
        # base 用卡上声明的默认基线，而不是 ref-heavy —— 参考实现自己就是 ref，
        # 拿它比 max_abs 恒为 0，等于没测。
        impl = "lightop-fused" if tag == "base" else tag
        try:
            fn = C.get_runner(impl, **kw)
            got, _mut = C.run_isolated(fn, *inp[:-1])
        except Exception as e:                       # noqa: BLE001
            # **不吞**：拿不到数就写 nonfinite=-1 并带原因，让 loop.py 判
            # ACCEPTED_NO_NUMERICS 而不是当成「精度合格」。
            rows.append({"tag": name, "error": "%s: %s" % (type(e).__name__, str(e)[:200]),
                         "max_abs": None, "nonfinite": -1})
            continue
        m = _stats(got, ref)
        rows.append({"tag": name, "max_abs": m["max_abs"],
                     "mean_abs": m["mean_abs"], "cosine": m["cosine"],
                     "max_ulp": _ulp_worst(got, ref), "nonfinite": _nonfinite(got)})

    parts = [ref] if torch.is_tensor(ref) else list(ref)
    ref_absmax = max((max(abs(float(t.max())), abs(float(t.min()))) for t in parts),
                     default=0.0)
    out = {"qscale": {"1": {"ref_absmax": ref_absmax, "rows": rows}}}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2) if not a.json else "→ %s" % a.json)
    # 有 error 行就退 1，让 loop.py 的调用方看得见
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
