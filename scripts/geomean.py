#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geomean —— 跨 shape 的几何平均加速比，作为主指标。

为什么主指标换成它
------------------
两条独立的理由，指向同一个结论：

1. **达成率会饱和，几何平均不会。**
   CompoundingKernel（arXiv:2611.19230）明确不用 SOL 分做奖励，
   理由是「SOL 在高加速比处会趋平」——优化到 80% 以后，
   每一点进步在分数上几乎看不出来，奖励信号就消失了。

2. **我们刚被达成率的分母坑过一次（差 10.9 倍）。**
   把 aiter 的 fp8 实现值 49.8 TF 当成 fp8 峰值，
   于是「还有 11 倍空间」被读成「8.9%，已到顶」——
   分母一错，结论方向就反。而「相对基线的比值」不需要峰值这个分母，
   少一个能出错的地方。

3. **几何平均防「挑一个好看的形状报数」。**
   算术平均会被一个 10× 的形状拉高；几何平均不会。
   这与论文用 209 题几何平均（4.40×）而不是最好那题是同一个道理。

口径
----
一切以**基线**为 1.0：`speedup = base_ms / cand_ms`（>1 表示更快）。
几何平均 = exp(mean(ln(speedup)))，等价于各形状加速比的连乘开 n 次方。

缺形状怎么办
------------
**不插值、不跳过、不假装通过。** 缺一个形状就报出来，并给出
「已覆盖 n/m」的覆盖率。只有完整覆盖才允许把 geomean 当结论写进卡。

用法
----
    from geomean import geomean_speedup, summarize_shapes
    r = geomean_speedup({"s1": (0.4, 1.0), "s2": (0.9, 1.0)}, expected=["s1","s2","s3"])
    print(r)

命令行：
    python3 -I scripts/geomean.py --demo
    python3 -I scripts/geomean.py --selftest
"""
import argparse
import json
import math
import sys


def geomean_speedup(pairs, expected=None, min_shapes=2):
    """pairs: {shape: (cand_ms, base_ms)} → 汇总 dict。

    返回字段：
      geomean          几何平均加速比（只算有值的形状）
      coverage         "n/m" 覆盖率
      complete         是否覆盖了 expected 里的每一个形状
      per_shape        {shape: speedup}
      missing          expected 里没测到的形状
      gate             能否当主结论用（complete 且 n >= min_shapes）
      reasons          gate 为 False 的原因
    """
    if not pairs:
        return {"geomean": None, "coverage": "0/0", "complete": False,
                "per_shape": {}, "missing": list(expected or []), "gate": False,
                "reasons": ["没有任何形状的数据"]}

    per, bad = {}, []
    for shape, v in pairs.items():
        try:
            cand, base = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            bad.append("%s: 数值无法解析" % shape)
            continue
        if base <= 0 or cand <= 0:
            bad.append("%s: 耗时必须 >0（cand=%s base=%s）" % (shape, cand, base))
            continue
        per[shape] = base / cand

    if not per:
        return {"geomean": None, "coverage": "0/0", "complete": False,
                "per_shape": {}, "missing": list(expected or []), "gate": False,
                "reasons": bad or ["没有可用形状"]}

    gm = math.exp(sum(math.log(x) for x in per.values()) / len(per))
    missing = [s for s in (expected or []) if s not in per]
    total = len(expected) if expected else len(per)
    complete = not missing

    reasons = list(bad)
    if not complete:
        reasons.append("缺形状 %s（不插值、不跳过）" % ",".join(missing))
    if len(per) < min_shapes:
        reasons.append("只有 %d 个形状，少于 %d（单形状的几何平均没有意义，"
                       "它就是那一个比值，防不了挑好看的形状）" % (len(per), min_shapes))

    return {
        "geomean": round(gm, 4),
        "coverage": "%d/%d" % (len(per), total),
        "complete": complete,
        "per_shape": {k: round(v, 4) for k, v in sorted(per.items())},
        "missing": missing,
        "gate": complete and len(per) >= min_shapes,
        "reasons": reasons,
    }


def summarize_shapes(r):
    """人读的一行摘要（写进卡片的 conclusion 用）。"""
    if r["geomean"] is None:
        return "几何平均加速比：无可用数据（%s）" % "；".join(r["reasons"])
    s = "几何平均 %.2fx（覆盖 %s）" % (r["geomean"], r["coverage"])
    if not r["gate"]:
        s += " caveat 不可作主结论：" + "；".join(r["reasons"])
    return s


def demo():
    cases = {
        "两形状都有": ({"s1": (0.4, 1.0), "s2": (0.5, 1.0)}, ["s1", "s2"]),
        "缺一个形状": ({"s1": (0.4, 1.0), "s2": (0.5, 1.0)}, ["s1", "s2", "s3"]),
        "单形状": ({"s1": (0.4, 1.0)}, ["s1"]),
        "一个形状特别好看": ({"s1": (0.1, 1.0), "s2": (1.0, 1.0), "s3": (1.0, 1.0)},
                            ["s1", "s2", "s3"]),
    }
    for name, (pairs, exp) in cases.items():
        print("== %s ==" % name)
        r = geomean_speedup(pairs, expected=exp)
        print("  " + summarize_shapes(r))
        print("  逐形状: %s" % r["per_shape"])
        print()
    print("注意最后一个：算术平均 4.00x，几何平均只有 2.15x ——")
    print("这正是几何平均要防的「挑一个好看的形状报数」。")


def selftest():
    # 基本正确性
    r = geomean_speedup({"a": (0.5, 1.0), "b": (0.5, 1.0)}, expected=["a", "b"])
    assert abs(r["geomean"] - 2.0) < 1e-9, r
    assert r["gate"] and r["complete"]
    print("[ok] 两形状 2x        geomean=%.2f coverage=%s" % (r["geomean"], r["coverage"]))

    # 缺形状必须挡住
    r = geomean_speedup({"a": (0.5, 1.0), "b": (0.5, 1.0)}, expected=["a", "b", "c"])
    assert not r["gate"] and r["missing"] == ["c"], r
    print("[ok] 缺形状被挡住      %s" % r["reasons"][0])

    # 单形状不许当主结论
    r = geomean_speedup({"a": (0.5, 1.0)}, expected=["a"])
    assert not r["gate"], r
    print("[ok] 单形状被挡住      %s" % r["reasons"][0])

    # 防「挑好看的形状」：算术平均会高估
    pairs = {"fast": (0.1, 1.0), "s2": (1.0, 1.0), "s3": (1.0, 1.0)}
    r = geomean_speedup(pairs, expected=["fast", "s2", "s3"])
    am = sum(pairs[k][1] / pairs[k][0] for k in pairs) / 3
    assert am > r["geomean"] * 1.5, (am, r["geomean"])
    print("[ok] 几何平均防挑数    算术 %.2fx vs 几何 %.2fx" % (am, r["geomean"]))

    # 非法数值
    r = geomean_speedup({"a": (0, 1.0), "b": (0.5, 1.0)}, expected=["a", "b"])
    assert not r["gate"] and any(">0" in x for x in r["reasons"]), r
    print("[ok] 非法耗时被拒      %s" % r["reasons"][0])
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="跨 shape 几何平均加速比")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--pairs", help='JSON: {"shape": [cand_ms, base_ms], ...}')
    ap.add_argument("--expected", help="逗号分隔的形状清单（判覆盖率用）")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.demo:
        demo()
        return 0
    if not a.pairs:
        ap.error("要么 --selftest / --demo / --pairs")
    pairs = json.loads(a.pairs)
    exp = [x.strip() for x in a.expected.split(",")] if a.expected else None
    r = geomean_speedup(pairs, expected=exp)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(summarize_shapes(r))
        print("逐形状: %s" % r["per_shape"])
    return 0 if r["gate"] or not exp else 1


if __name__ == "__main__":
    sys.exit(main())
