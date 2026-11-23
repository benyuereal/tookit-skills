#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate —— 按家族聚合，让「哪张卡优化得最好」变成一个可排序的问题。

为什么需要它
------------
卡片里 `ratio.value` 曾经混着四种量纲：加速比 7 / 达成率 5 / 绝对吞吐 10 / 非收益比值 2。
`2.39`（倍）与 `887`（GB/s）放在同一列，任何跨卡聚合都会错——
算不出全仓几何平均加速比，也就排不出「哪张卡优化得最好」。

E24 给每条 ratio 加了 `kind` 与 `unit` 之后，聚合才有意义：
  * **几何平均只在 `kind=speedup` 上算**。混入 utilization/throughput 会得出
    「平均 300 倍」这种荒谬数字（因为 887 也在里面）。
  * utilization 单独看中位数与分布（它是诊断量）。
  * throughput 按 unit（GB/s / TFLOPS）分组，不同单位不可合并。
  * ratio_other 不进任何聚合——它回答的不是「快了多少」。

为什么用几何平均而不是算术平均
------------------------------
几何平均防「一个好看的值拉高整体」。一个 10x 加两个 1x：
算术平均 4.00x，几何平均 2.15x。后者才是「典型情况有多好」。

用法
----
    python3 -I scripts/aggregate.py                     # 全家族
    python3 -I scripts/aggregate.py --family gemm       # 单个家族
    python3 -I scripts/aggregate.py --json
    python3 -I scripts/aggregate.py --selftest
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "Plugins/ops/op-atlas/cards"


def _load_cards(cards_dir=CARDS):
    import yaml
    out = []
    for p in sorted(Path(cards_dir).glob("*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        lv = d.get("last_verdict") or {}
        if lv.get("status") not in ("measured", "tuned"):
            continue
        r = lv.get("ratio") or {}
        out.append({
            "id": d.get("id") or p.stem,
            "family": d.get("family") or "?",
            "layer": d.get("layer") or "?",
            "status": lv.get("status"),
            "kind": r.get("kind"),
            "unit": r.get("unit"),
            "value": r.get("value"),
        })
    return out


def geomean(xs):
    xs = [float(x) for x in xs if x is not None and float(x) > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def median(xs):
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def aggregate(cards):
    """返回 {family: {...}}。每族给出四类统计，互不混算。"""
    fams = {}
    for c in cards:
        f = fams.setdefault(c["family"], {
            "speedup": [], "utilization": [], "throughput": {}, "ratio_other": [],
            "cards": [], "layers": set()})
        f["cards"].append(c["id"])
        f["layers"].add(c["layer"])
        k = c["kind"]
        v = c["value"]
        if k == "speedup":
            f["speedup"].append((c["id"], v))
        elif k == "utilization":
            f["utilization"].append((c["id"], v))
        elif k == "throughput":
            unit = c["unit"] or "?"
            f["throughput"].setdefault(unit, []).append((c["id"], v))
        elif k == "ratio_other":
            f["ratio_other"].append((c["id"], v))

    out = {}
    for fam, f in sorted(fams.items()):
        sp = [v for _, v in f["speedup"]]
        ut = [v for _, v in f["utilization"]]
        # 按 layer 分开算 speedup 几何平均：engine 与 kernel 不可同表比
        by_layer = {}
        for cid, v in f["speedup"]:
            c = next(x for x in cards if x["id"] == cid)
            by_layer.setdefault(c["layer"], []).append(v)
        out[fam] = {
            "n_cards": len(f["cards"]),
            "cards": f["cards"],
            "layers": sorted(f["layers"]),
            "speedup": {
                "n": len(sp),
                "geomean": round(geomean(sp), 3) if sp else None,
                "median": round(median(sp), 3) if sp else None,
                "best": max(f["speedup"], key=lambda kv: kv[1]) if sp else None,
                "by_layer": {ly: round(geomean(vs), 3) for ly, vs in sorted(by_layer.items())},
            },
            "utilization": {
                "n": len(ut),
                "median": round(median(ut), 3) if ut else None,
                "min": min(f["utilization"], key=lambda kv: kv[1]) if ut else None,
                "max": max(f["utilization"], key=lambda kv: kv[1]) if ut else None,
            },
            "throughput": {u: {"n": len(vs), "median": round(median([v for _, v in vs]), 1)}
                           for u, vs in f["throughput"].items()},
            "ratio_other": {"n": len(f["ratio_other"])},
            # 这族有没有「真实加速」——判推进层厚度的关键量
            "real_speedup_n": sum(1 for v in sp if v and v > 1.2),
        }
    return out


def render(agg, only_family=None):
    lines = []
    fams = [only_family] if only_family else sorted(agg)
    lines.append("%-14s %4s %8s %8s %6s %10s %s"
                 % ("家族", "卡数", "加速几何均", "加速中位", ">1.2x", "达成率中位", "绝对吞吐"))
    lines.append("-" * 96)
    for fam in fams:
        if fam not in agg:
            continue
        a = agg[fam]
        sp, ut = a["speedup"], a["utilization"]
        tp = ", ".join("%s×%.1f" % (u, v["median"]) for u, v in a["throughput"].items()) or "-"
        lines.append("%-14s %4d %8s %8s %6d %10s %s"
                     % (fam, a["n_cards"],
                        ("%.2fx" % sp["geomean"]) if sp["geomean"] is not None else "-",
                        ("%.2fx" % sp["median"]) if sp["median"] is not None else "-",
                        a["real_speedup_n"],
                        ("%.3f" % ut["median"]) if ut["median"] is not None else "-",
                        tp))
    lines.append("")
    tot_sp = [c for c in _ALL if c["kind"] == "speedup"]
    tot_real = [c for c in tot_sp if c["value"] and c["value"] > 1.2]
    lines.append("全仓：有数卡 %d 张 / 真实加速(>1.2x) %d 张 / 全仓几何平均 %s"
                 % (len(_ALL), len(tot_real),
                    ("%.2fx" % geomean([c["value"] for c in tot_sp]))
                    if tot_sp else "-"))
    # 按 layer 分开报，避免 engine 与 kernel 混算
    for ly in sorted({c["layer"] for c in tot_sp}):
        vs = [c["value"] for c in tot_sp if c["layer"] == ly]
        lines.append("  %-8s speedup %d 条，几何平均 %.2fx"
                     % (ly, len(vs), geomean(vs)))
    return "\n".join(lines)


_ALL = []


def selftest():
    cards = [
        {"id": "a", "family": "f1", "layer": "kernel", "kind": "speedup", "unit": "x", "value": 2.0},
        {"id": "b", "family": "f1", "layer": "kernel", "kind": "speedup", "unit": "x", "value": 8.0},
        {"id": "c", "family": "f1", "layer": "kernel", "kind": "throughput", "unit": "GB/s", "value": 887},
        {"id": "d", "family": "f2", "layer": "engine", "kind": "speedup", "unit": "x", "value": 1.69},
        {"id": "e", "family": "f2", "layer": "kernel", "kind": "speedup", "unit": "x", "value": 1.1},
    ]
    agg = aggregate(cards)
    # 1) 几何平均只用 speedup：f1 的 (2,8) -> 4.0，887 不许混进来
    assert abs(agg["f1"]["speedup"]["geomean"] - 4.0) < 1e-9, agg["f1"]["speedup"]
    print("[ok] 几何平均只用 speedup      f1 = %.2fx（GB/s 的 887 没混进来）"
          % agg["f1"]["speedup"]["geomean"])

    # 2) throughput 单独按 unit 分组
    assert agg["f1"]["throughput"]["GB/s"]["median"] == 887
    print("[ok] throughput 按 unit 分组   GB/s×%.0f" % agg["f1"]["throughput"]["GB/s"]["median"])

    # 3) layer 分开算：f2 的 engine 1.69 与 kernel 1.1 不能合成一个数
    bl = agg["f2"]["speedup"]["by_layer"]
    assert bl["engine"] == 1.69 and bl["kernel"] == 1.1, bl
    print("[ok] speedup 按 layer 分开     engine=%.2f kernel=%.2f" % (bl["engine"], bl["kernel"]))

    # 4) 真实加速计数（>1.2x）：f1 有 2 条（2.0, 8.0），f2 有 1 条（1.69）
    assert agg["f1"]["real_speedup_n"] == 2, agg["f1"]["real_speedup_n"]
    assert agg["f2"]["real_speedup_n"] == 1, agg["f2"]["real_speedup_n"]
    print("[ok] 真实加速计数              f1=%d f2=%d"
          % (agg["f1"]["real_speedup_n"], agg["f2"]["real_speedup_n"]))

    # 5) ratio_other 不进任何聚合
    cards2 = cards + [{"id": "z", "family": "f1", "layer": "kernel",
                       "kind": "ratio_other", "unit": "x", "value": 0.97}]
    agg2 = aggregate(cards2)
    assert abs(agg2["f1"]["speedup"]["geomean"] - 4.0) < 1e-9, "ratio_other 混进了几何平均"
    print("[ok] ratio_other 不进聚合      f1 仍是 %.2fx" % agg2["f1"]["speedup"]["geomean"])
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="按家族聚合实测结果（几何平均只在 speedup 上算）")
    ap.add_argument("--family", help="只看一个家族")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cards", default=str(CARDS))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    global _ALL
    _ALL = _load_cards(a.cards)
    if not _ALL:
        print("没有 measured/tuned 的卡", file=sys.stderr)
        return 2
    agg = aggregate(_ALL)
    if a.json:
        for v in agg.values():
            v["layers"] = list(v["layers"])
        print(json.dumps(agg, ensure_ascii=False, indent=2))
    else:
        print(render(agg, a.family))
    return 0


if __name__ == "__main__":
    sys.exit(main())
