#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roofline_golden —— 模板的**绝对正确性**校验。

为什么需要它（v5.1 §7.5，第五轮评审的第 2 条缺口）
--------------------------------------------------
原判据是「同一 gemm 卡两个 shape 的 bound 翻转」——那只测**方向敏感性**。

**洞**：一个 bytes 公式系统性差 2× 的模板，bound 翻转照样正确，
但会毒化所有 `remaining` 和 EV，而且错误安静地住在每张卡的 `roofline.json` 里。

这正是 S1 刚打完的那类错（助记符记错 → 判据本身错）。没有 golden，
十个模板就是十个潜在的助记符错误。

判据
----
用「已知良好优化」的算子交叉验证：
    公式算出的 bytes ÷ 实测时间 = 实测带宽
    实测带宽 ÷ 分母带宽(achievable) 必须落在 [0.6, 1.0]

落不进区间有两个可能，**必须区分**：
    模板错（公式少算/多算字节） 或 分母错（bw-ladder 口径问题）
不许二选一硬判。

用法
----
    python3 -I scripts/roofline_golden.py --selftest
    python3 -I scripts/roofline_golden.py --all
    python3 -I scripts/roofline_golden.py --template reduce-lastdim-2out
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TPL = REPO / "Plugins/ops/skills/op-roofline/templates.py"
SKILLS = REPO / "Plugins/ops/skills"
LADDER = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"

BAND = (0.6, 1.0)


def _templates():
    spec = importlib.util.spec_from_file_location("_tpl", str(TPL))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bw_denominator(pattern):
    """取 achievable 带宽。**不认 nominal**（1100 是产品名，永不当分母）。"""
    import yaml
    d = yaml.safe_load(LADDER.read_text(encoding="utf-8")) or {}
    key = {"r_only": "2_read_only", "w_only": "2_write_only",
           "rw_1to1": "2_rw_1to1", "rw_2to1": "2_rw_2to1"}.get(pattern)
    if not key:
        return None
    node = (d.get("layers") or {}).get(key) or {}
    v = node.get("achievable")
    return float(v) if v else None


# golden 用例：每个模板至少一条，来自**已有实测**的卡
GOLDEN = [
    {"template": "reduce-lastdim-2out", "card": "rmsnorm-fused",
     "params": {"M": 8192, "H": 4096, "dtype_bytes": 2},
     "t_us": 197.0, "pattern": "rw_1to1",
     "why": "lightop 实测 197.0 µs（三次独立，MAD 0.3~0.6）"},
    {"template": "matmul", "card": "gemm-bf16",
     "params": {"M": 8192, "N": 8192, "K": 8192, "dtype_bytes": 2},
     "t_us": 4404.6, "pattern": "rw_2to1",
     "why": "torch.mm 8192³ 实测 4.4046 ms"},
]


def evaluate_one(g, tpl_mod):
    r = tpl_mod.evaluate(g["template"], g["params"])
    if "error" in r:
        return {"error": r["error"]}
    denom = _bw_denominator(g["pattern"])
    if not denom:
        return {"error": "取不到 %s 档的 achievable 带宽" % g["pattern"]}
    byts = r["bytes"]
    achieved = byts / (g["t_us"] * 1e-6) / 1e9      # GB/s
    # **compute-bound 的模板不能用带宽分母校验**——GEMM 的 91 GB/s 不是「带宽差」，
    # 是它根本不在等带宽（AI=2731 ≫ ridge=161）。对它们要用算力利用率：
    #     achieved_tflops / peak_tflops 也应落在 [0.6, 1.0]
    # 两种尺子不能混用，这正是 v5.1 第一章说的「按 roofline 选指标」。
    if r.get("flops"):
        import yaml
        c = yaml.safe_load((REPO / "Plugins/ops/skills/op-roofline/constants.yaml")
                           .read_text(encoding="utf-8")) or {}
        peak = ((c.get("compute") or {}).get("bf16_gemm_peak_tf") or {}).get("value")
        tflops = r["flops"] / (g["t_us"] * 1e-6) / 1e12
        util = tflops / float(peak) if peak else None
        if util is None:
            return {"error": "取不到峰值算力"}
        lo, hi = BAND
        verdict = ("under" if util < lo else "over" if util > hi else "ok")
        return {"template": g["template"], "card": g["card"],
                "bytes": byts, "bytes_mb": round(byts / 1e6, 1),
                "t_us": g["t_us"], "metric": "MFU",
                "achieved": round(tflops, 1), "denominator": float(peak),
                "utilization": round(util, 3), "band": list(BAND),
                "verdict": verdict, "why": g["why"]}
    util = achieved / denom
    lo, hi = BAND
    if util < lo:
        verdict = "under"          # 实测带宽远低于分母：模板可能**多算**了字节
    elif util > hi:
        verdict = "over"           # 实测带宽超过分母：模板可能**少算**了字节，或分母错
    else:
        verdict = "ok"
    return {"template": g["template"], "card": g["card"],
            "bytes": byts, "bytes_mb": round(byts / 1e6, 1),
            "t_us": g["t_us"], "achieved_gbs": round(achieved, 1),
            "denominator_gbs": denom, "utilization": round(util, 3),
            "band": list(BAND), "verdict": verdict, "why": g["why"]}


def show(rows, as_json=False):
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return
    print("模板 golden 校验（bytes ÷ 实测时间 ÷ 分母，应落在 %.1f~%.1f）" % BAND)
    print("=" * 78)
    bad = 0
    for r in rows:
        if "error" in r:
            print("  ✗ %-22s %s" % (r.get("template"), r["error"]))
            bad += 1
            continue
        mark = {"ok": "✓", "under": "↓", "over": "↑"}[r["verdict"]]
        if r["verdict"] != "ok":
            bad += 1
        if r.get("metric") == "MFU":
            print("  %s %-22s %.1f MB / %.1f µs = %.0f TF ÷ %.0f = %.3f  [MFU]"
                  % (mark, r["template"], r["bytes_mb"], r["t_us"],
                     r["achieved"], r["denominator"], r["utilization"]))
        else:
            print("  %s %-22s %.1f MB / %.1f µs = %.0f GB/s ÷ %.0f = %.3f  [MBU]"
                  % (mark, r["template"], r["bytes_mb"], r["t_us"],
                     r["achieved_gbs"], r["denominator_gbs"], r["utilization"]))
        if r["verdict"] == "under":
            print("       ↓ 低于 %.1f：模板可能**多算**了字节（写多了份数？）" % BAND[0])
        elif r["verdict"] == "over":
            print("       ↑ 高于 %.1f：模板可能**少算**了字节，或分母口径错。"
                  "两者必须区分，不许二选一硬判" % BAND[1])
    print()
    if bad:
        print("不通过 %d 条 —— 落不进区间有两个可能：模板错 or 分母错（必须区分）" % bad)
    else:
        print("全部落在区间内")


def selftest():
    tpl = _templates()
    # 1) 正常：rmsnorm 应该落在区间内
    r = evaluate_one(GOLDEN[0], tpl)
    assert "error" not in r, r
    assert BAND[0] <= r["utilization"] <= BAND[1], r
    print("[ok] rmsnorm golden：%.1f MB / %.1f µs → util=%.3f ∈ 区间"
          % (r["bytes_mb"], r["t_us"], r["utilization"]))

    # 2) 反向：把字节人为放大 2×，必须判 over
    g2 = dict(GOLDEN[0])
    g2["params"] = dict(GOLDEN[0]["params"], dtype_bytes=4)   # 4B/elem = 字节翻倍
    r2 = evaluate_one(g2, tpl)
    assert r2["verdict"] == "over", r2
    print("[ok] 反向：字节算成 2× → util=%.2f 判 over（能抓到模板多算）" % r2["utilization"])

    # 3) 反向：字节人为减半，必须判 under
    g3 = dict(GOLDEN[0])
    g3["params"] = {"M": 8192, "H": 4096, "dtype_bytes": 1}
    g3["t_us"] = 197.0
    r3 = evaluate_one(g3, tpl)
    assert r3["verdict"] == "under", r3
    print("[ok] 反向：字节减半 → util=%.2f 判 under（能抓到模板少算）" % r3["utilization"])

    # 4) 分母必须来自 achievable，不是 nominal
    d = _bw_denominator("rw_1to1")
    assert d and d > 1500, d      # 1680；若取到 1100 就是错的
    print("[ok] 分母取 achievable：rw_1to1 = %.0f GB/s（不是 nominal 1100）" % d)

    print("\nselftest PASS（4 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="模板的绝对正确性校验")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--template")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    tpl = _templates()
    cases = [g for g in GOLDEN
             if not a.template or g["template"] == a.template]
    if not cases:
        print("没有匹配的 golden 用例", file=sys.stderr)
        return 2
    rows = [evaluate_one(g, tpl) for g in cases]
    show(rows, a.json)
    return 0 if all(r.get("verdict") == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
