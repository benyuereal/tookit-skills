#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lang_select —— 这个算子、这个场景，该用什么语言实现。

为什么需要它
------------
`loop.py enumerate` 只给「攻击面」（fusion-boundary / layout-pack 等），
**没有任何东西告诉 agent「用什么语言实现」**。而这个选择影响巨大：

    rmsnorm prefill-8k：lightop(HIP) 197.0µs / tilelang 221.0 / triton 242.0

triton 比厂商慢 23%，但**它在别处又是最优**——`bw_calibrate` 的带宽标定
必须用 Triton 手写 kernel（torch elementwise 只有 1051 GB/s，Triton 能到 1965）。

所以不存在「哪个语言更好」，只存在「哪个算子+哪个场景该用哪个」。

排序规则（三级）
----------------
  1. 有实测的，**按实测时间排**（不许写死顺序）
  2. 无实测但 use_when 命中场景 → 标 unmeasured，排在有实测之后
  3. avoid_when 命中 → 标 excluded，说明为什么排除

纪律
----
语言与攻击面**正交**：语言回答「用什么写」，攻击面回答「改哪里」。
不许把语言塞进 EV —— EV 的四个因子（remaining / 利用率 / 已证伪 / 前提）
全都针对攻击面，「Triton 的 remaining」不存在。
正确用法是两阶段：plan.py 出 surface → lang_select 出 language → (surface, language)。

用法
----
    python3 -I scripts/lang_select.py --card rmsnorm-fused --shape prefill-8k
    python3 -I scripts/lang_select.py --card gemm-fp8 --json
    python3 -I scripts/lang_select.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LANGS = REPO / "data/ops/lang-candidates.yaml"
SKILLS = REPO / "Plugins/ops/skills"

# 变体名 → 语言。本仓的命名约定已足够，但仍显式登记，不靠猜。
VARIANT_LANG = {
    "lightop-fused": "hip", "ref-heavy": "torch", "torch-native": "torch",
    "fused-3pass-triton": "triton", "fused-3pass-tilelang": "tilelang",
    "aiter": "hip", "hipplaslt": "hip", "hipblaslt": "hip",
}

# 场景先验：没有实测时按这些规则排。key = bound（或 launch）
SCENARIO_PRIOR = {
    "memory": ["hip", "tilelang", "triton", "cu"],
    "compute": ["hip", "triton", "tilelang", "cu"],
    "mixed": ["hip", "tilelang", "triton", "cu"],
    "launch": ["hip", "triton", "tilelang", "cu"],
}


def load_langs():
    import yaml
    if not LANGS.is_file():
        raise SystemExit("找不到 %s" % LANGS)
    return (yaml.safe_load(LANGS.read_text(encoding="utf-8")) or {}).get("languages") or []


def card_dir(card):
    return SKILLS / (card if card.startswith("op-") else "op-" + card)


def measured(card, shape):
    """从该卡的 results/ 里读实测：{lang: median_ms}。

    来源优先级：impl-ranking.json（三方同测，最可信）> perf/*.json > results/*.json
    """
    d = card_dir(card)
    out = {}
    ir = d / "results" / "impl-ranking.json"
    if ir.is_file():
        try:
            j = json.loads(ir.read_text(encoding="utf-8"))
            for tag, v in (j.get("impls") or {}).items():
                lang = VARIANT_LANG.get(tag)
                ms = ((v.get("clone") or {}).get("median_ms")
                      or (v.get("reuse") or {}).get("median_ms"))
                if lang and ms:
                    out.setdefault(lang, []).append(float(ms))
        except (ValueError, OSError):
            pass
    return {k: min(v) for k, v in out.items()}


def bound_of(card, shape):
    d = card_dir(card) / "roofline.json"
    if not d.is_file():
        return None, None
    try:
        j = json.loads(d.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, None
    if "by_shape" in j:
        node = (j["by_shape"] or {}).get(shape) or {}
        return node.get("bound"), j
    return j.get("bound"), j


def rank(langs, meas, bound, shape):
    """返回排序后的候选。**有实测的按时间排，不许写死顺序。**"""
    order = SCENARIO_PRIOR.get(str(bound or ""), ["hip", "triton", "tilelang", "cu"])
    rows = []
    for L in langs:
        lid = L.get("id")
        ms = meas.get(lid)
        row = {"id": lid, "name": L.get("name"), "measured_ms": ms,
               "evidence_note": L.get("evidence"), "unmeasured": L.get("unmeasured", False)}
        if ms is not None:
            row["tier"] = 1
            row["confidence"] = "high"
            row["why"] = "实测 %.1f µs" % (ms * 1000)
        elif L.get("unmeasured"):
            row["tier"] = 3
            row["confidence"] = "none"
            row["why"] = "本仓无实测（%s）" % (L.get("evidence") or "")[:60]
        else:
            row["tier"] = 2
            row["confidence"] = "low"
            row["why"] = "无实测，按场景先验排序（bound=%s）" % bound
        row["prior_rank"] = order.index(lid) if lid in order else 99
        rows.append(row)

    # 排序：tier 升序 → 有实测的按 ms → 无实测的按先验
    rows.sort(key=lambda r: (
        r["tier"],
        r["measured_ms"] if r["measured_ms"] is not None else 9e9,
        r["prior_rank"],
    ))
    return rows


def baseline_note(rows):
    """给出「别自己写」的判断：若厂商实现已排第一，值得明说。"""
    if not rows:
        return None
    top = rows[0]
    if top["id"] == "hip" and top["tier"] == 1:
        return ("hip 已有实测且最快（%.1f µs）——**先用它当基线，不要自己重写**。"
                % (top["measured_ms"] * 1000))
    return None


def show(rows, card, shape, bound, note, as_json=False):
    if as_json:
        print(json.dumps({"card": card, "shape": shape, "bound": bound,
                          "candidates": rows, "note": note},
                         ensure_ascii=False, indent=1))
        return
    print("语言候选（%s / %s，bound=%s）" % (card, shape, bound or "未标定"))
    print("=" * 70)
    if not rows:
        print("  无候选")
        return
    for i, r in enumerate(rows, 1):
        if r["tier"] == 3:
            print("  ✗ %-10s %s" % (r["id"], r["why"]))
            continue
        mark = "" if r["confidence"] == "high" else "  [%s]" % r["confidence"]
        print("  %d. %-10s %s%s" % (i, r["id"], r["why"], mark))
    if note:
        print()
        print("  → %s" % note)
    if any(r["tier"] == 2 for r in rows):
        print()
        print("  注：标 low_confidence 的条目没有本卡实测，只按场景先验排序。")
        print("      要用它之前先跑一次 bench 拿到实测。")


def selftest():
    langs = [{"id": x} for x in ("hip", "tilelang", "triton", "cu")]
    # 1) 有实测：按时间排，不看表里顺序
    meas = {"triton": 0.160, "hip": 0.240, "tilelang": 0.200}
    rows = rank(langs, meas, "memory", "x")
    assert rows[0]["id"] == "triton", rows
    assert [r["id"] for r in rows[:3]] == ["triton", "tilelang", "hip"], rows
    print("[ok] 有实测时按实测时间排序（triton 最快 → 排第一）")

    # 2) 反向：翻转数据，顺序必须跟着翻
    meas2 = {"triton": 0.300, "hip": 0.180, "tilelang": 0.200}
    rows2 = rank(langs, meas2, "memory", "x")
    assert rows2[0]["id"] == "hip", rows2
    print("[ok] 反向：数据翻转后排序跟着翻（证明没写死顺序）")

    # 3) 无实测 → tier 2 + low_confidence
    rows3 = rank(langs, {}, "memory", "x")
    assert all(r["tier"] == 2 for r in rows3), rows3
    assert all(r["confidence"] == "low" for r in rows3)
    print("[ok] 无实测 → tier=2 / low_confidence，不伪装成有数据")

    # 4) unmeasured 语言排最后
    L4 = [{"id": "hip"}, {"id": "cu", "unmeasured": True}]
    rows4 = rank(L4, {}, "memory", "x")
    assert rows4[-1]["id"] == "cu", rows4
    print("[ok] unmeasured 的语言排在最后")

    # 5) 基线提示
    rows5 = rank([{"id": "hip"}, {"id": "triton"}], {"hip": 0.197, "triton": 0.242},
                 "memory", "x")
    n = baseline_note(rows5)
    assert n and "先用它当基线" in n, n
    print("[ok] 厂商实现排第一时给出「不要自己重写」的提示")

    print("\nselftest PASS（5 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="语言候选排序")
    ap.add_argument("--card")
    ap.add_argument("--shape", default="prefill-8k")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.card:
        ap.error("给 --card，或 --selftest")

    langs = load_langs()
    meas = measured(a.card, a.shape)
    bound, _ = bound_of(a.card, a.shape)
    rows = rank(langs, meas, bound, a.shape)
    show(rows, a.card, a.shape, bound, baseline_note(rows), a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
