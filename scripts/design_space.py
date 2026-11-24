#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""design_space —— 枚举 tile 配置并预测哪个好。

为什么需要它（v6 的 C3）
------------------------
`derive.py` 能算**给定配置**的地板，`matmul.py` 能算**给定 tile** 的资源与预测。
但**没有任何东西枚举这些组合**，也不能回答「BM 128→64 会让 remaining
从 0.3 降到 0.1 还是升到 0.5」。

没有这个，agent 只能「试一个、看结果、再试」，**没有先验——那不是研发，是盲试**。

本工具做三件事
--------------
1. 枚举 (BM, BN, BK, warps, stages, split_k)
2. 用 `models/matmul.py` 的 `_legality` 过滤，**每个被拒的给出具体理由**
3. 用模型预测每个合法配置的 util / 耗时，按 predicted_remaining 排序

**关键纪律**：预测来自 `matmul.py` 的经验式，不是物理定律。
输出必带 `model_limit`，且只用于**排序**，不用于宣布结论。

用法
----
    python3 -I scripts/design_space.py --card gemm-bf16 --shape prefill-8k
    python3 -I scripts/design_space.py --m 8192 --n 8192 --k 8192 --json
    python3 -I scripts/design_space.py --selftest
"""
import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "Plugins/ops/skills/op-roofline/models/matmul.py"
CARDS = REPO / "Plugins/ops/op-atlas/cards"
SKILLS = REPO / "Plugins/ops/skills"

# 枚举维度。**故意给得不宽**——486 个组合已经够多，
# 再宽会让信号淹没在噪声里（且真实 tuner 也不会全扫）。
GRID = {
    "bm": (64, 128, 256),
    "bn": (64, 128, 256),
    "bk": (16, 32, 64),
    "warps": (4, 8),
    "stages": (2, 3, 4),
    "split_k": (1, 2, 4),
}


def _model():
    spec = importlib.util.spec_from_file_location("_mm", str(MODEL))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shape_of(card, shape_name):
    """从卡片 / skill 的 shapes.yaml 取 (M, N, K)。取不到返回 None。"""
    import yaml
    p = SKILLS / ("op-%s" % card.replace("op-", "")) / "shapes.yaml"
    if not p.is_file():
        return None
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    node = d.get(shape_name) or {}
    pr = node.get("params") or {}
    m, n, k = pr.get("M"), pr.get("N"), pr.get("K")
    if m and n and k:
        return int(m), int(n), int(k)
    return None


def enumerate_space(m, n, k, grid=None, dtype_bytes=2, lds_sub_rows=None):
    """枚举 + 过滤 + 预测。返回 {"configs": [...], "rejected": [...], "stats": {...}}。"""
    mm = _model()
    g = grid or GRID
    configs, rejected = [], []
    total = 0
    for bm, bn, bk, w, st, sk in itertools.product(
            g["bm"], g["bn"], g["bk"], g["warps"], g["stages"], g["split_k"]):
        total += 1
        cfg = {"bm": bm, "bn": bn, "bk": bk, "warps": w,
               "stages": st, "split_k": sk}
        r = mm.analyze(m, n, k, bm, bn, bk, w, st, sk,
                       dtype_bytes=dtype_bytes, lds_sub_rows=lds_sub_rows)
        if r.get("illegal"):
            rejected.append({**cfg, "reason": r["reject_reason"],
                             "resources": r["resources"]})
            continue
        configs.append({
            **cfg,
            "lds": r["resources"]["lds"],
            "vgpr": r["resources"]["vgpr"],
            "ctas_per_cu": r["resources"]["ctas_per_cu"],
            "grid": r["grid"],
            "t_floor_us": r["t_floor_us"],
            "predicted_mfu": r["predicted_mfu"],
            "predicted_us": r["predicted_us"],
            "predicted_remaining": round(1 / r["predicted_mfu"] - 1, 4)
            if r["predicted_mfu"] else None,
            "confidence": r["confidence"],
            "reason": r["predicted_reason"],
        })
    # 按 predicted_remaining 升序（越小越接近地板）
    configs.sort(key=lambda c: (c["predicted_remaining"] is None,
                                c["predicted_remaining"] or 9e9))
    return {"m": m, "n": n, "k": k, "configs": configs, "rejected": rejected,
            "stats": {"enumerated": total, "legal": len(configs),
                      "rejected": len(rejected)},
            "model_limit": mm.analyze(m, n, k)["model_limit"]}


def show(res, top=12, as_json=False):
    if as_json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    s = res["stats"]
    print("design space（M=%d N=%d K=%d）" % (res["m"], res["n"], res["k"]))
    print("枚举 %d / 合法 %d / 被拒 %d" % (s["enumerated"], s["legal"], s["rejected"]))
    print()
    if not res["configs"]:
        print("  **无合法配置** —— 这本身是个结论：tile 范围或形状有问题")
        return
    print("%-26s %-7s %-5s %-6s %-9s %-8s %s"
          % ("配置", "LDS", "VGPR", "CTA/CU", "floor(µs)", "预测util", "预测remaining"))
    for c in res["configs"][:top]:
        name = "BM%d BN%d BK%d w%d s%d k%d" % (
            c["bm"], c["bn"], c["bk"], c["warps"], c["stages"], c["split_k"])
        print("%-26s %-7d %-5d %-6d %-9.0f %-8.2f %+.0f%%"
              % (name, c["lds"], c["vgpr"], c["ctas_per_cu"],
                 c["t_floor_us"], c["predicted_mfu"],
                 c["predicted_remaining"] * 100))
    print()
    print("被拒原因分布：")
    from collections import Counter
    kinds = Counter()
    for r in res["rejected"]:
        for e in r["reason"]:
            key = e.split("=")[0].split("（")[0].split("%")[0][:14]
            kinds[key] += 1
    for k, v in kinds.most_common(5):
        print("  %-18s %d 个" % (k, v))
    print()
    print("⚠ " + res["model_limit"][:100])
    print("   预测只用于**排序**，不用于宣布结论。")


# ---------------------------------------------------------------- selftest

def selftest():
    # 1) 8192³ 能枚举出合法配置
    r = enumerate_space(8192, 8192, 8192)
    assert r["stats"]["enumerated"] == 486, r["stats"]
    assert r["stats"]["legal"] > 0, "8192³ 一个合法配置都没有，枚举范围有问题"
    print("[ok] 8192³：枚举 %d / 合法 %d" % (
        r["stats"]["enumerated"], r["stats"]["legal"]))

    # 2) 被拒的要给具体理由
    assert r["rejected"], "应该有一批被拒"
    assert all("reason" in x and x["reason"] for x in r["rejected"])
    print("[ok] 被拒配置都带具体理由（如 %r）" % r["rejected"][0]["reason"][0][:40])

    # 3) 排序：predicted_remaining 升序
    rem = [c["predicted_remaining"] for c in r["configs"]]
    assert rem == sorted(rem), "排序不对"
    print("[ok] 按 predicted_remaining 升序排列")

    # 4) **反向**：预测不许是常数
    utils = {c["predicted_mfu"] for c in r["configs"]}
    assert len(utils) > 1, "所有配置预测同一个 util，模型没起作用"
    print("[ok] 反向：预测值有 %d 种（不是常数）" % len(utils))

    # 5) **反向**：LDs 超限必须被拒，且理由含字节数
    big = [x for x in r["rejected"] if any("LDS" in e for e in x["reason"])]
    assert big, "应该有 LDS 超限被拒的"
    assert any("B >" in e for e in big[0]["reason"]), big[0]["reason"]
    print("[ok] 反向：LDS 超限被拒，理由含具体字节数")

    # 6) **反向**：形状太小时 grid < CU 也要被拒
    small = enumerate_space(256, 256, 8192)
    grid_rej = [x for x in small["rejected"] if any("grid" in e for e in x["reason"])]
    assert grid_rej, "小形状应有 grid 不足被拒的"
    print("[ok] 反向：小形状 %d 个配置因 grid < CU 被拒" % len(grid_rej))

    # 7) 输出必带 model_limit
    assert "model_limit" in r and "经验式" in r["model_limit"]
    print("[ok] 输出带 model_limit（预测是经验式，不是物理定律）")

    print("\nselftest PASS（7 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="枚举 tile 配置并预测")
    ap.add_argument("--card")
    ap.add_argument("--shape", default="prefill-8k")
    ap.add_argument("--m", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--lds-sub-rows", type=int, default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    if a.m and a.n and a.k:
        m, n, k = a.m, a.n, a.k
    elif a.card:
        got = shape_of(a.card, a.shape)
        if not got:
            print("ERROR: 卡片 %s 的 shapes.yaml 里没有 %s 的 M/N/K。"
                  "用 --m --n --k 显式给。" % (a.card, a.shape), file=sys.stderr)
            return 2
        m, n, k = got
    else:
        ap.error("给 --card --shape 或 --m --n --k，或 --selftest")

    res = enumerate_space(m, n, k, dtype_bytes=a.dtype_bytes,
                          lds_sub_rows=a.lds_sub_rows)
    show(res, top=a.top, as_json=a.json)
    return 0 if res["configs"] else 1


if __name__ == "__main__":
    sys.exit(main())
