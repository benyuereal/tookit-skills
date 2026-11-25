#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth —— 把「优化有多深」变成一个可以算出来的数（DS）。

为什么需要它
------------
`op-atlas/optimization-rules.yaml` 里写了三条规则（选对基准 / 懂算子 / 深挖），
但**没有一个能算出来的数**。所以谁都说不出「我们优化深度多少分」，
于是「补了 53 张卡」看起来和「做深了 3 张」一样。

DS（Depth Score）= 五个可机检维度，加权 100：

    25 × B   基线强度    跟谁比
    25 × U   达成率      离上限多远
    20 × S   跨 shape    一个形状还是多个
    15 × R   优化轮次    试过几轮
    15 × M   机制归因    知不知道为什么

五个维度都能从仓库现有数据算出来，**不需要新增任何手工填报字段**。
这是刻意的：DS 一旦需要人填，就会变成「谁填得漂亮谁分高」。

各维度的判据
------------
B 基线强度（四级）
  1.0  baseline 是厂商竞品（lightop/aiter/flash_mla/hipblaslt/hbird-gemm/boltops/deepgemm）
  0.6  baseline 是同库不同配置，或「融合 vs 分步」这类同族对照
  0.3  baseline 是分步拼接（多个 kernel 串起来）
  0.0  baseline 是 torch 参考实现 / correctness_only / 无 comparison
  特例：单算子 GEMM 的 torch.mm 算 1.0（它走 hipBLASLt）；
        但「torch 分步 / 三段式 / fp32 累积」算 0.0。

U 达成率
  U = min(ratio.value / 0.85, 1.0)，仅当 kind=utilization。
  没有 utilization 数据 → 0。**不许用 speedup 冒充**——速度比不知道离上限多远。

S 跨 shape
  geomean > 1.0（整体更慢）→ 0.0  **方向判据，2026-11-26 加**
    S 衡量的是「多形状下的通用性」。多形状测出整体更慢，
    证明的是这个实现不通用，不是它通过了通用性检验。
  geomean 有 >=3 个 shape → 1.0；2 个 → 0.5；1 个或无 → 0.0

R 轮次
  ledger.md 里 `### 轮 N` 的个数，R = min(轮数/3, 1.0)

M 机制归因
  有 PMC 证据 / from_attribution / 明确瓶颈类 → 1.0
  有机制文字但无 PMC → 0.5
  无 → 0

用法
----
    python3 -I scripts/depth.py                 # 全仓分布
    python3 -I scripts/depth.py --json
    python3 -I scripts/depth.py --family gemm
    python3 -I scripts/depth.py --card rmsnorm-fused
    python3 -I scripts/depth.py --selftest
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
SKILLS = REPO / "Plugins/ops/skills"

VENDOR_LIBS = {"lightop", "aiter", "flash-mla", "flash_mla", "flash-attn", "flash_attn",
               "hipblaslt", "hbird-gemm", "hbird", "boltops", "deepgemm", "deep-ep",
               "causal-conv1d", "flash-kda", "mhc-kernels", "torch"}

WEIGHTS = {"B": 25, "U": 25, "S": 20, "R": 15, "M": 15}


def _load(p):
    import yaml
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:                                            # noqa: BLE001
        return {}


def baseline_strength(card):
    """B：跟谁比。返回 (分数, 理由)。

    判序很重要，从「最硬」到「最软」：
      1. correctness_only / 无 comparison → 0
      2. baseline 是厂商库 → 1.0（**与 vs 文本无关**，lightop 就是 lightop）
      3. 单算子 torch.mm → 1.0（走 hipBLASLt）
      4. vs 里写「融合 vs / 同库」→ 0.6（同族对照，baseline 本身不是厂商库）
      5. vs 里写「分步 / 三段式 / fp32 累积」→ 0.3（分步拼接）
      6. 其余 → 0.3

    第 2 条必须在第 4/5 条之前：早期版本先判文本，导致
    `baseline: lightop-fused, verdict: "融合 vs 分步"` 被判成同族对照 0.6，
    而它明明是和厂商实现比。
    """
    lv = card.get("last_verdict") or {}
    comp = lv.get("comparison") or {}
    if not comp:
        return 0.0, "无 comparison（只有绝对值/达成率）"
    role = str(comp.get("baseline_role") or "")
    base = str(comp.get("baseline") or "")
    vs = str((lv.get("ratio") or {}).get("vs") or "") + str(comp.get("verdict") or "")

    if role == "correctness_only":
        return 0.0, "baseline_role=correctness_only"

    lib = base.split(".")[0].split("-")[0].lower()
    if any(v in lib for v in VENDOR_LIBS - {"torch"}):
        return 1.0, "厂商竞品 %s" % base

    # 单算子 torch.mm 走 hipBLASLt，与「torch 分步」不同
    if base.startswith("torch") or lib == "torch":
        if re.search(r"(gemm|matmul|\bmm\b)", base, re.I) and \
                not re.search(r"(分步|三段式|拆分|累积)", vs):
            return 1.0, "单算子 torch.mm（走 hipBLASLt）"
        return 0.0, "torch 参考实现"

    if re.search(r"(融合\s*vs|同库|同一实现)", vs):
        return 0.6, "同族对照（融合 vs 分步）"
    if re.search(r"(分步|三段式|拆分|fp32 累积)", vs):
        return 0.3, "baseline 是分步拼接"
    return 0.3, "未识别的 baseline 类型"


def utilization_score(card):
    """U：离上限多远。返回 (分数, 理由)。"""
    r = (card.get("last_verdict") or {}).get("ratio") or {}
    if r.get("kind") != "utilization":
        return 0.0, "无 utilization（kind=%s，不许用 speedup 冒充）" % r.get("kind")
    v = r.get("value")
    if not isinstance(v, (int, float)):
        return 0.0, "utilization 无值"
    return min(float(v) / 0.85, 1.0), "util=%.3f / 0.85" % v


def shape_score(card):
    """S：跨 shape 覆盖。返回 (分数, 理由)。"""
    lv = card.get("last_verdict") or {}
    gm = lv.get("geomean") or {}
    shapes = gm.get("shapes") or []
    val = gm.get("value")
    # **方向判据**（2026-11-26 业主批准）。S 衡量的是「多形状下的通用性」。
    # 如果多形状测出来整体更慢（geomean > 1），那证明的是「这个实现不通用」，
    # 不是「它通过了通用性检验」。旧版只看形状个数，geomean=0.5（快一倍）
    # 和 1.45（慢 45%）给一样的分 —— 实测让 rmsnorm-fused 拿 94 分冒充标杆。
    if isinstance(val, (int, float)) and val > 1.0:
        return 0.0, ("geomean=%.3f > 1（整体更慢），多形状证明的是不通用"
                     % val)
    if len(shapes) >= 3 and val is not None:
        return 1.0, "geomean 覆盖 %d 个 shape（%.3f）" % (len(shapes), val)
    if len(shapes) == 2:
        return 0.5, "geomean 覆盖 2 个 shape"
    # 退而求其次：看 regimes 有几档
    reg = lv.get("regimes") or {}
    measured = [k for k, v in reg.items()
                if isinstance(v, dict) and v.get("value") is not None]
    if len(measured) >= 2:
        return 0.5, "regimes 覆盖 %d 档" % len(measured)
    return 0.0, "单形状或无"


def round_score(card_id):
    """R：优化轮次。从 ledger.md 数 `### 轮 N`。返回 (分数, 理由)。"""
    for d in SKILLS.glob("op-%s" % card_id):
        led = d / "ledger.md"
        if led.is_file():
            n = len(re.findall(r"(?m)^###\s+轮\s*\d+", led.read_text(encoding="utf-8")))
            return min(n / 3.0, 1.0), "ledger 记录了 %d 轮" % n
    return 0.0, "无 ledger"


def mechanism_score(card):
    """M：机制归因。返回 (分数, 理由)。"""
    lv = card.get("last_verdict") or {}
    comp = lv.get("comparison") or {}
    # 有 attribution / PMC 证据
    if card.get("attribution") or comp.get("attribution"):
        return 1.0, "有 attribution 块"
    blob = json.dumps(card, ensure_ascii=False)
    if re.search(r"(PMC|pmc_|SQ_INSTS|GRBM|TCC_HIT|瓶颈|dominant)", blob):
        return 1.0, "有 PMC/瓶颈归因证据"
    np_ = lv.get("next_probe") or {}
    stop = str(np_.get("stop_if") or "")
    if any(w in stop for w in ("⇒", "因为", "由于", "机制", "说明", "证明")):
        return 0.5, "next_probe.stop_if 有机制说明"
    return 0.0, "无机制归因"


def score_card(card_id, card):
    B, b_why = baseline_strength(card)
    U, u_why = utilization_score(card)
    S, s_why = shape_score(card)
    R, r_why = round_score(card_id)
    M, m_why = mechanism_score(card)
    ds = round(WEIGHTS["B"] * B + WEIGHTS["U"] * U + WEIGHTS["S"] * S
               + WEIGHTS["R"] * R + WEIGHTS["M"] * M)
    return {"DS": ds, "B": round(B, 2), "U": round(U, 2), "S": round(S, 2),
            "R": round(R, 2), "M": round(M, 2),
            "status": card.get("status"), "family": card.get("family"),
            "why": {"B": b_why, "U": u_why, "S": s_why, "R": r_why, "M": m_why},
            "missing": [k for k, v in (("B", B), ("U", U), ("S", S), ("R", R), ("M", M))
                        if v == 0]}


def all_scores():
    out = {}
    for p in sorted(CARDS.glob("*.yaml")):
        d = _load(p)
        if not d.get("id"):
            continue
        out[d["id"]] = score_card(d["id"], d)
    return out


def report(scores):
    vals = [v["DS"] for v in scores.values()]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 1),
        "median": vals_sorted[len(vals_sorted) // 2],
        "min": vals_sorted[0], "max": vals_sorted[-1],
        "below_40": sum(1 for v in vals if v < 40),
        "above_70": sum(1 for v in vals if v >= 70),
        "measured_below_40": sum(1 for v in scores.values()
                                 if v["status"] in ("measured", "tuned") and v["DS"] < 40),
    }


def selftest():
    # B：四种基线
    cases = [
        ({"last_verdict": {"comparison": {"baseline_role": "correctness_only"}}}, 0.0),
        ({"last_verdict": {"comparison": {"baseline_role": "competitor",
                                          "baseline": "lightop-fused"}}}, 1.0),
        ({"last_verdict": {"comparison": {"baseline_role": "competitor",
                                          "baseline": "torch.mm"}}}, 1.0),
        ({"last_verdict": {"comparison": {"baseline_role": "competitor",
                                          "baseline": "lightop-fused",
                                          "verdict": "融合 vs 分步"}}}, 1.0),
        ({"last_verdict": {"comparison": {"baseline": "x", "verdict": "torch 三段式"}}}, 0.3),
        ({}, 0.0),
    ]
    for card, want in cases:
        got, why = baseline_strength(card)
        assert got == want, (card, got, want, why)
    # 非厂商 baseline + 「融合 vs 分步」→ 同族对照 0.6
    got, why = baseline_strength({"last_verdict": {"comparison": {
        "baseline": "in-house-fused", "verdict": "融合 vs 分步"}}})
    assert got == 0.6, (got, why)
    print("[ok] B 基线强度：correctness_only=0 / torch.mm=1.0 / 三段式=0.3 / 同族=0.6 / 无=0")

    # U：speedup 不许冒充
    u, why = utilization_score({"last_verdict": {"ratio": {"kind": "speedup", "value": 2.2}}})
    assert u == 0.0, (u, why)
    u, _ = utilization_score({"last_verdict": {"ratio": {"kind": "utilization",
                                                         "value": 0.85}}})
    assert abs(u - 1.0) < 1e-9, u
    print("[ok] U 达成率：speedup 不给分 / util 0.85 封顶")

    # S：geomean 形状数（用 <1 的值，否则会先被方向判据拦下）
    s, _ = shape_score({"last_verdict": {"geomean": {"value": 0.5,
                                                     "shapes": ["a", "b", "c"]}}})
    assert s == 1.0, s
    s, _ = shape_score({"last_verdict": {"geomean": {"value": 0.5, "shapes": ["a"]}}})
    assert s == 0.0, s
    print("[ok] S 跨 shape：3 个=1.0 / 1 个=0")

    # S 的方向判据（2026-11-26 加）：整体更慢时多形状不算通过
    s, why = shape_score({"last_verdict": {"geomean": {
        "value": 1.4496, "shapes": ["a", "b", "c"]}}})
    assert s == 0.0, (s, why)
    s, why = shape_score({"last_verdict": {"geomean": {
        "value": 1.1, "shapes": ["a", "b", "c"]}}})
    assert s == 0.0, (s, why)
    s, why = shape_score({"last_verdict": {"geomean": {
        "value": 1.0001, "shapes": ["a", "b", "c"]}}})
    assert s == 0.0, (s, why)
    s, why = shape_score({"last_verdict": {"geomean": {
        "value": 1.0, "shapes": ["a", "b", "c"]}}})
    assert s == 1.0, ("恰好 1.0（打平）应算通过边界", s, why)
    s, why = shape_score({"last_verdict": {"geomean": {
        "value": 0.5002, "shapes": ["a", "b", "c"]}}})
    assert s == 1.0, (s, why)
    print("[ok] S 方向判据：geomean>1 判 0（哪怕 3 个形状）；=1.0 与 <1 判满分")

    # 总分的量纲
    fake = {"status": "measured", "family": "norm",
            "last_verdict": {"comparison": {"baseline_role": "competitor",
                                            "baseline": "lightop-fused"},
                             "ratio": {"kind": "utilization", "value": 0.85}}}
    r = score_card("__selftest__", fake)
    assert r["DS"] == 25 + 25, r
    assert 0 <= r["DS"] <= 100
    print("[ok] DS 范围 0..100，本例 B+U = 50")

    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="优化深度评分（Depth Score）")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--family")
    ap.add_argument("--card")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    scores = all_scores()
    if a.card:
        s = scores.get(a.card)
        if not s:
            print("没有这张卡：%s" % a.card, file=sys.stderr)
            return 2
        print(json.dumps({a.card: s}, ensure_ascii=False, indent=2) if a.json
              else "%s  DS=%d  (B%.0f U%.0f S%.0f R%.0f M%.0f)\n%s"
                   % (a.card, s["DS"], s["B"] * 25, s["U"] * 25, s["S"] * 20,
                      s["R"] * 15, s["M"] * 15,
                      "\n".join("  %s: %s" % (k, v) for k, v in s["why"].items())))
        return 0

    if a.family:
        scores = {k: v for k, v in scores.items() if v["family"] == a.family}

    if a.json:
        print(json.dumps({"cards": scores, "summary": report(scores)},
                         ensure_ascii=False, indent=2))
        return 0

    print("%-26s %4s %4s %4s %4s %4s %4s  %s"
          % ("card", "DS", "B", "U", "S", "R", "M", "status"))
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]["DS"]):
        print("%-26s %4d %4.0f %4.0f %4.0f %4.0f %4.0f  %s"
              % (k, v["DS"], v["B"] * 25, v["U"] * 25, v["S"] * 20,
                 v["R"] * 15, v["M"] * 15, v["status"]))
    r = report(scores)
    print()
    print("n=%d  均值 %.1f  中位 %s  范围 %d..%d"
          % (r["n"], r["mean"], r["median"], r["min"], r["max"]))
    print("DS<40: %d 张（其中 measured/tuned %d 张）  DS>=70: %d 张"
          % (r["below_40"], r["measured_below_40"], r["above_70"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
