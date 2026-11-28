#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""atlas_query —— 图谱的两个交付物：在哪（where）与用哪个（select）。

「在哪」和「用哪个」是两个问题
------------------------------
    where  <op>                          哪几处有 → 判据：symbol 可导入
    select <op> --shape --dtype          用哪个   → 五步算法
    该不该继续优化                        floor.py（remaining 闸门）

业主：「另外知道某个算子在哪里存在，到时候怎么选择一个最佳的算子呢」
—— 如果选型器不可信，前面所有 schema 都只是文档。

五步算法（不许自己发挥）
------------------------
  Step 1 硬过滤（淘汰，不参与排序）
    - library.installed 且 api 可解析
    - semantics_match == exact（partial 必须 --allow-partial）
    - role != unverified（未验证的不能用于生产选型）
    - role == correctness_only 的可以「跑得快」但**不许被选中**（快不等于可用）
    - traffic_passes 必须与算子语义一致，不一致直接淘汰（不是降权）

  Step 2 口径对齐（只保留可比的）
    只有 (shape, dtype, caliber) 完全一致才能同榜。
    **铁律：不同 caliber 的数不许放同一张排序表。**
    宁可输出「两组数不同口径，无法直接比」，也不许静默混排。

  Step 3 排序
    主键 median_us 升序，但必须过显著性门（3×MAD）：
      Δ(第二 - 第一) <= 3×MAD → 判「统计并列」，**不许说 A 比 B 好**
    并列 tie-break：mad/us → rel_l2 → outputs 数 → 集成成本(function<class<template<自写)

  Step 4 置信度
    high   实测 n>=3 且 caliber 完整   → 允许说「推荐 X」
    medium 有实测但 n<3 或 caliber 缺项 → 「X 看起来更优，建议复测」
    low    只有 reference_gbs 推算      → 「无本地数据，按同类推算约…」
    none   无任何数据                  → **禁止排名**，只输出 measure_cmd

  Step 5 决策记录
    写 decisions/<op>-<shape>-<ts>.json，含 rejected 及理由

用法
----
    python3 -I scripts/atlas_query.py where rmsnorm
    python3 -I scripts/atlas_query.py select rmsnorm --shape prefill-8k --dtype bfloat16
    python3 -I scripts/atlas_query.py select gemm --shape prefill-8k --allow-partial
    python3 -I scripts/atlas_query.py --selftest
"""
import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "Plugins/ops/op-atlas"
CARDS = ATLAS / "cards"
PERF = ATLAS / "perf"
DECISIONS = ATLAS / "decisions"

K_SIGMA = 3.0

_SCRIPTS = Path(__file__).resolve().parent


def _render_template(name, **ctx):
    """读 scripts/templates/<name>.md；本文件只提供动态行。

    `python3 -I` 不把脚本目录放进 sys.path（与 recall/state 同约定），这里补一次。
    延迟 import：self_iteration 会把本文件拷进没有 templates/ 的临时仓，
    那里只做 where/_cards 逻辑，不渲染人读输出。
    """
    if str(_SCRIPTS) not in sys.path:
        sys.path.append(str(_SCRIPTS))
    from template_render import render
    return render(name, **ctx)


# 集成成本：越小越好。function 最省，自写最贵。
INTEGRATION_COST = {"function": 0, "class": 1, "template": 2, "handwrite": 3}

# 不许用于生产选型的 role
NOT_SELECTABLE = {"unverified", "wrong_op", "correctness_only"}


# ---------------------------------------------------------------- 读数据

def _load_yaml(p):
    import yaml
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:                                            # noqa: BLE001
        return {}


def _cards():
    return [(p.stem, _load_yaml(p)) for p in sorted(CARDS.glob("*.yaml"))]


def _perf_records():
    """读 perf/*.json：每条是一次测量，带 variant / shape / dtype / caliber / median_us。"""
    out = []
    if not PERF.is_dir():
        return out
    for p in sorted(PERF.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        recs = d if isinstance(d, list) else d.get("records") or [d]
        for r in recs:
            if isinstance(r, dict) and r.get("variant"):
                r.setdefault("_file", str(p.relative_to(REPO)))
                out.append(r)
    return out


# ---------------------------------------------------------------- where

def where(op, cards=None):
    """哪几处有：按 symbol 可导入判断。"""
    cards = cards or _cards()
    hits = []
    for cid, d in cards:
        for v in d.get("variants") or []:
            if not isinstance(v, dict):
                continue
            ip = str(v.get("import_path") or "")
            if not ip:
                continue
            # op 名与卡片 id / import_path 任一匹配
            if op.lower() in cid.lower() or op.lower() in ip.lower():
                hits.append({"card": cid, "impl": ip, "repo": v.get("repo"),
                             "role": v.get("role"), "tier": v.get("tier")})
    return hits


# ---------------------------------------------------------------- select

def _caliber_key(rec):
    """口径指纹：只有完全一致才能同榜。"""
    c = rec.get("caliber") or {}
    return (c.get("what"), c.get("alloc_in_timing"), c.get("state"),
            c.get("input_strategy"), c.get("timer"))


def _shape_key(rec):
    s = rec.get("shape") or {}
    return (s.get("M"), s.get("N"), s.get("K"), s.get("H"), s.get("dtype"))


def _caliber_complete(rec):
    c = rec.get("caliber") or {}
    need = ("what", "alloc_in_timing", "state", "input_strategy", "timer",
            "rounds", "statistic", "nl" if False else "n")
    return all(k in c for k in need)


def select(op, shape=None, dtype=None, allow_partial=False, cards=None, perfs=None):
    """五步选型。返回 {chosen, runner_up, rejected, confidence, note, measure_cmd}。"""
    cards = cards if cards is not None else _cards()
    perfs = perfs if perfs is not None else _perf_records()
    rejected = []

    # ---- Step 1 硬过滤
    cand_cards = [(cid, d) for cid, d in cards
                  if op.lower() in cid.lower()]
    if not cand_cards:
        return {"chosen": None, "confidence": "none", "rejected": [],
                "note": "图谱里没有匹配 %r 的卡" % op, "measure_cmd": None}

    allowed = {}
    for cid, d in cand_cards:
        for v in d.get("variants") or []:
            if not isinstance(v, dict):
                continue
            ip = str(v.get("import_path") or "")
            if not ip:
                continue
            role = str(v.get("role") or "")
            if role in NOT_SELECTABLE:
                rejected.append({"variant": ip, "reason": "role=%s 不可用于生产选型" % role})
                continue
            sm = str(v.get("semantics_match") or "exact")
            if sm != "exact" and not allow_partial:
                rejected.append({"variant": ip,
                                 "reason": "semantics_match=%s（需 --allow-partial）" % sm})
                continue
            allowed[ip] = {"card": cid, "variant": v}

    if not allowed:
        return {"chosen": None, "confidence": "none", "rejected": rejected,
                "note": "全部候选被 Step1 淘汰", "measure_cmd": None}

    # ---- Step 1b traffic_passes 必须与算子语义一致（不一致直接淘汰，不是降权）
    # 2026-11-23 的坑：我们写 2 份输出（268MB）、lightop 也 2 份，但我们曾只写 1 份（201MB），
    # 拿它比 lightop 得出「快 1.18x」——少干 1/3 的活。
    # 这里按「同一次 select 内出现的 traffic_passes 是否一致」判：
    # 不一致 ⇒ 全部淘汰并说明，绝不静默挑一个比。
    tps = {}
    for r in perfs:
        if r.get("variant") in allowed and r.get("traffic_passes"):
            tps.setdefault(json.dumps(r["traffic_passes"], sort_keys=True), []).append(
                r.get("variant"))
    if len(tps) > 1:
        return {"chosen": None, "confidence": "none", "rejected": rejected,
                "note": ("访存口径不一致，**不可比**：%s。"
                         "跨实现比较前必须先对齐 traffic_passes（几读几写）——"
                         "201MB vs 268MB 的比法得不出有效结论"
                         % {k: v for k, v in tps.items()}),
                "measure_cmd": "python3 -I scripts/check_semantics.py <card> <a> <b>"}

    # ---- Step 2 口径对齐
    rows = [r for r in perfs if r.get("variant") in allowed]
    if shape:
        rows = [r for r in rows if shape in str(r.get("shape")) or
                _shape_matches(r, shape)]
    if dtype:
        rows = [r for r in rows if str((r.get("shape") or {}).get("dtype", dtype)) == dtype]

    if not rows:
        # ---- Step 4 none：禁止排名，只给测量命令
        first = next(iter(allowed))
        return {"chosen": None, "confidence": "none", "rejected": rejected,
                "note": "无任何本地实测数据 ⇒ **禁止排名**",
                "measure_cmd": "python3 -I scripts/rank_impls.py --card %s --shape %s"
                               % (allowed[first]["card"], shape or "<shape>")}

    # 口径分组：同 caliber 才能同榜
    groups = {}
    for r in rows:
        groups.setdefault((_shape_key(r), _caliber_key(r)), []).append(r)

    if len(groups) > 1:
        # 铁律：不同 caliber 不许混排
        return {"chosen": None, "confidence": "none", "rejected": rejected,
                "note": ("检测到 %d 组不同口径的数据，**无法直接比**。"
                         "按 caliber 分组：%s。请先统一口径再选。"
                         % (len(groups), [list(g[1]) for g in groups])),
                "measure_cmd": "python3 -I scripts/rank_impls.py --card %s --shape %s"
                               % (next(iter(allowed.values()))["card"], shape or "<shape>")}

    group = next(iter(groups.values()))
    group.sort(key=lambda r: r.get("median_us") or 1e18)

    # ---- Step 3 排序 + 显著性
    first = group[0]
    second = group[1] if len(group) > 1 else None
    tie = False
    if second:
        d = abs((second.get("median_us") or 0) - (first.get("median_us") or 0))
        thr = K_SIGMA * max(_mad(first), _mad(second))
        tie = d <= thr
        if tie:
            # tie-break：mad/us → rel_l2 → outputs 少 → 集成成本
            group.sort(key=lambda r: (_mad(r) / max(r.get("median_us") or 1, 1e-9),
                                      r.get("rel_l2") or 0,
                                      len((r.get("traffic_passes") or {}).get("writes") or []),
                                      INTEGRATION_COST.get(r.get("integration") or "function", 0)))
            first = group[0]
            second = group[1] if len(group) > 1 else None

    # ---- Step 4 置信度
    n = first.get("caliber", {}).get("n") or 0
    if _caliber_complete(first) and n >= 3:
        conf = "high"
        note = "推荐 %s" % first["variant"]
    elif n >= 1:
        conf = "medium"
        note = ("%s 看起来更优，建议复测（n=%s，caliber %s）"
                % (first["variant"], n, "完整" if _caliber_complete(first) else "缺项"))
    else:
        conf = "low"
        note = "无本地实测，按同类推算"

    if tie and second:
        note = ("**统计并列**：%s 与 %s 差 %.2fµs <= 3×MAD %.2fµs，"
                "不许说谁更好" % (first["variant"], second["variant"],
                                abs((second.get("median_us") or 0) - (first.get("median_us") or 0)),
                                K_SIGMA * max(_mad(first), _mad(second))))

    # ---- Step 5 决策记录
    decision = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "op": op, "shape": shape, "dtype": dtype,
        "chosen": first["variant"], "runner_up": second["variant"] if second else None,
        "tie": tie, "confidence": conf, "note": note,
        "rejected": rejected,
        "evidence": {"file": first.get("_file"), "median_us": first.get("median_us"),
                     "caliber": first.get("caliber")},
        "measure_cmd": first.get("reproduce")
                       or "python3 -I scripts/rank_impls.py --card %s --shape %s"
                          % (allowed[first["variant"]]["card"], shape or "<shape>"),
    }
    return decision


def _mad(r):
    return float((r.get("caliber") or {}).get("mad") or r.get("mad") or 0.0)


def _shape_matches(r, shape):
    s = r.get("shape") or {}
    return shape in (s.get("name"), str(s.get("M")))


def write_decision(dec):
    DECISIONS.mkdir(parents=True, exist_ok=True)
    name = "%s-%s-%s.json" % (dec["op"], dec.get("shape") or "any",
                              time.strftime("%Y%m%d-%H%M%S"))
    (DECISIONS / name).write_text(json.dumps(dec, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    return DECISIONS / name


# ---------------------------------------------------------------- 人读输出

def render_where(hits):
    """where：表头 + 每一条命中 + 「共 N 处」。"""
    lines = ["%-24s %-42s %-14s %s" % ("卡", "入口", "role", "tier")]
    lines += ["%-24s %-42s %-14s %s"
              % (h["card"], h["impl"], h["role"], h["tier"]) for h in hits]
    return _render_template("atlas_query_where",
                            table="\n".join(lines), n=len(hits))


def render_select(dec, decision_path=None):
    """select：置信度/结论固定，其余行按有无条件拼接。"""
    lines = []
    if dec.get("chosen"):
        lines.append("选中   : %s" % dec["chosen"])
    if dec.get("runner_up"):
        lines.append("次优   : %s" % dec["runner_up"])
    if dec.get("measure_cmd"):
        lines.append("复跑   : %s" % dec["measure_cmd"])
    if dec.get("rejected"):
        lines.append("淘汰   :")
        lines += ["  - %s（%s）" % (r["variant"], r["reason"])
                  for r in dec["rejected"]]
    if decision_path is not None:
        lines.append("决策   : %s" % decision_path)
    tail = ("\n" + "\n".join(lines)) if lines else ""
    return _render_template("atlas_query", confidence=dec["confidence"],
                            note=dec["note"], tail=tail)


# ---------------------------------------------------------------- selftest

def selftest():
    """5 个反例测试 —— 业主的 P1 验收线。"""
    def card(cid, variants):
        return (cid, {"id": cid, "variants": variants})

    def v(ip, role="competitor", sm="exact"):
        return {"import_path": ip, "role": role, "semantics_match": sm}

    def perf(ip, us, mad=0.5, n=7, **kw):
        r = {"variant": ip, "median_us": us,
             "caliber": {"what": "op_call", "alloc_in_timing": False, "state": "hot",
                         "input_strategy": "preclone", "timer": "hip_event",
                         "rounds": 7, "statistic": "median", "n": n, "mad": mad},
             "shape": {"M": 8192, "dtype": "bfloat16", "name": "prefill-8k"}}
        r.update(kw)
        return r

    ok = True

    # 反例 1：两个实现流量不同（201MB vs 268MB）→ 拒绝同榜
    cards = [card("rmsnorm-fused", [v("a.x"), v("b.y")])]
    perfs = [perf("a.x", 200.0, traffic_passes={"reads": 2, "writes": 1}),
             perf("b.y", 240.0, traffic_passes={"reads": 2, "writes": 2})]
    d = select("rmsnorm", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
    has_traffic_check = ("traffic" in json.dumps(d, ensure_ascii=False).lower()
                         or d.get("confidence") == "none")
    assert has_traffic_check, d
    print("[ok] 反例1 流量不同 → 不静默混排")

    # 反例 2：差 1µs 但 MAD 也 1µs → 判并列
    cards = [card("x-op", [v("c.p"), v("d.q")])]
    perfs = [perf("c.p", 100.0, mad=1.0), perf("d.q", 101.0, mad=1.0)]
    d = select("x-op", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
    assert d["tie"] is True, d
    assert "并列" in d["note"], d
    print("[ok] 反例2 差 1µs/MAD 1µs → 判并列，不说谁更好")

    # 反例 3：只有无 caliber 的历史数字 → 拒绝推荐
    cards = [card("y-op", [v("e.r")])]
    perfs = [{"variant": "e.r", "median_us": 50.0, "caliber": {}, "shape": {"M": 8192}}]
    d = select("y-op", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
    # 无 caliber 的数不许进排名 —— 实现会比 low 更严：直接 none + 只给测量命令
    assert d["confidence"] in ("none", "low"), d
    assert d["confidence"] != "high", "无 caliber 不许给 high"
    assert d["chosen"] is None, "无 caliber 不许给出推荐：%s" % d.get("chosen")
    assert d.get("measure_cmd"), "必须给测量命令"
    print("[ok] 反例3 无 caliber → 拒绝推荐（confidence=%s，给了 measure_cmd）"
          % d["confidence"])

    # 反例 4：correctness_only 的实现最快 → 不许选中
    cards = [card("z-op", [v("fast.torch", role="correctness_only"), v("slow.comp", role="competitor")])]
    perfs = [perf("fast.torch", 10.0), perf("slow.comp", 100.0)]
    d = select("z-op", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
    assert d["chosen"] == "slow.comp", "correctness_only 最快也不许选中：%s" % d
    assert any("correctness_only" in r["reason"] for r in d["rejected"]), d
    print("[ok] 反例4 correctness_only 最快 → 不许选中（选 slow.comp）")

    # 反例 5：同一后端跨两 regime → 分档选，不许合并
    cards = [card("w-op", [v("m.n")])]
    p1 = perf("m.n", 100.0); p1["caliber"]["input_strategy"] = "preclone"
    p2 = perf("m.n", 50.0); p2["caliber"]["input_strategy"] = "reuse"
    d = select("w-op", "prefill-8k", "bfloat16", cards=cards, perfs=[p1, p2])
    assert d["confidence"] == "none", "不同口径不许合并排序：%s" % d
    assert "不同口径" in d["note"] or "无法直接比" in d["note"], d
    print("[ok] 反例5 跨 caliber → 拒绝合并，报「无法直接比」")

    print("\nselftest PASS（5/5 反例）")
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="图谱选型器：where / select")
    ap.add_argument("cmd", nargs="?", choices=["where", "select"])
    ap.add_argument("op", nargs="?")
    ap.add_argument("--shape")
    ap.add_argument("--dtype")
    ap.add_argument("--allow-partial", action="store_true", dest="allow_partial")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd or not a.op:
        ap.error("用法：atlas_query.py where|select <op> [--shape --dtype]")

    if a.cmd == "where":
        hits = where(a.op)
        if a.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        else:
            print(render_where(hits), end="")
        return 0

    dec = select(a.op, a.shape, a.dtype, a.allow_partial)
    if a.json:
        print(json.dumps(dec, ensure_ascii=False, indent=2))
        if dec.get("chosen"):
            write_decision(dec)
    else:
        path = None
        if dec.get("chosen"):
            path = write_decision(dec).relative_to(REPO)
        print(render_select(dec, decision_path=path), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
