#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_means —— 手工候选必须先写清算理，再开跑。

为什么需要它（借 v7.1 的「先写推理再跑」）
----------------------------------------
本仓现有闸门只看 `expect` 在不在、是不是数（check_expect.py）。结果是
「先跑再看」：候选写个 -8% 就能进队列，为什么这么改、预期卡在哪，全空。
研发与搜索的差别就在这里 —— 前者说得出机制和预期瓶颈，后者只是试。

但**不能照搬**：`advance --forge` 会程序化生成变体（scripts/advance.py
写 overlay，Plugins/ops/engine/loop.py 打 `_origin: advance-overlay`），
它们没有人类推理可写。所以规则只分两支：
  · 手工候选（无 _origin）→ 必须写齐 means / predicted_effect / expected_bottleneck
  · 自动生成候选（_origin: advance-overlay）→ 免检，但必须有 forge_evidence，
    证明它真过了编译体检；否则无法区分「真生成」和「手填冒充」。

判据（只有两条，别加第三条）
----------------------------
1. means：非空、去空白后 ≥15 字符、不是 TODO/待定/略 这类占位符。
   ≥15 是防「优化一下」「调参」—— 那种话等于没写，跑出来也无法归因。
2. predicted_effect：必须能被机器解析（-6%~-10% / -8% / 0%）。散文预期
   无法证伪：「省一次写回」涨了 3% 算不算兑现？没人说得清（check_expect 的教训）。
   数值端点复用 check_expect.NUMERIC，不另起一套正则，否则两处会漂移。
3. expected_bottleneck：断言预期卡在带宽/算力/发射/延迟/混合。取值必须是
   surfaces.yaml 里出现过的 class，或词表里的 bound 词。自造词（"gpu-bound"）
   无法与 loop.py 的天花板约束对齐，写下来也没人会读。

边界与语义
----------
纯函数式：路径进、问题列表出；不写文件、不改 YAML、不碰 GPU。只查 pending
（已 adopted/verified 的候选已经跑完，回头补 means 不是这道闸门的事）。
退出码：0 全过 / 1 有违规 / 2 用法错误或找不到文件。

用法
----
    python3 -I scripts/check_means.py Plugins/ops/skills/op-rmsnorm-fused
    python3 -I scripts/check_means.py <卡> --json
    python3 -I scripts/check_means.py --selftest
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SURFACES = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"

# 与 loop.py 的 BOUND_ALIASES 同一套词：迟绑定到 runtime 的 bound 判定上。
BOUND_WORDS = {"memory-bound", "compute-bound", "launch-bound",
               "latency-bound", "mixed"}
# 占位符不算「写了」。单列出来而不是靠长度判断：TODO 三个字符能过空判，
# 但它在任何语义下都不构成「怎么做」。
PLACEHOLDERS = {"todo", "tbd", "待定", "略", "待补", "占位", "同上",
                "n/a", "na", "-", "...", "无"}
MIN_MEANS = 15


def _load_check_expect():
    """按路径加载同目录的 check_expect.py，只为了复用它那条 NUMERIC 正则。"""
    p = Path(__file__).with_name("check_expect.py")
    spec = importlib.util.spec_from_file_location("_check_expect_for_means", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CE = _load_check_expect()
NUMERIC = CE.NUMERIC          # 复用：单值 -8% / 0% / 1.15x 的解析口径
RANGE = re.compile(r"[~～]")


def load_bottlenecks(surfaces=SURFACES):
    """合法 bottleneck 词 = surfaces.yaml 的 class ∪ 固定的 bound 词。"""
    import yaml
    raw = yaml.safe_load(Path(surfaces).read_text(encoding="utf-8")) or []
    classes = {str(s["class"]).strip().lower()
               for s in raw if isinstance(s, dict) and s.get("class")}
    return classes | BOUND_WORDS


def _means_problem(v):
    if v is None:
        return "缺 means —— 没说怎么做就开跑，这是搜索不是研发"
    s = str(v).strip()
    if not s:
        return "means 为空 —— 空串和没写一样"
    if s.lower() in PLACEHOLDERS:
        return "means=%r 是占位符 —— 占位符进队列，跑完也无法归因" % v
    if len(s) < MIN_MEANS:
        return "means=%r 只有 %d 字符（要求 ≥%d）—— 太短说明不了机制" % (
            v, len(s), MIN_MEANS)
    return None


def _effect_problem(v):
    """predicted_effect 必须可机检：单值或区间。散文一律拒。"""
    if v is None:
        return "缺 predicted_effect —— 没有可证伪的预期"
    s = str(v).strip()
    parts = RANGE.split(s)
    ok = (len(parts) == 2 and all(NUMERIC.match(x) for x in parts)) or \
         (len(parts) == 1 and bool(NUMERIC.match(s)))
    if ok:
        return None
    return ("predicted_effect=%r 不是可机检的数值（接受 -8%% / -6%%~-10%% / 0%%）"
            "—— 散文预期无法证伪" % v)


def _bottleneck_problem(v, valid):
    if v is None:
        return "缺 expected_bottleneck —— 没断言卡在哪，就无法判断方向对不对"
    s = str(v).strip().lower()
    if s in valid:
        return None
    return ("expected_bottleneck=%r 不在 surfaces.yaml 的 class / bound 词表"
            "（可选：%s）—— 自造词对不上天花板约束" % (v, ", ".join(sorted(valid))))


def check_card(card, surfaces=SURFACES):
    """返回 (problems, stats)。problems 为 None 表示 candidates.yaml 不存在。"""
    card = Path(card)
    p = card / "candidates.yaml"
    if not p.is_file():
        return None, {"card": card.name, "checked": 0}
    import yaml
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cands = [c for c in (d.get("candidates") or []) if isinstance(c, dict)]
    valid = load_bottlenecks(surfaces)
    problems, checked = [], 0
    for c in cands:
        # 只查 pending：adopted/verified 已经跑完，补 means 不改变任何决定。
        if str(c.get("status") or "pending") != "pending":
            continue
        checked += 1
        cid = c.get("id")
        if c.get("_origin") == "advance-overlay":
            if not c.get("forge_evidence"):
                problems.append({
                    "id": cid, "field": "forge_evidence",
                    "why": "缺 forge_evidence —— 自动生成候选必须证明过体检，"
                           "否则无法区分「真生成」和「手填冒充分自动产物」",
                })
            continue
        for field, fn in (
                ("means", lambda v: _means_problem(v)),
                ("predicted_effect", lambda v: _effect_problem(v)),
                ("expected_bottleneck",
                 lambda v: _bottleneck_problem(v, valid))):
            why = fn(c.get(field))
            if why:
                problems.append({"id": cid, "field": field, "why": why})
    return problems, {"card": card.name, "checked": checked}


def selftest():
    """反向用例比正向重要：每条判据都要证明「该报错时真会报错」。"""
    import tempfile
    import yaml

    def mk(cands):
        d = Path(tempfile.mkdtemp())
        (d / "candidates.yaml").write_text(
            yaml.dump({"candidates": cands}, allow_unicode=True),
            encoding="utf-8")
        return d

    full = {"id": "a", "status": "pending", "means": "把残差写回融进同一遍 kernel",
            "predicted_effect": "-8%", "expected_bottleneck": "memory-bound"}

    assert not check_card(mk([full]))[0], "完整手工候选应通过"

    r = dict(full, predicted_effect="-6%~-10%")
    assert not check_card(mk([r]))[0]
    print("[ok] 区间 predicted_effect（-6%~-10%）通过")

    p = check_card(mk([{k: v for k, v in full.items() if k != "means"}]))[0]
    assert p and "缺 means" in p[0]["why"], p
    print("[ok] 缺 means → 报错")

    p = check_card(mk([dict(full, means="TODO")]))[0]
    assert p and "占位符" in p[0]["why"], p
    print("[ok] means=TODO 占位符 → 报错")

    p = check_card(mk([dict(full, means="调参")]))[0]
    assert p and "字符" in p[0]["why"], p
    print("[ok] means 过短（调参）→ 报错")

    p = check_card(mk([dict(full, predicted_effect="省一次写回")]))[0]
    assert p and "可机检" in p[0]["why"], p
    print("[ok] 散文 predicted_effect → 报错")

    p = check_card(mk([dict(full, expected_bottleneck="gpu-bound")]))[0]
    assert p and "不在 surfaces.yaml" in p[0]["why"], p
    print("[ok] 自造 bottleneck（gpu-bound）→ 报错")

    p = check_card(mk([{"id": "auto-1", "status": "pending",
                        "_origin": "advance-overlay"}]))[0]
    assert p and "forge_evidence" in p[0]["why"], p
    print("[ok] 自动生成候选缺 forge_evidence → 报错")

    auto = {"id": "auto-1", "status": "pending", "_origin": "advance-overlay",
            "forge_evidence": {"profile": {"threads": 256}}}
    assert not check_card(mk([auto]))[0]
    print("[ok] 自动生成候选带 evidence → 免检通过（无需 means）")

    assert not check_card(mk([dict(full, status="verified")]))[0]
    print("[ok] 非 pending 候选不检查（已经跑完的不回头补）")

    print("\nselftest PASS（10 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="手工候选必须先写清算理（means/predicted_effect/expected_bottleneck）")
    ap.add_argument("card", nargs="?", help="技能卡目录（含 candidates.yaml）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.card:
        ap.error("给一张卡目录，或 --selftest")
    if not Path(a.card).is_dir():
        print("找不到卡片目录：%s" % a.card, file=sys.stderr)
        return 2
    probs, st = check_card(a.card)
    if probs is None:
        print("找不到 %s/candidates.yaml" % a.card, file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps({"card": st["card"], "checked": st["checked"],
                          "ok": not probs, "problems": probs},
                         ensure_ascii=False, indent=2))
    elif probs:
        print("✗ %s" % st["card"])
        for pr in probs:
            print("    %-22s [%s] %s" % (pr["id"], pr["field"], pr["why"]))
    else:
        print("✓ %s %d 个 pending 候选，推理齐" % (st["card"], st["checked"]))
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
