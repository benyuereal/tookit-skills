#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recall —— 跨卡记忆检索：上次这个 (bound, signal) 改了什么、涨了多少、是不是死路。

读 loop.py 写的索引 memory/ledger.jsonl，按 (bound, signal) / card 过滤，ts 倒序输出。
**不做数据库、不做索引、不做服务**——ledger 超 5000 行且单次超 1 秒才考虑（ENGINE.md §3.2）。

两条硬纪律：
  ① 每条结果必须带 citation（card + ts + verdict）。缺字段标 `incomplete_citation`，
     不许静默通过——没有 citation 的结果等于没有。
  ② `falsified=true` 的作用是**阻止重试**，不是提供灵感，必须显著标 `⚠ 已证伪`。

空结果不算错：输出「无历史记录」退出 0。坏 JSON 行跳过并 warning，不整体崩。

  python3 -I scripts/recall.py --bound memory --signal low_occupancy
  python3 -I scripts/recall.py --card op-rmsnorm-fused --json
  python3 -I scripts/recall.py --selftest
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "memory/ledger.jsonl"
CITE_FIELDS = ("card", "ts", "verdict")


def load(path):
    """读 ledger → (records, warnings)。坏行/非对象行跳过并记 warning。"""
    recs, warns, p = [], [], Path(path)
    if not p.exists():
        return recs, ["ledger 不存在：%s" % p]
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as e:
            warns.append("第 %d 行坏 JSON，已跳过：%s" % (i, e))
            continue
        if isinstance(r, dict):
            recs.append(r)
        else:
            warns.append("第 %d 行不是 JSON 对象，已跳过" % i)
    return recs, warns


def _tp(r):
    return r.get("trigger_pattern") if isinstance(r.get("trigger_pattern"), dict) else {}


def verdict_of(r):
    """loop.py 写 decision，索引样例写 verdict：两者都认。"""
    return r.get("verdict") or r.get("decision")


def citation_of(r):
    """→ (citation 字符串 | None, 缺失字段列表)。"""
    val = {"card": r.get("card"), "ts": r.get("ts"), "verdict": verdict_of(r)}
    missing = [k for k in CITE_FIELDS if val[k] in (None, "")]
    if missing:
        return None, missing
    return "%s %s verdict=%s" % (val["card"], val["ts"], val["verdict"]), []


def query(recs, bound=None, signal=None, card=None):
    """所有给定条件取 AND；bound/signal 优先读 trigger_pattern，回落顶层。"""
    out = [r for r in recs
           if (not bound or (_tp(r).get("bound") or r.get("bound")) == bound)
           and (not signal or (_tp(r).get("signal") or r.get("signal")) == signal)
           and (not card or r.get("card") == card)]
    # ts 倒序；同 ts 已证伪优先（它更重要）。缺 ts 排最后。
    return sorted(out, key=lambda r: (str(r.get("ts") or ""), r.get("falsified") is True),
                  reverse=True)


def speed(r):
    b, c = r.get("base_ms"), r.get("cand_ms")
    if not all(isinstance(x, (int, float)) for x in (b, c)) or not b:
        return ""
    pct = (1 - c / b) * 100
    return "  %sms vs %sms（%s %.1f%%）" % (c, b, "快" if pct > 0 else "慢", abs(pct)) \
        if pct else "  %sms vs %sms（持平）" % (c, b)


def render(r, idx=1):
    tp, (cite, missing) = _tp(r), citation_of(r)
    dead = r.get("falsified") is True
    lines = ["[%d] %s%s%s" % (idx, r.get("card") or "?",
                              " / %s" % r["tag"] if r.get("tag") else "",
                              "  ⚠ 已证伪" if dead else ""),
             "    direction  %s" % (r.get("direction") or tp.get("direction") or "-"),
             "    结果       %s%s" % (verdict_of(r) or "-", speed(r)),
             "    预期       %s" % (r.get("expected_effect") or "-"),
             "    证伪       %s" % ("⚠ 已证伪（falsified=true）" if dead else
                                    "未知" if r.get("falsified") is None else "false"),
             "    citation   %s" % (cite or "⚠ incomplete_citation（缺 %s）" % ",".join(missing))]
    if r.get("lesson"):
        lines.append("    教训       %s" % r["lesson"])
    return "\n".join(lines)


def emit(recs, warns, bound=None, signal=None, card=None, as_json=False):
    for w in warns:
        print("warning: %s" % w, file=sys.stderr)
    if as_json:
        rows = []
        for r in recs:
            cite, missing = citation_of(r)
            rows.append(dict(r, citation=cite, citation_status=
                             "incomplete_citation" if missing else "ok",
                             **({"citation_missing": missing} if missing else {})))
        print(json.dumps({"query": {"bound": bound, "signal": signal, "card": card},
                          "count": len(rows), "warnings": warns, "results": rows},
                         ensure_ascii=False, indent=2))
        return 0
    q = "、".join("%s=%s" % kv for kv in
                  (("card", card), ("bound", bound), ("signal", signal)) if kv[1])
    if not recs:
        print("无历史记录（按 %s 检索）" % (q or "全库（无过滤）"))
        return 0
    print("命中 %d 条（按 %s 检索）" % (len(recs), q or "全库（无过滤）"))
    for i, r in enumerate(recs, 1):
        print("\n" + render(r, i))
    return 0


def _chk(name, cond):
    print("[%s] %s" % ("ok" if cond else "FAIL", name))
    return bool(cond)


def selftest():
    tp = {"bound": "memory", "signal": "low_occupancy"}
    recs = [
        {"card": "a", "ts": "2026-11-24T10:00:00", "verdict": "significant",
         "trigger_pattern": tp, "expected_effect": "-11%", "base_ms": 0.195,
         "cand_ms": 0.161, "falsified": False},
        {"card": "b", "ts": "2026-11-24T11:00:00", "verdict": "significant_regression",
         "trigger_pattern": tp, "falsified": True},
        {"card": "c", "ts": "2026-11-24T11:00:00", "verdict": "noise",
         "trigger_pattern": {"bound": "compute", "signal": "wait_lds_high"},
         "falsified": None},
        {"card": "d", "ts": "2026-11-24T12:00:00",
         "trigger_pattern": {"bound": "compute", "signal": "wait_lds_high"}},
    ]
    got, (cite, missing) = query(recs, bound="memory", signal="low_occupancy"), citation_of(recs[3])
    checks = [
        ("反例1 bound+signal 过滤", [r["card"] for r in got] == ["b", "a"]),
        ("反例2 ts 倒序 + 同 ts 证伪优先", got[0]["falsified"] is True),
        ("反例3 card 过滤", [r["card"] for r in query(recs, card="c")] == ["c"]),
        ("反例4 空结果不报错", query(recs, bound="nope") == []),
        ("反例5 缺 verdict → incomplete_citation", cite is None and missing == ["verdict"]),
        ("反例6 证伪显著标记", "⚠ 已证伪" in render(recs[1], 1)),
        ("反例7 citation 三要素", citation_of(recs[0])[0] ==
         "a 2026-11-24T10:00:00 verdict=significant"),
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger.jsonl"
        p.write_text('{"card":"a"}\n{broken\n', encoding="utf-8")
        r2, w = load(p)
        checks.append(("反例8 坏 JSON 跳过并 warning", len(r2) == 1 and len(w) == 1))
    passed = sum([_chk(n, c) for n, c in checks])
    print("\nselftest %s（%d/%d 反例）" % (
        "PASS" if passed == len(checks) else "FAIL", passed, len(checks)))
    return 0 if passed == len(checks) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="跨卡记忆检索（按 bound/signal/card）")
    ap.add_argument("--bound")
    ap.add_argument("--signal")
    ap.add_argument("--card")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    recs, warns = load(a.ledger)
    return emit(query(recs, a.bound, a.signal, a.card), warns,
                a.bound, a.signal, a.card, a.json)


if __name__ == "__main__":
    sys.exit(main())
