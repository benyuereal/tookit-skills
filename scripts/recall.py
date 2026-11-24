#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recall —— 跨卡检索 memory/ledger.jsonl：按 (bound,signal)/card 过滤，ts 倒序输出。
不做数据库/索引/服务（超 5000 行且超 1 秒才考虑，docs/ENGINE.md §3.2）。硬纪律：
① 每条必须带 citation（card+ts+verdict），缺字段标 `incomplete_citation`；② falsified=true
是**阻止重试**不是灵感，必须显著标 `⚠ 已证伪`。空结果退出 0；坏 JSON 跳过并 warning。
"""
import argparse, json, sys, tempfile
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "memory/ledger.jsonl"
CITE = ("card", "ts", "verdict")
_tp = lambda r: r["trigger_pattern"] if isinstance(r.get("trigger_pattern"), dict) else {}
_v = lambda r: r.get("verdict") or r.get("decision")   # loop.py 写 decision，索引样例写 verdict

def load(path):
    """→ (records, warnings)。坏行/非对象行跳过并记 warning，不整体崩。"""
    p, recs, warns = Path(path), [], []
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

def citation_of(r):
    """→ (citation | None, 缺失字段)。缺任一字段不算有 citation。"""
    d = dict(card=r.get("card"), ts=r.get("ts"), verdict=_v(r))
    miss = [k for k in CITE if d[k] in (None, "")]
    return (None, miss) if miss else ("%s %s verdict=%s" % (d["card"], d["ts"], d["verdict"]), [])

def query(recs, bound=None, signal=None, card=None):
    """条件取 AND；bound/signal 先读 trigger_pattern 再回落顶层。ts 倒序，同 ts 已证伪优先。"""
    out = [r for r in recs
           if (not bound or (_tp(r).get("bound") or r.get("bound")) == bound)
           and (not signal or (_tp(r).get("signal") or r.get("signal")) == signal)
           and (not card or r.get("card") == card)]
    return sorted(out, key=lambda r: (str(r.get("ts") or ""), r.get("falsified") is True),
                  reverse=True)

def render(r, idx=1):
    tp, (cite, miss) = _tp(r), citation_of(r)
    dead, b, c = r.get("falsified") is True, r.get("base_ms"), r.get("cand_ms")
    spd = ("  %sms vs %sms（%s %.1f%%）" % (c, b, "快" if c < b else "慢", abs(1 - c / b) * 100)
           if isinstance(b, (int, float)) and isinstance(c, (int, float)) and b else "")
    rows = [("direction  ", r.get("direction") or tp.get("direction") or "-"),
            ("结果       ", (_v(r) or "-") + spd),
            ("预期       ", r.get("expected_effect") or "-"),
            ("证伪       ", "⚠ 已证伪（falsified=true）" if dead else
             "未知" if r.get("falsified") is None else "false"),
            ("citation   ", cite or "⚠ incomplete_citation（缺 %s）" % ",".join(miss))]
    if r.get("lesson"):
        rows.append(("教训       ", r["lesson"]))
    head = "%s%s%s" % (r.get("card") or "?", " / %s" % r["tag"] if r.get("tag") else "",
                       "  ⚠ 已证伪" if dead else "")
    return "\n".join(["[%d] %s" % (idx, head)] + ["    %s%s" % x for x in rows])

def emit(recs, warns, bound=None, signal=None, card=None, as_json=False):
    for w in warns:
        print("warning: %s" % w, file=sys.stderr)
    q = "、".join("%s=%s" % kv for kv in (("card", card), ("bound", bound),
                                         ("signal", signal)) if kv[1]) or "全库（无过滤）"
    if as_json:
        rows = []
        for r in recs:
            cite, miss = citation_of(r)
            rows.append(dict(r, citation=cite, citation_status="incomplete_citation" if miss else "ok",
                             **({"citation_missing": miss} if miss else {})))
        print(json.dumps({"query": dict(bound=bound, signal=signal, card=card),
                          "count": len(rows), "warnings": warns, "results": rows},
                         ensure_ascii=False, indent=2))
    elif not recs:
        print("无历史记录（按 %s 检索）" % q)
    else:
        print("命中 %d 条（按 %s 检索）" % (len(recs), q))
        for i, r in enumerate(recs, 1):
            print("\n" + render(r, i))
    return 0

def selftest():
    """8 个反例：过滤、排序、空结果、坏行、缺 citation、证伪标记。"""
    tp, hp = {"bound": "memory", "signal": "lo"}, {"bound": "compute", "signal": "lds"}
    R = [dict(card="a", ts="2026-11-24T10:00:00", verdict="significant", trigger_pattern=tp,
              base_ms=0.2, cand_ms=0.16, falsified=False),
         dict(card="b", ts="2026-11-24T11:00:00", verdict="sig_reg", trigger_pattern=tp,
              falsified=True),
         dict(card="c", ts="2026-11-24T11:00:00", verdict="noise", trigger_pattern=hp),
         dict(card="d", ts="2026-11-24T12:00:00", trigger_pattern=hp)]
    got, (cite, miss) = query(R, bound="memory", signal="lo"), citation_of(R[3])
    p = Path(tempfile.mkdtemp()) / "l.jsonl"
    p.write_text('{"card":"a"}\n{broken\n', encoding="utf-8")
    nrec, nwarn = len(load(p)[0]), len(load(p)[1])
    cases = [("过滤 bound+signal", [r["card"] for r in got] == ["b", "a"]),
             ("ts 倒序 + 同 ts 证伪优先", got[0]["falsified"] is True),
             ("过滤 card", [r["card"] for r in query(R, card="c")] == ["c"]),
             ("空结果不报错", query(R, bound="nope") == []),
             ("缺 verdict → incomplete_citation", cite is None and miss == ["verdict"]),
             ("证伪显著标记", "⚠ 已证伪" in render(R[1])),
             ("citation 三要素", citation_of(R[0])[0] == "a 2026-11-24T10:00:00 verdict=significant"),
             ("坏 JSON 跳过并 warning", (nrec, nwarn) == (1, 1))]
    for name, ok in cases:
        print("[%s] %s" % ("ok" if ok else "FAIL", name))
    n = sum(ok for _, ok in cases)
    print("\nselftest %s（%d/%d 反例）" % ("PASS" if n == len(cases) else "FAIL", n, len(cases)))
    return 0 if n == len(cases) else 1

def main(argv=None):
    ap = argparse.ArgumentParser(description="跨卡记忆检索（按 bound/signal/card）")
    for k in ("bound", "signal", "card"):
        ap.add_argument("--" + k)
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    recs, warns = load(a.ledger)
    return emit(query(recs, a.bound, a.signal, a.card), warns, a.bound, a.signal, a.card, a.json)

if __name__ == "__main__":
    sys.exit(main())
