#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate.py —— EV 事前校准：expected_effect 到底兑不兑现（v5.1 §6.4）。

为什么需要它：plan.py 的 EV 用 expected_effect 排序，但**没人验过它准不准**。
若某 surface 永远预测 +15%、实际只兑现 3%，EV 排序整体失真，agent 会一直在
高估的方向上打转。本脚本按 surface 汇总「预测中位数 / 实际中位数 / 比值」：
ratio < 0.5 = 系统性高估（EV 该打折），ratio > 1.5 = 低估。

口径（两个中位数必须同口径，否则比值没有意义）：
  predicted：expected_effect 去符号化。百分数是「耗时 delta」（"-11%" = 快 11%），
             倍数是「速度比」（"1.15x" = 快 15%）；两者都归一到「改善比例」：
                 pct → -value        x → value - 1        正 = 更快
  realized ：1 - cand_ms/base_ms                         正 = 更快（与上面同向）

数据：<skill>/runs/loop/*.jsonl（**只读**）。
不计入的记录：verdict/decision == "noise"，或 falsified 为 null/缺失（本轮不可信）。
expected_effect 解析不出**整串数值**的，计入 unparsable 单独报——不从散文里抠
数字、不猜范围（如 "−11.7~−11.9%" 一律 unparsable）。
缺 cand_ms/base_ms 的记录跳过并 warning。

用法：
  python3 -I scripts/calibrate.py --skill Plugins/ops/skills/op-rmsnorm-fused
  python3 -I scripts/calibrate.py --all
  python3 -I scripts/calibrate.py --json
  python3 -I scripts/calibrate.py --selftest
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
MIN_SAMPLES = 2

# 严格整串匹配：number 后必须紧跟 % 或 x/×，前后只允许空白。
# 不匹配 "10~15%"、"约 10%"、"1.12×/CTA" 这类描述——猜一个数比不报更糟。
_PCT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
_X_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*[x×]")

_REASON_LABEL = {
    "verdict_noise": "verdict=noise",
    "falsified_null": "falsified=null/缺失",
    "no_surface": "缺 surface id",
    "expected_missing": "expected_effect 缺失",
    "metrics_missing": "缺 cand_ms/base_ms",
}


def _norm(raw):
    """统一符号：U+2212 减号、全角＋％ｘ → ASCII，再去首尾空白。"""
    s = str(raw).strip()
    for old, new in (("−", "-"), ("＋", "+"), ("％", "%"),
                     ("ｘ", "x"), ("Ｘ", "X")):
        s = s.replace(old, new)
    return s


def parse_expected(raw):
    """expected_effect → (kind, value)；解析不了 → (None, None)。

    value 保留原始口径（不在这里做符号翻转）：
      ("pct", -0.11)  ← "-11%"
      ("x",   1.15)   ← "1.15x"
    需要统一成「改善比例」时调用 expected_improvement()。
    """
    if raw is None:
        return (None, None)
    s = _norm(raw)
    if not s:
        return (None, None)
    m = _PCT_RE.fullmatch(s)
    if m:
        return ("pct", float(m.group(1)) / 100.0)
    m = _X_RE.fullmatch(s)
    if m:
        v = float(m.group(1))
        # 倍数 <= 0 在物理上没有意义（速度不可能为 0 或负），按不可解析处理。
        return ("x", v) if v > 0 else (None, None)
    return (None, None)


def expected_improvement(kind, value):
    """原生口径 → 改善比例（正 = 更快，负 = 更慢）。"""
    if kind == "pct":
        return -value
    if kind == "x":
        return value - 1.0
    return None


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _ex_count(r):
    """一次校准结果里被排除的记录数（兼容列表 / 已聚合的整数两种形态）。"""
    v = r.get("excluded", 0)
    return len(v) if isinstance(v, list) else int(v or 0)


def _verdict_of(r):
    # loop.py 写 decision；早期/索取样例写 verdict。两者等权，verdict 优先。
    return r.get("verdict") or r.get("decision")


def _surface_of(r):
    tp = r.get("trigger_pattern")
    if not isinstance(tp, dict):
        tp = {}
    return r.get("direction") or tp.get("direction")


def calibrate_records(records, skill="", min_samples=MIN_SAMPLES):
    """纯函数：一组 loop 记录 → 按 surface 的校准结果。便于 selftest/测试直接调用。"""
    out = {"skill": skill, "records": len(records), "counted": 0,
           "rows": [], "unparsable": [], "skipped": [], "warnings": []}
    buckets = {}

    for r in records:
        if not isinstance(r, dict):
            out["skipped"].append({"skill": skill, "tag": None, "ts": None,
                                   "reason": "not_object", "detail": repr(r)})
            continue
        tag, ts = r.get("tag"), r.get("ts")

        # ---- 不可信记录：本轮不参与校准，也不进 unparsable。
        if _verdict_of(r) == "noise":
            out["skipped"].append({"skill": skill, "tag": tag, "ts": ts,
                                   "reason": "verdict_noise",
                                   "detail": "verdict/decision=noise"})
            continue
        if r.get("falsified") is None:
            out["skipped"].append({"skill": skill, "tag": tag, "ts": ts,
                                   "reason": "falsified_null",
                                   "detail": "falsified=%r" % (r.get("falsified"),)})
            continue

        surface = _surface_of(r)
        if not surface:
            msg = "[%s] tag=%s ts=%s 缺 surface id（direction 与 trigger_pattern.direction 均空），跳过" \
                  % (skill or "?", tag, ts)
            out["warnings"].append(msg)
            out["skipped"].append({"skill": skill, "tag": tag, "ts": ts,
                                   "reason": "no_surface", "detail": ""})
            continue

        raw = r.get("expected_effect")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            out["skipped"].append({"skill": skill, "tag": tag, "ts": ts,
                                   "reason": "expected_missing",
                                   "detail": "expected_effect=%r" % (raw,)})
            continue

        kind, value = parse_expected(raw)
        if kind is None:
            # 明确单独报：既不猜数，也不静默丢。
            out["unparsable"].append({
                "skill": skill, "surface": surface, "tag": tag, "ts": ts,
                "expected_effect": raw,
                "reason": "expected_effect 不是整串的百分数/倍数（散文或范围不猜）",
            })
            continue

        base, cand = r.get("base_ms"), r.get("cand_ms")
        if not (_is_num(base) and _is_num(cand) and base > 0 and cand > 0):
            msg = "[%s] tag=%s ts=%s 缺合法 cand_ms/base_ms（cand=%r base=%r），跳过" \
                  % (skill or "?", tag, ts, cand, base)
            out["warnings"].append(msg)
            out["skipped"].append({"skill": skill, "tag": tag, "ts": ts,
                                   "reason": "metrics_missing",
                                   "detail": "cand_ms=%r base_ms=%r" % (cand, base)})
            continue

        predicted = expected_improvement(kind, value)
        realized = 1.0 - cand / base
        buckets.setdefault(str(surface), []).append({
            "tag": tag, "ts": ts, "expected_effect": raw,
            "kind": kind, "value": value,
            "predicted": predicted, "realized": realized,
        })
        out["counted"] += 1

    out["excluded"] = len(out["skipped"])

    for surface in sorted(buckets):
        items = buckets[surface]
        n = len(items)
        predicted_med = statistics.median(i["predicted"] for i in items)
        realized_raw = statistics.median(i["realized"] for i in items)
        insufficient = n < max(1, int(min_samples))
        realized_med = None
        ratio = None
        note = ""

        if insufficient:
            note = "样本不足"
        else:
            realized_med = realized_raw
            if predicted_med == 0:
                note = "预测为 0，比率无定义"
            else:
                ratio = realized_med / predicted_med
                if ratio < 0:
                    note = "符号相反"
                elif ratio < 0.5:
                    note = "高估（EV 该打折）"
                elif ratio > 1.5:
                    note = "低估"

        out["rows"].append({
            "skill": skill, "surface": surface, "n": n,
            "predicted_med": predicted_med,
            "realized_med": realized_med,
            "ratio": ratio,
            "insufficient": insufficient,
            "note": note,
            "items": items,
        })
    return out


def load_jsonl(path):
    """→ (records, warnings)。坏行/非对象行跳过并 warning，空行不算坏行。"""
    p = Path(path)
    recs, warns = [], []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return recs, ["读不了 %s：%s" % (p, e)]
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as e:
            warns.append("%s:%d 坏 JSON，已跳过：%s" % (p, i, e))
            continue
        if isinstance(r, dict):
            recs.append(r)
        else:
            warns.append("%s:%d 不是 JSON 对象，已跳过" % (p, i))
    return recs, warns


def loop_files(skill_path):
    p = Path(skill_path)
    if p.is_file():
        return [p]
    loop = p / "runs/loop"
    return sorted(loop.glob("*.jsonl")) if loop.is_dir() else []


def skill_name_of(p):
    p = Path(p)
    if p.is_file():
        parts = p.parts
        if "runs" in parts:
            i = parts.index("runs")
            if i >= 1:
                return parts[i - 1]
        return p.stem
    if p.name == "loop" and p.parent.name == "runs":
        return p.parent.parent.name
    return p.name


def resolve_skill(arg, root):
    p = Path(arg).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    cand = root / p
    return cand.resolve() if cand.exists() else p.resolve()


def discover_skills(root):
    """--all：扫 Plugins/*/skills/*/runs/loop/*.jsonl 所在的 skill 目录。"""
    found = set()
    base = Path(root) / "Plugins"
    if not base.is_dir():
        return []
    for f in sorted(base.glob("*/skills/*/runs/loop/*.jsonl")):
        found.add(f.parents[2])
    return sorted(found)


def _fmt_pct(x):
    return "%.1f%%" % (x * 100.0)


def render(results, min_samples):
    """人读表格。返回 (是否无 rows, totals)。"""
    all_rows = [(r["skill"], row) for r in results for row in r["rows"]]
    all_skipped = [s for r in results for s in r["skipped"]]
    all_unparsable = [u for r in results for u in r["unparsable"]]
    totals = {
        "records": sum(r["records"] for r in results),
        "counted": sum(r["counted"] for r in results),
        "excluded": sum(_ex_count(r) for r in results),
        "unparsable": len(all_unparsable),
    }

    if all_rows:
        multi = len({r["skill"] for r in results}) > 1
        if multi:
            print("%-20s %-22s %3s %14s %13s %7s" %
                  ("skill", "surface", "n", "predicted_med", "realized_med", "ratio"))
        else:
            print("%-22s %3s %14s %13s %7s" %
                  ("surface", "n", "predicted_med", "realized_med", "ratio"))
        for skill, row in sorted(all_rows, key=lambda x: (x[0], x[1]["surface"])):
            real = "未兑现" if row["realized_med"] is None else _fmt_pct(row["realized_med"])
            ratio = "—" if row["ratio"] is None else "%.2f" % row["ratio"]
            suffix = ("   ← %s" % row["note"]) if row.get("note") else ""
            if multi:
                print("%-20s %-22s %3d %14s %13s %7s%s" %
                      (skill or "?", row["surface"], row["n"],
                       _fmt_pct(row["predicted_med"]), real, ratio, suffix))
            else:
                print("%-22s %3d %14s %13s %7s%s" %
                      (row["surface"], row["n"], _fmt_pct(row["predicted_med"]),
                       real, ratio, suffix))
    else:
        print("无记录：0 条可校准记录（读取 %d 条；排除 %d 条；unparsable %d 条）"
              % (totals["records"], totals["excluded"], totals["unparsable"]))

    if all_skipped:
        print("\n排除原因：")
        by_reason = {}
        for s in all_skipped:
            by_reason[s["reason"]] = by_reason.get(s["reason"], 0) + 1
        for reason, cnt in sorted(by_reason.items()):
            print("  %-24s %d" % (_REASON_LABEL.get(reason, reason), cnt))

    if all_unparsable:
        print("\nunparsable（%d 条，不计入中位数）：" % len(all_unparsable))
        for u in all_unparsable:
            print("  [%s] %-22s tag=%-20s expected_effect=%r"
                  % (u["skill"] or "?", u["surface"], u["tag"], u["expected_effect"]))

    if all_rows:
        print("\n合计：读取 %d 条，可校准 %d 条，排除 %d 条，unparsable %d 条"
              % (totals["records"], totals["counted"], totals["excluded"],
                 totals["unparsable"]))
    return (not all_rows), totals


def _payload(root, results, min_samples):
    return {
        "root": str(root),
        "min_samples": min_samples,
        "skills": [{"skill": r["skill"], "path": r.get("path"),
                    "records": r["records"], "counted": r["counted"],
                    "excluded": _ex_count(r), "unparsable": len(r["unparsable"])}
                   for r in results],
        "surfaces": [dict(row, skill=r["skill"])
                     for r in results for row in r["rows"]],
        "unparsable": [u for r in results for u in r["unparsable"]],
        "skipped": [s for r in results for s in r["skipped"]],
        "warnings": [w for r in results for w in r["warnings"]],
        "totals": {
            "records": sum(r["records"] for r in results),
            "counted": sum(r["counted"] for r in results),
            "excluded": sum(_ex_count(r) for r in results),
            "unparsable": sum(len(r["unparsable"]) for r in results),
        },
    }


def _selftest():
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    def rec(**kw):
        r = {"ts": "2026-01-01T00:00:00", "tag": "v1",
             "direction": "fusion-boundary", "expected_effect": "-10%",
             "base_ms": 1.0, "cand_ms": 0.95,
             "verdict": "significant", "falsified": False}
        r.update(kw)
        return r

    kind, val = parse_expected("-11%")
    check("parse '-11%' → pct -0.11", kind == "pct" and abs(val + 0.11) < 1e-12,
          "got %r %r" % (kind, val))
    kind, val = parse_expected("1.15x")
    check("parse '1.15x' → x 1.15 / +15%",
          kind == "x" and abs(val - 1.15) < 1e-12
          and abs(expected_improvement(kind, val) - 0.15) < 1e-12,
          "got %r %r" % (kind, val))
    check("parse prose → None",
          parse_expected("triton 融合 vs lightop 融合") == (None, None))
    check("parse range prose → None（不猜）",
          parse_expected("−11.7~−11.9%") == (None, None))

    empty = calibrate_records([])
    check("empty → no rows", empty["rows"] == [] and empty["records"] == 0)

    noise = calibrate_records([rec(), rec(tag="n", verdict="noise")])
    check("verdict=noise excluded",
          noise["counted"] == 1
          and any(s["reason"] == "verdict_noise" for s in noise["skipped"]))

    nullf = calibrate_records([rec(), rec(tag="ok"), rec(tag="f", falsified=None)])
    check("falsified=null excluded",
          nullf["counted"] == 2
          and any(s["reason"] == "falsified_null" for s in nullf["skipped"]))

    prose = calibrate_records([rec(expected_effect="纯文字描述")])
    check("prose → unparsable, not counted",
          prose["counted"] == 0 and len(prose["unparsable"]) == 1
          and prose["rows"] == [])

    ratio_res = calibrate_records([rec(), rec(tag="v2"), rec(tag="v3")])
    row = ratio_res["rows"][0]
    check("ratio: predicted 10%, realized 5% → 0.5",
          row["n"] == 3 and abs(row["predicted_med"] - 0.10) < 1e-12
          and abs(row["realized_med"] - 0.05) < 1e-12
          and abs(row["ratio"] - 0.5) < 1e-12, repr(row))

    missing = calibrate_records([rec(cand_ms=None)])
    check("missing cand/base → skipped + warning",
          missing["counted"] == 0 and missing["warnings"]
          and any(s["reason"] == "metrics_missing" for s in missing["skipped"]))

    mult = calibrate_records([rec(expected_effect="1.15x", cand_ms=0.85),
                              rec(tag="v2", expected_effect="1.15x", cand_ms=0.85)])
    mrow = mult["rows"][0]
    check("1.15x aggregated as +15%",
          abs(mrow["predicted_med"] - 0.15) < 1e-12
          and abs(mrow["realized_med"] - 0.15) < 1e-12
          and abs(mrow["ratio"] - 1.0) < 1e-12, repr(mrow))

    one = calibrate_records([rec()])
    check("n=1 → insufficient, no ratio",
          one["rows"][0]["insufficient"] and one["rows"][0]["ratio"] is None)

    multi = calibrate_records([rec(), rec(tag="v2"),
                               rec(direction="occupancy"), rec(tag="o2", direction="occupancy")])
    check("n per surface correct",
          {r["surface"]: r["n"] for r in multi["rows"]}
          == {"fusion-boundary": 2, "occupancy": 2})

    failed = [c for c in checks if not c[1]]
    for name, _ok, detail in failed:
        print("FAIL: %s%s" % (name, ("  (%s)" % detail) if detail else ""))
    print("selftest: %d/%d passed" % (len(checks) - len(failed), len(checks)))
    return 0 if not failed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="calibrate.py",
        description="EV 事前校准：按 surface 对比 expected_effect 与实际 delta（v5.1 §6.4）。")
    ap.add_argument("--skill", action="append", metavar="PATH",
                    help="skill 目录（或直接给 runs/loop/*.jsonl）；可重复")
    ap.add_argument("--all", action="store_true",
                    help="扫 <root>/Plugins/*/skills/*/runs/loop/*.jsonl")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--selftest", action="store_true", help="内置自检")
    ap.add_argument("--root", default=None,
                    help="仓库根（默认脚本上一级）；--all 用它定位 Plugins/")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                    help="少于该样本数只报预测、不报 ratio（默认 %d）" % MIN_SAMPLES)
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()
    if not a.skill and not a.all:
        ap.error("需要 --skill PATH 或 --all（或 --selftest）")

    root = Path(a.root).expanduser().resolve() if a.root else DEFAULT_ROOT
    targets = []
    if a.all:
        targets.extend(discover_skills(root))
    for s in a.skill or []:
        targets.append(resolve_skill(s, root))

    results, seen = [], set()
    for t in targets:
        key = str(Path(t).resolve()) + ("#file" if Path(t).is_file() else "")
        if key in seen:
            continue
        seen.add(key)
        files = loop_files(t)
        recs, warns = [], []
        if not files:
            warns.append("没有 runs/loop/*.jsonl：%s" % t)
        for f in files:
            rr, ww = load_jsonl(f)
            recs.extend(rr)
            warns.extend(ww)
        res = calibrate_records(recs, skill_name_of(t), min_samples=a.min_samples)
        res["warnings"] = warns + res["warnings"]
        res["path"] = str(Path(t).resolve())
        results.append(res)

    all_warnings = [w for r in results for w in r["warnings"]]
    for w in all_warnings:
        print("warning: %s" % w, file=sys.stderr)

    if a.json:
        print(json.dumps(_payload(root, results, a.min_samples),
                         ensure_ascii=False, indent=2))
    else:
        render(results, a.min_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
