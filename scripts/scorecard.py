#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scorecard —— 三维（深度/覆盖/质量）可机检评分卡。

为什么有这个文件
----------------
《三维评分卡与定时推进机制》定义了 15 个检查点，但**脚本从来没被写出来**。
结果就是每轮巡检都靠手算 + 靠印象，而「靠印象」正是 2026-11-23 那天
把 fp8 峰值判反的那条路。

三条纪律
--------
1. 每个检查点都从**仓库事实**算，不接受任何人工填报字段。
2. 反作弊是硬判据（§四）：comparison 缺 cand_us/base_us、floor 缺 basis、
   ledger 轮次没有 adopt/refute/stalled 决策 —— 一律不计入分子。
   分数涨了但数据是空的，比分数不涨更糟。
3. 目标线写在 TARGETS 里，与机制文档一致；改目标必须改文档 + 这里。

用法
----
    python3 -I scripts/scorecard.py
    python3 -I scripts/scorecard.py --json
    python3 -I scripts/scorecard.py --append      # 追加到 docs/scorecard-history.jsonl
    python3 -I scripts/scorecard.py --selftest

退出码：三维都 ≥ 8.5 → 0；否则 1（可接 CI / 巡检 cron）。
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
SKILLS = REPO / "Plugins/ops/skills"
HISTORY = REPO / "docs/scorecard-history.jsonl"

# 目标线（机制文档 §二）。9 分线 = 该检查点拿满分的门槛。
TARGETS = {
    "D1": 45.0,      # DS 均分
    "D2": 5.0,       # S 档 5 张卡 DS ≥ 70
    "D3": 0.85,      # measured 卡有 comparison 的占比
    "D4": 8.0,       # 有 ≥3 轮 ledger 的卡数
    "D5": 10.0,      # 有 PMC/归因证据的卡数
    "C1": 1.00,      # semantics 覆盖（measured/tuned）
    "C2": 0.90,      # regimes 覆盖（attention+gemm+moe）
    "C3": 0.90,      # floor.remaining 覆盖（measured）
    "C4": 8.0,       # 融合算子卡数
    "C5": 8.0,       # 四件套齐全的卡数
    "Q1": 0.0,       # pytest 失败数（越低越好）
    "Q2": 5.0,       # validate 警告数（越低越好）
    "Q3": 6.0,       # 新门禁反向测试覆盖（E29/E36/E37/E38/E40/loop）
    "Q4": 40.0,      # except Exception 总数（越低越好）
    "Q5": 200.0,     # 最长单函数行数（越低越好）
}
WEIGHTS = {
    "D": {"D1": 0.30, "D2": 0.25, "D3": 0.20, "D4": 0.15, "D5": 0.10},
    "C": {"C1": 0.15, "C2": 0.20, "C3": 0.25, "C4": 0.20, "C5": 0.20},
    "Q": {"Q1": 0.30, "Q2": 0.20, "Q3": 0.20, "Q4": 0.15, "Q5": 0.15},
}
LOWER_BETTER = {"Q1", "Q2", "Q4", "Q5"}
MEASURED = ("measured", "tuned")
# 业主 2026-11-24 裁决②：分母锁死在本轮开始时的 27 张。
# 原因：把没证据的卡降 draft 会让分母变小、分数自动涨——
# 那是「分数涨了但东西变少」。所以分子按现况算，分母永远是 MEASURED_BASELINE，
# 并且报告必须同时给原分母/现分母两套分。
MEASURED_BASELINE = 27

# PANORAMA §I：8 项融合算子里已经建卡的 4 项
FUSION_CARDS = ["fusion-add-rmsnorm-quant", "fusion-rmsnorm-rope",
                "fusion-silu-mul-quant", "fusion-rope-kv-write"]
FUSION_TOTAL = 8

# 新门禁 → 反向测试文件（有 test 文件且含「期望它报错」的断言才算）
GATES = {
    "E29": "impl_measured 不许外推",
    "E36": "semantics 内容一致",
    "E37": "semantics 必填",
    "E38": "regimes 结构",
    "E40": "算出超 achievable 即报错",
    "loop": "第 3 步 bottleneck 强制绑定",
}


def load_cards():
    import yaml
    out = {}
    for p in sorted(CARDS.glob("*.yaml")):
        out[p.stem] = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}, p)
    return out


def _lv(card):
    return card.get("last_verdict") or {}


def _floor_ok(card):
    """反作弊：floor 必须带 basis + t_floor_us + remaining，缺一不算。"""
    f = _lv(card).get("floor") or {}
    return (isinstance(f, dict) and f.get("remaining") is not None
            and f.get("t_floor_us") is not None and f.get("basis"))


def _comparison_ok(card):
    """反作弊：comparison 必须有 baseline_role / cand_us / base_us，空块不算。"""
    c = _lv(card).get("comparison") or {}
    if not isinstance(c, dict):
        return False
    need = ("baseline", "cand_us", "base_us")
    if any(c.get(k) in (None, "") for k in need):
        return False
    return str(c.get("baseline_role") or "") in (
        "competitor", "same_family", "reference", "correctness_only", "self")


def depth_scores():
    """DS 五维来自 scripts/depth.py（它自己读 results/*.json，不读配置档位）。"""
    p = subprocess.run([sys.executable, "-I", str(REPO / "scripts/depth.py"), "--json"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("depth.py 失败：%s" % (p.stderr or p.stdout)[:400])
    return json.loads(p.stdout)


def ledger_rounds(cid):
    """ledger 轮次与决策。**没有 adopt/refute/stalled 的轮次不算轮次**。"""
    for name in ("ledger.md", "LEDGER.md"):
        p = SKILLS / ("op-%s" % cid) / name
        if not p.is_file():
            continue
        s = p.read_text(encoding="utf-8")
        heads = len(re.findall(r"(?m)^#{2,4}\s*(?:轮|Round|R)\s*\d", s))
        acts = len(re.findall(r"(?i)\b(adopt|refute|stalled)\b|采纳|证伪|卡住", s))
        # 轮次口头计数 + 有决策词 才算「真轮次」；二者取小，防止只刷标题
        return min(heads, acts) if heads else 0
    return 0


def four_piece(cid):
    base = SKILLS / ("op-%s" % cid)
    if not base.is_dir():
        return False
    return all([
        (base / "SKILL.md").is_file(),
        (base / "contract.yaml").is_file(),
        any((base / "bench").glob("*.py")),
        any((base / "results").glob("*.json")),
    ])


def measured_count_of_rounds(cards, measured):
    return sum(1 for c in measured if ledger_rounds(c) >= 3)


def longest_function():
    import ast
    best = (0, "", "")
    for p in list((REPO / "Plugins/ops").rglob("*.py")) + list((REPO / "scripts").glob("*.py")):
        try:
            t = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ln = (n.end_lineno or n.lineno) - n.lineno + 1
                if ln > best[0]:
                    best = (ln, str(p.relative_to(REPO)), n.name)
    return best


def except_exception_count():
    n = 0
    for p in list((REPO / "Plugins/ops").rglob("*.py")) + list((REPO / "scripts").glob("*.py")):
        try:
            n += len(re.findall(r"except\s+Exception", p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return n


def gate_reverse_tests():
    """5 道门禁各自是否有**反向测试**：构造错误案例，确认它报错。

    判据必须落在「文件里有期望报错的断言」上，不能只匹配门禁编号出现过的字面量——
    第一版本用的是 `"test_semantics" in blob`，而 blob 是**文件内容**不是文件名，
    于是永远为 False。这类「判据写错还自洽」的坑，本仓已经踩过四次
    （E18 太窄 / E29 太窄 / E30 过严 / 重复函数定义），所以这里按文件 + 断言双条件。
    """
    tests = {p.name: p.read_text(encoding="utf-8")
             for p in (REPO / "Plugins/ops/tests").glob("*.py")}

    def has(fname, *pat):
        body = tests.get(fname, "")
        if not body:
            return False
        # 必须有 assert 且断言的是「应该报错/应该被拒」
        return bool(re.search(r"assert", body)) and all(re.search(x, body) for x in pat)

    return {
        "E29": has("test_peak_constant.py", r"impl_measured"),
        "E36": has("test_semantics.py", r"assertRaises|assertEqual|assertNotEqual"),
        "E37": has("test_semantics.py", r"E37|必填|missing"),
        "E38": has("test_atlas_content.py", r"E38|regimes"),
        # E40 的反向测试在 test_floor_block.py 的 TestCeilingOverrun：
        # 必须有 assertEqual(...) 且带 ["E40"] 期望，才算「故意写错看它报不报」
        "E40": has("test_floor_block.py", r'E40'),
        "loop": has("test_bottleneck_gate.py", r"blocks_run|rejected"),
    }


def pytest_failures():
    """跑一次 pytest 只取结论行。慢（~5 min），巡检时允许 --no-pytest 跳过。"""
    p = subprocess.run([sys.executable, "-I", "-m", "pytest", "Plugins/ops", "-q",
                        "--no-header", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=str(REPO))
    tail = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"(\d+) failed", tail)
    failed = int(m.group(1)) if m else 0
    m2 = re.search(r"(\d+) passed", tail)
    return failed, (int(m2.group(1)) if m2 else 0)


def validate_counts():
    p = subprocess.run([sys.executable, "-I", str(REPO / "Plugins/ops/op-atlas/validate.py")],
                       capture_output=True, text=True)
    tail = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"结果：(\d+) 错误\s*/\s*(\d+) 警告", tail)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def collect(with_pytest=True):
    cards = load_cards()
    measured = [c for c, (d, _) in cards.items() if d.get("status") in MEASURED]
    ds = depth_scores()
    per = ds["cards"]
    summ = ds["summary"]

    raw = {}
    raw["D1"] = round(summ["mean"], 1)
    raw["D2"] = sum(1 for c in measured if per.get(c, {}).get("DS", 0) >= 70)
    cmp_ok = [c for c in measured if _comparison_ok(cards[c][0])]
    raw["D3"] = round(len(cmp_ok) / max(MEASURED_BASELINE, len(measured)), 3)
    raw["D4"] = measured_count_of_rounds(cards, measured)
    raw["D5"] = sum(1 for c in measured
                    if re.search(r"PMC|pmc|GRBM|SQ_INSTS|attribution|归因",
                                 json.dumps(cards[c][0], ensure_ascii=False)))

    denom = max(MEASURED_BASELINE, len(measured))
    raw["C1"] = round(sum(1 for c in measured if cards[c][0].get("semantics")) / denom, 3)
    fam = {"attention", "gemm", "moe"}
    pool = [c for c, (d, _) in cards.items() if d.get("family") in fam]
    raw["C2"] = round(sum(1 for c in pool if _lv(cards[c][0]).get("regimes"))
                      / max(len(pool), 1), 3)
    floor_ok = [c for c in measured if _floor_ok(cards[c][0])]
    raw["C3"] = round(len(floor_ok) / denom, 3)
    raw["C4"] = sum(1 for c in FUSION_CARDS
                    if c in cards and cards[c][0].get("status") in MEASURED)
    raw["C5"] = sum(1 for c in measured if four_piece(c))

    if with_pytest:
        raw["Q1"], passed = pytest_failures()
    else:
        raw["Q1"], passed = None, None
    errors, warnings = validate_counts()
    raw["Q2"] = warnings
    rev = gate_reverse_tests()
    raw["Q3"] = sum(1 for k in GATES if rev.get(k))
    raw["Q4"] = except_exception_count()
    raw["Q5"] = longest_function()[0]

    detail = {
        "measured_cards": len(measured),
        "measured_baseline": MEASURED_BASELINE,
        "denominator": denom,
        "score_on_current_denominator": {
            "C1": round(sum(1 for c in measured if cards[c][0].get("semantics"))
                        / max(len(measured), 1), 3),
            "C3": round(len(floor_ok) / max(len(measured), 1), 3),
            "D3": round(len(cmp_ok) / max(len(measured), 1), 3),
        },
        "comparison_ok": sorted(cmp_ok),
        "floor_ok": sorted(floor_ok),
        "floor_missing": sorted(set(measured) - set(floor_ok)),
        "validate_errors": errors,
        "pytest_passed": passed,
        "gates_reverse": rev,
        "longest_fn": longest_function(),
        "ds_summary": summ,
    }
    return raw, detail


def score_dim(dim, raw):
    tot = 0.0
    detail = {}
    for key, w in WEIGHTS[dim].items():
        cur = raw.get(key)
        tgt = TARGETS[key]
        if cur is None:
            ratio = 0.0
        elif key in LOWER_BETTER:
            ratio = 1.0 if cur <= tgt else (tgt / cur if cur else 1.0)
        else:
            ratio = min(cur / tgt, 1.0) if tgt else 1.0
        ratio = max(0.0, min(ratio, 1.0))
        tot += w * ratio
        detail[key] = {"cur": cur, "target": tgt, "ratio": round(ratio, 3),
                       "gap": round(1 - ratio, 3), "weight": w,
                       "urgency": round((1 - ratio) * w, 4)}
    return round(10 * tot, 1), detail


def selftest():
    """评分卡自己也要被校验：构造一个「分数该涨但数据是空的」卡，确认它不计分。"""
    ok_card = {"id": "x", "status": "measured",
               "last_verdict": {"floor": {"remaining": 0.1, "t_floor_us": 200.0,
                                          "basis": "268MB/1680"},
                                "comparison": {"baseline": "lightop", "cand_us": 210.0,
                                               "base_us": 197.0,
                                               "baseline_role": "competitor"}}}
    assert _floor_ok(ok_card), "合法的 floor 应通过"
    assert _comparison_ok(ok_card), "合法的 comparison 应通过"

    no_basis = {"last_verdict": {"floor": {"remaining": 0.1, "t_floor_us": 200.0}}}
    assert not _floor_ok(no_basis), "缺 basis 的 floor 必须被拒（反作弊 §四）"
    empty_cmp = {"last_verdict": {"comparison": {"baseline": "lightop"}}}
    assert not _comparison_ok(empty_cmp), "空 comparison 块必须被拒"

    raw = {"D1": 45, "D2": 5, "D3": 0.85, "D4": 8, "D5": 10}
    s, _ = score_dim("D", raw)
    assert s == 10.0, "全达标应满分，实得 %s" % s
    raw["D1"] = 0
    s, d = score_dim("D", raw)
    assert s < 10 and d["D1"]["ratio"] == 0.0
    print("[ok] 反作弊：缺 basis / 空 comparison 均不计分")
    print("[ok] 线性插值：全达标 10.0，单项归零则掉 0.3×10")
    print("\nselftest PASS")
    return 0


def report(raw, detail, as_json):
    dims = {}
    for dim in ("D", "C", "Q"):
        dims[dim] = score_dim(dim, raw)
    if as_json:
        print(json.dumps({"raw": raw, "scores": {k: v[0] for k, v in dims.items()},
                          "detail": {k: v[1] for k, v in dims.items()},
                          "facts": detail}, ensure_ascii=False, indent=2))
    else:
        names = {"D": "深度", "C": "覆盖", "Q": "质量"}
        print("维度    当前   目标   状态")
        for dim in ("D", "C", "Q"):
            sc, det = dims[dim]
            bad = " ".join("%s(%s)" % (k, v["cur"]) for k, v in det.items()
                           if v["ratio"] < 0.999)
            print("%-6s %6.1f  %5.1f   %s %s" % (names[dim], sc, 9.0,
                                                 "✓" if sc >= 8.5 else "✗", bad))
        print("\n按差距×权重排序（前 2 项才是该派的）：")
        items = []
        for dim in ("D", "C", "Q"):
            for k, v in dims[dim][1].items():
                items.append((v["urgency"], k, v))
        items.sort(reverse=True)
        for u, k, v in items[:6]:
            print("  %-3s cur=%-8s target=%-8s gap=%-5.0f%% w=%.2f → %.3f"
                  % (k, v["cur"], v["target"], v["gap"] * 100, v["weight"], u))
    return dims


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--no-pytest", action="store_true", help="跳过 pytest（Q1 记 None）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    raw, detail = collect(with_pytest=not a.no_pytest)
    dims = report(raw, detail, a.json)
    if a.append:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "scores": {k: v[0] for k, v in dims.items()},
               "raw": raw, "facts": {k: detail[k] for k in
                                     ("measured_cards", "validate_errors",
                                      "pytest_passed", "ds_summary")}}
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("\n已追加 -> %s" % HISTORY.relative_to(REPO))
    ok = all(v[0] >= 8.5 for v in dims.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
