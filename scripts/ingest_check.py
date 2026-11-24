#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest_check —— 吸纳记录的守卫。

吸纳（ingest）**不是 loop 的一部分**。loop 是「优化一个算子直到极限」的迭代；
吸纳是「把一个已经做完的外部优化，拆开吸收进我们知识体」的一次性动作。

它的判据只有三样（业主原话）：
  ① 性能对比结果是否属实
  ② 是否有加速效果
  ③ 我们能否拆解吸收

本文件守的就是这三条**在记录层面**成立。它不跑 GPU、不改卡片、不重算数字 ——
数字来自实跑，实跑由人/agent 调 bench 完成并把结果写进 verification.json。
本文件只检查：**写下来的东西够不够让人相信，且三个月后还能复现。**

为什么必须有它：手写 JSON 的第一版 8 条记录里就有 2 条字段不齐
（Step3 用 measured_verdict 而不是 claim_verdict），没有任何东西会报错。
这正是「校验本身没被校验」。
"""
import argparse
import json
import sys
from pathlib import Path

# 每条记录必须有的字段。**少一个就拒收**，不许 setdefault 兜底 ——
# 兜底会让「没写」和「写了但值相同」不可区分。
REQUIRED = ("stage", "claim_verdict", "note")

# 判定的四档。**每一档的含义必须互斥**，否则「属实」会被滥用。
VERDICTS = {
    "exact": "实测与声明精确吻合（同量级且同方向，如 1.17x vs 1.17x）",
    "conforms": "实测落在声明区间内",
    "partially_conforms": "部分形状吻合、部分不吻合 —— **必须写明哪部分**",
    "refuted": "实测与声明矛盾",
    "not_reverified": "本次没跑 —— 允许存在，但不计入已验证",
}

# 顶层必须有。缺 provenance 的记录等于散文。
TOP_REQUIRED = ("source", "ingested_at", "card", "shape", "caliber", "records")


def check_record(r, i):
    probs = []
    for k in REQUIRED:
        if k not in r or r[k] in (None, ""):
            probs.append("records[%d] 缺 %s" % (i, k))
    v = r.get("claim_verdict")
    if v is not None and v not in VERDICTS:
        probs.append("records[%d] claim_verdict=%r 不在允许集合 %s"
                     % (i, v, sorted(VERDICTS)))
    # partially_conforms 必须写清哪部分 —— 否则它就是「大概对」的托词
    if v == "partially_conforms" and "哪部分" not in str(r.get("note", "")) \
            and "低于" not in str(r.get("note", "")) and "高于" not in str(r.get("note", "")):
        probs.append("records[%d] partially_conforms 但 note 没说清哪部分不吻合" % i)
    # 声称了实测就必须有数字；声称 exact/conforms 却无任何数值 = 空口
    if v in ("exact", "conforms", "refuted"):
        has_num = any(k in r for k in ("measured_gbs", "measured_x", "measured"))
        if not has_num:
            probs.append("records[%d] claim_verdict=%s 但没有任何实测字段" % (i, v))
    return probs


def check(doc):
    """返回问题列表。空列表 = 通过。"""
    probs = []
    if not isinstance(doc, dict):
        return ["顶层必须是 object"]
    for k in TOP_REQUIRED:
        if k not in doc:
            probs.append("缺顶层字段 %s" % k)
    cal = doc.get("caliber") or {}
    for k in ("measure", "baseline"):
        if not cal.get(k):
            probs.append("缺 caliber.%s —— 没有测量口径的数字不可比" % k)
    if not cal.get("denominator_gbs"):
        probs.append("缺 caliber.denominator_gbs —— 带宽分母不明则达成率无意义")

    recs = doc.get("records") or []
    if not recs:
        probs.append("records 为空 —— 没有记录就没有吸纳")
    for i, r in enumerate(recs):
        probs += check_record(r, i)

    # 汇总必须与实际记录一致（防「声称验证了 5 条，实际只有 3 条」）
    s = doc.get("summary") or {}
    if "claims_verified" in s:
        actual = sum(1 for r in recs
                     if r.get("claim_verdict") in ("exact", "conforms",
                                                   "partially_conforms", "refuted"))
        if int(s["claims_verified"]) != actual:
            probs.append("summary.claims_verified=%s 与实际已复验记录数 %d 不符"
                         % (s["claims_verified"], actual))
    return probs


def selftest():
    """**反向优先**：每条判据都要证明它该报错时真会报错。"""
    good = {
        "source": "/x", "ingested_at": "t", "card": "c", "shape": "s",
        "caliber": {"measure": "m", "baseline": "b", "denominator_gbs": 1870},
        "records": [{"stage": "a", "claim_verdict": "conforms",
                     "note": "n", "measured_gbs": 500}],
        "summary": {"claims_verified": 1},
    }
    assert check(good) == [], check(good)
    print("[ok] 干净记录通过")

    bad = json.loads(json.dumps(good))
    del bad["records"][0]["claim_verdict"]
    assert any("缺 claim_verdict" in p for p in check(bad))
    print("[ok] 反向：缺字段被拒（第一版手写 JSON 就是这么坏的）")

    bad = json.loads(json.dumps(good))
    bad["records"][0]["claim_verdict"] = "差不多对"
    assert any("不在允许集合" in p for p in check(bad))
    print("[ok] 反向：自造判定词被拒")

    bad = json.loads(json.dumps(good))
    bad["records"][0]["claim_verdict"] = "exact"
    del bad["records"][0]["measured_gbs"]
    assert any("没有任何实测字段" in p for p in check(bad))
    print("[ok] 反向：声称 exact 却无实测数字被拒")

    bad = json.loads(json.dumps(good))
    bad["records"][0]["claim_verdict"] = "partially_conforms"
    bad["records"][0]["note"] = "还行"
    assert any("没说清哪部分" in p for p in check(bad))
    print("[ok] 反向：partially_conforms 含糊其辞被拒")

    bad = json.loads(json.dumps(good))
    del bad["caliber"]["denominator_gbs"]
    assert any("分母不明" in p for p in check(bad))
    print("[ok] 反向：缺带宽分母被拒")

    bad = json.loads(json.dumps(good))
    bad["summary"]["claims_verified"] = 9
    assert any("不符" in p for p in check(bad))
    print("[ok] 反向：汇总数与实际记录数不符被拒")
    print("selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="verification.json 或其所在目录")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.path:
        ap.error("需要 path 或 --selftest")
    p = Path(a.path)
    if p.is_dir():
        cands = sorted(p.glob("verification.json"))
        if not cands:
            print("目录下没有 verification.json：%s" % p)
            return 2
        p = cands[0]
    if not p.is_file():
        print("找不到 %s" % p)
        return 2
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        print("不是合法 JSON：%s" % e)
        return 2
    probs = check(doc)
    if a.json:
        print(json.dumps({"path": str(p), "ok": not probs, "problems": probs},
                         ensure_ascii=False, indent=2))
    else:
        print(("✓ " if not probs else "✗ ") + str(p))
        for x in probs:
            print("  - " + x)
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
