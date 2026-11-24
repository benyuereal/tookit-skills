#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest_check —— 吸纳记录的守卫。

吸纳（ingest）**不是 loop 的一部分**。loop 是「优化一个算子直到极限」的迭代；
吸纳是「把一个已经做完的外部优化，拆开吸收进我们知识体」的一次性动作。

判据只有三样（业主原话）：
  ① 性能对比结果是否属实  ② 是否有加速效果  ③ 我们能否拆解吸收

本文件守这三条**在记录层面**成立，不跑 GPU、不改卡片、不重算数字。
它校验两种格式：
  - verification.json（旧，手写散文，check_verification 原样保留 —— 不破坏历史）
  - records.jsonl（新，schema 见 Plugins/ops/ingest/schema.yaml；一行一条）

新格式比旧格式多守一条最关键的：**provenance.quote 必须能在 provenance.file 的
provenance.line 上 grep 到**。这是防编造的唯一硬判据 —— 没有它，手写的
「文档说 1.89x」和真跑出来的 1.89x 在文件里长得一模一样。

另一条：extract:human 的字段缺席**不算错**，但必须列进 human_pending。
缺了又不说，才会报错 —— 那等于把「没判」伪装成「判了」。
"""
import argparse
import json
import sys
from pathlib import Path

# ── 旧格式（verification.json）的判据，原样保留 ────────────────────────────
REQUIRED = ("stage", "claim_verdict", "note")
VERDICTS = {
    "exact": "实测与声明精确吻合（同量级且同方向，如 1.17x vs 1.17x）",
    "conforms": "实测落在声明区间内",
    "partially_conforms": "部分形状吻合、部分不吻合 —— **必须写明哪部分**",
    "refuted": "实测与声明矛盾",
    "not_reverified": "本次没跑 —— 允许存在，但不计入已验证",
}
TOP_REQUIRED = ("source", "ingested_at", "card", "shape", "caliber", "records")

# ── 新格式（records.jsonl）的判据 ──────────────────────────────────────────
NEW_VERDICTS = ("exact", "conforms", "partially_conforms", "refuted",
                "not_reverified", "not_applicable")
KINDS = ("claim", "measurement", "define", "caliber", "manifest")
METRICS = ("bandwidth_gbs", "range_gbs", "latency_us", "speedup_x", "pct",
           "count", "outcome", "abs_error", "define_value")
UNITS = ("gbs", "us", "x", "pct", "count", "bytes", "string")
# 这些字段只能人填。机器填了它们 = 伪造人的判断，同样要报错。
HUMAN = ("our_measured_value", "verdict_note", "our_significance", "mechanism",
         "preconditions", "architecture_applicable", "absorb", "absorb_target",
         "absorb_action")
# 各 kind 的三件套必填（机器字段）
KIND_REQUIRED = {
    "claim": ("metric", "claim_text", "claimed_value", "claim_unit",
              "baseline_desc", "claim_verdict"),
    "measurement": ("metric", "claim_text", "claimed_value", "claim_unit",
                    "baseline_desc", "claim_verdict"),
    "define": ("metric", "claim_text", "claimed_value", "claim_unit"),
    "caliber": ("metric", "claim_text", "claimed_value", "claim_unit"),
    "manifest": (),
}
ENVELOPE = ("schema", "id", "kind", "stage", "provenance")
PROV_REQUIRED = ("file", "line", "quote")


def check_record(r, i):
    """旧格式：单条记录。保留原逻辑与报错文案。"""
    probs = []
    for k in REQUIRED:
        if k not in r or r[k] in (None, ""):
            probs.append("records[%d] 缺 %s" % (i, k))
    v = r.get("claim_verdict")
    if v is not None and v not in VERDICTS:
        probs.append("records[%d] claim_verdict=%r 不在允许集合 %s"
                     % (i, v, sorted(VERDICTS)))
    if v == "partially_conforms" and "哪部分" not in str(r.get("note", "")) \
            and "低于" not in str(r.get("note", "")) and "高于" not in str(r.get("note", "")):
        probs.append("records[%d] partially_conforms 但 note 没说清哪部分不吻合" % i)
    if v in ("exact", "conforms", "refuted"):
        has_num = any(k in r for k in ("measured_gbs", "measured_x", "measured"))
        if not has_num:
            probs.append("records[%d] claim_verdict=%s 但没有任何实测字段" % (i, v))
    return probs


def check_verification(doc):
    """旧格式：整份 verification.json。"""
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

    s = doc.get("summary") or {}
    if "claims_verified" in s:
        actual = sum(1 for r in recs
                     if r.get("claim_verdict") in ("exact", "conforms",
                                                   "partially_conforms", "refuted"))
        if int(s["claims_verified"]) != actual:
            probs.append("summary.claims_verified=%s 与实际已复验记录数 %d 不符"
                         % (s["claims_verified"], actual))
    return probs


# 兼容旧调用名
check = check_verification


def _check_provenance(p, rec_id):
    """防编造核心：file 可读、line 在范围内、quote 就在那一行。"""
    probs = []
    for k in PROV_REQUIRED:
        if k not in p:
            probs.append("%s provenance 缺 %s" % (rec_id, k))
    if probs:
        return probs
    f, ln, q = p["file"], p["line"], p["quote"]
    if not isinstance(q, str) or len(q.strip()) < 8:
        probs.append("%s provenance.quote 太短（<%d 字符）—— 无法证明任何事"
                     % (rec_id, 8))
        return probs
    # file 必须相对 root 且不逃逸；校验器不许被 ../../etc/passwd 牵着走
    if Path(f).is_absolute() or ".." in Path(f).parts:
        probs.append("%s provenance.file 必须是 rel 路径且不得含 '..'：%s"
                     % (rec_id, f))
        return probs
    path = Path(p.get("_root", ".")) / f
    if not path.is_file():
        probs.append("%s provenance.file 不存在或不可读：%s" % (rec_id, f))
        return probs
    if not isinstance(ln, int) or ln < 1:
        probs.append("%s provenance.line 必须是正整数：%r" % (rec_id, ln))
        return probs
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        probs.append("%s provenance.file 读不了：%s" % (rec_id, e))
        return probs
    if ln > len(lines):
        probs.append("%s provenance.line=%d 超出文件行数 %d"
                     % (rec_id, ln, len(lines)))
    elif q not in lines[ln - 1]:
        # 允许跨行引用时退一步：整份文件里能 grep 到就放行，但提示行号漂了。
        if q not in "\n".join(lines):
            probs.append("%s provenance.quote 在 %s 中 grep 不到（疑似编造）：%r"
                         % (rec_id, f, q[:60]))
        else:
            probs.append("%s provenance.quote 在文件里但不在第 %d 行（行号漂移）"
                         % (rec_id, ln))
    return probs


def check_new_record(r, i, root):
    """新格式：单条。返回 (问题列表, human_pending 列表)。"""
    probs = []
    rec_id = r.get("id") or "records[%d]" % i
    for k in ENVELOPE:
        if k not in r:
            probs.append("%s 缺信封字段 %s" % (rec_id, k))
    if r.get("schema") != "mhc-ingest/1":
        probs.append("%s schema=%r 不是 mhc-ingest/1" % (rec_id, r.get("schema")))
    kind = r.get("kind")
    if kind not in KINDS:
        probs.append("%s kind=%r 不在 %s" % (rec_id, kind, list(KINDS)))
    if r.get("metric") not in METRICS and kind != "manifest":
        probs.append("%s metric=%r 不在允许集合" % (rec_id, r.get("metric")))
    if r.get("claim_unit") not in UNITS and kind != "manifest":
        probs.append("%s claim_unit=%r 不在允许集合" % (rec_id, r.get("claim_unit")))

    v = r.get("claim_verdict")
    if kind not in ("manifest",):
        if v is None:
            probs.append("%s 缺 claim_verdict" % rec_id)
        elif v not in NEW_VERDICTS:
            probs.append("%s claim_verdict=%r 不在允许集合" % (rec_id, v))

    # 条件必填：每个 kind 的三件套
    for k in KIND_REQUIRED.get(kind, ()):
        if r.get(k) in (None, ""):
            if k not in (r.get("missing") or {}):
                probs.append("%s 缺 %s 且未记 missing（不许静默省略）"
                             % (rec_id, k))
    # 带宽类必须有分母 —— 沿用旧判据
    if r.get("metric") in ("bandwidth_gbs", "range_gbs") and not r.get("denominator_gbs"):
        probs.append("%s metric=%s 但缺 denominator_gbs —— 分母不明则达成率无意义"
                     % (rec_id, r.get("metric")))

    # ── human_pending 的契约 ───────────────────────────────────────────────
    present_human = [h for h in HUMAN if h in r]
    pending = r.get("human_pending") or []
    if not isinstance(pending, list):
        probs.append("%s human_pending 必须是 list" % rec_id)
        pending = []
    # 必填 human 字段：缺席必须列 pending；在场不能同时列 pending
    for h in ("architecture_applicable", "absorb"):
        if h in r and h in pending:
            probs.append("%s %s 既有值又在 human_pending（自相矛盾）"
                         % (rec_id, h))
        if h not in r and h not in pending and kind != "manifest":
            probs.append("%s 缺必填 human 字段 %s 且未列 human_pending —— "
                         "「没判」必须显式可见" % (rec_id, h))
    for h in pending:
        if h not in HUMAN:
            probs.append("%s human_pending 里有未知字段 %s" % (rec_id, h))
        elif h in r:
            probs.append("%s %s 既有值又在 human_pending" % (rec_id, h))
    # missing 只许装机器字段；把人的事赖给机器是错
    miss = r.get("missing") or {}
    if not isinstance(miss, dict):
        probs.append("%s missing 必须是 object" % rec_id)
        miss = {}
    for k, why in miss.items():
        if k in HUMAN:
            probs.append("%s missing 里出现 human 字段 %s —— 人没判不能算机器缺失"
                         % (rec_id, k))
        if not (isinstance(why, str) and why.strip()):
            probs.append("%s missing.%s 必须写原因，不许空" % (rec_id, k))

    # ── 属实判定的防伪：没有我方复跑就不许说吻合 ───────────────────────────
    if v in ("exact", "conforms", "partially_conforms", "refuted"):
        if "our_measured_value" not in r:
            probs.append("%s claim_verdict=%s 却没有 our_measured_value —— "
                         "文档自证不算复现" % (rec_id, v))

    # ── 吸收的完整性 ───────────────────────────────────────────────────────
    if r.get("absorb") is True:
        for k in ("absorb_target", "absorb_action"):
            if not r.get(k):
                probs.append("%s absorb=true 但缺 %s" % (rec_id, k))
    # 显著性判据必须与数值一致（机器算的，可审计）
    if r.get("metric") == "speedup_x" and isinstance(r.get("claimed_value"), (int, float)):
        expect = ("significant" if r["claimed_value"] >= 1.03 else
                  "marginal" if r["claimed_value"] >= 1.005 else "none")
        if r.get("significance") and r["significance"] != expect:
            probs.append("%s significance=%s 与 claimed_value=%s 不符（应 %s）"
                         % (rec_id, r["significance"], r["claimed_value"], expect))

    probs += _check_provenance(r.get("provenance") or {}, rec_id)
    return probs


def check_records(records, root="."):
    """新格式：整份 records.jsonl（含 manifest）。返回 (问题, human_pending 汇总)。"""
    probs = []
    if not records:
        return ["records 为空 —— 没有记录就没有吸纳"], []
    man = records[0]
    if man.get("kind") != "manifest":
        probs.append("第一行必须是 kind=manifest")
    source_root = man.get("source_root")
    if not source_root:
        probs.append("manifest 缺 source_root —— 无法校验 provenance.file 是否可读")
    base = Path(source_root) if source_root else Path(root)

    seen = set()
    pend = []
    for i, r in enumerate(records[1:], 1):
        r = dict(r)
        r["provenance"] = dict(r.get("provenance") or {})
        r["provenance"]["_root"] = str(base)
        for h in r.get("human_pending") or []:
            pend.append("%s:%s" % (r.get("id", i), h))
        probs += check_new_record(r, i, base)
        rid = r.get("id")
        if rid in seen:
            probs.append("id 重复：%s —— 同一结论被吸了两次" % rid)
        seen.add(rid)

    # manifest 计数必须与实际一致（防「声称抽了 8 条实际 3 条」，沿用旧精神）
    counts = man.get("counts") or {}
    actual = {}
    for r in records[1:]:
        actual[r.get("kind")] = actual.get(r.get("kind"), 0) + 1
    if counts and counts != actual:
        probs.append("manifest.counts=%s 与实际 %s 不符" % (counts, actual))
    return probs, pend


def selftest():
    """**反向优先**：每条判据都要证明它该报错时真会报错。"""
    # ── 旧格式的 7 条，一条不减 ────────────────────────────────────────────
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

    # ── 新格式：合成一个临时语料，正向通过、逐个字段反向 ───────────────────
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "R.md").write_text("l1\nl2\n上界是 **1870 GB/s**，别写错行号。\n",
                            encoding="utf-8")
    man = {"kind": "manifest", "schema": "mhc-ingest/1",
           "source_root": str(d), "counts": {"claim": 1}}
    rec = {"schema": "mhc-ingest/1", "id": "a", "kind": "claim",
           "stage": "total", "metric": "speedup_x", "claim_text": "q",
           "claim_unit": "x", "claimed_value": 1.89, "baseline_desc": "b",
           "claim_verdict": "not_reverified",
           "provenance": {"file": "R.md", "line": 3,
                          "quote": "上界是 **1870 GB/s**，别写错行号。"},
           "human_pending": ["our_measured_value", "verdict_note",
                             "our_significance", "mechanism", "preconditions",
                             "architecture_applicable", "absorb"]}
    probs, pend = check_records([man, rec])
    assert probs == [], probs
    assert "a:architecture_applicable" in pend
    print("[ok] 新：干净记录通过，human_pending 不等于错而是被列出")

    def mut(fn):
        r = json.loads(json.dumps(rec))
        fn(r)
        return check_records([man, r])[0]

    assert any("grep 不到" in p for p in mut(
        lambda r: r["provenance"].update(quote="文档说 2.3x，其实没有这句")))
    print("[ok] 新反向：quote 在原文不存在 → 「疑似编造」被拒（核心判据）")

    assert any("行号漂移" in p for p in mut(
        lambda r: r["provenance"].update(line=1)))
    print("[ok] 新反向：quote 在文件里但行号不对 → 拒")

    assert any("不存在或不可读" in p for p in mut(
        lambda r: r["provenance"].update(file="NOPE.md")))
    print("[ok] 新反向：provenance.file 不可读 → 拒")

    assert any("不得含 '..'" in p for p in mut(
        lambda r: r["provenance"].update(file="../../etc/passwd")))
    print("[ok] 新反向：provenance.file 路径逃逸 → 拒")

    assert any("太短" in p for p in mut(
        lambda r: r["provenance"].update(quote="x")))
    print("[ok] 新反向：quote 是占位符 → 拒")

    assert any("our_measured_value" in p for p in mut(
        lambda r: r.update(claim_verdict="conforms")))
    print("[ok] 新反向：说 conforms 却没有我方复跑 → 拒（文档自证不算）")

    assert any("未列 human_pending" in p for p in mut(
        lambda r: r["human_pending"].remove("architecture_applicable")))
    print("[ok] 新反向：必填 human 字段没判又不说 → 拒")

    assert any("human_pending 里有未知字段" in p for p in mut(
        lambda r: r["human_pending"].append("whatever")))
    print("[ok] 新反向：human_pending 塞未知字段 → 拒")

    assert any("既有值又在 human_pending" in p for p in mut(
        lambda r: r.update(absorb=True, absorb_target="t", absorb_action="a")))
    print("[ok] 新反向：打了分又列成待办 → 拒（矛盾）")

    assert any("人没判不能算机器缺失" in p for p in mut(
        lambda r: r.update(missing={"absorb": "不想判"})))
    print("[ok] 新反向：human 字段塞进 missing → 拒")

    assert any("缺 denominator_gbs" in p for p in mut(
        lambda r: r.update(metric="bandwidth_gbs")))
    print("[ok] 新反向：带宽结论没有分母 → 拒")

    assert any("significance" in p for p in mut(
        lambda r: r.update(significance="none")))
    print("[ok] 新反向：显著性与数值不符 → 拒（机器算的必须可审计）")

    probs, _ = check_records([man, rec, dict(rec)])
    assert any("id 重复" in p for p in probs)
    print("[ok] 新反向：id 重复 → 拒")

    bad_man = dict(man, counts={"claim": 9})
    probs, _ = check_records([bad_man, rec])
    assert any("counts" in p for p in probs)
    print("[ok] 新反向：manifest 计数与实际不符 → 拒")

    print("selftest PASS")
    return 0


def _load_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="records.jsonl / verification.json 或其所在目录")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.path:
        ap.error("需要 path 或 --selftest")
    p = Path(a.path)
    if p.is_dir():
        jl = sorted(p.glob("records.jsonl"))
        js = sorted(p.glob("verification.json"))
        cands = jl or js
        if not cands:
            print("目录下没有 records.jsonl / verification.json：%s" % p)
            return 2
        p = cands[0]
    if not p.is_file():
        print("找不到 %s" % p)
        return 2

    pend = []
    if p.suffix == ".jsonl":
        try:
            records = _load_jsonl(p)
        except ValueError as e:
            print("不是合法 JSONL：%s" % e)
            return 2
        probs, pend = check_records(records, p.parent)
    else:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            print("不是合法 JSON：%s" % e)
            return 2
        probs = check_verification(doc)

    if a.json:
        print(json.dumps({"path": str(p), "ok": not probs, "problems": probs,
                          "human_pending": pend}, ensure_ascii=False, indent=2))
    else:
        print(("✓ " if not probs else "✗ ") + str(p))
        for x in probs:
            print("  - " + x)
        if pend:
            print("  待办（人不判不算错，但必须看见）：%d 处 human_pending"
                  % len(pend))
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
