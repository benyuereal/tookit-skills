#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_expect —— 候选的 expect 必须是数值。

为什么需要它（2026-11-24 实测发现）
----------------------------------
`calibrate.py` 要算「预测 vs 实际」的比值，前提是 `expect` 能解析成数。
但真实数据里 op-rmsnorm-fused 的三条 expect 全是散文：

    "triton 融合 vs lightop 融合（同语义 2读2写）"
    "inplace=True 时省一份残差写回"
    "hidden 非 2 的幂时向量化宽度受限造成的损失"

结果 **7 条记录 0 条可校准** —— EV 的事前校准根本用不了。

散文 expect 还有一个更深的毛病：**它无法被证伪**。
"能省一次写回" 涨了 3% 算不算兑现？没人说得清。写 `-11%` 就能。

判据
----
`expect` 必须是 `-?\d+(\.\d+)?%` 或 `\d+(\.\d+)?x`。
没有数值的必须在 `expect_basis` 里写明为什么（通常是"未实测"），
且 `expect_confidence` 标 low。

用法
----
    python3 -I scripts/check_expect.py --all
    python3 -I scripts/check_expect.py --skill Plugins/ops/skills/op-rmsnorm-fused
    python3 -I scripts/check_expect.py --selftest
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / "Plugins/ops/skills"

NUMERIC = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*[%x]\s*$")


def check_skill(skill_dir):
    """返回 (problems, stats)。"""
    import yaml
    p = Path(skill_dir) / "candidates.yaml"
    if not p.is_file():
        # **没有 candidates.yaml 不是问题**：只有参与优化循环的 skill 才需要它
        # （op-roofline / op-hip-compile 这类纯知识 skill 本来就没有）。
        # 第一版把它报成错误，结果 15 个 skill 全被误报。
        # 返回空 problems + 空 stats，调用方据此跳过。
        return [], {}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cands = d.get("candidates") or []
    probs, ok = [], 0
    for c in cands:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if str(c.get("status") or "pending") != "pending":
            continue
        e = c.get("expect")
        if e is None:
            probs.append((cid, "缺 expect —— 没有预期就无从校准，也无从证伪"))
            continue
        if NUMERIC.match(str(e)):
            ok += 1
        else:
            # 散文允许，但必须带 low_confidence + basis（明说"未实测"）
            if str(c.get("expect_confidence") or "") != "low":
                probs.append((cid, "expect=%r 不是数值，且未标 expect_confidence: low —— "
                                   "散文预期无法校准，也无法证伪" % e))
            elif not str(c.get("expect_basis") or "").strip():
                probs.append((cid, "expect=%r 不是数值且缺 expect_basis —— "
                                   "必须写明为什么给不出数" % e))
            else:
                ok += 1
    return probs, {"skill": Path(skill_dir).name, "candidates": len(cands),
                   "numeric": ok}


def selftest():
    import tempfile
    import yaml

    def mk(cands):
        d = Path(tempfile.mkdtemp())
        (d / "candidates.yaml").write_text(
            yaml.dump({"candidates": cands}, allow_unicode=True), encoding="utf-8")
        return d

    # 1) 数值版本 pass
    d = mk([{"id": "a", "status": "pending", "expect": "-11%"}])
    probs, st = check_skill(d)
    assert not probs, probs
    print("[ok] 数值 expect（-11%%）通过")

    # 2) 倍数写法也认
    d = mk([{"id": "a", "status": "pending", "expect": "1.15x"}])
    assert not check_skill(d)[0]
    print("[ok] 倍数 expect（1.15x）通过")

    # 3) 散文 + 无 low_confidence → 报错
    d = mk([{"id": "a", "status": "pending", "expect": "省一次写回"}])
    probs, _ = check_skill(d)
    assert probs and "无法证伪" in probs[0][1], probs
    print("[ok] 散文 expect 未标 low → 报错：%s" % probs[0][1][:40])

    # 4) 散文 + low + basis → 放行
    d = mk([{"id": "a", "status": "pending", "expect": "省一次写回",
             "expect_confidence": "low", "expect_basis": "未在本仓实测"}])
    assert not check_skill(d)[0]
    print("[ok] 散文 + low + basis → 放行（明说未实测）")

    # 5) 散文 + low 但无 basis → 报错
    d = mk([{"id": "a", "status": "pending", "expect": "省一次写回",
             "expect_confidence": "low"}])
    probs, _ = check_skill(d)
    assert probs and "缺 expect_basis" in probs[0][1], probs
    print("[ok] 散文 + low 但无 basis → 报错")

    # 6) 缺 expect → 报错
    d = mk([{"id": "a", "status": "pending"}])
    probs, _ = check_skill(d)
    assert probs and "缺 expect" in probs[0][1]
    print("[ok] 缺 expect → 报错")

    # 7) 非 pending 的不管
    d = mk([{"id": "a", "status": "adopted"}])
    assert not check_skill(d)[0]
    print("[ok] 非 pending 候选不检查（已完成的不管）")

    print("\nselftest PASS（7 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="expect 必须是数值")
    ap.add_argument("--skill", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    dirs = a.skill or (sorted(p for p in SKILLS.glob("op-*") if p.is_dir())
                       if a.all else [])
    if not dirs:
        ap.error("给 --skill 或 --all，或 --selftest")
    bad = 0
    for d in dirs:
        probs, st = check_skill(d)
        if probs:
            print("✗ %s" % Path(d).name)
            for cid, why in probs:
                print("    %-20s %s" % (cid, why))
            bad += len(probs)
        elif st.get("candidates"):
            print("✓ %-24s %d 个候选，%d 条可校准"
                  % (st["skill"], st["candidates"], st["numeric"]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
