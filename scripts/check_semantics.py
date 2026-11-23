#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_semantics —— 比较两个实现前，先断言它们的「活」是一样的。

为什么需要它
------------
2026-11-23 我们拿「只写 1 份输出（201 MB）」的 kernel 去比 lightop 的
「写 2 份（268 MB）」，得出「快 1.18×」——**少干 1/3 的活然后说自己快**。
等语义对齐后实际慢 6.7%。

`variants.yaml` 里其实早写了「会原地改写输入」，但它只说了改写、没说输出几份。
信息在，但不在机器能读的位置。这个脚本把「能机读」补上。

用法
----
    python3 -I scripts/check_semantics.py <card_dir> a b       # 比较两个变体
    python3 -I scripts/check_semantics.py <card_dir> --list    # 列出所有变体的语义
    python3 -I scripts/check_semantics.py <card_dir> --selftest

退出码：0 = 可比较；1 = 不可比较（会说明差在哪）；2 = 用法/文件问题。
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CRITICAL = ("inputs", "outputs", "traffic_passes")


def load_semantics(card_dir):
    """读 <card>/variants.yaml 的 semantics 块。返回 {name: semantics}。"""
    import yaml
    p = Path(card_dir) / "variants.yaml"
    if not p.is_file():
        raise SystemExit("找不到 %s" % p)
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = {}
    for v in d.get("variants") or []:
        if not isinstance(v, dict) or not v.get("name"):
            continue
        sem = v.get("semantics")
        if sem is not None:
            out[v["name"]] = sem
    return out


def compare(a_name, a_sem, b_name, b_sem):
    """返回 (可比?, 差异清单)。"""
    diffs = []
    for k in CRITICAL:
        va, vb = a_sem.get(k), b_sem.get(k)
        if va != vb:
            diffs.append("%s: %s=%r vs %s=%r" % (k, a_name, va, b_name, vb))
    # 输出集按集合比：顺序不同不算差异，少一份才算
    oa, ob = set(a_sem.get("outputs") or []), set(b_sem.get("outputs") or [])
    if oa != ob:
        only_a, only_b = sorted(oa - ob), sorted(ob - oa)
        diffs.append("输出集不同：%s 独有 %s，%s 独有 %s（少写一份 = 少干一份活）"
                     % (a_name, only_a or "无", b_name, only_b or "无"))
    return (not diffs), diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description="比较前先断言两个实现的活一样")
    ap.add_argument("card_dir", nargs="?", help="卡片目录")
    ap.add_argument("names", nargs="*", help="两个变体名")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.card_dir:
        ap.error("要么给卡片目录，要么 --selftest")

    sem = load_semantics(a.card_dir)
    if not sem:
        print("该卡没有任何变体声明 semantics —— 这是最该补的一项。", file=sys.stderr)
        print("在 variants.yaml 里给每个变体加：\n"
              "  semantics:\n    inputs: [x, ...]\n    outputs: [out, ...]\n"
              "    mutates: [...]\n    traffic_passes: N", file=sys.stderr)
        return 2

    if a.list or not a.names:
        print("%-24s %-28s %-6s %-6s %s" % ("变体", "输出", "遍数", "改写", "输入"))
        for n, s in sem.items():
            print("%-24s %-28s %-6s %-6s %s"
                  % (n, ",".join(s.get("outputs") or []), s.get("traffic_passes"),
                     ",".join(s.get("mutates") or []) or "-",
                     ",".join(s.get("inputs") or [])))
        return 0

    if len(a.names) != 2:
        ap.error("要给两个变体名，或 --list")
    x, y = a.names
    for n in (x, y):
        if n not in sem:
            print("变体 %r 没有 semantics 声明" % n, file=sys.stderr)
            return 2
    ok, diffs = compare(x, sem[x], y, sem[y])
    if ok:
        print("✅ %s 与 %s 的访存契约一致，可以比" % (x, y))
        return 0
    print("❌ %s 与 %s 不可比：" % (x, y))
    for d in diffs:
        print("   - " + d)
    print("\n**不对齐就是在比两件不同的活。** 先改实现或改基线，别急着出 ratio。")
    return 1


def selftest():
    # 等语义 → 可比
    a = {"inputs": ["x", "r"], "outputs": ["out", "res_out"], "traffic_passes": 4}
    ok, d = compare("A", a, "B", dict(a))
    assert ok and not d, d
    print("[ok] 等语义 → 可比")

    # 少一份输出 → 不可比（这正是今天的坑）
    b = {"inputs": ["x", "r"], "outputs": ["out"], "traffic_passes": 3}
    ok, d = compare("ours", b, "lightop", a)
    assert not ok
    assert any("少写一份" in x for x in d), d
    print("[ok] 少一份输出 → 不可比：%s" % d[-1])

    # 输入集不同 → 不可比
    c = {"inputs": ["x"], "outputs": ["out"], "traffic_passes": 2}
    ok, d = compare("A", c, "B", {"inputs": ["x", "r"], "outputs": ["out"],
                                  "traffic_passes": 2})
    assert not ok and any("inputs" in x for x in d), d
    print("[ok] 输入集不同 → 不可比")

    # 遍数不同 → 不可比
    ok, d = compare("A", {"inputs": ["x"], "outputs": ["out"], "traffic_passes": 2},
                    "B", {"inputs": ["x"], "outputs": ["out"], "traffic_passes": 3})
    assert not ok and any("traffic_passes" in x for x in d), d
    print("[ok] 遍数不同 → 不可比")
    print("\nselftest PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
