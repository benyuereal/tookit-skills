#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""索引卡内容质量守卫（不只是 YAML 结构对不对）。

结构校验（validate.py 的 E1~E11）只能保证「字段齐全、指向可解析」，
拦不住一种更隐蔽的腐化：**字段填了，但填的是空话**。

典型形态：`not_for: ["常规"]`、`pros: ["可控"]`、`cons: ["维护成本"]`。
这些条目结构上完全合法，但没有任何信息量——读者看完仍然不知道该选谁。
本轮实测发现 154 条这类条目，占全部场景条目的多数。
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
CARDS = REPO / "Plugins/ops/op-atlas/cards"

# 纯泛词：出现即视为没有信息量（短但具体的词如「蝶形并行」不在此列）
GENERIC = {
    "常规", "通用", "生产", "简单", "可控", "维护成本", "维护成本高",
    "兜底", "查表命中", "查表", "回退", "灵活", "成熟", "够用", "慢",
    "大核", "大 K", "非投机", "算法对照", "常规形状", "通用形状",
}
MIN_LEN = 6


def _items(text, field):
    out = []
    for m in re.finditer(r"(?ms)^    %s:\n((?:      - .*\n)+)" % field, text):
        out += re.findall(r'      - "([^"]*)"', m.group(1))
    return out


class TestAtlasContentQuality(unittest.TestCase):
    def test_no_generic_only_entries(self):
        bad = []
        for p in sorted(CARDS.glob("*.yaml")):
            t = p.read_text(encoding="utf-8")
            for f in ("best_for", "not_for", "pros", "cons"):
                for it in _items(t, f):
                    if it in GENERIC:
                        bad.append("%s %s: %r（纯泛词，无信息量）" % (p.stem, f, it))
        self.assertEqual(bad, [],
                         "以下条目结构合法但没有信息量，读者看完仍不知该选谁：\n"
                         + "\n".join(bad))

    def test_primary_impl_has_real_scenario(self):
        """tier=primary 的条目，best_for 至少有一条实质内容。"""
        bad = []
        for p in sorted(CARDS.glob("*.yaml")):
            t = p.read_text(encoding="utf-8")
            for blk in re.split(r"(?m)^(?=  - repo: )", t):
                if "tier: primary" not in blk:
                    continue
                items = re.findall(r'(?ms)^    best_for:\n((?:      - .*\n)+)', blk)
                if not items:
                    continue
                vals = re.findall(r'      - "([^"]*)"', items[0])
                if all(v in GENERIC or len(v) < MIN_LEN for v in vals):
                    bad.append("%s: primary 的 best_for 全是泛词 %r" % (p.stem, vals))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_no_fabricated_ratio(self):
        """last_verdict.status=none 时 ratio 必须是 null（没测就不许有倍数）。"""
        bad = []
        for p in sorted(CARDS.glob("*.yaml")):
            t = p.read_text(encoding="utf-8")
            if re.search(r"(?m)^  status: none", t) and not re.search(
                    r"(?m)^  ratio: null", t):
                bad.append(p.stem)
        self.assertEqual(bad, [], "这些卡 status=none 但有 ratio：%s" % bad)


if __name__ == "__main__":
    unittest.main()
