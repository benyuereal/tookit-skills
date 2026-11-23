#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L7：每个家族的映射率 ≥80%。

为什么按家族卡：优化一个算子时，你会先落到它的**家族**（gemm / quant / attention），
如果那个家族的映射率低，就会卡在「不知道调哪个函数」上。
整体 77% 看着还行，但 gemm 只有 50% 时，优化 MoE（两个 GEMM 占 89-96%）
会直接卡住——整体指标掩盖了结构性缺口。

映射率定义：(有 import_path 或 l1_exempt_reason 的 impl 数) / 该家族 impl 总数
  l1_exempt_reason 算达标：源码树/自研确实没有 库.函数，但已说明原因。
"""
import collections
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
THRESHOLD = 0.8


class TestFamilyMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        import yaml
        cls.rates = {}
        tot = collections.Counter(); ok = collections.Counter()
        for p in CARDS.glob("*.yaml"):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            fam = d.get("family")
            for i in d.get("variants") or []:
                tot[fam] += 1
                if i.get("import_path") or i.get("l1_exempt_reason"):
                    ok[fam] += 1
        cls.rates = {f: ok[f] / tot[f] for f in tot}
        cls.counts = dict(tot)

    def test_every_family_above_threshold(self):
        low = {f: (round(r * 100, 1), self.counts[f])
               for f, r in self.rates.items() if r < THRESHOLD}
        self.assertEqual(
            low, {},
            "以下家族映射率低于 %.0f%%（优化该族算子时会卡在「不知道调哪个函数」）：\n%s"
            % (THRESHOLD * 100,
               "\n".join("  %s: %.1f%% (%d impl)" % (f, v[0], v[1])
                         for f, v in sorted(low.items()))))

    def test_report_all_families(self):
        """不是失败条件，但把全量打印出来，便于看到哪一族在临界。"""
        for f, r in sorted(self.rates.items(), key=lambda x: x[1]):
            print("  %-14s %5.1f%%  (%d impl)" % (f, r * 100, self.counts[f]))


if __name__ == "__main__":
    unittest.main()
