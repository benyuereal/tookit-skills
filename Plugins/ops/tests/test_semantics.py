#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较前必须先断言「两个实现的活是一样的」。

2026-11-23 的硬伤：我们拿「只写 1 份输出（201 MB）」的 kernel 去比 lightop 的
「写 2 份（268 MB）」，得出「快 1.18×」——**少干 1/3 的活然后说自己快**。
等语义对齐后实际慢 6.7%。

`variants.yaml` 里早写了「会原地改写输入」，但**只说了改写、没说输出几份**。
信息在，但不在机器能读的位置。

本测试守四件事：
  A. 少一份输出 → 判不可比（这正是今天的坑）
  B. 输入集不同 → 判不可比
  C. 遍数不同 → 判不可比
  D. 真实卡的 rmsnorm-fused：ref-heavy 与 lightop-fused 必须被判不可比
     （它是 oracle 不是性能基线，这个区分必须是机器可判的）
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts/check_semantics.py"
CARD = REPO / "Plugins/ops/skills/op-rmsnorm-fused"


def _run(*args):
    return subprocess.run([sys.executable, "-I", str(TOOL)] + list(args),
                          capture_output=True, text=True)


class TestSemanticsAlignment(unittest.TestCase):
    def test_selftest_passes(self):
        r = _run("--selftest")
        self.assertIn("selftest PASS", r.stdout, r.stdout + r.stderr)

    def test_real_card_lists_semantics(self):
        r = _run(str(CARD), "--list")
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ("ref-heavy", "lightop-fused", "fused-3pass-triton"):
            self.assertIn(name, r.stdout)

    def test_equal_semantics_are_comparable(self):
        r = _run(str(CARD), "fused-3pass-triton", "lightop-fused")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("可以比", r.stdout)

    def test_ref_heavy_is_not_a_performance_baseline(self):
        """ref-heavy 只输出 out，lightop 输出两份 ⇒ 不可比。

        这是今天最重要的发现之一：**A/B 的基线本身选错了**。
        """
        r = _run(str(CARD), "ref-heavy", "lightop-fused")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("不可比", r.stdout)
        self.assertIn("少写一份", r.stdout)

    def test_ref_heavy_declared_not_comparable(self):
        """variants.yaml 里要显式写 not_comparable_to，读者才不会误用。"""
        import yaml
        d = yaml.safe_load((CARD / "variants.yaml").read_text(encoding="utf-8"))
        ref = next(v for v in d["variants"] if v["name"] == "ref-heavy")
        self.assertEqual(ref.get("status"), "reference")
        nc = (ref.get("semantics") or {}).get("not_comparable_to") or []
        self.assertIn("lightop-fused", nc)
        self.assertIn("fused-3pass-triton", nc)

    def test_run_isolated_preserves_all_outputs(self):
        """run_isolated 不许丢弃第 2 份输出——这是「少写一份」藏得住的原因。"""
        src = (CARD / "_common.py").read_text(encoding="utf-8")
        self.assertIn("def _clone_out", src,
                      "缺 _clone_out：tuple 输出会被静默丢弃（2026-11-23 的坑）")
        self.assertIn("isinstance(out, (tuple, list))", src)


if __name__ == "__main__":
    unittest.main()
