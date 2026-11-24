#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""variant_forge 的测试：参数覆盖必须真的生效，且失败必须报错。

核心纪律来自 `surfaces.yaml` 的教训：**产出「看起来改了但没改」的变体比没有更糟**，
因为它会被当成实验证据。所以每一条 patch 都要：
  - 锚点恰好出现 1 次（0 次或 >1 次都报错）
  - 生成后回读验证新值真的在
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_vf", str(REPO / "scripts/variant_forge.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestVariantForge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()
        cls.src, err = cls.m.load_parent("triton-reduce")
        if err:
            raise unittest.SkipTest("父模板不可用：%s" % err)

    def test_selftest_passes(self):
        self.assertEqual(self.m.selftest(), 0)

    def test_num_warps_override(self):
        new, changes, errs = self.m.apply_patch(self.src, {"num_warps": "8"})
        self.assertEqual(errs, [])
        self.assertIn("num_warps=8", new)
        self.assertNotIn("num_warps=4", new)

    def test_block_override(self):
        new, _, errs = self.m.apply_patch(self.src, {"BLOCK": "2048"})
        self.assertEqual(errs, [])
        self.assertIn("eps, 2048,", new)

    def test_two_patches_differ(self):
        a, _, _ = self.m.apply_patch(self.src, {"num_warps": "8"})
        b, _, _ = self.m.apply_patch(self.src, {"num_warps": "8", "BLOCK": "4096"})
        self.assertNotEqual(a, b, "两个不同 patch 生成相同结果")

    def test_unknown_param_rejected(self):
        """反向：不认识的参数必须报错，不许静默忽略。"""
        _, _, errs = self.m.apply_patch(self.src, {"num_frobs": "3"})
        self.assertTrue(errs)
        self.assertIn("不支持", errs[0])

    def test_out_of_range_rejected(self):
        """反向：num_warps=0 越界。"""
        _, _, errs = self.m.apply_patch(self.src, {"num_warps": "0"})
        self.assertTrue(errs)
        self.assertIn("越界", errs[0])

    def test_missing_anchor_rejected(self):
        """反向：锚点找不到必须报错 —— 这是防「假变体」的核心。"""
        broken = self.src.replace("num_warps=4", "num_warps=XX")
        _, _, errs = self.m.apply_patch(broken, {"num_warps": "8"})
        self.assertTrue(errs, "锚点被改掉后应该报错")
        self.assertIn("0 次", errs[0])

    def test_duplicate_anchor_rejected(self):
        """反向：锚点出现多次会误改别处，必须报错。"""
        dup = self.src + "\n# num_warps=4 (duplicate in comment)\n"
        _, _, errs = self.m.apply_patch(dup, {"num_warps": "8"})
        self.assertTrue(errs, "锚点重复应该报错")
        self.assertIn("2 次", errs[0])

    def test_verify_catches_noop(self):
        """反向：新旧值相同，verify 必须抓到。"""
        _, changes, _ = self.m.apply_patch(self.src, {"num_warps": "4"})
        verrs = self.m.verify(self.src, changes, self.src)
        self.assertTrue(verrs, "新旧值相同时 verify 应该报错")

    def test_generated_code_parses(self):
        import ast
        new, _, _ = self.m.apply_patch(self.src, {"num_warps": "8", "BLOCK": "2048"})
        ast.parse(new)

    def test_diff_shows_changed_line(self):
        new, _, _ = self.m.apply_patch(self.src, {"num_warps": "8"})
        d = self.m.diff_text(self.src, new)
        self.assertIn("num_warps=8", d)
        self.assertIn("-def run", d)


if __name__ == "__main__":
    unittest.main()
