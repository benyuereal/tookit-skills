#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""design_space 的测试：枚举 → 过滤 → 排序。

核心价值不是「能算出数」，是**被拒的配置必须给出具体理由**。
没有理由的拒绝等于「静默丢弃」，agent 无从知道是形状错了还是 tile 选错了。
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ds", str(REPO / "scripts/design_space.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestDesignSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_selftest_passes(self):
        self.assertEqual(self.m.selftest(), 0)

    def test_enumeration_size(self):
        r = self.m.enumerate_space(8192, 8192, 8192)
        self.assertEqual(r["stats"]["enumerated"], 486,
                         "3×3×3×2×3×3 = 486，枚举数不对")

    def test_has_legal_configs(self):
        r = self.m.enumerate_space(8192, 8192, 8192)
        self.assertGreater(r["stats"]["legal"], 0,
                           "8192³ 一个合法配置都没有，枚举范围有问题")

    def test_every_rejection_has_reason(self):
        """**被拒必须给理由** —— 这是本工具的核心价值。"""
        r = self.m.enumerate_space(8192, 8192, 8192)
        for x in r["rejected"]:
            self.assertTrue(x.get("reason"), "被拒配置没有理由：%s" % x)

    def test_sorted_by_predicted_remaining(self):
        r = self.m.enumerate_space(8192, 8192, 8192)
        rem = [c["predicted_remaining"] for c in r["configs"]]
        self.assertEqual(rem, sorted(rem), "排序不对")

    def test_prediction_not_constant(self):
        """反向：预测值不许是常数，否则模型没起作用。"""
        r = self.m.enumerate_space(8192, 8192, 8192)
        utils = {c["predicted_mfu"] for c in r["configs"]}
        self.assertGreater(len(utils), 1, "所有配置预测同一个 util")

    def test_lds_overflow_has_byte_count(self):
        """反向：LDS 超限的拒绝理由必须含具体字节数，不能只说「超了」。"""
        r = self.m.enumerate_space(8192, 8192, 8192)
        big = [x for x in r["rejected"] if any("LDS" in e for e in x["reason"])]
        self.assertTrue(big, "应有 LDS 超限被拒的")
        self.assertTrue(any("B >" in e for e in big[0]["reason"]),
                        "理由要含字节数：%s" % big[0]["reason"])

    def test_small_shape_rejected_by_grid(self):
        """反向：形状太小时 grid < CU 也要被拒。"""
        r = self.m.enumerate_space(256, 256, 8192)
        grid_rej = [x for x in r["rejected"] if any("grid" in e for e in x["reason"])]
        self.assertTrue(grid_rej, "小形状应有 grid 不足被拒的")

    def test_output_has_model_limit(self):
        r = self.m.enumerate_space(8192, 8192, 8192)
        self.assertIn("model_limit", r)
        self.assertIn("经验式", r["model_limit"])

    def test_no_legal_configs_still_works(self):
        """极端形状下可能 0 合法 —— 不许崩，要能报出来。"""
        r = self.m.enumerate_space(16, 16, 16)
        self.assertEqual(r["stats"]["legal"], 0)
        self.assertEqual(len(r["configs"]), 0)
        self.assertGreater(len(r["rejected"]), 0)


if __name__ == "__main__":
    unittest.main()
