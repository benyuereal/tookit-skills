#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floor 口径的回归测试（从 floor.py 的 selftest 搬来，单一数据源仍是 floor.py）。

守的不是「代码能跑」，而是三个被真实事故验证过的口径：
  · 两个地板取慢的（compute vs memory）
  · launch 是加性项（小 kernel 的地板 = 搬运 + 43µs）
  · 归约要打折（0.79）
"""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_floor():
    spec = importlib.util.spec_from_file_location(
        "_floor_caliber", str(REPO / "Plugins/ops/engine/floor.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass 需要模块已注册
    spec.loader.exec_module(mod)
    return mod


class TestFloorCaliber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = _load_floor()

    def test_table_cases(self):
        """数据表里的 7 条单跑用例：一行一个场景。"""
        for c in self.f.caliber_cases():
            with self.subTest(case=c.name):
                r = self.f.analyze(c.given, self.f._fake_ladder())
                for key, want in c.expect.items():
                    if key == "has_error":
                        self.assertIn("error", r, r)
                    elif key == "advice_contains":
                        self.assertIn(want, r.get("advice", ""), r)
                    else:
                        self.assertEqual(r[key], want, (c.name, r))

    def test_launch_is_additive(self):
        self.f._check_launch_additive(self.f._fake_ladder_with_reduction())

    def test_below_launch_floor_reported_honestly(self):
        self.f._check_below_launch(self.f._fake_ladder_with_reduction())

    def test_reduction_discount_widens_floor(self):
        self.f._check_reduction_discount(self.f._fake_ladder_with_reduction())

    def test_optimized_mode_refuses_to_fake_pass(self):
        """`python -O` 会剥掉 assert —— selftest 必须拒绝运行，而不是静默 PASS。"""
        p = subprocess.run(
            [sys.executable, "-O", "-I", str(REPO / "scripts/floor.py"), "--selftest"],
            capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("python -O", p.stdout + p.stderr)
