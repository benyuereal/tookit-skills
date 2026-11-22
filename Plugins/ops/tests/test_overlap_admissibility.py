#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重叠结论准入条件（方法类判据）的守卫。

同一个坑踩过两次：
  阶段 3 —— 1GB copy(1.6ms) 配 8192 GEMM(27.5ms)，同机器测出 92% 与 0%
  阶段 4 —— 512MB D2H(17.4ms) 配 routed(0.31ms)，测出 18.4%（噪声）

教训写成 README 不够，必须固化成检查——这跟「unittest 空跑」「lint 没进 CI」
是同一个病：计数器给了，闸门没关。
"""
import importlib.util
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
CASE = REPO / "Plugins/ops/skills/op-optimize/cases/002-moe-pipeline/stage4_shared.py"


def _load():
    spec = importlib.util.spec_from_file_location("_s4_guard", str(CASE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOverlapAdmissibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_rejects_mismatched_window(self):
        """17.4 ms vs 0.31 ms（阶段 4 踩过的真实组合）必须被拒。"""
        with self.assertRaises(SystemExit) as cm:
            self.m.assert_window(17.371, 0.312)
        self.assertEqual(cm.exception.code, 2)

    def test_accepts_comparable_window(self):
        r = self.m.assert_window(1.175, 0.323)
        self.assertGreaterEqual(r, 0.2)

    def test_waiver_requires_all_three(self):
        # 只给一项不够
        for kwargs in ({}, {"waiver_reason": "x"},
                       {"waiver_reason": "x", "noise_argument": "y"}):
            with self.assertRaises(SystemExit):
                self.m.assert_window(17.371, 0.312, **kwargs)
        # 三项齐全才放行
        r = self.m.assert_window(17.371, 0.312, waiver_reason="x",
                                 noise_argument="y", cross_check="z")
        self.assertLess(r, 0.2)

    def test_both_scripts_call_the_guard(self):
        """两个重叠脚本都必须真的调用 assert_window，不能只定义不调用。"""
        for name in ("stage3_overlap.py", "stage4_shared.py"):
            src = (CASE.parent / name).read_text(encoding="utf-8")
            calls = src.count("assert_window(") - src.count("def assert_window(")
            self.assertGreaterEqual(calls, 1,
                                    "%s 定义了 assert_window 但没调用——闸门没关" % name)


if __name__ == "__main__":
    unittest.main()
