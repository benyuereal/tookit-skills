#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""expect 必须是数值（否则 EV 校准用不了）。

背景：calibrate.py 跑真实数据得出 **0 条可校准**，
根因是 candidates.yaml 的 expect 全是散文（"triton 融合 vs lightop 融合"）。
散文预期还有更深的毛病：**它无法被证伪**——
"能省一次写回" 涨了 3% 算不算兑现？没人说得清。写 `-11%` 就能。
"""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_check_expect", str(REPO / "scripts/check_expect.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestCheckExpect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_selftest_passes(self):
        self.assertEqual(self.m.selftest(), 0)

    def test_real_repo_has_no_prose_pending_expect(self):
        """真实仓库里，所有 pending 候选的 expect 必须可校准。

        **这条会在新加散文 expect 时失败** —— 那是正确的：
        它逼着写候选时就给一个可证伪的数，或者明说"未实测"。
        """
        r = subprocess.run([sys.executable, "-I",
                            str(REPO / "scripts/check_expect.py"), "--all"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0,
                         "有候选的 expect 不可校准：\n%s" % r.stdout)


if __name__ == "__main__":
    unittest.main()
