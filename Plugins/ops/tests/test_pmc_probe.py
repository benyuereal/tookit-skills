#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pmc_probe 的反向测试：三态必须分得清，不许把「没采到」当成 0。

背景：本仓第 7 次踩「校验本身没被校验」——所以新工具的每一条判据，
都要有一个「故意写错，确认它报出来」的用例。
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_pmc_probe", str(REPO / "scripts/pmc_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FULL = """Index,KernelName,grd,wgr,lds,arch_vgpr,sgpr,wave_size,GRBM_COUNT,SQ_INSTS_VALU,SQ_LDS_BANK_CONFLICT
0,k1,1024,256,16384,80,48,64,10000,5000,7
1,k1,1024,256,16384,80,48,64,9000,4500,3
"""

ZERO = """Index,KernelName,GRBM_COUNT,SQ_LDS_BANK_CONFLICT
0,k1,10000,0
"""

MISSING = """Index,KernelName,GRBM_COUNT
0,k1,10000
"""


class TestPmcProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_present_sums_instance_columns(self):
        r = self.m.probe(FULL)
        self.assertEqual(r["counters"]["GRBM_COUNT"]["status"], "present")
        self.assertEqual(r["counters"]["GRBM_COUNT"]["total"], 19000.0)

    def test_all_zero_is_not_zero(self):
        """列存在但全 0 必须是 all_zero，不能是 present(0)。

        这条防的是：把「没采到 / 没启用」读成「没发生」，
        于是得出「bank conflict 没问题」——正是本仓最怕的那种假结论。
        """
        r = self.m.probe(ZERO)
        self.assertEqual(r["counters"]["SQ_LDS_BANK_CONFLICT"]["status"], "all_zero")

    def test_missing_is_missing(self):
        r = self.m.probe(MISSING)
        self.assertEqual(r["counters"]["SQ_LDS_BANK_CONFLICT"]["status"], "missing")
        self.assertEqual(r["counters"]["SQ_WAVES"]["status"], "missing")

    def test_banned_includes_missing_and_all_zero(self):
        usable, banned = self.m.verdict(self.m.probe(ZERO))
        self.assertIn("SQ_LDS_BANK_CONFLICT", banned,
                      "all_zero 不许进判据可用集")
        self.assertIn("GRBM_COUNT", usable)

    def test_unparsable_value_not_silently_zero(self):
        r = self.m.probe("Index,KernelName,GRBM_COUNT\n0,k1,abc\n")
        self.assertEqual(r["counters"]["GRBM_COUNT"]["status"], "unparsable")

    def test_empty_csv_raises(self):
        with self.assertRaises(ValueError):
            self.m.probe("Index,KernelName\n")

    def test_placement_read_from_csv(self):
        """放置条件必须能从 PMC CSV 直取，不必依赖 .so。"""
        r = self.m.probe(FULL)
        self.assertEqual(r["placement"]["arch_vgpr"], 80.0)
        self.assertEqual(r["placement"]["wgr"], 256.0)
        self.assertEqual(r["placement"]["wave_size"], 64.0)

    def test_instance_suffix_columns_are_summed(self):
        """TA_*/TCC_* 是 TA_TA_BUSY[0..15] 这种实例列，必须求和不是取一个。"""
        txt = ("Index,KernelName,TA_TA_BUSY[0],TA_TA_BUSY[1],TA_TA_BUSY[2]\n"
               "0,k1,10,20,30\n")
        r = self.m.probe(txt)
        self.assertEqual(r["counters"]["TA_TA_BUSY"]["status"], "present")
        self.assertEqual(r["counters"]["TA_TA_BUSY"]["total"], 60.0)
        self.assertEqual(r["counters"]["TA_TA_BUSY"]["columns"], 3)


if __name__ == "__main__":
    unittest.main()
