#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E39 的反向测试：floor 块必须自洽，否则它只是一个好看的数字。

为什么单独一个文件：E39 是本轮新加的判据，而本仓历史上出现过
「新判据写进去了却从未生效」（validate.py 里同名函数定义两次，Python 静默用后者），
以及「判据太窄/太宽」各一次。新门禁必须自带反向测试，否则等于没加。

每个测试都构造一个**错的** floor 块，确认 E39 报出来；再构造一个对的，确认不报。
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "_val_floor", str(REPO / "Plugins/ops/op-atlas/validate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOOD = {
    "remaining": 0.198,
    "t_floor_us": 202.3,
    "t_actual_us": 242.2,
    "binding": "memory",
    "basis": "268.4 MB / (1680 GB/s × 0.79)",
    "verdict": "structural",
    "denominator_source": "Plugins/ops/op-atlas/bw-ladder.yaml",
}


class TestFloorBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = _load_validator()

    def run_check(self, block):
        errs = []
        self.v._check_floor_block({"floor": block}, "test.yaml",
                                  lambda code, where, msg: errs.append((code, msg)))
        return errs

    def test_good_block_passes(self):
        self.assertEqual(self.run_check(dict(GOOD)), [],
                         "合法的 floor 块不应报错")

    def test_missing_basis_is_an_error(self):
        bad = dict(GOOD)
        bad.pop("basis")
        errs = self.run_check(bad)
        self.assertTrue(any("basis" in m for _, m in errs),
                        "缺 basis 必须报错（这正是「填个数字糊弄」的入口）")

    def test_missing_t_floor_is_an_error(self):
        bad = dict(GOOD)
        bad.pop("t_floor_us")
        self.assertTrue(self.run_check(bad), "缺 t_floor_us 必须报错")

    def test_remaining_must_match_times(self):
        bad = dict(GOOD)
        bad["remaining"] = 0.05          # 与 242.2/202.3-1=0.198 不符
        errs = self.run_check(bad)
        self.assertTrue(any("不符" in m for _, m in errs),
                        "remaining 与 t_actual/t_floor 不一致必须报错（改了耗时没重算）")

    def test_verdict_must_match_remaining(self):
        bad = dict(GOOD)
        bad["remaining"] = 3.97           # +397%，却写 at_floor
        bad["verdict"] = "at_floor"
        errs = self.run_check(bad)
        self.assertTrue(any("verdict" in m for _, m in errs),
                        "verdict 与 remaining 区间不符必须报错")

    def test_below_floor_must_say_below_floor(self):
        blk = dict(GOOD, t_floor_us=43.3, t_actual_us=34.2,
                   remaining=-0.21, verdict="below_floor")
        self.assertEqual(self.run_check(blk), [])

    def test_no_floor_block_is_fine(self):
        errs = []
        self.v._check_floor_block({}, "test.yaml",
                                  lambda c, w, m: errs.append((c, m)))
        self.assertEqual(errs, [], "没写 floor 的卡不该被 E39 罚款（是 C3 覆盖率的事）")

    def test_unknown_field_rejected(self):
        bad = dict(GOOD, magic=1)
        self.assertTrue(self.run_check(bad), "floor 未知字段必须报错")


class TestCeilingOverrun(unittest.TestCase):
    """E40：算出来的带宽/算力不许超过本机 achievable 的 102%。

    这一组是「判据本身要被判据」的产物——第一版写完看起来对，
    跑反向测试才发现两个真 bug：
      ① 用 bf16 的 279 去卡 fp8 的 498.5 → fp8 卡被误判成错；
      ② 改完后又用 fp8 的 579.7 去放行 bf16 的 400 → bf16 超标蒙混过关。
    所以这里两类精度各测「合法」与「超标」两侧，不许只测一边。
    """

    @classmethod
    def setUpClass(cls):
        cls.v = _load_validator()

    def peaks(self):
        return self.v._achievable_peaks(REPO)

    def run_check(self, floor):
        errs = []
        self.v._check_floor_block({"floor": floor}, "t.yaml",
                                  lambda c, w, m: errs.append((c, m)), self.peaks())
        return [c for c, _ in errs]

    BASE = {"remaining": 0.05, "t_floor_us": 100.0, "t_actual_us": 105.0,
            "basis": "x", "verdict": "at_floor", "binding": "memory"}

    def test_nominal_is_not_a_cap(self):
        """1100 标称绝不能被当成 achievable 上限。"""
        peaks = self.peaks()
        self.assertNotIn(1100.0, peaks["bw"],
                         "bw-ladder 的 1_nominal(1100) 不许进 achievable 集合")

    def test_bandwidth_within_cap_passes(self):
        self.assertEqual(
            self.run_check(dict(self.BASE, achieved_gbs=1600.0)), [])

    def test_bandwidth_over_cap_is_error(self):
        self.assertEqual(
            self.run_check(dict(self.BASE, achieved_gbs=2100.0)), ["E40"])

    def test_bandwidth_inside_tolerance_passes(self):
        """1965×1.02=2004，落在容差内不算错（sclk 档位波动是 2%）。"""
        self.assertEqual(
            self.run_check(dict(self.BASE, achieved_gbs=2000.0)), [])

    def test_bf16_compute_over_cap_is_error(self):
        self.assertEqual(
            self.run_check(dict(self.BASE, achieved_tf=400.0, precision="bf16")),
            ["E40"], "bf16 400 TF 超过 279 的 102%，必须报错")

    def test_fp8_is_not_judged_by_bf16_cap(self):
        """498.5 TF 是 fp8 的正常值，不许被 bf16 的 279 卡住。"""
        self.assertEqual(
            self.run_check(dict(self.BASE, achieved_tf=498.5, precision="fp8")), [])

    def test_fp8_over_cap_is_error(self):
        self.assertEqual(
            self.run_check(dict(self.BASE, achieved_tf=700.0, precision="fp8")),
            ["E40"], "fp8 700 TF 超过 579.7 的 102%，必须报错")


if __name__ == "__main__":
    unittest.main()
