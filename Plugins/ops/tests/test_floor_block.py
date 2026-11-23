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


if __name__ == "__main__":
    unittest.main()
