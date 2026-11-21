#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主张门（claim_gate）的行为守卫 —— 「薄约束」的核心。

只有一条原则：**约束面 = 断言面，不是动作面。**
  - run 永不拒绝：什么实验都能跑，包括 --no-acc 快速筛方向
  - report / verify / adopt 才看门：记录不够格就不能升级成交付结论

三条硬门各封一个谎言：
  caliber_incomplete  封「分母漂移」
  not_significant     封「跑两次看着差不多」
  accuracy_missing    封「快了但算错了」
"""
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "Plugins/ops/skills/op-optimize/loop.py"

GOOD_CALIBER = {"what": "op_call", "alloc": "false", "state": "hot",
                "shape": "s128k32768", "device": "gfx938"}


def _load_loop():
    spec = importlib.util.spec_from_file_location("_loop_gate", str(LOOP))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_loop_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rec(**over):
    r = {"ts": "2026-11-22T00:00:00", "card": "tmp", "tag": "t", "skv": 32768,
         "base_ms": 0.78, "base_mad": 0.004, "cand_ms": 0.69, "mad": 0.003,
         "delta_pct": -11.7, "decision": "ACCEPTED_BITEXACT", "why": "显著",
         "caliber": dict(GOOD_CALIBER), "significant": True, "acc_ok": True}
    r.update(over)
    return r


class TestClaimGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = _load_loop()

    def test_all_three_gates_pass(self):
        ok, reasons = self.loop.claim_gate(_rec())
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_missing_caliber_blocks(self):
        for key in ("what", "alloc", "state", "shape", "device"):
            cal = dict(GOOD_CALIBER)
            cal.pop(key)
            ok, reasons = self.loop.claim_gate(_rec(caliber=cal))
            self.assertFalse(ok, "缺 %s 应该被拦" % key)
            self.assertIn("caliber_incomplete", reasons)

    def test_unstated_caliber_blocks(self):
        cal = dict(GOOD_CALIBER, state="unstated")
        ok, reasons = self.loop.claim_gate(_rec(caliber=cal))
        self.assertFalse(ok)
        self.assertIn("caliber_incomplete", reasons)

    def test_insignificant_blocks(self):
        ok, reasons = self.loop.claim_gate(_rec(significant=False))
        self.assertFalse(ok)
        self.assertIn("not_significant", reasons)

    def test_missing_accuracy_blocks(self):
        ok, reasons = self.loop.claim_gate(_rec(acc_ok=False))
        self.assertFalse(ok)
        self.assertIn("accuracy_missing", reasons)

    def test_no_acc_flag_caps_verdict_at_held(self):
        """--no-acc 跑出来的记录永远过不了主张门（比较可以跑，主张不许下）。"""
        ok, reasons = self.loop.claim_gate(_rec(acc_ok=False))
        self.assertFalse(ok)
        self.assertEqual(reasons, ["accuracy_missing"])

    def test_multiple_failures_all_reported(self):
        ok, reasons = self.loop.claim_gate(
            _rec(caliber={}, significant=False, acc_ok=False))
        self.assertFalse(ok)
        self.assertEqual(sorted(reasons),
                         ["accuracy_missing", "caliber_incomplete", "not_significant"])


class TestReportHeld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = _load_loop()

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.card = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _report(self, all_flag):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.loop.cmd_report(SimpleNamespace(card=self.card, all=all_flag))
        return buf.getvalue()

    def test_held_records_hidden_by_default(self):
        self.loop.append_record(self.card, _rec(acc_ok=False))
        out = self._report(all_flag=False)
        self.assertNotIn("maxn32", out)
        self.assertIn("未达主张门槛", out)
        self.assertIn("accuracy_missing", out)

    def test_all_flag_shows_held_with_reasons(self):
        self.loop.append_record(self.card, _rec(tag="maxn32", acc_ok=False))
        out = self._report(all_flag=True)
        self.assertIn("maxn32", out)
        self.assertIn("held:", out)

    def test_qualifying_record_renders_normally(self):
        self.loop.append_record(self.card, _rec(tag="maxn32"))
        out = self._report(all_flag=False)
        self.assertIn("maxn32", out)
        self.assertNotIn("未达主张门槛", out)


if __name__ == "__main__":
    unittest.main()


class TestCaliberNormalisation(unittest.TestCase):
    """口径字段位置不统一时的归一：不能因为放错地方就判「口径不全」。"""

    @classmethod
    def setUpClass(cls):
        cls.loop = _load_loop()

    def test_device_falls_back_to_env_fingerprint(self):
        rec = _rec(caliber={"what": "op_call", "alloc": True, "state": "hot"},
                   env={"gfx": "gfx938:sramecc+", "device": "BW1100"},
                   shape={"s": 128, "skv": 32768})
        ok, reasons = self.loop.claim_gate(rec)
        self.assertTrue(ok, reasons)

    def test_shape_falls_back_to_record_shape(self):
        rec = _rec(caliber={"what": "op_call", "alloc": True, "state": "hot",
                            "device": "gfx938"}, shape={"s": 128, "skv": 32768})
        ok, reasons = self.loop.claim_gate(rec)
        self.assertTrue(ok, reasons)

    def test_env_cpu_does_not_count_as_device(self):
        rec = _rec(caliber={"what": "op_call", "alloc": True, "state": "hot"},
                   env={"device": "cpu"}, shape={"s": 1})
        ok, reasons = self.loop.claim_gate(rec)
        self.assertFalse(ok)
        self.assertIn("caliber_incomplete", reasons)

    def test_thermal_alias_still_accepted(self):
        rec = _rec(caliber={"what": "op_call", "alloc": True, "thermal": "hot",
                            "device": "gfx938"}, shape={"s": 1})
        ok, reasons = self.loop.claim_gate(rec)
        self.assertTrue(ok, reasons)

    def test_alloc_false_is_a_valid_statement(self):
        """alloc=False 是「明确不含分配」，不是「没写」。"""
        rec = _rec(caliber={"what": "op_call", "alloc": False, "state": "hot",
                            "device": "gfx938"}, shape={"s": 1})
        ok, reasons = self.loop.claim_gate(rec)
        self.assertTrue(ok, reasons)
