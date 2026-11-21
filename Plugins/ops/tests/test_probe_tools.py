#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三个基础探针的单测：--selftest 通过 + 口径纯函数正确 + 无 GPU 时明确降级。

被测脚本：<repo>/scripts/probe_stream.py、probe_pattern.py、pmc.py。
全部用例不碰 GPU、不调 hipcc，可在无卡环境跑（CI 即无卡）。
"""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(REPO / "scripts" / filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stream = _load("probe_stream_under_test", "probe_stream.py")
pattern = _load("probe_pattern_under_test", "probe_pattern.py")
pmc = _load("pmc_under_test", "pmc.py")


def _run_selftest(mod):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(["--selftest"])
    return rc, buf.getvalue()


class TestSelftests(unittest.TestCase):
    """每个工具都必须有可跑的自检，且无需 GPU。"""

    def test_probe_stream_selftest(self):
        rc, out = _run_selftest(stream)
        self.assertEqual(rc, 0, out)
        self.assertIn("selftest PASS", out)

    def test_probe_pattern_selftest(self):
        rc, out = _run_selftest(pattern)
        self.assertEqual(rc, 0, out)
        self.assertIn("selftest PASS", out)

    def test_pmc_selftest(self):
        rc, out = _run_selftest(pmc)
        self.assertEqual(rc, 0, out)
        self.assertIn("selftest PASS", out)


class TestStreamCalibers(unittest.TestCase):
    """copy 的分母必须是 R+W（2×），不能和 read/write 混。"""

    def test_payload_bytes(self):
        self.assertEqual(stream.payload_bytes("read", 100), 100)
        self.assertEqual(stream.payload_bytes("write", 100), 100)
        self.assertEqual(stream.payload_bytes("copy", 100), 200)

    def test_calibers_are_distinct(self):
        c = {p: stream.caliber(p) for p in stream.PHASES}
        self.assertEqual(len(set(c.values())), 3, c)
        self.assertIn("2×", c["copy"])

    def test_check_uses_max_size(self):
        results = {256: {"read": {"value": 1700.0}}, 1024: {"read": {"value": 1750.0}}}
        checks = stream.check(results, {"memory.stream_read_peak.value": "1724"})
        self.assertEqual(checks[0]["probed"], 1750.0)
        self.assertEqual(checks[0]["probed_size_mib"], 1024)

    def test_check_flags_nominal_as_denominator(self):
        results = {1024: {"read": {"value": 1750.0}}}
        checks = stream.check(results, {"memory.stream_read_peak.value": "1100"})
        self.assertEqual(checks[0]["status"], "warn")


class TestPatternRules(unittest.TestCase):
    def test_runs_must_be_multiple_of_16(self):
        self.assertEqual(pattern.parse_runs(["16", "linear"]), [16, "linear"])
        with self.assertRaises(SystemExit):
            pattern.parse_runs(["24"])

    def test_ceiling_excludes_linear(self):
        fake = {"records": [
            {"run_bytes": 256, "achieved_gbs": 1049.0, "pct_of_linear": 60.0},
            {"run_bytes": "linear", "achieved_gbs": 1870.0, "pct_of_linear": 100.0}]}
        best = pattern.ceiling(fake)
        self.assertEqual(best["run_bytes"], 256)

    def test_no_torch_degrades_without_crashing(self):
        # 模拟无 torch：main 必须明确报「无法探测」并退 2，而不是抛栈或给默认数
        orig = pattern._torch
        pattern._torch = lambda: (None, "模拟：没有 torch")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = pattern.main(["--total-mib", "64", "--runs", "256"])
        finally:
            pattern._torch = orig
        self.assertEqual(rc, 2)
        self.assertIn("无法探测", buf.getvalue())


class TestPmcRules(unittest.TestCase):
    """四条硬规则：稳态行、排噪声、跨实例求和、常数不写死。"""

    CSV = ("Index,KernelName,GRBM_COUNT,BeginNs,EndNs,TA_TA_BUSY[0],TA_TA_BUSY[1]\n"
           "0,_ZN2at6native3fooEii,900,0,1000,,\n"
           "1,main_kernel<S=128>,1400000,0,9000,10,7\n"
           "2,main_kernel<S=128>,1000000,0,12000,11,9\n"
           "3,main_kernel<S=128>,1010000,0,11000,12,8\n")

    def test_steady_row_is_min_grbm_not_first(self):
        res = pmc.parse(self.CSV)
        self.assertTrue(res["kernel"].startswith("main_kernel"))
        self.assertEqual(res["launches"], 3)
        self.assertEqual(res["row_index"], "2")

    def test_noise_excluded(self):
        res = pmc.parse(self.CSV)
        self.assertIn("_ZN2at6native3fooEii", res["excluded_noise"])

    def test_ta_instances_are_summed(self):
        res = pmc.parse(self.CSV)
        self.assertEqual(res["counters"]["TA_TA_BUSY"]["value"], 20.0)  # 11+9

    def test_cu_not_hardcoded(self):
        self.assertTrue(str(pmc.parse(self.CSV)["cu"]).startswith("unknown("))
        self.assertEqual(pmc.parse(self.CSV, cu=64)["cu"], 64)

    def test_unparseable_input_errors(self):
        self.assertIn("error", pmc.parse("Index,KernelName\n0,foo\n"))
        self.assertIn("error", pmc.parse(self.CSV, kernel="nope"))


if __name__ == "__main__":
    unittest.main()
