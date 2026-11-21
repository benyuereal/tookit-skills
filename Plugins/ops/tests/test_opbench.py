#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opbench.py 的单测：--selftest 通过 + median/MAD 计算正确性 + 显著性/地板门。

被测脚本：<repo>/scripts/opbench.py（共享脚本，本目录只测不 workload）。
无卡环境可跑：统计与判据是纯函数；只有「GPU event 计时口径」一条用例在
无 DCU/torch 时显式 skip（原因见 skipUnless 文案）。
"""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


opbench = _load("opbench_under_test", REPO / "scripts" / "opbench.py")


class TestSelftest(unittest.TestCase):
    def test_selftest_passes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = opbench.selftest()
        out = buf.getvalue()
        self.assertEqual(rc, 0, out)
        self.assertIn("selftest PASS", out)

    def test_cli_selftest_passes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = opbench.main(["--selftest"])
        self.assertEqual(rc, 0)
        self.assertIn("selftest PASS", buf.getvalue())


class TestSummarize(unittest.TestCase):
    def test_median_and_mad_odd_count(self):
        s = opbench.summarize([1.0, 1.1, 0.9, 1.0, 1.2])
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["median_ms"], 1.0, places=12)
        self.assertAlmostEqual(s["mad_ms"], 0.1, places=12)
        self.assertAlmostEqual(s["min_ms"], 0.9, places=12)
        self.assertAlmostEqual(s["max_ms"], 1.2, places=12)
        self.assertAlmostEqual(s["mean_ms"], 1.04, places=12)

    def test_median_and_mad_even_count(self):
        # median([1,2,3,4]) = 2.5；偏差 [1.5,0.5,0.5,1.5] 的中位数 = 1.0
        s = opbench.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(s["median_ms"], 2.5, places=12)
        self.assertAlmostEqual(s["mad_ms"], 1.0, places=12)

    def test_single_sample_has_zero_mad(self):
        s = opbench.summarize([5.0])
        self.assertEqual(s["n"], 1)
        self.assertAlmostEqual(s["median_ms"], 5.0)
        self.assertEqual(s["mad_ms"], 0.0)

    def test_constant_samples_are_bit_stable(self):
        s = opbench.summarize([0.842] * 7)
        self.assertAlmostEqual(s["median_ms"], 0.842, places=12)
        self.assertEqual(s["mad_ms"], 0.0)

    def test_none_values_are_dropped(self):
        s = opbench.summarize([1.0, None, 3.0])
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["median_ms"], 2.0)

    def test_empty_input_is_not_a_number(self):
        s = opbench.summarize([], unit="us")
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["median_us"])
        self.assertIsNone(s["mad_us"])

    def test_mad_is_not_standard_deviation(self):
        # 一个离群点把样本拉开，但 MAD 仍由中位数附近决定：devs=[0,0,0,9] -> 0
        s = opbench.summarize([1.0, 1.0, 1.0, 10.0])
        self.assertAlmostEqual(s["median_ms"], 1.0)
        self.assertEqual(s["mad_ms"], 0.0)


class TestSignificant(unittest.TestCase):
    def setUp(self):
        self.base = {"median_ms": 10.0, "mad_ms": 0.2}

    def test_faster_beyond_noise_passes(self):
        ok, ratio, why = opbench.significant(self.base, {"median_ms": 8.0, "mad_ms": 0.2})
        self.assertTrue(ok, why)
        self.assertAlmostEqual(ratio, 0.8, places=12)
        self.assertIn("显著", why)

    def test_delta_equal_to_noise_is_inconclusive(self):
        # |Δ| = 3 × MAD = 0.6，判据是「> k×MAD」，等号仍算噪声内
        ok, ratio, why = opbench.significant(self.base, {"median_ms": 9.4, "mad_ms": 0.2})
        self.assertFalse(ok, why)
        self.assertIn("噪声内", why)
        self.assertAlmostEqual(ratio, 0.94, places=12)

    def test_slower_beyond_noise_does_not_pass(self):
        ok, ratio, why = opbench.significant(self.base, {"median_ms": 11.0, "mad_ms": 0.2})
        self.assertFalse(ok, why)
        self.assertGreater(ratio, 1.0)

    def test_k_is_configurable(self):
        cand = {"median_ms": 9.4, "mad_ms": 0.2}
        self.assertFalse(opbench.significant(self.base, cand, k=3.0)[0])
        self.assertTrue(opbench.significant(self.base, cand, k=2.0)[0])

    def test_lower_is_better_false_for_throughput(self):
        ok, ratio, _ = opbench.significant(100.0, 150.0, lower_is_better=False)
        self.assertTrue(ok)
        self.assertAlmostEqual(ratio, 1.5)
        ok2, _, _ = opbench.significant(100.0, 80.0, lower_is_better=False)
        self.assertFalse(ok2)

    def test_plain_floats_have_zero_mad(self):
        ok, ratio, why = opbench.significant(10.0, 9.0)
        self.assertTrue(ok, why)
        self.assertAlmostEqual(ratio, 0.9)

    def test_missing_value_cannot_pass(self):
        ok, ratio, why = opbench.significant({"median_ms": None}, {"median_ms": 8.0})
        self.assertFalse(ok)
        self.assertIsNone(ratio)
        self.assertIn("缺少数值", why)

    def test_accepts_summarize_output(self):
        s = opbench.summarize([8.0, 8.1, 7.9])
        ok, ratio, why = opbench.significant(opbench.summarize([10.0, 10.1, 9.9]), s)
        self.assertTrue(ok, why)
        self.assertAlmostEqual(ratio, 0.8, places=6)


class TestFloorAndMetric(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        status, ratio = opbench.floor_status(1020, 1200)
        self.assertEqual(status, "at_floor")
        self.assertAlmostEqual(ratio, 0.85, places=12)

    def test_below_threshold_has_headroom(self):
        status, ratio = opbench.floor_status(1019, 1200)
        self.assertEqual(status, "headroom")
        self.assertLess(ratio, 0.85)

    def test_zero_peak_is_unknown(self):
        self.assertEqual(opbench.floor_status(1.0, 0), ("unknown", 0.0))

    def test_custom_threshold(self):
        self.assertEqual(opbench.floor_status(900, 1000, threshold=0.9)[0], "at_floor")
        self.assertEqual(opbench.floor_status(899, 1000, threshold=0.9)[0], "headroom")

    def test_emit_metric_format(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            line = opbench.emit_metric(op="demo", variant="v1", shape="s1", ms=1.2345, status="ok")
        self.assertTrue(line.startswith("METRIC "))
        self.assertIn("ms=1.2345", line)
        self.assertIn("status=ok", line)
        self.assertEqual(buf.getvalue().strip(), line)


class TestBenchFn(unittest.TestCase):
    def test_bench_fn_returns_structure_on_cpu(self):
        s = opbench.bench_fn(lambda: sum(range(1000)), warmup_ms=2, rep_ms=5,
                             repeats=2, min_iters=1)
        self.assertEqual(s["n"], 2)
        self.assertIsNotNone(s["median_ms"])
        self.assertGreaterEqual(s["median_ms"], 0.0)
        self.assertIn(s["caliber"]["timer"], ("wall_clock", "cuda_event"))
        self.assertEqual(s["caliber"]["state"], "hot")

    @unittest.skipUnless(
        opbench._cuda_available(),
        "无可用 DCU/GPU（torch 未安装或 torch.cuda.is_available() 为 False）："
        "无法验证 bench_fn 走 CUDA/HIP event 计时口径；统计与判据部分已由其它用例覆盖",
    )
    def test_bench_fn_uses_cuda_event_on_gpu(self):
        s = opbench.bench_fn(lambda: None, warmup_ms=5, rep_ms=10, repeats=2, min_iters=2)
        self.assertEqual(s["caliber"]["timer"], "cuda_event")
        self.assertEqual(s["n"], 2)


if __name__ == "__main__":
    unittest.main()
