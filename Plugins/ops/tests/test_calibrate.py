#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/calibrate.py 的口径测试。

calibrate 是说「expected_effect 有没有兑现」的唯一入口，它能骗人的地方有五处，
本文件逐条钉住：

  A. 空数据 → 「无记录」退出 0
  B. 数值解析：百分数与倍数两种口径都归一到「改善比例」，符号不能反
  C. 不可信记录（verdict=noise / falsified=null）不计入，且原因可查
  D. expected_effect 是散文 → 进 unparsable，不许猜数、不许静默丢
  E. ratio = realized / predicted；n 按 surface 分开数

反向用例（C/D/E 的一部分）与正向用例同等重要：一个只验证「能算出数」的测试，
挡不住「把噪声当证据」这类最贵的错。
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts/calibrate.py"


def _load():
    spec = importlib.util.spec_from_file_location("_calibrate", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(**kw):
    """一条 loop 记录的默认形状：预测 -10%，实际 base 1.0 → cand 0.95（+5%）。"""
    r = {"ts": "2026-01-01T00:00:00", "card": "op-rmsnorm-fused",
         "tag": "v1", "direction": "fusion-boundary",
         "expected_effect": "-10%",
         "base_ms": 1.0, "cand_ms": 0.95,
         "verdict": "significant", "falsified": False}
    r.update(kw)
    return r


def write_loop(lines):
    """lines 里 str 原样写入（造坏行），dict 序列化成 JSON；返回 skill 目录。"""
    d = Path(tempfile.mkdtemp())
    (d / "runs/loop").mkdir(parents=True)
    p = d / "runs/loop/20260101.jsonl"
    p.write_text("\n".join(
        x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
        for x in lines) + "\n", encoding="utf-8")
    return d


def run(*args):
    return subprocess.run([sys.executable, "-I", str(TOOL)] + list(args),
                          capture_output=True, text=True)


class TestParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_percent_negative(self):
        kind, val = self.m.parse_expected("-11%")
        self.assertEqual(kind, "pct")
        self.assertAlmostEqual(val, -0.11, places=12)
        # 改善比例：快 11%
        self.assertAlmostEqual(self.m.expected_improvement(kind, val), 0.11, places=12)

    def test_multiplier(self):
        kind, val = self.m.parse_expected("1.15x")
        self.assertEqual(kind, "x")
        self.assertAlmostEqual(val, 1.15, places=12)
        # 1.15x 速度 → 改善 +15%，不是 +1.15
        self.assertAlmostEqual(self.m.expected_improvement(kind, val), 0.15, places=12)

    def test_unicode_minus_and_fullwidth(self):
        self.assertEqual(self.m.parse_expected("−17.5%")[1] * 100, -17.5)
        self.assertEqual(self.m.parse_expected("1.2×")[1], 1.2)

    def test_prose_is_unparsable(self):
        # 反向：纯文字 → 不猜数
        self.assertEqual(self.m.parse_expected("triton 融合 vs lightop 融合（同语义 2读2写）"),
                         (None, None))
        self.assertEqual(self.m.parse_expected("约 10%"), (None, None))
        self.assertEqual(self.m.parse_expected("−11.7~−11.9%"), (None, None))

    def test_nonpositive_multiplier_is_bad(self):
        self.assertEqual(self.m.parse_expected("0x"), (None, None))
        self.assertEqual(self.m.parse_expected("-1.2x"), (None, None))


class TestCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def cal(self, *lines, **kw):
        return self.m.calibrate_records([x for x in lines], **kw)

    # ---- A 空数据
    def test_empty_has_no_rows(self):
        out = self.cal()
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["records"], 0)

    def test_empty_skill_dir_exit_zero_says_no_records(self):
        d = Path(tempfile.mkdtemp())  # 没有 runs/loop
        r = run("--skill", str(d))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("无记录", r.stdout)

    # ---- E ratio / n
    def test_ratio_three_records(self):
        # 预测 10%，实际 5% → ratio 0.5
        out = self.cal(rec(), rec(tag="v2"), rec(tag="v3"))
        row = out["rows"][0]
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["predicted_med"], 0.10, places=12)
        self.assertAlmostEqual(row["realized_med"], 0.05, places=12)
        self.assertAlmostEqual(row["ratio"], 0.5, places=12)
        self.assertEqual(out["counted"], 3)

    def test_n_counted_per_surface(self):
        out = self.cal(rec(), rec(tag="v2"),
                       rec(direction="occupancy", tag="o1"),
                       rec(direction="occupancy", tag="o2"))
        got = {r["surface"]: r["n"] for r in out["rows"]}
        self.assertEqual(got, {"fusion-boundary": 2, "occupancy": 2})
        self.assertEqual(out["counted"], 4)

    def test_single_sample_insufficient_no_ratio(self):
        out = self.cal(rec())
        row = out["rows"][0]
        self.assertTrue(row["insufficient"])
        self.assertIsNone(row["ratio"])
        self.assertEqual(row["n"], 1)

    def test_realized_uses_cand_over_base(self):
        # cand > base → 变慢，realized 为负
        out = self.cal(rec(cand_ms=1.1), rec(tag="v2", cand_ms=1.1))
        self.assertAlmostEqual(out["rows"][0]["realized_med"], -0.1, places=12)

    # ---- B 倍数口径参与聚合
    def test_multiplier_aggregated_as_improvement(self):
        out = self.cal(rec(expected_effect="1.15x", cand_ms=0.85),
                       rec(tag="v2", expected_effect="1.15x", cand_ms=0.85))
        row = out["rows"][0]
        self.assertAlmostEqual(row["predicted_med"], 0.15, places=12)
        self.assertAlmostEqual(row["realized_med"], 0.15, places=12)
        self.assertAlmostEqual(row["ratio"], 1.0, places=12)

    # ---- C 不可信记录
    def test_noise_excluded(self):
        # 反向：verdict=noise 不计入
        out = self.cal(rec(verdict="noise"))
        self.assertEqual(out["counted"], 0)
        self.assertEqual(out["rows"], [])
        self.assertTrue(any(s["reason"] == "verdict_noise" for s in out["skipped"]))

    def test_decision_noise_excluded(self):
        # loop.py 实际写的是 decision，不是 verdict
        out = self.cal({"direction": "fusion-boundary", "expected_effect": "-10%",
                        "base_ms": 1.0, "cand_ms": 0.95,
                        "decision": "INCONCLUSIVE", "falsified": None})
        self.assertEqual(out["counted"], 0)
        self.assertTrue(any(s["reason"] == "falsified_null" for s in out["skipped"]))

    def test_falsified_null_excluded(self):
        # 反向：falsified=null 不计入
        out = self.cal(rec(), rec(tag="ok"), rec(tag="f", falsified=None))
        self.assertEqual(out["counted"], 2)
        self.assertEqual(out["rows"][0]["n"], 2)
        self.assertTrue(any(s["reason"] == "falsified_null" for s in out["skipped"]))

    # ---- D 散文 → unparsable，不进中位数
    def test_prose_goes_unparsable_not_median(self):
        # 反向：一条可解析 + 一条散文 → n 只数可解析那条；散文单列
        out = self.cal(rec(), rec(tag="p", expected_effect="triton 融合 vs lightop 融合"),
                       rec(tag="v2"))
        self.assertEqual(out["counted"], 2)
        self.assertEqual(out["rows"][0]["n"], 2)
        self.assertEqual(len(out["unparsable"]), 1)
        self.assertEqual(out["unparsable"][0]["tag"], "p")
        self.assertIn("triton", out["unparsable"][0]["expected_effect"])

    def test_all_prose_yields_no_rows_but_reports(self):
        out = self.cal(rec(expected_effect="纯文字描述"))
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["counted"], 0)
        self.assertEqual(len(out["unparsable"]), 1)

    # ---- 缺 metrics → 跳过并 warning
    def test_missing_metrics_skipped_with_warning(self):
        out = self.cal(rec(cand_ms=None))
        self.assertEqual(out["counted"], 0)
        self.assertTrue(out["warnings"])
        self.assertTrue(any(s["reason"] == "metrics_missing" for s in out["skipped"]))

    def test_zero_base_skipped(self):
        out = self.cal(rec(base_ms=0))
        self.assertEqual(out["counted"], 0)
        self.assertTrue(any(s["reason"] == "metrics_missing" for s in out["skipped"]))

    def test_direction_falls_back_to_trigger_pattern(self):
        out = self.cal(rec(direction=None,
                           trigger_pattern={"direction": "vmem-vector-width"}),
                       rec(tag="v2", direction=None,
                           trigger_pattern={"direction": "vmem-vector-width"}))
        self.assertEqual(out["rows"][0]["surface"], "vmem-vector-width")


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_json_output_shape(self):
        d = write_loop([rec(), rec(tag="v2"), rec(tag="v3")])
        r = run("--skill", str(d), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        payload = json.loads(r.stdout)
        self.assertIn("surfaces", payload)
        self.assertEqual(payload["totals"]["counted"], 3)
        row = payload["surfaces"][0]
        self.assertEqual(row["surface"], "fusion-boundary")
        self.assertAlmostEqual(row["ratio"], 0.5, places=12)

    def test_bad_json_line_skipped_with_warning(self):
        d = write_loop([rec(), rec(tag="v2"), "{not json"])
        r = run("--skill", str(d), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        payload = json.loads(r.stdout)
        self.assertEqual(payload["totals"]["counted"], 2)
        self.assertTrue(payload["warnings"])

    def test_human_table_has_columns(self):
        d = write_loop([rec(), rec(tag="v2")])
        r = run("--skill", str(d))
        self.assertEqual(r.returncode, 0, r.stderr)
        for col in ("surface", "n", "predicted_med", "realized_med", "ratio"):
            self.assertIn(col, r.stdout)
        self.assertIn("fusion-boundary", r.stdout)

    def test_requires_target(self):
        r = run()
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
