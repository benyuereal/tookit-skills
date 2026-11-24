#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""matmul tile 模型的测试。

关键约束：**这个模型必须能被实测检验**，不是自洽的数字游戏。
唯一的外部锚点是 hbird-gemm 的同机实测（/workspace/hbird-gemm/docs/DESIGN.md）：
  ISA floor           1.6017 ms = 686.5 TF
  我们最优结构 256×256×64  1.6708 ms = 658.1 TF （= floor 的 96%）
  hipBLASLt            1.8931 ms = 581 TF
本测试用这三个点回测。
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "Plugins/ops/skills/op-roofline/models/matmul.py"


def _load():
    spec = importlib.util.spec_from_file_location("_mm", str(MODEL))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestModelSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_model_spec_contract(self):
        """derive.py 靠 MODEL_SPEC 动态分发，字段必须齐。"""
        spec = self.m.MODEL_SPEC
        for k in ("fn", "args", "peak", "bound"):
            self.assertIn(k, spec, "MODEL_SPEC 缺 %s" % k)
        self.assertEqual(spec["fn"], "analyze")

    def test_args_match_signature(self):
        import inspect
        sig = inspect.signature(self.m.analyze)
        for a in self.m.MODEL_SPEC["args"]:
            self.assertIn(a, sig.parameters, "MODEL_SPEC.args 里的 %r 不在 analyze 签名里" % a)


class TestFloor(unittest.TestCase):
    """地板必须与 constants.yaml 的分母自洽。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_8192_floor_is_compute_bound(self):
        r = self.m.analyze(8192, 8192, 8192)
        self.assertEqual(r["bound"], "compute",
                         "8192³ 的 AI=2731 ≫ ridge，必然 compute-bound")
        # 2*8192^3 / 279e12 = 3941 µs
        self.assertAlmostEqual(r["t_floor_us"], 3941.0, delta=3941 * 0.05)

    def test_ai_matches_template(self):
        """与 templates.py 的 matmul 模板必须给同一个 AI。"""
        r = self.m.analyze(8192, 8192, 8192)
        # templates: ai = 2*N^3 / (3*N^2*2) = N/3 = 2730.67
        self.assertAlmostEqual(r["ai"], 8192 / 3, delta=1.0)


class TestResources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_bm_doubles_accumulator(self):
        """BM 翻倍 → 累加器翻倍 → VGPR 涨。"""
        a = self.m._resources(128, 128, 64, 8, 2, 2)
        b = self.m._resources(256, 128, 64, 8, 2, 2)
        self.assertGreater(b["vgpr"], a["vgpr"])

    def test_bk_doubles_lds(self):
        a = self.m._resources(128, 128, 64, 8, 2, 2)
        b = self.m._resources(128, 128, 128, 8, 2, 2)
        self.assertEqual(b["lds"], a["lds"] * 2, "BK 翻倍 LDS 必须翻倍")

    def test_stages_increase_lds(self):
        a = self.m._resources(128, 128, 64, 8, 2, 2)
        b = self.m._resources(128, 128, 64, 8, 3, 2)
        self.assertGreater(b["lds"], a["lds"])

    def test_lds_rows_independent_of_accumulator(self):
        """**关键**：lds_rows 只影响 LDS，不影响 VGPR。

        hbird 的 256×256 tile 用 32 行子块搬 A，LDS 只 8KB，
        但累加器仍是 128 VGPR。第一版把整块 bm 传进 LDS 公式，
        LDS 腿被压到 1 CTA/CU —— 比真实值严 4 倍。
        """
        a = self.m._resources(256, 256, 64, 8, 2, 2)
        b = self.m._resources(256, 256, 64, 8, 2, 2, lds_rows=32)
        self.assertEqual(a["vgpr"], b["vgpr"],
                         "改 lds_rows 不许影响 VGPR —— 累加器只由 bm*bn 决定")
        self.assertLess(b["lds"], a["lds"], "lds_rows 变小 LDS 应变小")


class TestLegality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_lds_overflow_rejected(self):
        r = self.m.analyze(8192, 8192, 8192, bm=512, bn=512, bk=64,
                           warps=8, stages=4, b_via_vgpr=False)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("LDS" in e for e in r["reject_reason"]))

    def test_k_not_divisible_rejected(self):
        r = self.m.analyze(8192, 8192, 8192, bk=48)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("整除" in e for e in r["reject_reason"]))

    def test_bk_not_16_multiple_rejected(self):
        r = self.m.analyze(8192, 8192, 8192, bk=24)
        self.assertTrue(r.get("illegal"), r)

    def test_grid_too_small_rejected(self):
        # M=N=256, bm=bn=256 → grid=1 < 64
        r = self.m.analyze(256, 256, 8192, bm=256, bn=256, bk=64)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("grid" in e for e in r["reject_reason"]))

    def test_illegal_has_no_prediction(self):
        """illegal 配置不许硬给一个 predicted_us。"""
        r = self.m.analyze(8192, 8192, 8192, bm=512, bn=512, bk=64, stages=4)
        self.assertNotIn("predicted_us", r)


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def _ok(self, **kw):
        base = dict(m=8192, n=8192, k=8192, bm=128, bn=128, bk=64,
                    warps=8, stages=2)
        base.update(kw)
        return self.m.analyze(**base)

    def test_prediction_not_constant(self):
        """不同配置必须给不同预测 —— 否则模型没在算。"""
        vals = {self._ok(bm=bm, bn=bn, warps=w).get("predicted_mfu")
                for bm, bn, w in ((128, 128, 8), (256, 256, 8), (64, 64, 4))}
        self.assertGreater(len(vals), 1, "预测值是常数，模型没起作用：%s" % vals)

    def test_hbird_anchor_256x256(self):
        """**最关键的验收**：用 hbird 的实测配置回测。

        hbird 实测：256×256×64 tile + 32 行子块 → 2 CTAs/CU，658.1 TF
        ⇒ util = 658.1/686.5 = 0.958（对 ISA floor）
        但我们的分母是 achievable 279 TF（bf16）而不是 686.5（fp8 ISA floor），
        所以这里只验**驻留数**与**预测在合理区间**。
        """
        r = self._ok(bm=256, bn=256, lds_sub_rows=32)
        self.assertEqual(r["resources"]["ctas_per_cu"], 2,
                         "hbird 实测 2 CTAs/CU，模型必须一致")
        self.assertGreater(r["predicted_mfu"], 0.8,
                           "256×256 是 hbird 的最优 tile，预测不该低于 0.8")

    def test_small_tile_lower_than_big(self):
        """小 tile 的预测不该高于大 tile（驻留多但单块 work 少）。"""
        small = self._ok(bm=64, bn=64, warps=4, lds_sub_rows=32)
        big = self._ok(bm=256, bn=256, lds_sub_rows=32)
        self.assertLessEqual(small["predicted_mfu"], big["predicted_mfu"] + 0.01)

    def test_output_declares_model_limit(self):
        """每个预测必须带 model_limit —— 它是经验式，不是物理定律。"""
        r = self._ok()
        self.assertIn("model_limit", r)
        self.assertIn("经验式", r["model_limit"])
        self.assertIn(r["confidence"], ("low", "medium", "high"))


if __name__ == "__main__":
    unittest.main()
