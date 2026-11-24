#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reduce-lastdim tile 模型的测试。

关键约束：**这个模型必须能被实测检验**，不是自洽的数字游戏。
唯一的外部锚点是 op-rmsnorm-fused 的同机实测
（skills/op-rmsnorm-fused/results/impl-ranking.json + ledger.md 2026-11-24）：

  形状 M=8192 H=4096 bf16，2读2写 = 268435456 B，分母 1680 GB/s ⇒ 159.78 µs
  lightop（shfl）   197.0 µs  → 0.811
  tilelang          221.0 µs  → 0.723
  triton（lds）     242.0 µs  → 0.660

本测试用这三个点回测，并检查三条机制方向（反向断言）。
"""
import importlib.util
import inspect
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "Plugins/ops/skills/op-roofline/models/reduce_lastdim.py"
TEMPLATES = REPO / "Plugins/ops/skills/op-roofline/templates.py"

ANCHOR_SHAPE = dict(M=8192, H=4096, outputs=4)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestModelSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_rl")

    def test_model_spec_contract(self):
        """derive.py 靠 MODEL_SPEC 动态分发，字段必须齐。"""
        spec = self.m.MODEL_SPEC
        for k in ("fn", "args", "peak", "bound"):
            self.assertIn(k, spec, "MODEL_SPEC 缺 %s" % k)
        self.assertEqual(spec["fn"], "analyze")
        self.assertEqual(spec["peak"], "bf16")
        self.assertEqual(spec["bound"], "memory")

    def test_args_match_signature(self):
        sig = inspect.signature(self.m.analyze)
        for a in self.m.MODEL_SPEC["args"]:
            self.assertIn(a, sig.parameters,
                          "MODEL_SPEC.args 里的 %r 不在 analyze 签名里" % a)

    def test_callable_with_positional_args(self):
        """按 MODEL_SPEC.args 顺序位置调用必须能跑通。"""
        args = [8192, 4096, 1, 4, 2, "shfl"]
        r = self.m.analyze(*args)
        self.assertNotIn("TypeError", str(r))


class TestBytes(unittest.TestCase):
    """流量必须与 templates.py 的分档一致。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_rl")
        cls.t = _load(TEMPLATES, "_tl")

    def test_2out_bytes_268435456(self):
        r = self.m.analyze(**ANCHOR_SHAPE)
        self.assertEqual(r["bytes"], 268435456,
                         "M=8192 H=4096 bf16 2读2写 = 4*M*H*2")

    def test_2out_matches_template(self):
        r = self.m.analyze(**ANCHOR_SHAPE)
        t = self.t.evaluate("reduce-lastdim-2out", {"M": 8192, "H": 4096,
                                                    "dtype_bytes": 2})
        self.assertEqual(r["bytes"], t["bytes"],
                         "bytes 必须与 templates.py reduce-lastdim-2out 完全一致")

    def test_1out_matches_template(self):
        r = self.m.analyze(M=8192, H=4096, outputs=2)
        t = self.t.evaluate("reduce-lastdim", {"M": 8192, "H": 4096,
                                              "dtype_bytes": 2})
        self.assertEqual(r["bytes"], t["bytes"],
                         "bytes 必须与 templates.py reduce-lastdim 完全一致")

    def test_t_memory_matches_2out_template(self):
        """t_memory = bytes / 1680 GB/s；与模板算出的 bytes 同一分母 ⇒ ±10%。"""
        r = self.m.analyze(**ANCHOR_SHAPE)
        t = self.t.evaluate("reduce-lastdim-2out", {"M": 8192, "H": 4096,
                                                    "dtype_bytes": 2})
        t_mem = t["bytes"] / 1680e9 * 1e6
        self.assertAlmostEqual(r["t_memory_us"], t_mem, delta=t_mem * 0.10,
                               msg="t_memory 与 templates reduce-lastdim-2out 不一致")
        # 同时锚定绝对值：268435456 / 1680e9 = 159.78 µs
        self.assertAlmostEqual(r["t_memory_us"], 159.8, delta=1.0)


class TestResources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_rl")

    def test_shfl_lds_zero(self):
        r = self.m.analyze(**ANCHOR_SHAPE, reduce_impl="shfl")
        self.assertEqual(r["resources"]["lds"], 0,
                         "shfl 归约全在寄存器，LDS 必须为 0")

    def test_lds_impl_uses_lds(self):
        r = self.m.analyze(**ANCHOR_SHAPE, reduce_impl="lds")
        self.assertEqual(r["resources"]["lds"], 1 * 4096 * 4,
                         "lds 路径 LDS = block_rows*H*4")

    def test_lds_scales_with_block_rows(self):
        a = self.m.analyze(**ANCHOR_SHAPE, reduce_impl="lds", block_rows=1)
        b = self.m.analyze(**ANCHOR_SHAPE, reduce_impl="lds", block_rows=4)
        self.assertEqual(b["resources"]["lds"], a["resources"]["lds"] * 4,
                         "LDS 占用必须与 block_rows 成正比")

    def test_reduce_depth_grows_with_H(self):
        """反向：H=4096 的归约深度必须 > H=512。"""
        deep = self.m.analyze(M=8192, H=4096, outputs=4)
        shallow = self.m.analyze(M=8192, H=512, outputs=4)
        self.assertGreater(deep["resources"]["reduce_depth"],
                           shallow["resources"]["reduce_depth"],
                           "H 越大归约树越深，深度必须单调不减")
        # 具体数：H=4096 warps=4 → ceil(log2(4096/256)) + ceil(log2(4)) = 4+2
        self.assertEqual(deep["resources"]["reduce_depth"], 6)
        # H=512 → ceil(log2(2)) + 2 = 1+2
        self.assertEqual(shallow["resources"]["reduce_depth"], 3)


class TestLegality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_rl")

    def test_block_rows_zero_rejected(self):
        r = self.m.analyze(**ANCHOR_SHAPE, block_rows=0)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("block_rows" in e for e in r["reject_reason"]))

    def test_h_not_power_of_two_rejected(self):
        r = self.m.analyze(M=8192, H=4095, outputs=4)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("2 的幂" in e for e in r["reject_reason"]))

    def test_unknown_reduce_impl_rejected(self):
        r = self.m.analyze(**ANCHOR_SHAPE, reduce_impl="magic")
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("reduce_impl" in e for e in r["reject_reason"]))

    def test_lds_overflow_rejected(self):
        # block_rows=32, H=4096, lds → 32*4096*4 = 512KB > 64KB
        r = self.m.analyze(M=8192, H=4096, outputs=4,
                          block_rows=32, reduce_impl="lds")
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("LDS" in e for e in r["reject_reason"]))

    def test_illegal_has_no_prediction(self):
        r = self.m.analyze(**ANCHOR_SHAPE, block_rows=0)
        self.assertNotIn("predicted_us", r)
        self.assertEqual(r["confidence"], "none")


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_rl")

    def _ok(self, **kw):
        base = dict(**ANCHOR_SHAPE)
        base.update(kw)
        return self.m.analyze(**base)

    def test_three_impls_give_three_predictions(self):
        """三种 reduce_impl 必须给三个不同预测。"""
        vals = {self._ok(reduce_impl=x)["predicted_util"]
                for x in ("shfl", "tilelang", "lds")}
        self.assertEqual(len(vals), 3, "三种实现给了重复预测：%s" % vals)

    def test_anchor_backtest(self):
        """**最关键的验收**：三个实测锚点回测。

        lightop 0.811 / tilelang 0.723 / triton 0.660，误差 ±2%。
        """
        expect = {"shfl": 0.811, "tilelang": 0.723, "lds": 0.660}
        measured_us = {"shfl": 197.0, "tilelang": 221.0, "lds": 242.0}
        for impl, want in expect.items():
            r = self._ok(reduce_impl=impl)
            self.assertAlmostEqual(r["predicted_util"], want, delta=0.02,
                                   msg="%s 锚点回测偏差过大：%s vs %.3f"
                                       % (impl, r["predicted_util"], want))
            self.assertAlmostEqual(r["predicted_us"], measured_us[impl],
                                   delta=measured_us[impl] * 0.05,
                                   msg="%s 的 predicted_us 与实测 %s µs 不符"
                                       % (impl, measured_us[impl]))

    def test_shfl_beats_lds(self):
        """反向：shfl 预测 > lds 预测。"""
        self.assertGreater(self._ok(reduce_impl="shfl")["predicted_util"],
                           self._ok(reduce_impl="lds")["predicted_util"],
                           "寄存器归约必须优于 LDS 树归约")

    def test_shfl_beats_tilelang_beats_lds(self):
        """严格序：shfl > tilelang > lds。"""
        s = self._ok(reduce_impl="shfl")["predicted_util"]
        t = self._ok(reduce_impl="tilelang")["predicted_util"]
        l = self._ok(reduce_impl="lds")["predicted_util"]
        self.assertGreater(s, t)
        self.assertGreater(t, l)

    def test_vec_width_up_raises_prediction(self):
        """反向：vec_width 2→4，预测上升。"""
        v2 = self._ok(vec_width=2)["predicted_util"]
        v4 = self._ok(vec_width=4)["predicted_util"]
        self.assertGreater(v4, v2, "vec_width 2→4 应提升有效带宽")

    def test_vec_width_capped(self):
        """vec_width 超过 4（dwordx4）不再加分。"""
        v4 = self._ok(vec_width=4)["predicted_util"]
        v8 = self._ok(vec_width=8)["predicted_util"]
        self.assertEqual(v4, v8, "vec_width>4 应封顶")

    def test_prediction_not_constant(self):
        """反向：预测不许是常数，否则模型没在算。"""
        vals = {self._ok(reduce_impl=i, vec_width=v, warps=w)["predicted_util"]
                for i, v, w in (("shfl", 2, 4), ("lds", 2, 4),
                                ("shfl", 4, 8), ("tilelang", 4, 2))}
        self.assertGreater(len(vals), 2, "预测值是常数，模型没起作用：%s" % vals)

    def test_output_declares_model_limit_and_confidence(self):
        """每个预测必须带 model_limit + confidence —— 它是经验式，不是物理定律。"""
        r = self._ok()
        self.assertIn("model_limit", r)
        self.assertIn("经验式", r["model_limit"])
        self.assertIn(r["confidence"], ("low", "medium", "high"))
        self.assertIn("predicted_reason", r)

    def test_illegal_output_still_has_model_limit(self):
        r = self.m.analyze(**ANCHOR_SHAPE, block_rows=0)
        self.assertIn("model_limit", r)
        self.assertIn("reject_reason", r)

    def test_upper_bound_respected(self):
        """再好的配置也不能超过 UTIL_CEILING。"""
        r = self._ok(reduce_impl="shfl", vec_width=4, warps=8)
        self.assertLessEqual(r["predicted_util"], self.m.UTIL_CEILING)


if __name__ == "__main__":
    unittest.main()
