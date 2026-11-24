#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""attention-decode tile 模型的测试。

关键约束：**这个模型必须能被同机实测检验**，不是自洽的数字游戏。
外部锚点全部来自 op-optimize 的两个 case（BW1100 同机实测）：
  case 006 mla-absorb (H=32 D=576 block=64)
      B=1→0.073  B=8→0.259  B=32→0.353
      KV=2048→0.128 8192→0.259 16384→0.312 32768→0.345
  case 008 paged (H=32 D=128 block=64, KV=8192)
      B=1→0.006  B=8→0.021  B=32→0.027  B=64→0.029
本测试用这些点回测，并检查并行度规律（反向）。"""
import importlib.util
import inspect
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "Plugins/ops/skills/op-roofline/models/attention_decode.py"
TEMPLATES = REPO / "Plugins/ops/skills/op-roofline/templates.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestModelSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_ad")

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


class TestAnchors(unittest.TestCase):
    """回测两个 case 的实测点。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_ad")

    def test_batch1_mla(self):
        """B=1 → 0.073（±0.05）。"""
        r = self.m.analyze(8192, 1, 32, 576)
        self.assertAlmostEqual(r["predicted_util"], 0.073, delta=0.05)

    def test_batch32_mla(self):
        """B=32 → 0.353（±0.10）。"""
        r = self.m.analyze(8192, 32, 32, 576)
        self.assertAlmostEqual(r["predicted_util"], 0.353, delta=0.10)

    def test_batch8_mla(self):
        """B=8 → 0.259（±0.05）。"""
        r = self.m.analyze(8192, 8, 32, 576)
        self.assertAlmostEqual(r["predicted_util"], 0.259, delta=0.05)

    def test_kv_sweep_mla(self):
        """KV 扫（B=8, D=576）四个点全部 ±0.05。"""
        for kv, meas in ((2048, 0.128), (8192, 0.259),
                         (16384, 0.312), (32768, 0.345)):
            r = self.m.analyze(kv, 8, 32, 576)
            self.assertAlmostEqual(r["predicted_util"], meas, delta=0.05,
                                   msg="KV=%d 预测 %.3f 偏离实测 %.3f"
                                       % (kv, r["predicted_util"], meas))

    def test_paged_low_ceiling(self):
        """paged D=128 四个点全部 ±0.01 —— 这是 0.03 档的关键。"""
        for b, meas in ((1, 0.006), (8, 0.021), (32, 0.027), (64, 0.029)):
            r = self.m.analyze(8192, b, 32, 128)
            self.assertAlmostEqual(r["predicted_util"], meas, delta=0.01,
                                   msg="B=%d 预测 %.3f 偏离实测 %.3f"
                                       % (b, r["predicted_util"], meas))
        # paged 与 mla 差一个数量级，模型必须表现出来
        paged = self.m.analyze(8192, 32, 32, 128)["predicted_util"]
        mla = self.m.analyze(8192, 32, 32, 576)["predicted_util"]
        self.assertLess(paged * 5, mla, "D=128 与 D=576 的达成率必须差一个量级")


class TestParallelismLaws(unittest.TestCase):
    """**反向**：规律错了模型就没在算。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_ad")

    def test_b1_less_than_b32(self):
        """**反向**：B=1 必须严格小于 B=32（并行度规律）。"""
        lo = self.m.analyze(8192, 1, 32, 576)["predicted_util"]
        hi = self.m.analyze(8192, 32, 32, 576)["predicted_util"]
        self.assertLess(lo, hi, "B=1 预测 %.3f 不小于 B=32 %.3f" % (lo, hi))

    def test_split_raises_util(self):
        """**反向**：num_split 3→1 预测下降（split 增加并行度）。"""
        s3 = self.m.analyze(8192, 8, 32, 576, num_split=3)["predicted_util"]
        s1 = self.m.analyze(8192, 8, 32, 576, num_split=1)["predicted_util"]
        self.assertLess(s1, s3,
                        "num_split 从 3 降到 1 时预测必须下降（%f vs %f）" % (s1, s3))

    def test_prediction_not_constant(self):
        """**反向**：不同配置必须给不同预测 —— 否则模型没在算。"""
        vals = {self.m.analyze(8192, b, 32, d)["predicted_util"]
                for b, d in ((1, 576), (8, 576), (32, 576), (32, 128))}
        self.assertGreater(len(vals), 1, "预测值是常数，模型没起作用：%s" % vals)


class TestCrossCheck(unittest.TestCase):
    """流量必须与 templates.py 的 attention-decode 同口径。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_ad")
        cls.tpl = _load(TEMPLATES, "_tpl")

    def test_bytes_match_template(self):
        """models 的 bytes 与 templates 的 attention-decode 一致（±10%）。"""
        for kv, b, h, d in ((8192, 8, 32, 576), (8192, 32, 32, 128),
                            (2048, 1, 32, 576)):
            mine = self.m.analyze(kv, b, h, d)["bytes"]
            # templates: attention-decode 的 S 就是 kv_len，B=batch，h=heads，D=head_dim
            tpl = self.tpl.evaluate("attention-decode",
                                    {"B": b, "S": kv, "h": h, "D": d,
                                     "dtype_bytes": 2})
            self.assertAlmostEqual(mine, tpl["bytes"], delta=tpl["bytes"] * 0.10,
                                   msg="kv=%d b=%d d=%d: models=%d templates=%d"
                                       % (kv, b, d, mine, tpl["bytes"]))


class TestLegality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_ad")

    def test_block_size_zero_rejected(self):
        r = self.m.analyze(8192, 8, 32, 576, block_size=0)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("block_size" in e for e in r["reject_reason"]))

    def test_kv_smaller_than_block_rejected(self):
        r = self.m.analyze(32, 8, 32, 576, block_size=64)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("kv_len" in e for e in r["reject_reason"]))

    def test_head_dim_not_multiple_of_8_rejected(self):
        r = self.m.analyze(8192, 8, 32, 100)
        self.assertTrue(r.get("illegal"), r)
        self.assertTrue(any("head_dim" in e for e in r["reject_reason"]))

    def test_illegal_has_no_prediction(self):
        """illegal 配置不许硬给 predicted_util。"""
        r = self.m.analyze(8192, 8, 32, 576, block_size=0)
        self.assertNotIn("predicted_util", r)
        self.assertEqual(r["confidence"], "none")


class TestOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load(MODEL, "_ad")

    def test_model_limit_and_confidence(self):
        """每个预测必须带 model_limit + confidence，且写明反推来源。"""
        r = self.m.analyze(8192, 8, 32, 576)
        self.assertIn("model_limit", r)
        self.assertIn("经验式", r["model_limit"])
        self.assertIn(r["confidence"], ("low", "medium", "high"))
        self.assertIn("calibration", r)
        self.assertIn("anchors", r["calibration"])
        self.assertGreaterEqual(len(r["calibration"]["anchors"]), 2)

    def test_low_parallelism_hint(self):
        """grid < 64 时输出里含 low_parallelism 提示。

        heads=1, batch=1, kv=64, block=64 ⇒ grid=1 < 64。
        """
        r = self.m.analyze(64, 1, 1, 576, block_size=64)
        self.assertTrue(r["low_parallelism"])
        self.assertIn("low_parallelism", r)
        self.assertIn("并行", r["low_parallelism_note"])

    def test_high_parallelism_no_hint(self):
        """grid 足够大时不应误报 low_parallelism。"""
        r = self.m.analyze(8192, 32, 32, 576)
        self.assertFalse(r["low_parallelism"])


if __name__ == "__main__":
    unittest.main()
