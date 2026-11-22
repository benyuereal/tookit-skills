#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算力峰值常数的自洽守卫。

背景（2026-11-23）
------------------
constants.yaml 的 bf16 峰值长期写着「反推 185 TF」（由 19.7 条 MFMA/cycle 推出）。
同机实测 hipBLASLt 大 GEMM 能到 **310 TF** —— 实测值高于声称的峰值。
一个读数超过「峰值」，只有两种可能：测得不对，或峰值常数错了。查下来是后者。

两个具体的错：
  1. 标定用的是 N=2048（0.097 ms）。这个形状固定开销占比大，根本不能代表峰值；
     同一次扫描里 N=6144 是 310 TF，N=2048 只有 178 TF。
  2. 由错峰值反解出的 19.7 条/cycle 被写回模型，又被当成分母去算别的东西，
     于是「达成率」全线高估 ~1.7×（MoE grouped GEMM 记成 0.43，实际 0.26）。

这与带宽那次是同一个坑：**分母用错了，结论方向就反了**
（1100 标称 vs 1724 实测；185 反推 vs 310 实测）。

本测试守三件事，都不需要显卡：
  A. constants.yaml 必须有实测口径的 bf16_gemm_peak_tf，且带 verified_on + source；
  B. 反解出的 mma_issue_peak 必须与「峰值 ÷ (8192 × sclk)」自洽（相对误差 < 5%）；
  C. 模型里的硬编码常量必须与 constants.yaml 一致，不许各写各的。
"""
import importlib.util
import io
import re
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOFLINE = REPO / "Plugins/ops/skills/op-roofline"
CONSTANTS = ROOFLINE / "constants.yaml"
MODELS = ROOFLINE / "models"
MFMA_FLOPS = 8192


def _load(text=None):
    """constants.yaml 只用到两级缩进 + 标量，自己解析，不依赖 pyyaml。"""
    import yaml
    return yaml.safe_load(CONSTANTS.read_text(encoding="utf-8"))


def _get(node, key, default=None):
    if not isinstance(node, dict):
        return default
    v = node.get(key, default)
    return v


def _scalar(node, key, default=None):
    v = _get(node, key, default)
    if isinstance(v, dict):
        v = v.get("value", default)
    return v


class TestPeakConstant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        cls.c = _load()

    def test_bf16_peak_is_measured_not_derived(self):
        """峰值必须标 status: measured，并带 verified_on 与 source。

        为什么卡这么死：185 那个值标的是 derived（由另一个 derived 量推出），
        链条一开始就错，后面全是自洽的错。实测值才能当分母。
        """
        node = _get(self.c.get("compute"), "bf16_gemm_peak_tf")
        self.assertIsNotNone(node, "constants.yaml 缺 compute.bf16_gemm_peak_tf（实测峰值）")
        self.assertEqual(node.get("status"), "measured",
                         "bf16_gemm_peak_tf 必须是 measured；derived 值不能当达成率分母")
        for k in ("source", "verified_on"):
            self.assertTrue(node.get(k), "bf16_gemm_peak_tf 缺 %s" % k)
        self.assertGreater(float(node["value"]), 0)

    def test_mma_issue_peak_consistent_with_peak(self):
        """反解的 MFMA/cycle 必须与「峰值 ÷ (8192 × sclk)」自洽。

        这条抓的是：有人改了峰值却忘了改发射率（或反之），
        两个数各说各话，模型和结论就会分叉。
        """
        peak = float(_scalar(self.c.get("compute"), "bf16_gemm_peak_tf"))
        issue = float(_scalar(self.c.get("compute"), "mma_issue_peak"))
        clk_mhz = _scalar(self.c.get("compute"), "bf16_gemm_peak_tf_at_sclk_mhz")
        if clk_mhz is None:
            clk_mhz = _get(self.c.get("compute"), "bf16_gemm_peak_tf").get(
                "at_sclk_mhz", 1200)
        expect = peak * 1e12 / (MFMA_FLOPS * float(clk_mhz) * 1e6)
        dev = abs(issue - expect) / expect
        self.assertLess(
            dev, 0.05,
            "mma_issue_peak=%.1f 与「%.0f TF ÷ (8192 × %s MHz) = %.1f」不符（偏差 %.1f%%）；"
            "改了一个必须同步改另一个" % (issue, peak, clk_mhz, expect, dev * 100))

    def test_model_constant_matches_constants_yaml(self):
        """models/sparse_mla.py 的 MMA_PEAK_PER_CYCLE 必须来自 constants.yaml。"""
        declared = float(_scalar(self.c.get("compute"), "mma_issue_peak"))
        path = MODELS / "sparse_mla.py"
        spec = importlib.util.spec_from_file_location("_peak_model", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with redirect_stderr(io.StringIO()):
            got = mod.MMA_PEAK_PER_CYCLE
        self.assertAlmostEqual(
            got, declared, places=6,
            msg="sparse_mla.MMA_PEAK_PER_CYCLE=%.2f 与 constants.yaml 的 %.2f 不一致"
                % (got, declared))

    def test_models_do_not_hardcode_stale_19_7(self):
        """扫全仓，19.7 这个已被否证的发射率不许再出现在**可执行代码**里。

        文档里作为「历史更正」出现是允许的（要说明为什么改），注释里同理；
        但 .py 的代码行里再拿它算东西就是回归。

        判据用 tokenize 而不是切 '#'：docstring 里的 19.7 不是注释，
        切 '#' 抓不到，会把本测试自己的说明文字也算成违规（第一版就是这样）。
        """
        import tokenize
        bad = []
        for path in REPO.glob("Plugins/**/*.py"):
            try:
                toks = list(tokenize.generate_tokens(
                    io.StringIO(path.read_text(encoding="utf-8", errors="ignore")).readline))
            except (tokenize.TokenError, IndentationError):        # pragma: no cover
                continue
            # 只保留 NUMBER token：19.7 作为数值字面量出现才算「当常数用」。
            # 字符串/注释/docstring 里的 19.7 是 prose，不是常量。
            for tok in toks:
                if tok.type == tokenize.NUMBER and tok.string == "19.7":
                    bad.append("%s:%d: %s" % (path.relative_to(REPO), tok.start[0],
                                              tok.line.strip()))
        self.assertEqual(bad, [], "以下位置仍把已否证的 19.7 条/cycle 当常数用：\n"
                         + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
