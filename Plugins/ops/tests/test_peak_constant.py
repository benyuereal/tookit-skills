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

    def test_every_compute_peak_declares_its_caliber(self):
        """每个算力常数必须声明口径：nominal / achievable / impl_measured。

        这是 2026-11-23 那次误判的根因，也是本仓库最贵的一类错误。
        当时把 aiter opus 的 fp8 实测 49.8 TF 当成了「fp8 峰值」，
        而它其实是**某一条实现**的值（impl_measured），
        achievable 应该是「最佳实现实测」，nominal 才是硬件规格。

        混用三者会让结论**方向反过来**：
          impl_measured 当 nominal → 把 11 倍空间判成「已到顶」→ 放弃优化（最危险）
          nominal 当 achievable   → 把「已到顶」判成「还有空间」→ 白花时间

        三层语义在 constants.yaml 的 compute 段开头有完整说明。
        """
        c = self.c.get("compute") or {}
        allowed = {"nominal", "achievable", "impl_measured"}
        bad = []
        for key, node in c.items():
            if not isinstance(node, dict):
                continue
            unit = str(node.get("unit") or "").upper()
            if "TFLOPS" not in unit:          # 只守算力类
                continue
            cal = node.get("caliber")
            if cal is None:
                bad.append("%s: 缺 caliber（必须标明 nominal/achievable/impl_measured）" % key)
            elif cal not in allowed:
                bad.append("%s: caliber=%r 非法（只允许 %s）" % (key, cal, sorted(allowed)))
        self.assertEqual(bad, [], "以下算力常数口径未标明：\n" + "\n".join(bad))

    def test_reproducible_only_where_it_makes_sense(self):
        """`reproduce:` 只对实测类（achievable / impl_measured）强制。

        nominal 峰值来自芯片规格，**本来就不可能复跑**——强行要求脚本，
        只会逼人伪造一个「复跑脚本」来骗过测试。
        这正是 540 那次的反面教材：它声称 measured，实际来源是不可执行的 kb 记录。

        所以规则是：
          achievable / impl_measured  → 必须带 reproduce，且文件必须存在
          nominal                     → 必须带 source 说明规格出处，不要求脚本
        """
        c = self.c.get("compute") or {}
        bad = []
        for key, node in c.items():
            if not isinstance(node, dict):
                continue
            if "TFLOPS" not in str(node.get("unit") or "").upper():
                continue
            cal = node.get("caliber")
            rep = node.get("reproduce")
            if cal in ("achievable", "impl_measured"):
                if not rep:
                    bad.append("%s(%s): 缺 reproduce（实测值必须能复跑）" % (key, cal))
                    continue
                path = str(rep).split("#")[0].strip()
                if not (REPO / path).is_file():
                    bad.append("%s(%s): reproduce 指向不存在的文件 %s" % (key, cal, path))
            elif cal == "nominal":
                if not node.get("source"):
                    bad.append("%s(nominal): 缺 source（规格出处必须写明）" % key)
        self.assertEqual(bad, [], "口径与可复跑性不匹配：\n" + "\n".join(bad))

    def test_native_fp8_judged_by_same_library_ratio(self):
        """判「是否原生 fp8」必须用**同库同形状**对照，这条要留证据。

        2026-11-23 的误判根因：拿 aiter 的 fp8（49.8 TF）去比 hipBLASLt 的 bf16（310 TF），
        得出「fp8 比 bf16 慢 5.3×」→ 进而误判「fp8 硬件不行」。
        但那比的是**两个库**，不是两种精度。

        正确的测法是同库同形状：aiter fp8 vs aiter bf16。
        实测耗时比 0.52~0.63（fp8 反而快 1.6~1.9×），
        ⇒ 本机**有**原生 fp8 路径，不是 dequant 回退。

        本测试守两件事：
          ① fp8_native_speedup 这个常数存在且 >1（证明原生路径存在）
          ② 常数的 note 里必须写明「同库同形状」，防止有人又拿跨库比
        """
        c = self.c.get("compute") or {}
        node = c.get("fp8_native_speedup")
        self.assertIsNotNone(
            node, "缺 fp8_native_speedup——没有它就无法判断本机 fp8 是原生还是 dequant 回退")
        val = float(_scalar(c, "fp8_native_speedup"))
        self.assertGreater(
            val, 1.0,
            "fp8_native_speedup=%.2f ≤1 说明同库对照下 fp8 没比 bf16 快；"
            "那才是真的 dequant 回退，需要重新定性" % val)
        note = str(node.get("note") or "") + str(node.get("caveat") or "") \
            + str(node.get("source") or "")
        self.assertIn("同库", note,
                      "fp8_native_speedup 的说明里必须点明「同库同形状」——"
                      "跨库比较会把库差距记成精度差距，这正是 11-23 误判的根因")

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


class TestBandwidthModes(unittest.TestCase):
    """E18 必须接受 bw-ladder 的四档，而不是只认单一常数。

    2026-11-23：1724 被替换成实测四档（纯读 1965 / 纯写 1692 / 1:1 1680 / 2:1 1730）。
    单一 stream_read_peak 只代表纯读，但 R2W2 的算子该用 1:1 档。
    E18 早期只认单一常数，会把正确的分档引用判成错——这就是「判据本身没被校验」。

    本测试守两头：
      A. 四档都能通过
      B. 档位之外的值（1100 标称 / 1724 作废值）仍被拦
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_val_bw", str(REPO / "Plugins/ops/op-atlas/validate.py"))
        cls.v = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.v)
        except SystemExit:                                   # pragma: no cover
            raise unittest.SkipTest("validate.py 导入失败")

    def test_four_modes_read_from_ladder(self):
        modes = self.v._bandwidth_modes(REPO)
        self.assertEqual(len(modes), 4, "bw-ladder 应有四档，实得 %s" % sorted(modes))
        for m in modes:
            self.assertGreater(m, 1500, "档位值异常低：%s" % m)

    def test_stream_read_peak_matches_ladder_read_only(self):
        import yaml
        c = yaml.safe_load((REPO / "Plugins/ops/skills/op-roofline/constants.yaml")
                           .read_text(encoding="utf-8"))
        peak = c["memory"]["stream_read_peak"]["value"]
        modes = self.v._bandwidth_modes(REPO)
        self.assertIn(float(peak), modes,
                      "constants 的 stream_read_peak=%s 必须是 bw-ladder 的一档" % peak)
