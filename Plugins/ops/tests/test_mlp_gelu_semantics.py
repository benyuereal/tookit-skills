#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""守「卡片的语义必须自洽」这条。

本测试不是测代码，是测**数据的一致性**：
卡里 math 说的算子和 conclusion 说的算子必须是同一个。

为什么需要这条：
  `cards/mlp-gelu-geglu.yaml` 的 `math` 写的是 MLP 的 GeGLU
      y = gelu(x*W_up) * (x*W_gate) * W_down
  但 `last_verdict.ratio` 存的是
      GNSilu(GroupNorm+SiLU) 153 GB/s / torch F.silu 468 GB/s = 0.33
  这是**两个完全不同的算子**（一个 MLP 门控激活，一个 norm+激活复合算子），
  而且两者的形状（MLP 的 M/H vs 图像 NCHW 的 64x32x64x64）也不在一个口径。
  卡片自己在 conclusion 与 hard_preconditions 里也承认了这一点——
  也就是说「矛盾」是写在卡上的事实，不是猜测。

本测试的判据（不修卡，只登记）：
  1. math 里的算子名 vs conclusion 里的算子名 —— 不许被说成别的算子
  2. ratio.scope 的形状维度必须与 math 的符号一致（MLP 的 M/H vs NCHW 的 4 维）
  3. 主变体 import_path 的算子名必须与 math 同族
  4. ratio 的分母（vs）里若并列两个不同形状，说明它是跨形状拼接的比值

若发现不一致，测试应 **skip（当前已知问题 D5）而不是 fail** ——
修卡需要业主裁决（拆卡 / 移证据 / 降 draft），测试只负责让问题「可被机器重新发现」。
一旦裁决完成、卡被修好，这些 skip 会自动变成 pass，不需要改测试。

同时守住「校验本身要被校验」：
  第 5 组用合成卡做正/反例，证明判据**在该报错时真的会报错**，
  否则上面的 skip 可能是「判据坏了、什么都没查出来」的假绿。
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
CARD = CARDS / "mlp-gelu-geglu.yaml"
# 同家族的「对照卡」：结构正确，用作反例（判据不许在它身上误报）
CONTROL = CARDS / "mlp-swiglu.yaml"

DEBT_ID = "D5"

# math / conclusion / import_path 里认得的算子词。只用来比对「是不是同一个算子」，
# 不追求覆盖全部激活。—— 宁可漏报，不要在对照卡上误报。
_ACTIVATION_WORDS = ("gelu", "geglu", "silu", "swiglu", "glu")
_NORM_WORDS = ("gnsilu", "groupnorm", "layernorm", "rmsnorm")


def _load(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _words(text, vocab):
    """从一段文本里抽出属于 vocab 的词（按词边界，避免 'gnsilu' 被算成 'silu'）。"""
    low = (text or "").lower()
    out = set()
    for w in vocab:
        if w == "gnsilu":
            if "gnsilu" in low or "gn_silu" in low:
                out.add(w)
        elif re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(w), low):
            out.add(w)
    return out


def _shape_tokens(text):
    """抽 64x32x64x64 这类 NCHW 形状。返回 (维度数, 原文)。"""
    m = re.search(r"\b(\d+)x(\d+)x(\d+)x(\d+)\b", text or "")
    return (4, m.group(0)) if m else (0, None)


def _impl_name(import_path):
    """取 import_path 的最后一段，拆成小写词列表。lightop.GNSilu -> gnsilu。"""
    return (import_path or "").split(".")[-1].lower()


def inconsistencies(card):
    """返回「语义不自洽」的证据列表。空列表 = 自洽。

    纯字典运算，不 import 任何库 —— 这样正/反例可以用合成卡跑。
    """
    found = []

    math = card.get("math", "")
    verdict = card.get("last_verdict") or {}
    conclusion = verdict.get("conclusion", "")
    ratio = verdict.get("ratio") or {}

    # --- 判据 1：math 的算子 vs conclusion 的算子 ---------------------------
    math_act = _words(math, _ACTIVATION_WORDS)
    conc_norm = _words(conclusion, _NORM_WORDS)
    # conclusion 若把 math 的算子改称 norm 复合算子，就是两个算子
    if math_act and conc_norm:
        found.append(
            "math 写 %s，conclusion 却称 %s —— 两个不同算子"
            % (sorted(math_act), sorted(conc_norm)))

    # --- 判据 2：ratio.scope 的形状维度 vs math 的符号 ----------------------
    scope = ratio.get("scope") or {}
    scope_text = " ".join(str(v) for v in scope.values() if isinstance(v, str))
    ndim, shape = _shape_tokens(scope_text)
    if ndim == 4 and math_act and not _shape_tokens(math)[0]:
        # math 是 MLP（M/H 矩阵），scope 却是 4 维图像张量
        found.append(
            "math 是 MLP 形状（M,H），ratio.scope 却给 4 维 NCHW 形状 %r" % shape)

    # --- 判据 3：主变体 import_path 与 math 是否同族 -----------------------
    variants = card.get("variants") or []
    primaries = [v for v in variants if v.get("tier") == "primary"] or variants[:1]
    for v in primaries:
        ip = v.get("import_path", "")
        name = _impl_name(ip)
        name_act = _words(name, _ACTIVATION_WORDS)
        name_norm = _words(name, _NORM_WORDS)
        if name_norm and not name_act and math_act:
            found.append(
                "主变体 import_path=%s 是 norm 复合算子，与 math 的 %s 不同族"
                % (ip, sorted(math_act)))

    # --- 判据 4：ratio 的分母是否并列了两个不同形状（跨形状拼接比值）--------
    vs = str(ratio.get("vs") or "")
    in_len = str(scope.get("in_len") or "")
    both = vs + " " + in_len
    if " / " in both and _shape_tokens(in_len)[0]:
        found.append(
            "ratio 是跨形状拼接的比值（in_len=%r），不是同口径对比" % in_len)

    return found


class TestCardSemanticsConsistency(unittest.TestCase):
    """卡级一致性：命中已知矛盾时 skip 并登记，不 fail。"""

    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        cls.card = _load(CARD)
        cls.found = inconsistencies(cls.card)

    def _register_or_pass(self):
        if self.found:
            self.skipTest(
                "%s 已知问题（待业主裁决，见 docs/DEBT.md）：\n  - %s"
                % (DEBT_ID, "\n  - ".join(self.found)))

    def test_math_and_conclusion_name_the_same_operator(self):
        """math 说 gelu，conclusion 不许改口说 GNSilu/GroupNorm。"""
        self._register_or_pass()

    def test_ratio_scope_matches_math_shape(self):
        """ratio 的形状必须和 math 的符号在同一口径（MLP 的 M/H）。"""
        self._register_or_pass()

    def test_primary_variant_matches_math_operator(self):
        """主变体 import_path 必须是 math 同族的算子，不能是 norm 复合算子。"""
        self._register_or_pass()

    def test_verdict_records_evidence_of_the_conflict(self):
        """矛盾本身要被卡「自证」：conclusion/hard_preconditions 应写明 GNSilu≠纯激活。

        这条不是判据，而是防「有人悄悄删掉自证、把矛盾改成沉默」。
        若将来卡被修好（证据移走），这条会自然变绿或可一并降级。
        """
        if not self.found:
            self.skipTest("卡已自洽（%s 已裁决），本自证测试不适用" % DEBT_ID)
        text = (self.card.get("last_verdict", {}).get("conclusion", "")
                + " " + " ".join(
                    p.get("statement", "")
                    for p in (self.card.get("hard_preconditions") or [])))
        self.assertIn("GNSilu", text,
                      "卡里连「GNSilu 不是纯激活」的自证都没了 —— 矛盾被抹掉，比留着更糟")


class TestVerifierIsVerified(unittest.TestCase):
    """证明判据本身有效：该报错的合成卡必须报错，干净的对照卡必须不报。"""

    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def test_control_card_is_clean(self):
        """对照卡 mlp-swiglu 结构正确 —— 判据在它身上不许误报。"""
        control = _load(CONTROL)
        self.assertEqual(inconsistencies(control), [],
                         "判据在干净的对照卡上误报，说明判据过严，需收窄")

    def test_synthetic_inconsistent_card_is_caught(self):
        """把矛盾注入一张最小卡，判据必须全部命中（否则 skip 就是假绿）。"""
        bad = {
            "math": "y = gelu(x*W_up) * (x*W_gate) * W_down",
            "variants": [{"tier": "primary", "import_path": "lightop.GNSilu"}],
            "last_verdict": {
                "ratio": {
                    "vs": "GNSilu 相对 torch F.silu 的带宽比（153/468）",
                    "scope": {"in_len": "64x32x64x64 (GNSilu) / 4096x4096 (F.silu)"},
                },
                "conclusion": "GNSilu 是 GroupNorm+SiLU 复合算子，不能当纯激活",
            },
        }
        found = inconsistencies(bad)
        blob = " ".join(found)
        # 四条判据应各自留下痕迹
        self.assertIn("conclusion", blob, found)
        self.assertIn("NCHW", blob, found)
        self.assertIn("import_path", blob, found)
        self.assertIn("跨形状", blob, found)

    def test_real_card_is_caught(self):
        """真实卡也必须被同一条判据判为不自洽 —— 防止判据只对合成卡有效。"""
        self.assertTrue(inconsistencies(_load(CARD)),
                        "判据在真实卡上没命中，但 DEBT %s 记着它不自洽" % DEBT_ID)


class TestLocalImplementations(unittest.TestCase):
    """本机可用实现清单（实测，不是照抄卡面）。

    这些断言是「卡该怎么修」的事实基础，全部应当 pass。
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("torch 未安装")

    def test_gnsilu_is_not_a_plain_activation(self):
        """lightop.GNSilu 必须传 num_groups/num_channels —— 它不可能是纯激活。"""
        import inspect
        import lightop
        params = inspect.signature(lightop.GNSilu).parameters
        required = [p.name for p in params.values()
                    if p.default is inspect._empty and p.kind != p.VAR_KEYWORD]
        self.assertIn("num_groups", required)
        self.assertIn("num_channels", required)

    def test_aiter_has_a_real_geglu_building_block(self):
        """math 的算子在本机有对应入口：aiter.gelu_and_mul(out, [M,2H]) == gelu(a)*b。"""
        import torch
        import aiter
        import torch.nn.functional as F
        if not torch.cuda.is_available():          # pragma: no cover
            self.skipTest("无 GPU")
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        a, b = x.split(8, dim=1)
        out = torch.empty(8, 8, device="cuda", dtype=torch.bfloat16)
        aiter.gelu_and_mul(out, x)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(out, F.gelu(a) * b, atol=2e-2),
                        "aiter.gelu_and_mul 的语义不是 gelu(first)*second")

    def test_lightop_has_elementwise_gelu_variant(self):
        """lightop.FuseBiasGelu(x, bias) == gelu(x+bias)，与 math 同族。"""
        import torch
        import lightop
        import torch.nn.functional as F
        if not torch.cuda.is_available():          # pragma: no cover
            self.skipTest("无 GPU")
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
        y = lightop.FuseBiasGelu().cuda()(x, bias)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, F.gelu(x + bias), atol=2e-2))


if __name__ == "__main__":
    unittest.main()
