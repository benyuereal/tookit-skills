#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-rmsnorm-fused adapter 契约守卫。"""
import sys
import unittest
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))

REQUIRED = ("build_inputs", "reference", "runner_kwargs", "get_runner", "run_isolated")

# 当前确实没有 adapter 的变体。VARIANTS 里新增名字时必须同步这里，
# 否则 test_all_declared_variants_resolvable 会红。
# 2026-11-23 现状：ref-heavy / lightop-fused / fused-3pass-triton /
# fused-3pass-tilelang 都已实现；torch-native 仍是空壳。
UNIMPLEMENTED = {"torch-native"}


class TestAdapterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 用文件路径加载，不用模块名：三张卡都有 _common.py，
        # `import _common` 会命中 sys.modules 里第一个加载的那个，
        # 导致后跑的卡实际在测前一张卡的 adapter。
        import importlib.util
        path = CARD / "_common.py"
        spec = importlib.util.spec_from_file_location(
            "_common_%s" % CARD.name.replace("-", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        cls.c = mod
        cls.shape = mod.load_shape("tiny")
        cls.params = dict(cls.shape.get("params") or {})

    def test_required_symbols_present(self):
        missing = [s for s in REQUIRED if not callable(getattr(self.c, s, None))]
        self.assertEqual(missing, [], "adapter 缺必需符号：%s" % missing)

    def test_build_inputs_returns_tuple_plus_meta(self):
        out = self.c.build_inputs(self.params, seed=0, device="cpu")
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[-1], dict, "最后一项必须是 meta dict")

    def test_reference_runs_and_is_finite(self):
        inp = self.c.build_inputs(self.params, seed=0, device="cpu")
        ref = self.c.reference(*inp[:-1], params=self.params)
        # reference() 按 semantics.outputs 返回**两份**（out, residual_out）。
        # 测试原先按单输出写，参考改成双输出后没跟着改——这是"改了一处没改另一处"。
        outs = ref if isinstance(ref, (tuple, list)) else (ref,)
        for i, t in enumerate(outs):
            self.assertTrue(bool(t.isfinite().all()),
                            "参考输出第 %d 份含 NaN/Inf" % i)

    def test_build_inputs_is_deterministic(self):
        a = self.c.build_inputs(self.params, seed=7, device="cpu")
        b = self.c.build_inputs(self.params, seed=7, device="cpu")
        self.assertTrue(bool((a[0] == b[0]).all()), "同 seed 输入不一致")

    def test_reference_runner_matches_reference(self):
        inp = self.c.build_inputs(self.params, seed=0, device="cpu")
        ref = self.c.reference(*inp[:-1], params=self.params)
        run = self.c.get_runner("ref-heavy", **self.c.runner_kwargs(self.params))
        got = run(*inp[:-1], **self.c.runner_kwargs(self.params))
        if isinstance(ref, tuple):
            for r, g in zip(ref, got):
                self.assertTrue(bool((r == g).all()), "ref-heavy 与 reference 不一致")
        else:
            self.assertTrue(bool((ref == got).all()), "ref-heavy 与 reference 不一致")

    def test_get_runner_refuses_without_backend(self):
        """确实没有 adapter 的变体必须明确拒绝，而不是返回假实现。

        2026-11-23 这条测试红了：它断言「除 ref-heavy 外全部抛 RuntimeError」，
        但 lightop-fused / fused-3pass-tilelang 已经实现，不再抛。
        改成「只对未实现的变体断言拒绝」——保留测试意图，不删测试。
        """
        for name in self.c.VARIANTS:
            if name not in UNIMPLEMENTED:
                continue
            with self.assertRaises(RuntimeError, msg="%s 应拒绝" % name):
                self.c.get_runner(name)

    def test_all_declared_variants_resolvable(self):
        """这才是真正该守的：声明的变体要么能返回 callable，要么在 UNIMPLEMENTED 白名单里。

        白名单外抛 RuntimeError = adapter 与 VARIANTS 脱节（新增了名字但没实现，
        或者实现坏了被异常吞掉）。两种都该红。
        """
        for name in self.c.VARIANTS:
            try:
                fn = self.c.get_runner(name)
            except RuntimeError:
                self.assertIn(name, UNIMPLEMENTED,
                              "变体 %s 抛了 RuntimeError 但不在 UNIMPLEMENTED 里——"
                              "要么补实现，要么加进白名单" % name)
                continue
            self.assertTrue(callable(fn), "%s 返回的不是 callable" % name)

    def test_get_runner_rejects_unknown_variant(self):
        with self.assertRaises(ValueError):
            self.c.get_runner("no-such-variant")


if __name__ == "__main__":
    unittest.main()
