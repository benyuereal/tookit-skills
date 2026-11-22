#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-gemm-fp8 adapter 契约守卫。

adapter 是 opcheck 与 bench 的唯一数学来源，它坏了所有门都会失去意义。
四条断言：
  1. 契约要求的五个符号都在（缺一个 opcheck 直接 rc=2）
  2. build_inputs 返回 (inp..., meta)，且 meta 是 dict
  3. reference 能被 build_inputs 的输出直接喂通，且输出形状正确、数值有限
  4. 没有后端时 get_runner 必须抛——不许返回假实现
"""
import sys
import unittest
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))

REQUIRED = ("build_inputs", "reference", "runner_kwargs", "get_runner", "run_isolated")

TINY = {"M": 4, "N": 256, "K": 256, "_name": "tiny"}


class TestAdapterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 用文件路径加载，不用模块名：多张卡都有 _common.py，
        # `import _common` 会命中 sys.modules 里第一个加载的那个。
        import importlib.util
        path = CARD / "_common.py"
        spec = importlib.util.spec_from_file_location(
            "_common_%s" % CARD.name.replace("-", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        cls.c = mod

    def test_required_symbols_present(self):
        missing = [s for s in REQUIRED if not callable(getattr(self.c, s, None))]
        self.assertEqual(missing, [], "adapter 缺必需符号：%s" % missing)

    def test_build_inputs_returns_tuple_plus_meta(self):
        out = self.c.build_inputs(TINY, seed=0, device="cpu")
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[-1], dict, "最后一项必须是 meta dict")
        self.assertEqual(len(out) - 1, len(self.c.INPUT_NAMES))

    def test_reference_runs_and_is_finite(self):
        inp = self.c.build_inputs(TINY, seed=0, device="cpu")
        ref = self.c.reference(*inp[:-1], params=TINY)
        self.assertEqual(tuple(ref.shape), (TINY["M"], TINY["N"]))
        self.assertTrue(bool(ref.isfinite().all()), "参考输出含 NaN/Inf")

    def test_build_inputs_is_deterministic(self):
        a = self.c.build_inputs(TINY, seed=7, device="cpu")
        b = self.c.build_inputs(TINY, seed=7, device="cpu")
        self.assertTrue(bool((a[0] == b[0]).all()), "同 seed 输入不一致")

    def test_get_runner_refuses_without_backend(self):
        """宁可红，也不许静默通过——没有后端就必须抛。"""
        with self.assertRaises(RuntimeError):
            self.c.get_runner("tilelang")

    def test_get_runner_rejects_unknown_variant(self):
        with self.assertRaises(ValueError):
            self.c.get_runner("no-such-variant")

    def test_unavailable_variant_is_absent_from_registry(self):
        """hipblaslt 在 gfx938 上无 FP8 方案，不该出现在可选变体里。"""
        self.assertNotIn("hipblaslt", self.c.VARIANTS)

    def test_run_isolated_returns_poison_list(self):
        inp = self.c.build_inputs(TINY, seed=0, device="cpu")
        fn = lambda *args: args[0] + 1
        out, poisoned = self.c.run_isolated(fn, *inp[:-1])
        self.assertIsInstance(poisoned, list)
        self.assertEqual(poisoned, [], "未修改输入的实现不该被报为改了输入")


if __name__ == "__main__":
    unittest.main()
