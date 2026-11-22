#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-moe-topk-gating adapter 契约守卫。"""
import sys
import unittest
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))

REQUIRED = ("build_inputs", "reference", "runner_kwargs", "get_runner", "run_isolated")


class TestAdapterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        cls.c = importlib.import_module("_common")
        cls.shape = cls.c.load_shape("tiny")
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
        parts = ref if isinstance(ref, tuple) else (ref,)
        for part in parts:
            if hasattr(part, "isfinite"):
                self.assertTrue(bool(part.isfinite().all()), "参考输出含 NaN/Inf")

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
        for name in self.c.VARIANTS:
            if name == "ref-heavy":
                continue
            with self.assertRaises(RuntimeError):
                self.c.get_runner(name)

    def test_get_runner_rejects_unknown_variant(self):
        with self.assertRaises(ValueError):
            self.c.get_runner("no-such-variant")


if __name__ == "__main__":
    unittest.main()
