#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反作弊四件套的测试。每道检查都要有正反例。

背景（v5.1 §八）：公开审计里两个真实作弊案例
  KDA 分布感知  —— 只在被检查时切精确数学，计时时用近似
  MinGRU 覆盖   —— 按 num_envs 分派：检查的形状走精确，bench 的形状走快

共同点：agent 没变强，它学会了骗评测。且「高级工作 + 一处作弊」，人眼难看出来。
这些测试的作用是：**作弊模式必须能被机检抓到**，不是靠人看。
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_anticheat", str(REPO / "scripts/anticheat.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Vec(list):
    """让 *3 / *0.01 / 取负 的语义与 tensor 一致。

    纯 list 的 `[1,2]*3` 是复制而不是数值乘 —— 用它做测试会自伤。
    """
    def __mul__(self, k):
        return Vec(v * k for v in self)

    __rmul__ = __mul__

    def __neg__(self):
        return Vec(-v for v in self)


class TestShapeSeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_disjoint_passes(self):
        r = self.m.check_shape_separation(["s1024", "s2048"], ["s8192"])
        self.assertTrue(r["ok"], r)

    def test_overlap_fails_and_names_it(self):
        r = self.m.check_shape_separation(["s1024", "s8192"], ["s8192", "s4096"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["overlap"], ["s8192"],
                         "必须具体指出哪个形状交集，否则无从修")

    def test_empty_side_fails(self):
        """单边为空不能判 pass —— 没有 bench 形状等于没有分离。"""
        r = self.m.check_shape_separation(["s1"], [])
        self.assertFalse(r["ok"])


class TestAstScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_clean_kernel(self):
        r = self.m.scan_forbidden("def k(x):\n    return x + 1\n")
        self.assertTrue(r["ok"], r)

    def test_shape_dispatch_caught(self):
        """MinGRU 那类：按 num_envs / n 分派。"""
        src = "def k(n):\n    if n <= 2048:\n        return fast(n)\n    return slow(n)\n"
        r = self.m.scan_forbidden(src)
        self.assertFalse(r["ok"])
        self.assertTrue(r["hits"])
        self.assertEqual(r["hits"][0]["line"], 2)

    def test_magic_number_caught(self):
        src = "def k(M):\n    if M == 8192:\n        return approx(M)\n"
        r = self.m.scan_forbidden(src)
        self.assertFalse(r["ok"])

    def test_stat_branch_caught(self):
        src = "def k(x):\n    if x.mean() > 0.5:\n        return approx(x)\n"
        r = self.m.scan_forbidden(src)
        self.assertFalse(r["ok"])

    def test_output_declares_its_limitation(self):
        """AST 结果必须自带局限说明。

        这条是防止人对 AST 产生假信心：它只能抓字面量分支，
        更隐蔽的统计量分派（先算 hash 再分派）抓不到。
        """
        r = self.m.scan_forbidden("def k(x):\n    return x\n")
        self.assertIn("limit", r)
        self.assertIn("没作弊", r["limit"],
                      "必须明确写出「不许用 AST 结果当没作弊的证据」")

    def test_comment_is_ignored(self):
        """注释里写着作弊代码不算。"""
        src = "def k(x):\n    # if n <= 2048: fast()\n    return x\n"
        r = self.m.scan_forbidden(src)
        self.assertTrue(r["ok"], r)


class TestPerturbation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_good_impl_passes_all_transforms(self):
        f = lambda t: Vec(v * 2 for v in t)                   # noqa: E731
        r = self.m.make_perturbations(f, [Vec([1.0, 2.0])], f)
        self.assertTrue(r["ok"], r)
        self.assertEqual(len(r["probes"]), 3)

    def test_stat_dispatch_caught(self):
        """核心用例：按输入统计量分派的实现必须被抓。

        AST 抓不到这种（它要先算 mean 才知道走哪条分支），
        只有把输入分布打散才能暴露。
        """
        def cheat(t):
            m = sum(t) / len(t)
            return Vec(v * 2 if m < 2.0 else v for v in t)     # 均值大时偷工

        def ref(t):
            return Vec(v * 2 for v in t)

        r = self.m.make_perturbations(cheat, [Vec([1.0, 2.0])], ref)
        self.assertFalse(r["ok"], "按统计量分派必须被抓")
        failed = [p["transform"] for p in r["probes"] if not p["ok"]]
        self.assertTrue(failed, "必须指出哪档变换失败")

    def test_exception_becomes_failure_not_skip(self):
        """实现抛异常要算失败，不能静默跳过。"""
        def boom(t):
            raise RuntimeError("boom")
        r = self.m.make_perturbations(boom, [Vec([1.0])], boom)
        self.assertFalse(r["ok"])
        self.assertIn("异常", r["probes"][0]["detail"])


class TestInputIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_good_impl_passes_ab(self):
        run = lambda inp: Vec(v + 1 for v in inp)             # noqa: E731
        r = self.m.check_input_isolation(run, run, Vec([1.0]), Vec([2.0]))
        self.assertTrue(r["ok"], r)

    def test_cache_cheat_caught(self):
        """**最关键的一条**：缓存 kernel 必须被抓。

        v5.1 修正过这条的语义：
          错：连调两次断言逐位相同 —— 缓存作弊恰恰让两次相同
          对：换输入再测 —— 缓存返回 A 的结果，对不上 ref(B)
        这个测试就钉住「换输入」这一半。
        """
        cache = {}

        def run_cache(inp):
            if "k" in cache:
                return cache["k"]
            cache["k"] = Vec(v + 1 for v in inp)
            return cache["k"]

        def ref(inp):
            return Vec(v + 1 for v in inp)

        r = self.m.check_input_isolation(run_cache, ref, Vec([1.0]), Vec([9.0]))
        self.assertFalse(r["ok"], "缓存 kernel 必须被抓")
        self.assertIn("缓存", r.get("diagnosis", ""),
                      "诊断必须指出这是缓存/分派作弊")

    def test_atomics_not_penalized(self):
        """atomics 的合法非确定性不能被 bitexact 断言误伤。"""
        run = lambda inp: Vec(v + 1 for v in inp)             # noqa: E731
        r = self.m.check_input_isolation(run, run, Vec([1.0]), Vec([2.0]),
                                         source="atomicAdd(x, 1)")
        self.assertTrue(r["ok"])
        self.assertTrue(any(c.get("skipped") for c in r["checks"]),
                        "atomics kernel 的 bitexact 检查应标 skipped")


class TestFileHash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_hash_stable(self):
        import tempfile
        p = Path(tempfile.mkdtemp()) / "k.txt"
        p.write_text("v1", encoding="utf-8")
        h = self.m.sha256_of(p)
        self.assertTrue(self.m.check_file_hash_stable(p, h)["ok"])

    def test_hash_change_detected(self):
        import tempfile
        p = Path(tempfile.mkdtemp()) / "k.txt"
        p.write_text("v1", encoding="utf-8")
        h = self.m.sha256_of(p)
        p.write_text("v2", encoding="utf-8")
        r = self.m.check_file_hash_stable(p, h)
        self.assertFalse(r["ok"])
        self.assertIn("改过", r["reason"])


class TestSelftest(unittest.TestCase):
    def test_selftest_passes(self):
        self.assertEqual(_load().selftest(), 0)


if __name__ == "__main__":
    unittest.main()
