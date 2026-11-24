#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISA surface 的可用性判据。

背景（v5.1 §四）：`isa-policy` 这个 class 原先只有 2 条，且**都是死路**
（slc-policy / prefetch-depth-isa）——一个 class 里全是死路 = 这个 class
在「可攻击面」意义上是空的，而 bound-to-class.yaml 在三档里都把它列为可试。

本测试守四件事：
  A. 4 条新 surface 的 class / stage 合法
  B. 每条都有**可执行**的 precondition_check（不是"永久失败"，也不是空话）
  C. 每条 evidence 都写明「静态证据，动态占比未知」（防止把静态计数当动态权重）
  D. 反向：census 文档里不许再出现 `library_gfx938` 这个**不存在**的路径
     （允许出现在"它不存在"的说明里）
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ISA_SURF = REPO / "data/ops/isa-surfaces.yaml"
CENSUS = REPO / "Plugins/ops/skills/op-hip-compile/references/isa-census.md"

REQUIRED_IDS = {"vmem-vector-width", "mls-path", "lds-read-form", "epilogue-expansion"}


class TestIsaSurfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        if not ISA_SURF.is_file():
            raise unittest.SkipTest("isa-surfaces.yaml 尚未建立")
        cls.d = yaml.safe_load(ISA_SURF.read_text(encoding="utf-8")) or {}
        cls.surfs = cls.d.get("surfaces") or []

    def test_four_required_surfaces_exist(self):
        got = {s.get("id") for s in self.surfs}
        self.assertEqual(got, REQUIRED_IDS,
                         "ISA surface 缺失：%s" % (REQUIRED_IDS - got))

    def test_class_and_stage_legal(self):
        for s in self.surfs:
            self.assertIn(s.get("class"), ("isa-policy", "pipeline"),
                          "%s 的 class 非法" % s.get("id"))
            self.assertIn(s.get("stage"), ("S3", "S4", "S5"),
                          "%s 的 stage 非法" % s.get("id"))

    def test_precondition_check_is_executable_ish(self):
        """每种 check 必须含一个真实工具名。

        「永久失败」和「默认通过」是同一类错误，只是方向相反 —— 这条守前者。
        """
        tools = ("dccobjdump", "grep", "hipcc", "llvm-readelf")
        for s in self.surfs:
            c = str(s.get("precondition_check") or "")
            self.assertTrue(c.strip(), "%s 缺 precondition_check" % s.get("id"))
            self.assertTrue(any(t in c for t in tools),
                            "%s 的 precondition_check 里没有可执行工具：%r"
                            % (s.get("id"), c))

    def test_every_evidence_marks_static_limits(self):
        """每条 evidence 必须写明静态证据 / 动态占比未知。

        这是 v5.1 第四轮评审的第 4 条缺口：静态计数 ≠ 动态权重。
        没有这句，agent 会把 dwordx2 的静态计数当成「能省 2×」。
        """
        for s in self.surfs:
            ev = str(s.get("evidence") or "")
            self.assertIn("静态", ev,
                          "%s 的 evidence 没标「静态证据」——会被当成动态权重"
                          % s.get("id"))

    def test_expected_gain_is_bounded_not_promised(self):
        """expected_gain 必须是上界表述（含「上界」或「需 PMC 定级」）。"""
        for s in self.surfs:
            g = str(s.get("expected_gain") or "")
            self.assertTrue("上界" in g or "需 PMC" in g,
                            "%s 的 expected_gain 没标上界：%r" % (s.get("id"), g))

    def test_census_does_not_cite_nonexistent_path(self):
        """反向：census 里 `library_gfx938` 只能出现在「它不存在」的说明里。

        原文档把 `library_gfx938` 当真实路径引用，而该目录不存在
        （实测为 library_bw_nmz）——这是「判据本身记错了」的又一个例子。
        """
        if not CENSUS.is_file():
            self.skipTest("census 不存在")
        txt = CENSUS.read_text(encoding="utf-8")
        for line in txt.splitlines():
            if "library_gfx938" in line:
                ok = ("不存在" in line or "不在" in line or "更正" in line)
                self.assertTrue(ok,
                                "census 把不存在的路径当真实引用：%r" % line.strip())

    def test_discipline_section_present(self):
        """文件级纪律段必须存在——它是四条 surface 的共同免责。"""
        disc = str(self.d.get("discipline") or "")
        self.assertIn("静态", disc)
        self.assertIn("PMC", disc)
        print("\n纪律段：%s" % disc[:80])


if __name__ == "__main__":
    unittest.main()
