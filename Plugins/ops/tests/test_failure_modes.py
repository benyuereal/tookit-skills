#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""失败模式库守卫（阶段 5）。

为什么单独一个库：`candidates.yaml:refuted` 记的是「某个改动没收益」，
这里记的是「这条路根本走不通」。两者的处置完全不同：
  没收益 → 换个方向再试
  走不通 → 别在这上面花时间，改基础设施或换实现

三条断言：
  1. ≥6 条，且 ≥3 条 class: hard
  2. hard 类每条必须有可执行的 check
  3. 每条必须有 scope（不写环境，结论无法复用）
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
LIB = REPO / "Plugins/ops/skills/op-moe-w8a8/references/failure-modes.yaml"

# check 必须含可执行锚点（命令 / 函数调用 / 报错串）
EXECUTABLE = re.compile(r"python3|scripts/|grep |assert |→|Error|ValueError|NameError|RuntimeError")


class TestFailureModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        cls.d = yaml.safe_load(LIB.read_text(encoding="utf-8")) or {}
        cls.modes = cls.d.get("failure_modes") or []

    def test_counts(self):
        self.assertGreaterEqual(len(self.modes), 6, "失败模式少于 6 条")
        hard = [m for m in self.modes if m.get("class") == "hard"]
        self.assertGreaterEqual(len(hard), 3, "hard 类少于 3 条")

    def test_hard_modes_have_executable_check(self):
        bad = []
        for m in self.modes:
            if m.get("class") != "hard":
                continue
            chk = str(m.get("check") or "")
            if not chk:
                bad.append("%s: hard 类缺 check" % m.get("id"))
            elif not EXECUTABLE.search(chk):
                bad.append("%s: check 无可执行锚点 %r" % (m.get("id"), chk[:50]))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_every_mode_has_scope(self):
        bad = [m.get("id") for m in self.modes if not str(m.get("scope") or "").strip()]
        self.assertEqual(bad, [], "这些条目缺 scope（结论无法复用）：%s" % bad)

    def test_ids_unique(self):
        ids = [m.get("id") for m in self.modes]
        self.assertEqual(len(ids), len(set(ids)), "id 有重复")


if __name__ == "__main__":
    unittest.main()
