#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate.py 的结构守卫：不许有重复定义。

2026-11-23 踩到：`_check_impl_to_ceiling` 与 `_check_pros_cons_substance`
各被定义两次（一次在插入新版时没删旧的）。Python 静默用后一个覆盖前一个，
于是：
  - E29 的新判据（扫 ceiling.note / modify_path.how）写进去了却不生效
  - E30 的白名单改动同样不生效
两个都是「代码看起来改了，实际没跑」。

这类 bug 无法靠读 diff 发现——两段代码都在文件里，语法合法。
只能用结构检查抓。
"""
import ast
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"


class TestValidatorStructure(unittest.TestCase):
    def test_no_duplicate_function_definitions(self):
        tree = ast.parse(VALIDATE.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        dup = [k for k, v in Counter(names).items() if v > 1]
        self.assertEqual(dup, [],
                         "validate.py 有重复定义的函数（后一个静默覆盖前一个）：%s" % dup)

    def test_critical_checks_defined_exactly_once(self):
        tree = ast.parse(VALIDATE.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        for fn in ("_check_impl_to_ceiling", "_check_pros_cons_substance",
                   "_check_evidence", "_check_baseline_ref", "_check_depth_rules"):
            self.assertEqual(names.count(fn), 1,
                             "%s 定义数 = %d（应为 1）" % (fn, names.count(fn)))

    def test_every_check_function_is_called(self):
        """定义了却没调用的检查函数 = 失效的门禁。"""
        src = VALIDATE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = [n.name for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("_check_")]
        uncalled = [n for n in names if src.count(n) <= 1]
        self.assertEqual(uncalled, [], "这些检查函数只定义没调用：%s" % uncalled)


if __name__ == "__main__":
    unittest.main()
