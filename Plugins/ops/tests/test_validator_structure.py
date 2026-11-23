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


class TestSemanticsListIntegrity(unittest.TestCase):
    """semantics 的列表项不许被逗号切碎。

    2026-11-23 实测：YAML 里写 `outputs: [y (B,M,N)]` 没加引号，
    任何 round-trip（yaml.dump / 自动 formatter）都会把它按逗号拆成
    ['y (B', 'M', 'N)'] —— 直接触发 E36「outputs 数与 writes 不符」，
    而错误信息会指向「输出份数不对」，与真实原因（YAML 引号）无关。

    判据：列表项不允许以 '(' 结尾或含未配对的括号。
    """
    CARDS = REPO / "Plugins/ops/op-atlas/cards"

    def test_list_items_not_split_by_comma(self):
        import yaml
        bad = []
        for p in sorted(self.CARDS.glob("*.yaml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            sem = d.get("semantics") or {}
            for key in ("inputs", "outputs"):
                for item in sem.get(key) or []:
                    if not isinstance(item, str):
                        continue
                    opens, closes = item.count("("), item.count(")")
                    if opens != closes:
                        bad.append("%s %s: %r（括号不配对，多半是逗号切碎）"
                                   % (p.stem, key, item))
                    elif item.rstrip().endswith(","):
                        bad.append("%s %s: %r（以逗号结尾）" % (p.stem, key, item))
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
