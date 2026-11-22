#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 映射粒度守卫：import_path 必须指向**可调用对象**。

为什么需要这条：L1 定义是「从算子名精确落到 库.函数」。
但最初的核查用的是 `hasattr(模块, 名字)` —— 它只能确认「这个名字存在」，
**不能确认「它是函数」**。

实测抓到 6 处模块级映射（4 张卡）：
  lightop.moe_align_sum           → 模块，真实函数是 .moe_sum
  lightop.fuse_rmsnorm_rope       → 模块，真实函数是 .rms_rotary_embedding_fuse
  deepgemm.group_gemm_weight_pack → 模块，真实函数是 .marlin_bf16_masked_weight
  flash_kda.cp                    → 模块，真实函数是 .build_cp_layout_cpu

拿模块名当函数名，读者照着调会得到 "module is not a callable object"。
"""
import importlib
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[3]
CARDS = REPO / "Plugins/ops/op-atlas/cards"


def _resolve(ip):
    """逐层解析 import_path，返回 (是否可调用, 解析到的类型名)。"""
    parts = ip.split(".")
    for k in range(len(parts), 0, -1):
        mod = ".".join(parts[:k - 1])
        attr = parts[k - 1]
        try:
            m = importlib.import_module(mod) if mod else None
        except Exception:
            continue
        if m is not None and hasattr(m, attr):
            o = getattr(m, attr)
            return callable(o), type(o).__name__
    return None, "unresolvable"


class TestImportPathCallable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        import yaml
        cls.entries = []
        for p in sorted(CARDS.glob("*.yaml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            for i in d.get("impls") or []:
                if i.get("import_path"):
                    cls.entries.append((p.stem, i["import_path"]))

    def test_every_import_path_is_callable(self):
        not_callable, unresolvable = [], []
        for card, ip in self.entries:
            ok, kind = _resolve(ip)
            if ok is None:
                unresolvable.append("%s -> %s" % (card, ip))
            elif not ok:
                not_callable.append("%s -> %s（是 %s，不是函数）" % (card, ip, kind))
        self.assertEqual(
            not_callable, [],
            "以下 import_path 指向模块/类属性而非可调用函数：\n" + "\n".join(not_callable))
        self.assertEqual(
            unresolvable, [],
            "以下 import_path 无法解析（库未装或拼写错）：\n" + "\n".join(unresolvable))


if __name__ == "__main__":
    unittest.main()
