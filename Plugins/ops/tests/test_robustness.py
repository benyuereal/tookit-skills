#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""鲁棒性检查（未见过的 shape + stress 数值）必须**能拦住东西**。

来源：CompoundingKernel（arXiv:2611.19230）的裁决是五类，
其中「鲁棒性」在**未提交的 shape + stress 数值**上做：
L2/079 通过了所有 shape，但 FP16 stress 输入溢出，
他们宁可从 1.62× 退到 0.48× 也不交。

我们的检查原来只覆盖提交的形状 —— 等于「考试题和练习题一样」。
业主的验收要求很明确：**至少有一例因此被判不通过；一例都没抓到说明检查没生效。**

所以本测试的核心不是「这个 kernel 对不对」，而是「这套门有没有效力」：
用 `--selftest-flaw` 注入一个已知有缺陷的变体，看门抓不抓得到。
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPCHECK = REPO / "scripts/opcheck.py"
CARD = REPO / "Plugins/ops/skills/op-fused-add-rmsnorm"


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, "-I", str(OPCHECK), "robustness"] + list(args),
                          capture_output=True, text=True, timeout=timeout)


class TestRobustnessGate(unittest.TestCase):
    def test_gate_catches_injected_flaw(self):
        """注入已知缺陷 → 两类检查都要抓到（这是门有生效的唯一证据）。"""
        r = _run("--contract", str(CARD / "contract.yaml"),
                 "--variants", "ref-heavy", "--shape", "decode-m1",
                 "--selftest-flaw")
        self.assertIn("__selftest_flaw__", r.stdout, "注入的缺陷变体没被跑到")
        caught = re.search(r"抓到 (\d+) 例不通过", r.stdout)
        self.assertIsNotNone(caught, r.stdout[-500:])
        self.assertGreater(int(caught.group(1)), 0,
                           "注入的缺陷一例都没抓到 —— 鲁棒性门没生效")
        # 两类都要有覆盖：A 类是 max_abs 超限，B 类是 inf/nan
        self.assertRegex(r.stdout, r"FAIL\s+H=\S+\s+__selftest_flaw__ max_abs=",
                         "A 类（未见 shape）没抓到")
        self.assertRegex(r.stdout, r"__selftest_flaw__ 输出含 inf/nan",
                         "B 类（stress 数值）没抓到")

    def test_unrunnable_variant_is_not_counted_as_catch(self):
        """「跑不起来」不许算抓到 —— 否则缺后端会被记成「检查有效」。

        第一版就是这样：aiter-norm 未接入，在 12 个用例里全部报错，
        被记成「抓到 12 例」——信号被淹掉，真实问题（0 例）反而看不见。
        """
        r = _run("--contract", str(CARD / "contract.yaml"),
                 "--variants", "ref-heavy,aiter-norm", "--shape", "decode-m1")
        self.assertIn("ERROR", r.stdout)
        self.assertIn("不计入抓到", r.stdout)
        m = re.search(r"抓到 (\d+) 例不通过", r.stdout)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 0,
                         "跑不起来的变体不该计入抓到；实得 %s 例" % m.group(1))

    def test_reference_overflow_is_skipped_not_caught(self):
        """参考实现自己也溢出 → SKIP，不许算抓到。

        CompoundingKernel 的「抓到」定义是「参考正常、被测异常」。
        如果参考也溢出，我们什么都没证明。
        """
        r = _run("--contract",
                 str(REPO / "Plugins/ops/skills/op-moe-topk-gating/contract.yaml"),
                 "--variants", "ref-heavy", "--shape", "decode-m1")
        if "参考实现自己也溢出" in r.stdout:
            self.assertRegex(r.stdout, r"SKIP\s+huge\s+参考实现自己也溢出")
            m = re.search(r"抓到 (\d+) 例不通过", r.stdout)
            self.assertIsNotNone(m)

    def test_unseen_shapes_are_really_unseen(self):
        """A 类用的形状必须不在卡片 shapes.yaml 里，否则等于没测。"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_opcheck_rt", str(OPCHECK))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(REPO / "scripts"))
        spec.loader.exec_module(mod)

        sys.path.insert(0, str(REPO / "scripts"))
        from opverify import parse_yaml
        known = parse_yaml((CARD / "shapes.yaml").read_text(encoding="utf-8")) or {}
        proto = dict((known[list(known)[0]] or {}).get("params") or {})
        self.assertTrue(proto, "测试前提：该卡应有已知形状")
        for sp in mod.UNSEEN_SHAPES:
            derived = mod._derive_unseen(proto, sp)
            if derived is None:
                continue
            key = tuple(sorted((k, v) for k, v in derived.items()
                               if isinstance(v, (int, float))))
            for kname, node in known.items():
                kp = tuple(sorted((k, v) for k, v in ((node or {}).get("params") or {}).items()
                                  if isinstance(v, (int, float))))
                self.assertNotEqual(key, kp,
                                    "A 类形状 %s 与卡片已知形状 %s 撞了" % (key, kname))

    def test_selftest_flaw_flag_exists(self):
        r = subprocess.run([sys.executable, "-I", str(OPCHECK), "robustness", "--help"],
                           capture_output=True, text=True)
        self.assertIn("--selftest-flaw", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
