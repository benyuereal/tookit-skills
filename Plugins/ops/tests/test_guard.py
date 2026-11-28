#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engine/guard.py 的接受标准（PLAN-OPTIMIZE 优化项 C）。

agent 试图自己跑 bench 时，必须是**代码级拒绝**，不能只靠 prompt 劝。
只读查看（ls/cat/sed/grep）必须放行 —— 否则连"看看有哪些文件"都被挡。
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "_guard_mod", str(REPO / "Plugins/ops/engine/guard.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = _load_guard()

    def _rejects(self, cmd):
        with self.assertRaises(self.g.GuardViolation):
            self.g.guard_no_bench(cmd)

    def _allows(self, cmd):
        self.g.guard_no_bench(cmd)      # 不抛就是放行

    def test_rejects_running_ab_py(self):
        self._rejects("python3 Plugins/ops/skills/op-x/bench/ab.py --help")
        self._rejects(["python3", "Plugins/ops/skills/op-x/bench/ab.py"])
        self._rejects("cd card && python bench/ab.py --variants v1")
        self._rejects("python3 .../bench/acc_probe.py --json out.json")

    def test_rejects_executing_anything_under_bench(self):
        self._rejects("bash -c 'python3 Plugins/ops/skills/op-x/bench/bench.py'")
        self._rejects("hipcc .../bench/foo.cu -o /tmp/x")

    def test_allows_readonly_inspection(self):
        self._allows("ls bench/")
        self._allows("cat Plugins/ops/skills/op-x/bench/ab.py")
        self._allows("sed -n 1,40p bench/ab.py")
        self._allows("grep -n variants bench/ab.py")

    def test_allows_engine_loop(self):
        self._allows("python3 -I Plugins/ops/engine/loop.py -C card run tag")
        self._allows("python3 -I Plugins/ops/engine/orchestrator.py --selfcheck")
